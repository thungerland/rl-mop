import gymnasium as gym
import minigrid
import numpy as np
import torch
import torch.nn.functional as F
import argparse
from pathlib import Path
from tqdm.auto import trange
from collections import defaultdict

from minigrid.wrappers import *
from minigrid.envs.babyai import *

from mixture_of_experts import MixtureOfExpertsPolicy, reset_hidden_on_done, compute_lpc


class EvalVectorEnv:
    """
    Simplified vectorized environment for evaluation.
    Similar to VectorBabyAIEnv but without expert bots and with position tracking.
    """

    def __init__(self, task_id: str, num_envs: int, device: torch.device):
        from transformers import AutoTokenizer, AutoModel
        import torch.nn as nn

        self.task_id = task_id
        self.num_envs = num_envs
        self.device = device

        self.envs = []
        self.obs_list = []
        self.missions = []

        # Language encoding (same as training)
        self.tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
        self.lang_encoder = AutoModel.from_pretrained("distilbert-base-uncased").to(self.device)
        self.lang_encoder.eval()
        for p in self.lang_encoder.parameters():
            p.requires_grad = False

        self.lang_proj = nn.Linear(768, 128).to(device)
        for p in self.lang_proj.parameters():
            p.requires_grad = False

        self.lang_embs = torch.zeros(num_envs, 128, device=device)

        # Metrics
        self.episode_steps = np.zeros(num_envs, dtype=int)
        self.total_episodes = 0
        self.successful_episodes = 0
        self.sum_path_ratio = 0.0
        self.num_successful_with_ratio = 0

        # Create environments
        for i in range(num_envs):
            env = gym.make(task_id)
            env = env.unwrapped
            obs, _ = env.reset()

            mission = obs["mission"]
            self.lang_embs[i] = self._encode_mission(mission)

            obs = obs.copy()
            obs.pop('mission', None)
            obs = self._add_carrying_flag(env, obs)

            self.envs.append(env)
            self.obs_list.append(obs)
            self.missions.append(mission)

        self.action_space = self.envs[0].action_space

    def _add_carrying_flag(self, env, obs: dict) -> dict:
        base = getattr(env, "unwrapped", env)
        obs["carrying_flag"] = 0 if getattr(base, "carrying", None) is None else 1
        return obs

    def _encode_mission(self, mission: str) -> torch.Tensor:
        inputs = self.tokenizer(
            [mission],
            padding=True,
            truncation=True,
            return_tensors="pt"
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            out = self.lang_encoder(**inputs)

        emb = out.last_hidden_state[:, 0]
        emb = self.lang_proj(emb)
        return emb.squeeze(0).detach()

    def _compute_optimal_steps(self, env, max_steps=500):
        """Compute optimal steps using BabyAIBot on a cloned environment."""
        import copy
        from minigrid.utils.baby_ai_bot import BabyAIBot

        env_clone = copy.deepcopy(env)
        bot = BabyAIBot(env_clone)

        steps = 0
        done = False
        truncated = False

        while not done and not truncated and steps < max_steps:
            a = bot.replan(None)
            _, _, done, truncated, _ = env_clone.step(a.value)
            steps += 1

        return steps

    def get_agent_positions(self):
        """Get current agent position for each environment."""
        positions = []
        for env in self.envs:
            pos = tuple(env.agent_pos)
            positions.append(pos)
        return positions

    def reset(self):
        """Reset all environments."""
        self.obs_list = []
        self.expert_steps = np.zeros(self.num_envs, dtype=int)

        for i, env in enumerate(self.envs):
            obs, _ = env.reset()

            mission = obs["mission"]
            self.missions[i] = mission
            self.lang_embs[i] = self._encode_mission(mission)

            obs = obs.copy()
            obs.pop('mission', None)
            obs = self._add_carrying_flag(env, obs)

            self.obs_list.append(obs)
            self.episode_steps[i] = 0
            self.expert_steps[i] = self._compute_optimal_steps(env)

        return list(self.obs_list)

    def step(self, actions):
        """Step all environments."""
        assert len(actions) == self.num_envs
        next_obs_list = []
        terminated = np.zeros(self.num_envs, dtype=bool)
        truncated = np.zeros(self.num_envs, dtype=bool)

        for i, env in enumerate(self.envs):
            action = int(actions[i])
            obs, reward, term, trunc, info = env.step(action)

            self.episode_steps[i] += 1
            terminated[i] = term
            truncated[i] = trunc

            obs = obs.copy()
            obs.pop('mission', None)
            obs = self._add_carrying_flag(env, obs)

            episode_over = term or trunc

            if episode_over:
                self.total_episodes += 1
                if term:
                    self.successful_episodes += 1
                    if self.expert_steps[i] > 0:
                        ratio = self.episode_steps[i] / self.expert_steps[i]
                        self.sum_path_ratio += ratio
                        self.num_successful_with_ratio += 1

                # Reset this environment
                obs, _ = env.reset()
                mission = obs["mission"]
                self.missions[i] = mission
                self.lang_embs[i] = self._encode_mission(mission)

                obs = obs.copy()
                obs.pop('mission', None)
                obs = self._add_carrying_flag(env, obs)

                self.episode_steps[i] = 0
                self.expert_steps[i] = self._compute_optimal_steps(env)

            next_obs_list.append(obs)

        self.obs_list = next_obs_list
        return next_obs_list, terminated, truncated


def encode_obs_batch(obs_list, device):
    """Encode observations to tensor (same as training)."""
    imgs = []
    dirs = []
    carry = []

    for obs in obs_list:
        img = torch.tensor(obs["image"], dtype=torch.float32, device=device)
        img = img.permute(2, 0, 1).reshape(-1)
        imgs.append(img)
        dirs.append(int(obs["direction"]))
        carry.append(int(obs.get("carrying_flag", 0)))

    imgs = torch.stack(imgs, dim=0)
    dirs = torch.tensor(dirs, dtype=torch.long, device=device)
    dirs_onehot = F.one_hot(dirs, num_classes=4).float()
    carry = torch.tensor(carry, dtype=torch.float32, device=device).unsqueeze(-1)

    x = torch.cat([imgs, dirs_onehot, carry], dim=-1)
    return x


def load_checkpoint(checkpoint_path, device):
    """Load checkpoint and reconstruct policy."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint['config']

    # Reconstruct policy from config
    policy = MixtureOfExpertsPolicy(
        input_dim=config['input_dim'],
        intermediate_dim=config.get('intermediate_dim', 256),
        expert_hidden_sizes=config.get('expert_hidden_sizes', [32, 64, 128]),
        router_hidden_size=config.get('router_hidden_size', 64),
        num_actions=7,  # BabyAI default
        lang_dim=config.get('lang_dim', 128)
    ).to(device)

    policy.load_state_dict(checkpoint['policy_state_dict'])
    policy.eval()

    return policy, config


def evaluate(policy, vec_env, num_episodes, device):
    """
    Run evaluation episodes and collect routing data.

    Returns:
        metrics: dict with success_rate, path_ratio, mean_lpc
        routing_data: list of (position, routing_weights, lpc) tuples
    """
    policy.eval()
    num_envs = vec_env.num_envs

    # Initialize hidden states
    h = policy.init_hidden(num_envs, device)

    # Storage for routing data: (position, routing_weights per layer)
    routing_data = []

    # Run until we have enough episodes
    vec_env.reset()
    episodes_completed = 0

    with torch.no_grad():
        pbar = trange(num_episodes, desc="Evaluating")
        while episodes_completed < num_episodes:
            # Get current positions before stepping
            positions = vec_env.get_agent_positions()

            # Encode observations
            obs_batch = encode_obs_batch(vec_env.obs_list, device)
            lang_embs = vec_env.lang_embs

            # Forward pass with routing info
            logits, h, routing_info = policy(obs_batch, lang_embs, h, return_routing_info=True)

            # Compute per-sample LPC for this batch
            batch_lpc = compute_lpc(routing_info, policy.layer_expert_sizes)

            # Collect routing data for each environment
            for i in range(num_envs):
                pos = positions[i]
                # Collect routing weights from all layers
                layer_routing = {}
                for layer_name, layer_info in routing_info.items():
                    weights = layer_info['router_weights'][i].cpu().numpy()
                    layer_routing[layer_name] = weights

                # Compute per-sample LPC for this specific environment
                sample_lpc = 0.0
                for layer_idx, expert_sizes in enumerate(policy.layer_expert_sizes):
                    layer_key = f'layer_{layer_idx}'
                    weights = routing_info[layer_key]['router_weights'][i]  # (num_experts,)
                    sizes_squared = torch.tensor(
                        [s ** 2 for s in expert_sizes],
                        dtype=weights.dtype,
                        device=weights.device
                    )
                    sample_lpc += (weights * sizes_squared).sum().item()

                routing_data.append((pos, layer_routing, sample_lpc))

            # Sample actions (greedy for eval)
            actions = logits.argmax(dim=-1).cpu().numpy()

            # Step environments
            _, terminated, truncated = vec_env.step(actions)
            episode_over = terminated | truncated

            # Reset hidden states for finished episodes
            dones_tensor = torch.from_numpy(episode_over).to(device)
            h = reset_hidden_on_done(h, dones_tensor)

            # Count completed episodes
            new_episodes = episode_over.sum()
            episodes_completed += new_episodes
            pbar.update(new_episodes)

        pbar.close()

    # Compute metrics
    success_rate = vec_env.successful_episodes / vec_env.total_episodes if vec_env.total_episodes > 0 else 0.0
    path_ratio = vec_env.sum_path_ratio / vec_env.num_successful_with_ratio if vec_env.num_successful_with_ratio > 0 else float('nan')
    mean_lpc = sum(rd[2] for rd in routing_data) / len(routing_data) if routing_data else 0.0

    metrics = {
        'success_rate': success_rate,
        'path_ratio': path_ratio,
        'mean_lpc': mean_lpc,
        'total_episodes': vec_env.total_episodes,
        'successful_episodes': vec_env.successful_episodes,
    }

    return metrics, routing_data


def main():
    parser = argparse.ArgumentParser(description='Evaluate MoE Policy')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to checkpoint file')
    parser.add_argument('--task_id', type=str, default=None,
                        help='Task to evaluate on (default: same as training)')
    parser.add_argument('--num_envs', type=int, default=16,
                        help='Number of parallel environments')
    parser.add_argument('--num_episodes', type=int, default=100,
                        help='Number of episodes to evaluate')

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load checkpoint
    print(f"Loading checkpoint: {args.checkpoint}")
    policy, config = load_checkpoint(args.checkpoint, device)

    # Use training task if not specified
    task_id = args.task_id or config['task_id']
    print(f"Evaluating on task: {task_id}")

    # Create evaluation environment
    vec_env = EvalVectorEnv(task_id, args.num_envs, device)

    # Run evaluation
    metrics, routing_data = evaluate(policy, vec_env, args.num_episodes, device)

    # Print results
    print("\n" + "="*50)
    print("EVALUATION RESULTS")
    print("="*50)
    print(f"Task: {task_id}")
    print(f"Episodes: {metrics['total_episodes']}")
    print(f"Success Rate: {metrics['success_rate']:.2%}")
    print(f"Path Ratio: {metrics['path_ratio']:.2f}")
    print(f"Mean LPC: {metrics['mean_lpc']:.2f}")
    print(f"Routing samples collected: {len(routing_data)}")
    print("="*50)

    # Return data for further analysis (when called as module)
    return metrics, routing_data


if __name__ == "__main__":
    main()
