import gymnasium as gym
import minigrid
import numpy as np
import torch
import torch.nn.functional as F
import argparse
import threading
from pathlib import Path
from tqdm.auto import trange
from collections import defaultdict

from minigrid.wrappers import *
from minigrid.envs.babyai import *

from mixture_of_experts import MixtureOfExpertsPolicy, reset_hidden_on_done, compute_lpc


class ReplanTimeout(Exception):
    """Raised when bot.replan() takes too long."""
    pass


def replan_with_timeout(bot, prev_action, timeout_seconds=5):
    """Call bot.replan() with a wall-clock timeout using threading."""

    result = [None]
    exception = [None]

    def target():
        try:
            result[0] = bot.replan(prev_action)
        except Exception as e:
            exception[0] = e

    thread = threading.Thread(target=target)
    thread.start()
    thread.join(timeout=timeout_seconds)

    if thread.is_alive():
        # Thread still running - timeout occurred
        # Note: thread will continue in background but we ignore its result
        raise ReplanTimeout("bot.replan() timed out")

    if exception[0] is not None:
        raise exception[0]

    return result[0]


class EvalVectorEnv:
    """
    Simplified vectorized environment for evaluation.
    Similar to VectorBabyAIEnv but without expert bots and with position tracking.
    """

    def __init__(self, task_id: str, num_envs: int, device: torch.device, max_steps: int = None, lang_dim: int = 32):
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

        self.lang_proj = nn.Linear(768, lang_dim).to(device)
        for p in self.lang_proj.parameters():
            p.requires_grad = False

        self.lang_embs = torch.zeros(num_envs, lang_dim, device=device)
        self.max_steps = max_steps

        # Metrics
        self.episode_steps = np.zeros(num_envs, dtype=int)
        self.total_episodes = 0
        self.successful_episodes = 0
        self.sum_path_ratio = 0.0
        self.num_successful_with_ratio = 0
        self.bot_plan_failures = 0

        # Create environments
        for i in range(num_envs):
            if self.max_steps is not None:
                env = gym.make(task_id, max_steps=self.max_steps)
            else:
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

    def load_lang_proj(self, state_dict):
        """Load lang_proj weights from checkpoint and re-encode all missions."""
        self.lang_proj.load_state_dict(state_dict)
        # Re-encode all missions with the correct lang_proj weights
        for i, mission in enumerate(self.missions):
            self.lang_embs[i] = self._encode_mission(mission)

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
            a = replan_with_timeout(bot, None, timeout_seconds=5)
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

    def _extract_env_context(self, env) -> dict:
        """Extract positions of all relevant objects from the environment grid.

        Returns a dict with:
            - grid_size: (width, height)
            - agent_start_pos: (x, y) agent position at episode start
            - agent_start_room: (top_x, top_y) room top-left corner (multi-room envs only)
            - room_grid_shape: (num_cols, num_rows) room grid layout (multi-room envs only)
            - goals: list of (x, y, color) for Goal objects
            - doors: list of (x, y, color, is_open, is_locked)
            - keys: list of (x, y, color)
            - balls: list of (x, y, color)
            - boxes: list of (x, y, color)
        """
        context = {
            'grid_size': (env.grid.width, env.grid.height),
            'agent_start_pos': tuple(int(c) for c in env.agent_pos),
            'goals': [],
            'doors': [],
            'keys': [],
            'balls': [],
            'boxes': [],
        }

        # Detect room info if this is a multi-room environment (e.g. BabyAI RoomGrid)
        if hasattr(env, 'room_from_pos'):
            try:
                room = env.room_from_pos(*env.agent_pos)
                context['agent_start_room'] = tuple(int(c) for c in room.top)
                if hasattr(env, 'room_grid'):
                    context['room_grid_shape'] = (len(env.room_grid[0]), len(env.room_grid))
            except Exception:
                pass

        for j in range(env.grid.height):
            for i in range(env.grid.width):
                cell = env.grid.get(i, j)
                if cell is None:
                    continue

                obj_type = cell.type
                color = getattr(cell, 'color', None)

                if obj_type == 'goal':
                    context['goals'].append((i, j, color))
                elif obj_type == 'door':
                    is_open = getattr(cell, 'is_open', False)
                    is_locked = getattr(cell, 'is_locked', False)
                    context['doors'].append((i, j, color, is_open, is_locked))
                elif obj_type == 'key':
                    context['keys'].append((i, j, color))
                elif obj_type == 'ball':
                    context['balls'].append((i, j, color))
                elif obj_type == 'box':
                    context['boxes'].append((i, j, color))

        return context

    def get_env_contexts(self):
        """Get environment context for each environment."""
        return [self._extract_env_context(env) for env in self.envs]

    def _reset_single_env(self, i: int, env, max_retries=10):
        """Reset a single environment, retrying if the expert bot can't plan.

        Each failure increments self.bot_plan_failures. Returns the observation.
        """
        for attempt in range(max_retries):
            obs, _ = env.reset()

            mission = obs["mission"]
            obs = obs.copy()
            obs.pop('mission', None)
            obs = self._add_carrying_flag(env, obs)

            try:
                expert_steps = self._compute_optimal_steps(env)
            except Exception as e:
                self.bot_plan_failures += 1
                print(f"[WARNING] Bot failed to plan on fresh {self.task_id} "
                      f"(env {i}, attempt {attempt+1}/{max_retries}): {e}")
                continue

            self.missions[i] = mission
            self.lang_embs[i] = self._encode_mission(mission)
            self.episode_steps[i] = 0
            self.expert_steps[i] = expert_steps
            return obs

        raise RuntimeError(
            f"Bot failed to plan on {max_retries} consecutive fresh resets "
            f"of {self.task_id} (env {i}). Environment may be broken."
        )

    def reset(self):
        """Reset all environments."""
        self.obs_list = [None] * self.num_envs
        self.expert_steps = np.zeros(self.num_envs, dtype=int)

        for i, env in enumerate(self.envs):
            self.obs_list[i] = self._reset_single_env(i, env)

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
                obs = self._reset_single_env(i, env)

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
    """Load checkpoint and reconstruct policy.

    Returns:
        policy: MixtureOfExpertsPolicy with loaded weights
        config: dict with training config
        lang_proj_state_dict: state dict for lang_proj layer (or None if not saved)
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint['config']

    # Reconstruct policy from config
    policy = MixtureOfExpertsPolicy(
        input_dim=config['input_dim'],
        intermediate_dim=config.get('intermediate_dim', 256),
        expert_hidden_sizes=config.get('expert_hidden_sizes', [32, 64, 128]),
        router_hidden_size=config.get('router_hidden_size', 64),
        num_actions=config.get('num_actions', 7),  # fallback for old checkpoints
        lang_dim=config.get('lang_dim', 32)
    ).to(device)

    policy.load_state_dict(checkpoint['policy_state_dict'])
    policy.eval()

    # Extract lang_proj weights if saved (for backwards compatibility)
    lang_proj_state_dict = checkpoint.get('lang_proj_state_dict', None)

    return policy, config, lang_proj_state_dict


def evaluate(policy, vec_env, num_episodes, device):
    """
    Run evaluation episodes and collect routing data.

    Returns:
        metrics: dict with success_rate, path_ratio, mean_lpc
        routing_data: list of dicts with keys:
            position, layer_routing, lpc, env_context, carrying, action_logits
            where env_context contains grid_size, goals, doors, keys, balls, boxes
    """
    policy.eval()
    num_envs = vec_env.num_envs

    # Initialize hidden states
    h = policy.init_hidden(num_envs, device)

    # Storage for routing data: (position, routing_weights per layer, lpc, env_context)
    routing_data = []

    # Run until we have enough episodes
    vec_env.reset()
    episodes_completed = 0

    # Capture environment contexts (object positions) for each environment
    # These are captured at episode start and updated when environments reset
    env_contexts = vec_env.get_env_contexts()

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

            # Compute per-sample LPC for this batch (unused, kept for potential future use)
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

                # Include environment context, carrying state, and action logits
                carrying = int(vec_env.obs_list[i].get('carrying_flag', 0))
                routing_data.append({
                    'position': pos,
                    'layer_routing': layer_routing,
                    'lpc': sample_lpc,
                    'env_context': env_contexts[i],
                    'carrying': carrying,
                    'action_logits': logits[i].cpu().numpy(),
                })

            # Sample actions (greedy for eval)
            actions = logits.argmax(dim=-1).cpu().numpy()

            # Step environments
            _, terminated, truncated = vec_env.step(actions)
            episode_over = terminated | truncated

            # Reset hidden states for finished episodes
            dones_tensor = torch.from_numpy(episode_over).to(device)
            h = reset_hidden_on_done(h, dones_tensor)

            # Update environment contexts for environments that just reset
            # (vec_env.step() internally resets environments when episode ends)
            for i in range(num_envs):
                if episode_over[i]:
                    env_contexts[i] = vec_env._extract_env_context(vec_env.envs[i])

            # Count completed episodes
            new_episodes = episode_over.sum()
            episodes_completed += new_episodes
            pbar.update(new_episodes)

        pbar.close()

    # Compute metrics
    success_rate = vec_env.successful_episodes / vec_env.total_episodes if vec_env.total_episodes > 0 else 0.0
    path_ratio = vec_env.sum_path_ratio / vec_env.num_successful_with_ratio if vec_env.num_successful_with_ratio > 0 else float('nan')
    mean_lpc = sum(rd['lpc'] for rd in routing_data) / len(routing_data) if routing_data else 0.0

    metrics = {
        'success_rate': success_rate,
        'path_ratio': path_ratio,
        'mean_lpc': mean_lpc,
        'total_episodes': vec_env.total_episodes,
        'successful_episodes': vec_env.successful_episodes,
        'bot_plan_failures': vec_env.bot_plan_failures,
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
    parser.add_argument('--max_steps', type=int, default=None,
                        help='Max steps per episode (default: use training config, or env default)')

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load checkpoint
    print(f"Loading checkpoint: {args.checkpoint}")
    policy, config, lang_proj_state_dict = load_checkpoint(args.checkpoint, device)

    # Use training task if not specified
    task_id = args.task_id or config['task_id']
    print(f"Evaluating on task: {task_id}")

    # Create evaluation environment (CLI --max_steps overrides training config)
    max_steps = args.max_steps if args.max_steps is not None else config.get('max_steps')
    vec_env = EvalVectorEnv(task_id, args.num_envs, device, max_steps=max_steps, lang_dim=config.get('lang_dim', 32))

    # Load lang_proj weights if available
    if lang_proj_state_dict is not None:
        vec_env.load_lang_proj(lang_proj_state_dict)
        print("Loaded lang_proj weights from checkpoint")
    else:
        print("Warning: No lang_proj weights in checkpoint, using random initialization")

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
    print(f"Bot plan failures: {metrics['bot_plan_failures']}")
    print(f"Routing samples collected: {len(routing_data)}")
    print("="*50)

    # Return data for further analysis (when called as module)
    return metrics, routing_data


if __name__ == "__main__":
    main()
