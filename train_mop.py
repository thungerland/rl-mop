import gymnasium as gym
import minigrid
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import copy
import wandb
import random
import argparse
import yaml
import threading
from pathlib import Path

from torch.utils.data import Dataset, DataLoader, random_split
from tqdm.auto import trange
from minigrid.wrappers import *
from minigrid.envs.babyai import *
from minigrid.utils.baby_ai_bot import BabyAIBot
from PIL import Image
from collections import deque

# Mixture of Experts import
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


class VectorBabyAIEnv:
    """
    Simple N-parallel wrapper around a BabyAI task.

    - All envs use the same task id (e.g. "BabyAI-GoToRedBall-v0")
    - Keeps its own list of BabyAIBot experts, one per env
    - Auto-resets envs when they finish
    """

    def __init__(self, task_id: str, num_envs: int, device: torch.device, max_steps: int = None, lang_dim: int = 32):

        from transformers import AutoTokenizer, AutoModel
        self.task_id = task_id
        self.num_envs = num_envs
        self.device = device
        self.max_steps = max_steps  # None = use environment default

        self.envs = []
        self.bots = []
        self.obs_list = [] # list of observation dicts with the 'image', 'direction', 'mission' keys
        self.missions = [] # We want to store the language separately from numerical observations

        # Store the last action executed in each env (None at episode start)
        # This will be relevant as the BabyAIBot must replan according to the action taken under the GRU policy
        self.previous_actions = [None] * num_envs

        # Flag to track envs where bot.replan() failed (agent stuck in unreachable state)
        self.force_reset = [False] * num_envs

        # Language encoding
        self.tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
        self.lang_encoder = AutoModel.from_pretrained("distilbert-base-uncased").to(self.device)
        self.lang_encoder.eval()

        # Logging objects
        self.episode_steps = np.zeros(num_envs, dtype=int)
        self.episode_success = np.zeros(num_envs, dtype=bool)
        self.expert_steps = np.zeros(num_envs, dtype=int) # bot steps
        self.total_episodes = 0
        self.successful_episodes = 0
        self.recent_successes = deque(maxlen=50)
        self.sum_path_ratio = 0.0
        self.num_successful_with_ratio = 0
        self.recent_path_ratios = deque(maxlen=50) # metric to track improvements in planning efficiency
        self.bot_plan_failures = 0  # count of fresh-env resets where bot failed to plan

        for p in self.lang_encoder.parameters():
            p.requires_grad = False

        self.lang_proj = nn.Linear(768, lang_dim).to(device)
        for p in self.lang_proj.parameters():
            p.requires_grad = False

        # Initialise cache for language embeddings
        self.lang_embs = torch.zeros(num_envs, lang_dim, device=device)

        # Create N envs + bots + initial states
        for i in range(num_envs):
            if self.max_steps is not None:
                env = gym.make(task_id, max_steps=self.max_steps)
            else:
                env = gym.make(task_id)
            env = env.unwrapped
    
            obs, _ = env.reset() # ignore info from .reset() using _
            mission = obs["mission"] # extract langauge modality
            self.lang_embs[i] = self._encode_mission(mission)
            obs = obs.copy() # defensive to not mutate the environment
            # Remove mission key from obs
            obs.pop('mission', None)
            # Add flag to inform agent whether or not they are carrying an object (e.g. key, box etc.)
            obs = self._add_carrying_flag(env, obs)
    
            # Create a bot linked to each of the N envs 
            bot = BabyAIBot(env)
    
            self.envs.append(env)
            self.bots.append(bot)
            self.obs_list.append(obs)
            self.missions.append(mission)
    
        # Store action space dim 
        self.action_space = self.envs[0].action_space

    def _add_carrying_flag(self, env, obs: dict) -> dict:
        # In BabyAI, env.carrying is None or an object instance
        base = getattr(env, "unwrapped", env)
        # Add a carrying_flag key to the observation dict
        obs["carrying_flag"] = 0 if getattr(base, "carrying", None) is None else 1
        return obs

    def _encode_mission(self, mission: str) -> torch.Tensor:
        inputs = self.tokenizer(
            [mission],
            padding=True,
            truncation=True,
            return_tensors="pt"
        )
        inputs = {k: v.to(self.lang_embs.device) for k, v in inputs.items()}
    
        with torch.no_grad():
            out = self.lang_encoder(**inputs)
    
        emb = out.last_hidden_state[:, 0]  # (1, 768)
        emb = self.lang_proj(emb)          # (1, 128)
        return emb.squeeze(0).detach()

    # Helper function to compute the optimal no. of steps (the no. of steps the bot would have taken)
    # for completing a task.
    def compute_expert_steps_from_clone(self, env, max_steps=500):
        # We need to clone the environment since otherwise if we step on it, the agent can no longer solve the env from its initial configuration
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

    def _reset_single_env(self, i, max_retries=10):
        """Reset env i, ensuring the expert bot can plan from the fresh state.

        If the bot fails, re-reset (up to max_retries) until we get a solvable
        instance.  Each failure increments self.bot_plan_failures so the count
        is visible in wandb.
        """
        env = self.envs[i]

        for attempt in range(max_retries):
            obs, _ = env.reset()
            mission = obs["mission"]
            obs = obs.copy()
            obs.pop('mission', None)
            obs = self._add_carrying_flag(env, obs)

            try:
                expert_steps = self.compute_expert_steps_from_clone(env)
            except Exception as e:
                self.bot_plan_failures += 1
                print(f"[WARNING] Bot failed to plan on fresh {self.task_id} "
                      f"(env {i}, attempt {attempt+1}/{max_retries}): {e}")
                continue

            # Success — update all state for this env slot
            self.obs_list[i] = obs
            self.missions[i] = mission
            self.lang_embs[i] = self._encode_mission(mission)
            self.bots[i] = BabyAIBot(env)
            self.expert_steps[i] = expert_steps
            self.episode_steps[i] = 0
            self.episode_success[i] = False
            self.previous_actions[i] = None
            return obs

        raise RuntimeError(
            f"Bot failed to plan on {max_retries} consecutive fresh resets "
            f"of {self.task_id} (env {i}). Environment may be broken."
        )

    def reset(self):
        """
        Reset all envs and return list of observations.
        Also re-creates the bots so they are synced with the new envs.
        """
        self.obs_list = [None] * self.num_envs
        self.previous_actions = [None] * self.num_envs

        for i in range(self.num_envs):
            self._reset_single_env(i)

        return list(self.obs_list)

    def step(self, actions):
        """
        Step all envs in parallel.

        actions: iterable of length num_envs with ints in the action space.

        Returns:
            next_obs_list: list of obs dicts (length num_envs)
            terminated: np.array shape [num_envs]  (env finished successfully at this step)
            terminated: np.array shape [num_envs]  (env finished unsuccessfully at this step e.g. timeout/ out of bounds)
        """
        assert len(actions) == self.num_envs
        next_obs_list = []
        terminated = np.zeros(self.num_envs, dtype=bool)
        truncated = np.zeros(self.num_envs, dtype=bool)

        for i, env in enumerate(self.envs):

            # Handle forced reset (agent stuck in unreachable state)
            if self.force_reset[i]:
                self.force_reset[i] = False
                # Treat as truncation for metrics
                self.total_episodes += 1
                self.recent_successes.append(0)

                self._reset_single_env(i)

                next_obs_list.append(self.obs_list[i])
                terminated[i] = False
                truncated[i] = True
                continue

            # Step the env with the provided action
            action = int(actions[i])
            obs, reward, term, trunc, info = env.step(action)

            # Increment agent step count
            self.episode_steps[i] += 1

            episode_over = term or trunc
            terminated[i] = term
            truncated[i]  = trunc
            
            obs = obs.copy()
            obs.pop('mission', None)
            obs = self._add_carrying_flag(env, obs)

            # Log success/failure
            if episode_over:
                self.total_episodes += 1
                if term:
                    self.successful_episodes += 1
                    self.episode_success[i] = True

                    # recent success tracking
                    self.recent_successes.append(1)

                    # increment path ratio metric
                    ratio = self.episode_steps[i] / self.expert_steps[i]
                    self.sum_path_ratio += ratio
                    self.num_successful_with_ratio += 1
                    self.recent_path_ratios.append(ratio)

                else: # truncation
                    self.episode_success[i] = False

                    # recent success tracking
                    self.recent_successes.append(0)

                # Reset the i-th environment and its corresponding bot, if it has terminated
                self._reset_single_env(i)
                obs = self.obs_list[i]
            else:
                self.previous_actions[i] = action

            # Update the observation for each environment which will be the state reached after action or a reset start state 
            next_obs_list.append(obs)

        self.obs_list = next_obs_list
        
        return next_obs_list, terminated, truncated

    def get_expert_actions(self):
        """
        Query BabyAIBot for each env's current observation/state.

        Returns:
            expert_actions: list[int] of length num_envs
        """
        actions = []
        for i, bot in enumerate(self.bots):
            prev = self.previous_actions[i]
            try:
                a_opt = replan_with_timeout(bot, prev, timeout_seconds=5)
                actions.append(a_opt.value)
            except (ReplanTimeout, AssertionError, Exception):
                # Bot can't find path or timed out - agent is stuck
                # Mark for forced reset and return dummy action
                self.force_reset[i] = True
                actions.append(0)  # dummy action, env will be reset before stepping

        return actions

def encode_obs_batch(obs_list, device):
    """
    obs_list: list of length N, each obs is a dict with the keys 'image', 'direction'

    Returns:
        x: tensor (N, obs_dim_without_language)
    """
    # Storage for images and directions as before
    # Instead of storing each timestep of an episode, we now only store the current observation but for each env
    imgs = []
    dirs = []
    carry = []

    for obs in obs_list:
        img = torch.tensor(obs["image"], dtype=torch.float32, device=device) # (7,7,3)
        img = img.permute(2,0,1) # C*H*W in case we want to use a CNN embedding later on (3,7,7)
        img = img.reshape(-1)  # (3,7,7) -> (3*7*7,) 
        imgs.append(img)

        dirs.append(int(obs["direction"]))
        carry.append(int(obs.get("carrying_flag", 0)))

    imgs = torch.stack(imgs, dim=0) # (N, 3*7*7)
    
    dirs = torch.tensor(dirs, dtype=torch.long, device=device) # (N, )
    dirs_onehot = F.one_hot(dirs, num_classes=4).float() # (N, 4)
    
    carry = torch.tensor(carry, dtype=torch.float32, device=device).unsqueeze(-1)  # (N, 1)

    x = torch.cat([imgs, dirs_onehot, carry], dim=-1) # (N, input_dim)
    return x.to(device) 
    

def detach_hidden(h):
    """
    Detach hidden states from computation graph.

    Args:
        h: List of (h_router, h_experts) tuples, one per layer

    Returns:
        Detached hidden states with same structure
    """
    h_detached = []
    for h_router, h_experts in h:
        h_router_detached = h_router.detach()
        h_experts_detached = [h_exp.detach() for h_exp in h_experts]
        h_detached.append((h_router_detached, h_experts_detached))
    return h_detached


def train_unroll_moe(policy, optimizer, vec_env, h, unroll_len, device, lpc_alpha=0.0):
    """
    One truncated-backprop through time (BPTT) unroll for MoE policy.

    Args:
        policy: MixtureOfExpertsPolicy
        optimizer: torch optimizer
        vec_env: VectorBabyAIEnv
        h: List of (h_router, h_experts) tuples, one per layer
        unroll_len: int
        device: "cuda" or "cpu"
        lpc_alpha: float, weight for LPC regularization (0.0 = disabled)

    Returns:
        h_new: updated hidden states after unroll (detached from graph)
        avg_loss: scalar (float)
        avg_acc: scalar (float)
        avg_lpc: scalar (float), average LPC over the unroll
    """
    policy.train()
    num_envs = vec_env.num_envs
    use_lpc = lpc_alpha > 0.0

    total_loss = 0.0
    total_lpc = 0.0
    total_correct = 0
    total_count = 0

    # Truncated BPTT: detach hidden states
    h = detach_hidden(h)

    for t in range(unroll_len):
        # Encode current obs
        obs_batch = encode_obs_batch(vec_env.obs_list, device)  # (N, input_dim)

        # Get expert actions as labels for this timestep
        expert_actions = vec_env.get_expert_actions()  # list of length N
        expert_actions = torch.tensor(expert_actions, dtype=torch.long, device=device)

        # Forward MoE policy for one step
        lang_embs = vec_env.lang_embs
        logits, h, routing_info = policy(obs_batch, lang_embs, h, return_routing_info=use_lpc)

        # Compute the supervised loss between policy and expert
        step_loss = F.cross_entropy(logits, expert_actions)

        # Compute LPC if enabled
        if use_lpc:
            step_lpc = compute_lpc(routing_info, policy.layer_expert_sizes)
            total_lpc += step_lpc.item()
            step_loss = step_loss + lpc_alpha * step_lpc

        total_loss += step_loss

        # Compute the accuracy
        with torch.no_grad():
            preds = logits.argmax(dim=-1)  # (N,)
            correct = (preds == expert_actions).sum().item()
            total_correct += correct
            total_count += num_envs

        # Extract the student actions
        with torch.no_grad():
            student_actions = preds.detach().cpu().numpy()

        # Step the envs according to student actions
        next_obs_list, terminated, truncated = vec_env.step(student_actions)
        episode_over = terminated | truncated

        # Reset hidden states for envs where the episode has finished
        dones_tensor = torch.from_numpy(episode_over).to(device=device)
        h = reset_hidden_on_done(h, dones_tensor)

    # Backprop through the unroll
    avg_loss = total_loss / unroll_len
    optimizer.zero_grad()
    avg_loss.backward()
    optimizer.step()

    avg_acc = total_correct / max(total_count, 1)
    avg_lpc = total_lpc / unroll_len if use_lpc else 0.0

    # Detach hidden states for next unroll
    h = detach_hidden(h)

    return h, avg_loss.item(), avg_acc, avg_lpc


def save_checkpoint(policy, optimizer, config, update, checkpoint_dir, task_id, trial, lang_proj_state_dict):
    """
    Save model checkpoint to disk.

    Structure: checkpoint_dir/task_id/trial_N/checkpoint_final.pt
    """
    # Create directory structure
    save_dir = Path(checkpoint_dir) / task_id / f"trial_{trial}"
    save_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = save_dir / "checkpoint_final.pt"

    checkpoint = {
        'policy_state_dict': policy.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'config': config,
        'update': update,
        'lang_proj_state_dict': lang_proj_state_dict,
    }

    torch.save(checkpoint, checkpoint_path)
    print(f"Checkpoint saved to {checkpoint_path}")
    return checkpoint_path


def load_config(config_path, args):
    """Load config from YAML and override with command-line args."""
    # Load base config
    import builtins
    with builtins.open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Override with command-line arguments
    if args.task_id is not None:
        config['task_id'] = args.task_id
    if args.trial is not None:
        config['trial'] = int(args.trial)
    if args.num_envs is not None:
        config['num_envs'] = int(args.num_envs)
    if args.unroll_len is not None:
        config['unroll_len'] = int(args.unroll_len)
    if args.num_updates is not None:
        config['num_updates'] = int(args.num_updates)
    if args.lr is not None:
        config['lr'] = float(args.lr)
    if args.log_interval is not None:
        config['log_interval'] = int(args.log_interval)
    if args.wandb_project is not None:
        config['wandb_project'] = args.wandb_project
    if args.expert_hidden_sizes is not None:
        # Support both single-layer and multi-layer formats (matching YAML syntax):
        # Single layer: "[32,64,128]" -> [32, 64, 128]
        # Multi-layer: "[[32,64],[64,128],[128,256]]" -> [[32, 64], [64, 128], [128, 256]]
        import ast
        config['expert_hidden_sizes'] = ast.literal_eval(args.expert_hidden_sizes)
    if args.intermediate_dim is not None:
        config['intermediate_dim'] = int(args.intermediate_dim)
    if args.router_hidden_size is not None:
        config['router_hidden_size'] = int(args.router_hidden_size)
    if args.lpc_alpha is not None:
        config['lpc_alpha'] = float(args.lpc_alpha)
    if args.max_steps is not None:
        config['max_steps'] = int(args.max_steps)

    # Ensure all numeric config values are the correct type
    config['trial'] = int(config['trial'])
    config['num_envs'] = int(config['num_envs'])
    config['unroll_len'] = int(config['unroll_len'])
    config['num_updates'] = int(config['num_updates'])
    config['lr'] = float(config['lr'])
    config['log_interval'] = int(config['log_interval'])
    config['input_dim'] = int(config['input_dim'])
    config['lang_dim'] = int(config['lang_dim'])

    # MoE-specific defaults
    if 'expert_hidden_sizes' not in config:
        config['expert_hidden_sizes'] = [32, 64, 128]
    if 'intermediate_dim' not in config:
        config['intermediate_dim'] = 256
    if 'router_hidden_size' not in config:
        config['router_hidden_size'] = 64
    if 'lpc_alpha' not in config:
        config['lpc_alpha'] = 0.0

    return config


def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Train BabyAI with Mixture of Experts (MoE) Policy')
    parser.add_argument('--config', type=str, default='config.yaml',
                        help='Path to config file')
    parser.add_argument('--task_id', type=str, default=None,
                        help='BabyAI task ID (e.g., BabyAI-ActionObjDoor-v0)')
    parser.add_argument('--trial', type=int, default=None,
                        help='Trial number for this run')
    parser.add_argument('--num_envs', type=int, default=None,
                        help='Number of parallel environments')
    parser.add_argument('--unroll_len', type=int, default=None,
                        help='Unroll length for BPTT')
    parser.add_argument('--num_updates', type=int, default=None,
                        help='Number of training updates')
    parser.add_argument('--lr', type=float, default=None,
                        help='Learning rate')
    parser.add_argument('--log_interval', type=int, default=None,
                        help='Logging interval')
    parser.add_argument('--wandb_project', type=str, default=None,
                        help='Wandb project name')
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed')
    parser.add_argument('--expert_hidden_sizes', type=str, default=None,
                        help='Expert hidden sizes. Single layer: "[32,64,128]". Multi-layer: "[[32,64],[64,128],[128,256]]"')
    parser.add_argument('--intermediate_dim', type=int, default=None,
                        help='Intermediate dimension for layer communication')
    parser.add_argument('--router_hidden_size', type=int, default=None,
                        help='Router GRU hidden size')
    parser.add_argument('--lpc_alpha', type=float, default=None,
                        help='LPC regularization weight (0.0 = disabled)')
    parser.add_argument('--max_steps', type=int, default=None,
                        help='Max steps per episode (None = use environment default)')

    # Checkpoint arguments
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints',
                        help='Directory to save checkpoints')

    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config, args)

    # Extract config values
    task_id = config['task_id']
    trial = config['trial']
    num_envs = config['num_envs']
    unroll_len = config['unroll_len']
    num_updates = config['num_updates']
    lr = config['lr']
    log_interval = config['log_interval']
    wandb_project = config['wandb_project']
    input_dim = config['input_dim']
    lang_dim = config['lang_dim']

    # Set random seed if provided
    if args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create vectorized env
    max_steps = config.get('max_steps', None)
    vec_env = VectorBabyAIEnv(task_id, num_envs, device, max_steps=max_steps, lang_dim=lang_dim)
    obs_list = vec_env.reset()
    num_actions = vec_env.action_space.n
    config['num_actions'] = num_actions

    # Create MoE policy
    # Get MoE-specific config parameters or use defaults
    expert_hidden_sizes = config.get('expert_hidden_sizes', [32, 64, 128])
    intermediate_dim = config.get('intermediate_dim', 256)
    router_hidden_size = config.get('router_hidden_size', 64)
    lpc_alpha = config.get('lpc_alpha', 0.0)

    policy = MixtureOfExpertsPolicy(
        input_dim=input_dim,
        intermediate_dim=intermediate_dim,
        expert_hidden_sizes=expert_hidden_sizes,
        router_hidden_size=router_hidden_size,
        num_actions=num_actions,
        lang_dim=lang_dim
    ).to(device)

    optimizer = optim.Adam(policy.parameters(), lr=lr)

    # Initialize hidden states for router and experts
    h = policy.init_hidden(num_envs, device)

    # wandb init
    wandb.init(
        project=wandb_project,
        name=f"{task_id}-t{trial}",
        config=config,
        reinit=True
    )


    # -------------------------
    # Training loop
    # -------------------------
    for update in trange(1, num_updates + 1):

        # One truncated backprop through time (BPTT) update
        h, avg_loss, avg_acc, avg_lpc = train_unroll_moe(
            policy,
            optimizer,
            vec_env,
            h,
            unroll_len,
            device,
            lpc_alpha
        )

        # -------------------------
        # Logging
        # -------------------------
        if update % log_interval == 0:

            # Success rate which is global and cumulative
            if vec_env.total_episodes > 0:
                success_rate = vec_env.successful_episodes / vec_env.total_episodes
            else:
                success_rate = 0.0

            # Success rate which is local and recent 
            if len(vec_env.recent_successes) > 0:
                recent_success_rate = np.mean(vec_env.recent_successes)
            else:
                recent_success_rate = float("nan")


            # Path efficiency (computed for successful episodes only)
            if vec_env.num_successful_with_ratio > 0:
                mean_path_ratio = vec_env.sum_path_ratio / vec_env.num_successful_with_ratio
            else:
                mean_path_ratio = float("nan")

            if len(vec_env.recent_path_ratios) > 0:
                recent_path_ratio = np.mean(vec_env.recent_path_ratios)
            else:
                recent_path_ratio = float("nan")

            log_dict = {
                "update": update,
                "loss": avg_loss,
                "accuracy": avg_acc,
                "success_rate/recent": recent_success_rate,
                "success_rate/cumulative": success_rate,
                "path_ratio/recent": recent_path_ratio,
                "episodes/total": vec_env.total_episodes,
                "episodes/bot_plan_failures": vec_env.bot_plan_failures,
            }
            if lpc_alpha > 0.0:
                log_dict["lpc"] = avg_lpc
            wandb.log(log_dict)

            print(
                f"Update {update:04d} | "
                f"loss: {avg_loss:.3f} | "
                f"acc: {avg_acc:.3f} | "
                f"success_rate (recent): {recent_success_rate:.2f} | "
                f"path_ratio (recent): {recent_path_ratio:.2f} | "
                f"total_episodes: {vec_env.total_episodes}"
            )

    # Save final checkpoint
    checkpoint_dir = args.checkpoint_dir
    save_checkpoint(policy, optimizer, config, num_updates, checkpoint_dir, task_id, trial, vec_env.lang_proj.state_dict())

    wandb.finish()


if __name__ == "__main__":
    main()