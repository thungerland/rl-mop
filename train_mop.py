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
from pathlib import Path

from torch.utils.data import Dataset, DataLoader, random_split
from tqdm.auto import trange
from minigrid.wrappers import *
from minigrid.envs.babyai import *
from minigrid.utils.baby_ai_bot import BabyAIBot
from PIL import Image
from collections import deque

# MoP imports
from mop_config import Config
from stateful_mop import StatefulMoPPolicy

class VectorBabyAIEnv:
    """
    Simple N-parallel wrapper around a BabyAI task.

    - All envs use the same task id (e.g. "BabyAI-GoToRedBall-v0")
    - Keeps its own list of BabyAIBot experts, one per env
    - Auto-resets envs when they finish
    """

    def __init__(self, task_id: str, num_envs: int, device: torch.device):
        
        from transformers import AutoTokenizer, AutoModel
        self.task_id = task_id
        self.num_envs = num_envs
        self.device = device

        self.envs = []
        self.bots = []
        self.obs_list = [] # list of observation dicts with the 'image', 'direction', 'mission' keys
        self.missions = [] # We want to store the language separately from numerical observations

        # Store the last action executed in each env (None at episode start)
        # This will be relevant as the BabyAIBot must replan according to the action taken under the GRU policy
        self.previous_actions = [None] * num_envs

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

        for p in self.lang_encoder.parameters():
            p.requires_grad = False

        self.lang_proj = nn.Linear(768, 128).to(device)
        for p in self.lang_proj.parameters():
            p.requires_grad = False

        # Initialise cache for language embeddings
        self.lang_embs = torch.zeros(num_envs, 128, device=device)

        # Create N envs + bots + initial states 
        for i in range(num_envs):
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
            a = bot.replan(None)
            _, _, done, truncated, _ = env_clone.step(a.value)
            steps += 1

        return steps

    def reset(self):
        """
        Reset all envs and return list of observations.
        Also re-creates the bots so they are synced with the new envs.
        """
        self.obs_list = []
        self.previous_actions = [None] * self.num_envs
        # self.dones[:] = False
        
        for i, env in enumerate(self.envs):
            obs, _ = env.reset()
            
            mission = obs["mission"]
            obs = obs.copy()
            obs.pop('mission', None)
            obs = self._add_carrying_flag(env, obs)
            
            self.obs_list.append(obs)
            self.missions[i] = mission
            self.lang_embs[i] = self._encode_mission(mission)
            
            # Recreate each bot for each fresh env
            self.bots[i] = BabyAIBot(env)

            # Initialise episode-level metrics
            self.expert_steps[i] = self.compute_expert_steps_from_clone(env)
            self.episode_steps[i] = 0
            self.episode_success[i] = False

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
                obs, _ = env.reset()
                mission = obs["mission"]
                self.missions[i] = mission
                self.lang_embs[i] = self._encode_mission(mission)
                
                obs = obs.copy()
                obs.pop('mission', None)
                obs = self._add_carrying_flag(env, obs)

                # Reset bot
                self.bots[i] = BabyAIBot(env)

                # Prepare episode-level metrics for the new episode
                self.expert_steps[i] = self.compute_expert_steps_from_clone(env)
                self.episode_steps[i] = 0
                self.episode_success[i] = False

                # Robustness: store the action that was executed.
                # But if we reset, the next step is a new episode -> prev should be None.
                self.previous_actions[i] = None
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
            a_opt = bot.replan(prev) # bot needs to replan according to the actual last taken action
            actions.append(a_opt.value)

        return actions

class GRUPolicy(nn.Module):
    def __init__(self, input_dim, hidden_size, num_actions, lang_dim=128):
        super().__init__()

        # --- Base GRU modules ---
        self.gru = nn.GRU(
            input_size=input_dim + lang_dim, 
            hidden_size=hidden_size, 
            batch_first=True
        )
        self.head = nn.Linear(hidden_size, num_actions)

    def forward(self, x, lang_embs, h):
        """
        x:         (batch, input_dim)        these are the numerical obs (image+dir+carry) for one timestep
        lang_embs: (batch, lang_dim)
        h:         (1, batch, hidden_size)   GRU hidden state
        """
        
        # Concatenate language to obs 
        x = torch.cat([x, lang_embs], dim=1)  # (batch, input_dim + 128])
        
        # GRU step 
        x = x.unsqueeze(1)            # (batch, 1, input_dim + 128)
        out, h_new = self.gru(x, h)   # h: (1, batch, hidden_size)
        out = out.squeeze(1)          # (batch, hidden_size)
        logits = self.head(out)       # (batch, num_actions)
        
        return logits, h_new


def create_babyai_mop_config(
    input_dim: int = 280,  # 3*7*7 + 4 + 1 + 128
    num_actions: int = 7,
    intermediate_dim: int = 256,
    router_dim: int = 64,
    layers: list = None,
    device: str = "cuda",
    **kwargs
) -> Config:
    """
    Create a MoP Config suitable for BabyAI tasks.

    Args:
        input_dim: Total input dimension (image + direction + carrying + language)
        num_actions: Number of actions in the action space
        intermediate_dim: Hidden dimension for intermediate layers
        router_dim: Hidden dimension for router GRUs
        layers: List of expert configurations, e.g., ["32,64,128", "64,128"]
        device: Device to use
        **kwargs: Additional config parameters

    Returns:
        Config object for MoP model
    """
    if layers is None:
        # Default: 2 layers with 3 experts each of increasing size
        layers = ["32,64,128", "64,128,256"]

    # Required config parameters for MoP
    config_dict = {
        'input_dim': input_dim,
        'output_dim': num_actions,
        'intermediate_dim': intermediate_dim,
        'router_dim': router_dim,
        'layers': layers,
        'device': device,
        'task_id': 'babyai',  # Single task
        'task_dim': 32,  # Not used since single task
        'disable_task_embedding_layer': True,  # Single task, no need for task embeddings
        'disable_wandb': True,  # We handle wandb separately
        'disable_fixation_loss': True,
        'disable_task_performance_scaling': True,
        'expert_cost_exponent': 2.0,
        'cost_based_loss_alpha': 0.0,  # Start with no complexity penalty
        'cost_based_loss_epsilon': 0.0,
        'dropout_max_prob': None,
        'dropout_router_weight_threshold': None,
        'early_stopping_threshold': None,
        'ephemeral': False,
        'learning_rate': 0.001,  # Placeholder, we control this in optimizer
        'num_epochs': 1,  # Placeholder
        'num_steps': 1000,  # Placeholder
        'batch_size': 16,  # Placeholder, we control this with num_envs
        'checkpoint': None,
        'run_id': 'babyai_mop',
    }

    # Override with any provided kwargs
    config_dict.update(kwargs)

    return Config.from_dict(config_dict, migrate=False)

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
    

def train_unroll(policy, optimizer, vec_env, h, unroll_len, device):
    """
    One truncated-backprop through time (BPTT) unroll of length `unroll_len`.

    Args:
        policy: GRUPolicy
        optimizer: torch optimizer
        vec_env: VectorBabyAIEnv
        h: GRU hidden state, shape [1, num_envs, hidden_size]
        unroll_len: int
        device: "cuda" or "cpu"

    Returns:
        new_h: updated hidden state after unroll (detached from graph)
        avg_loss: scalar (float)
        avg_acc: scalar (float)
    """
    policy.train()
    num_envs = vec_env.num_envs

    total_loss = 0.0
    total_correct = 0
    total_count = 0

    # truncated BPTT: don't backprop through previous unrolls
    h = h.detach()

    # invalid_action_ids = invalid_action_ids or []

    for t in range(unroll_len):
        # Encode current obs 
        obs_batch = encode_obs_batch(vec_env.obs_list, device) # (N, input_dim)

        # Get expert actions as labels for this timestep
        expert_actions = vec_env.get_expert_actions() # list of length N
        expert_actions = torch.tensor(expert_actions, dtype=torch.long, device=device)

        # Forward GRU policy for one step
        lang_embs = vec_env.lang_embs
        logits, h = policy(obs_batch, lang_embs, h) # logits [N, num_actions], h [1, N, hidden_size]

        # Compute the supervised loss between policy and expert
        step_loss = F.cross_entropy(logits, expert_actions)
        total_loss += step_loss

        # Compute the accuracy
        with torch.no_grad():
            preds = logits.argmax(dim=-1) # (N,)
            correct = (preds == expert_actions).sum().item() # scalar
            total_correct += correct
            total_count += num_envs

        # Extract the student actions (suggested by the GRU policy) to step the envs
        with torch.no_grad():
            student_actions = preds.detach().cpu().numpy()

        # Step the envs according to student actions 
        next_obs_list, terminated, truncated = vec_env.step(student_actions)
        episode_over = terminated | truncated 

        # Reset hidden state for envs where the episode has finished 
        dones_tensor = torch.from_numpy(episode_over.astype(int)).to(device=device)  # [N]
        done_mask = dones_tensor.unsqueeze(0).unsqueeze(-1) # [1,N,1]

        # Where done_mask==1, zero out the hidden state
        h = h * (1.0 - done_mask)

    # Backprop through the unroll
    avg_loss = total_loss / unroll_len
    optimizer.zero_grad()
    avg_loss.backward()
    optimizer.step()

    avg_acc = total_correct / max(total_count, 1)

    # Detach hidden state for next unroll
    h = h.detach()

    return h, avg_loss.item(), avg_acc


def train_unroll_mop(policy, optimizer, vec_env, hidden_states, unroll_len, device):
    """
    One truncated-backprop through time (BPTT) unroll for MoP policy.

    Args:
        policy: StatefulMoPPolicy
        optimizer: torch optimizer
        vec_env: VectorBabyAIEnv
        hidden_states: Dict of hidden states for all routers and experts
        unroll_len: int
        device: "cuda" or "cpu"

    Returns:
        new_hidden_states: updated hidden states after unroll (detached from graph)
        avg_loss: scalar (float)
        avg_acc: scalar (float)
    """
    policy.train()
    num_envs = vec_env.num_envs

    total_loss = 0.0
    total_correct = 0
    total_count = 0

    # Truncated BPTT: detach all hidden states
    hidden_states = {k: v.detach() for k, v in hidden_states.items()}

    for t in range(unroll_len):
        # Encode current obs
        obs_batch = encode_obs_batch(vec_env.obs_list, device)  # (N, input_dim)

        # Get expert actions as labels for this timestep
        expert_actions = vec_env.get_expert_actions()  # list of length N
        expert_actions = torch.tensor(expert_actions, dtype=torch.long, device=device)

        # Forward MoP policy for one step
        lang_embs = vec_env.lang_embs
        logits, hidden_states = policy(obs_batch, lang_embs, hidden_states)

        # Compute the supervised loss between policy and expert
        step_loss = F.cross_entropy(logits, expert_actions)
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
        dones_tensor = torch.from_numpy(episode_over.astype(int)).to(device=device)  # [N]

        # For MoP, we need to reset all hidden states (routers and experts)
        for key in hidden_states.keys():
            # hidden_states[key] has shape [1, N, hidden_dim]
            done_mask = dones_tensor.unsqueeze(0).unsqueeze(-1)  # [1, N, 1]
            hidden_states[key] = hidden_states[key] * (1.0 - done_mask)

    # Backprop through the unroll
    avg_loss = total_loss / unroll_len
    optimizer.zero_grad()
    avg_loss.backward()
    optimizer.step()

    avg_acc = total_correct / max(total_count, 1)

    # Detach hidden states for next unroll
    hidden_states = {k: v.detach() for k, v in hidden_states.items()}

    return hidden_states, avg_loss.item(), avg_acc


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
    if args.hidden_size is not None:
        config['hidden_size'] = int(args.hidden_size)
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
    if args.mop_layers is not None:
        config['mop_layers'] = args.mop_layers.split(';')
    if args.mop_intermediate_dim is not None:
        config['mop_intermediate_dim'] = int(args.mop_intermediate_dim)
    if args.mop_router_dim is not None:
        config['mop_router_dim'] = int(args.mop_router_dim)

    # Ensure all numeric config values are the correct type
    config['trial'] = int(config['trial'])
    config['num_envs'] = int(config['num_envs'])
    config['hidden_size'] = int(config.get('hidden_size', 64))  # Default for compatibility
    config['unroll_len'] = int(config['unroll_len'])
    config['num_updates'] = int(config['num_updates'])
    config['lr'] = float(config['lr'])
    config['log_interval'] = int(config['log_interval'])
    config['input_dim'] = int(config['input_dim'])
    config['lang_dim'] = int(config['lang_dim'])

    # MoP-specific defaults
    if 'mop_layers' not in config:
        config['mop_layers'] = ["32,64,128", "64,128,256"]
    if 'mop_intermediate_dim' not in config:
        config['mop_intermediate_dim'] = 256
    if 'mop_router_dim' not in config:
        config['mop_router_dim'] = 64

    return config


def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Train BabyAI with Mixture-of-Pathways (MoP) Policy')
    parser.add_argument('--config', type=str, default='config.yaml',
                        help='Path to config file')
    parser.add_argument('--task_id', type=str, default=None,
                        help='BabyAI task ID (e.g., BabyAI-ActionObjDoor-v0)')
    parser.add_argument('--trial', type=int, default=None,
                        help='Trial number for this run')
    parser.add_argument('--num_envs', type=int, default=None,
                        help='Number of parallel environments')
    parser.add_argument('--hidden_size', type=int, default=None,
                        help='GRU hidden size (kept for compatibility, not used in MoP)')
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
    parser.add_argument('--mop_layers', type=str, default=None,
                        help='MoP layer configuration (e.g., "32,64,128;64,128,256")')
    parser.add_argument('--mop_intermediate_dim', type=int, default=None,
                        help='MoP intermediate dimension')
    parser.add_argument('--mop_router_dim', type=int, default=None,
                        help='MoP router dimension')

    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config, args)

    # Extract config values
    task_id = config['task_id']
    trial = config['trial']
    num_envs = config['num_envs']
    hidden_size = config['hidden_size']
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
    vec_env = VectorBabyAIEnv(task_id, num_envs, device)
    obs_list = vec_env.reset()
    num_actions = vec_env.action_space.n

    # Create MoP policy
    # Get MoP-specific config parameters or use defaults
    mop_layers = config.get('mop_layers', ["32,64,128", "64,128,256"])
    mop_intermediate_dim = config.get('mop_intermediate_dim', 256)
    mop_router_dim = config.get('mop_router_dim', 64)

    mop_config = create_babyai_mop_config(
        input_dim=input_dim + lang_dim,  # Combined obs + language
        num_actions=num_actions,
        intermediate_dim=mop_intermediate_dim,
        router_dim=mop_router_dim,
        layers=mop_layers,
        device=str(device)
    )

    policy = StatefulMoPPolicy(mop_config).to(device)
    optimizer = optim.Adam(policy.parameters(), lr=lr)

    # Initialize all hidden states for MoP (routers + experts)
    hidden_states = policy.init_hidden_states(num_envs, device)

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
        hidden_states, avg_loss, avg_acc = train_unroll_mop(
            policy,
            optimizer,
            vec_env,
            hidden_states,
            unroll_len,
            device
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

            wandb.log({
                "update": update,
                "loss": avg_loss,
                "accuracy": avg_acc,
                "success_rate/recent": recent_success_rate,
                "success_rate/cumulative": success_rate,
                "path_ratio/recent": recent_path_ratio,
                "episodes/total": vec_env.total_episodes,
            })

            print(
                f"Update {update:04d} | "
                f"loss: {avg_loss:.3f} | "
                f"acc: {avg_acc:.3f} | "
                f"success_rate (recent): {recent_success_rate:.2f} | "
                f"path_ratio (recent): {recent_path_ratio:.2f} | "
                f"total_episodes: {vec_env.total_episodes}"
            )
        
if __name__ == "__main__":
    main()