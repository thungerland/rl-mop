#!/usr/bin/env python3
"""
Batch evaluation script for RL-MoP experiments.

Scans modal_checkpoints/, evaluates all models, extracts hyperparameters,
and saves results to CSV with routing data cached for visualization.

Usage:
    python batch_eval.py                           # Evaluate all new checkpoints
    python batch_eval.py --force                   # Re-evaluate all
    python batch_eval.py --num_episodes 200        # More episodes
    python batch_eval.py --task BabyAI-GoToRedBall-v0  # Filter by task
    python batch_eval.py --skip_routing            # Skip routing cache
"""

import argparse
import json
import numpy as np
import torch
import pandas as pd
from pathlib import Path
from datetime import datetime
from tqdm import tqdm

from eval_mop import load_checkpoint, evaluate, EvalVectorEnv


def discover_checkpoints(checkpoint_dir: str, task_filter: str = None, trial_filter: int = None,
                         seed_filter: int = None, update_filter: int = None,
                         final_only: bool = False) -> list[Path]:
    """Scan checkpoint directory for checkpoint files.

    Handles two path layouts:
      Old: task_id/trial_N/checkpoint_final.pt          (3 parts, legacy)
      New: task_id/trial_N/seed_S/checkpoint_<U>.pt     (4 parts)

    By default, checkpoint_final.pt aliases in the new layout are skipped to avoid
    double-counting (the numbered file is canonical, update is read from the filename).

    With final_only=True, only checkpoint_final.pt files are returned. The actual
    update number is read from the checkpoint config at eval time, so it correctly
    reflects whatever step training ended on rather than being fixed to num_updates.
    Old-style 3-part checkpoint_final.pt is always included (backward compat).
    """
    checkpoints = []
    base = Path(checkpoint_dir)

    if not base.exists():
        print(f"Warning: Checkpoint directory '{checkpoint_dir}' does not exist.")
        return checkpoints

    glob_pattern = "**/checkpoint_final.pt" if final_only else "**/checkpoint_*.pt"

    for checkpoint_path in base.glob(glob_pattern):
        parts = checkpoint_path.relative_to(base).parts

        if len(parts) == 3:
            # Old layout: task_id/trial_N/checkpoint_final.pt
            task_id, trial_str, filename = parts
            seed = None
        elif len(parts) == 4:
            # New layout: task_id/trial_N/seed_S/checkpoint_<U>.pt
            task_id, trial_str, seed_str, filename = parts
            try:
                seed = int(seed_str.split('_')[1])
            except (IndexError, ValueError):
                print(f"Warning: Could not parse seed from {checkpoint_path}, skipping.")
                continue
        else:
            print(f"Warning: Unexpected path depth {checkpoint_path}, skipping.")
            continue

        # Parse update from filename
        name = filename.replace('.pt', '')      # "checkpoint_500" or "checkpoint_final"
        update_str = name.split('_', 1)[1]      # "500" or "final"
        if update_str == 'final':
            if len(parts) == 4 and not final_only:
                # New-style final alias in default mode: skip — numbered file is canonical
                continue
            # Otherwise (final_only=True, or old 3-part): include; update resolved from config
            update = None
        else:
            if final_only:
                # Should not happen with checkpoint_final.pt glob, but be safe
                continue
            try:
                update = int(update_str)
            except ValueError:
                print(f"Warning: Could not parse update from {checkpoint_path}, skipping.")
                continue

        try:
            trial = int(trial_str.split('_')[1])
        except (IndexError, ValueError):
            print(f"Warning: Could not parse trial from {checkpoint_path}, skipping.")
            continue

        # Apply filters
        if task_filter and task_id != task_filter:
            continue
        if trial_filter is not None and trial != trial_filter:
            continue
        if seed_filter is not None and seed != seed_filter:
            continue
        if update_filter is not None and update != update_filter:
            continue

        checkpoints.append(checkpoint_path)

    return sorted(checkpoints)


def load_existing_results(results_path: str) -> pd.DataFrame:
    """Load existing CSV or return empty DataFrame with correct schema."""
    if Path(results_path).exists():
        return pd.read_csv(results_path)

    # Return empty DataFrame with expected columns
    return pd.DataFrame(columns=[
        'checkpoint_path', 'task_id', 'trial', 'seed', 'update', 'num_episodes',
        'success_rate', 'path_ratio', 'mean_lpc', 'bot_plan_failures',
        'num_updates', 'unroll_len', 'lr', 'lpc_alpha',
        'expert_hidden_sizes', 'intermediate_dim', 'router_hidden_size',
        'max_steps', 'lang_dim', 'evaluated_at'
    ])


def checkpoint_already_evaluated(checkpoint_path: Path, results_df: pd.DataFrame, base_dir: Path) -> bool:
    """Check if checkpoint is already in results (by relative path)."""
    if results_df.empty:
        return False
    rel_path = str(checkpoint_path.relative_to(base_dir.parent))
    return rel_path in results_df['checkpoint_path'].values


def build_result_row(checkpoint_path: Path, metrics: dict, config: dict,
                     num_episodes: int, base_dir: Path) -> dict:
    """Build a result row dictionary from evaluation results and config."""
    return {
        'checkpoint_path': str(checkpoint_path.relative_to(base_dir.parent)),
        'task_id': config['task_id'],
        'trial': int(config['trial']),
        'seed': int(config['seed']) if config.get('seed') is not None else None,
        'update': int(config['update']) if config.get('update') is not None else None,
        'num_episodes': int(num_episodes),
        'success_rate': float(metrics['success_rate']),
        'path_ratio': float(metrics['path_ratio']),
        'mean_lpc': float(metrics['mean_lpc']),
        'bot_plan_failures': int(metrics.get('bot_plan_failures', 0)),
        # Hyperparameters from config
        'num_updates': int(config.get('num_updates')) if config.get('num_updates') else None,
        'unroll_len': int(config.get('unroll_len')) if config.get('unroll_len') else None,
        'lr': float(config.get('lr')) if config.get('lr') else None,
        'lpc_alpha': float(config.get('lpc_alpha')) if config.get('lpc_alpha') is not None else None,
        'expert_hidden_sizes': json.dumps(config.get('expert_hidden_sizes')),
        'intermediate_dim': int(config.get('intermediate_dim')) if config.get('intermediate_dim') else None,
        'router_hidden_size': int(config.get('router_hidden_size')) if config.get('router_hidden_size') else None,
        'max_steps': int(config.get('max_steps')) if config.get('max_steps') else None,
        'lang_dim': int(config.get('lang_dim', 32)),
        'evaluated_at': datetime.now().isoformat(),
    }


def save_routing_data(routing_data: list, checkpoint_path: Path, config: dict,
                      metrics: dict, num_episodes: int, cache_dir: str):
    """Save routing data to JSON file for visualization caching."""
    task_id = config['task_id']
    trial = config['trial']
    seed = config.get('seed')
    update_val = config.get('update')

    cache_path = Path(cache_dir) / task_id / f"trial_{trial}"
    if seed is not None:
        cache_path = cache_path / f"seed_{seed}"
    cache_path = cache_path / (f"update_{update_val}" if update_val is not None else "update_unknown")
    cache_path.mkdir(parents=True, exist_ok=True)

    # Deduplicate env_context: store once per unique episode layout, reference by index
    # per timestep. env_context is identical across all timesteps of an episode, so
    # storing it per-timestep inflates file size massively for long episodes / large mazes.
    episodes_json = []
    context_to_idx = {}
    routing_json = []

    for sample in routing_data:
        pos = sample['position']
        layer_routing = sample['layer_routing']
        lpc = sample['lpc']
        env_context = sample['env_context']
        carrying = sample.get('carrying', 0)
        door_unlocked = sample.get('door_unlocked', 0)
        action_logits = sample.get('action_logits')
        t_step = sample.get('t_step')
        t_unlocked = sample.get('t_unlocked')
        t_pick = sample.get('t_pick')
        t_drop = sample.get('t_drop')
        dist_to_door = sample.get('dist_to_door')
        dist_to_key = sample.get('dist_to_key')
        dist_to_target = sample.get('dist_to_target')

        context_key = str(env_context)
        if context_key not in context_to_idx:
            context_to_idx[context_key] = len(episodes_json)
            episodes_json.append(env_context)
        episode_idx = context_to_idx[context_key]

        layer_routing_json = {}
        for k, v in layer_routing.items():
            if hasattr(v, 'tolist'):
                layer_routing_json[k] = v.tolist()
            else:
                layer_routing_json[k] = list(v)

        logits_arr = np.array(action_logits, dtype=np.float64)
        logits_shifted = logits_arr - logits_arr.max()
        exp_l = np.exp(logits_shifted)
        probs = exp_l / exp_l.sum()
        entry = {
            'episode': episode_idx,
            'position': [int(p) for p in pos],
            'layer_routing': layer_routing_json,
            'lpc': float(lpc),
            'carrying': int(carrying),
            'door_unlocked': int(door_unlocked),
            'action_logits': [float(v) for v in action_logits],
            'action': int(np.argmax(logits_arr)),
            'entropy': float(-np.sum(probs * np.log(probs + 1e-9))),
            't_step': int(t_step) if t_step is not None else None,
            't_unlocked': int(t_unlocked) if t_unlocked is not None else None,
            't_pick': int(t_pick) if t_pick is not None else None,
            't_drop': int(t_drop) if t_drop is not None else None,
            'dist_to_door': float(dist_to_door) if dist_to_door is not None else None,
            'dist_to_key': float(dist_to_key) if dist_to_key is not None else None,
            'dist_to_target': float(dist_to_target) if dist_to_target is not None else None,
        }
        routing_json.append(entry)

    cache_data = {
        'checkpoint_path': str(checkpoint_path),
        'num_episodes': int(num_episodes),
        'evaluated_at': datetime.now().isoformat(),
        'expert_hidden_sizes': config.get('expert_hidden_sizes'),
        'metrics': {
            'success_rate': float(metrics['success_rate']),
            'path_ratio': float(metrics['path_ratio']),
            'mean_lpc': float(metrics['mean_lpc']),
            'bot_plan_failures': int(metrics.get('bot_plan_failures', 0)),
        },
        'episodes': episodes_json,
        'routing_data': routing_json,
    }

    with open(cache_path / 'routing_data.json', 'w') as f:
        json.dump(cache_data, f)


def main():
    parser = argparse.ArgumentParser(
        description='Batch evaluate MoE checkpoints',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument('--checkpoint_dir', type=str, default='modal_checkpoints',
                        help='Directory containing checkpoints (default: modal_checkpoints)')
    parser.add_argument('--results_path', type=str, default='evaluation_results.csv',
                        help='Path to results CSV file (default: evaluation_results.csv)')
    parser.add_argument('--cache_dir', type=str, default='evaluation_cache',
                        help='Directory for routing data cache (default: evaluation_cache)')
    parser.add_argument('--num_episodes', type=int, default=100,
                        help='Number of evaluation episodes (default: 100)')
    parser.add_argument('--num_envs', type=int, default=16,
                        help='Number of parallel environments (default: 16)')
    parser.add_argument('--task', type=str, default=None,
                        help='Filter by task_id (e.g., BabyAI-GoToRedBall-v0)')
    parser.add_argument('--trial', type=int, default=None,
                        help='Filter by trial number')
    parser.add_argument('--seed', type=int, default=None,
                        help='Filter by seed number')
    parser.add_argument('--update', type=int, default=None,
                        help='Filter by update/checkpoint step (e.g. 1500)')
    parser.add_argument('--final-only', action='store_true',
                        help='Evaluate only checkpoint_final.pt files (actual last update, not fixed num_updates)')
    parser.add_argument('--force', action='store_true',
                        help='Re-evaluate all checkpoints (ignore existing results)')
    parser.add_argument('--skip_routing', action='store_true',
                        help='Skip saving routing data cache (faster)')

    args = parser.parse_args()

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Discover checkpoints
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoints = discover_checkpoints(args.checkpoint_dir, args.task, args.trial,
                                       args.seed, args.update, args.final_only)
    print(f"Found {len(checkpoints)} checkpoint(s) in {args.checkpoint_dir}")

    if not checkpoints:
        print("No checkpoints to evaluate.")
        return

    # Load existing results
    results_df = load_existing_results(args.results_path)
    if 'lang_dim' in results_df.columns:
        results_df['lang_dim'] = results_df['lang_dim'].fillna(128).astype(int)
    print(f"Existing results: {len(results_df)} rows")

    # Filter to unevaluated checkpoints (unless --force)
    if args.force:
        to_evaluate = checkpoints
        print(f"Force mode: will evaluate all {len(to_evaluate)} checkpoints")
    else:
        to_evaluate = [
            cp for cp in checkpoints
            if not checkpoint_already_evaluated(cp, results_df, checkpoint_dir)
        ]
        print(f"New checkpoints to evaluate: {len(to_evaluate)}")

    if not to_evaluate:
        print("All checkpoints already evaluated. Use --force to re-evaluate.")
        return

    # Evaluate each checkpoint
    new_rows = []

    for checkpoint_path in tqdm(to_evaluate, desc="Evaluating checkpoints"):
        try:
            # Load model
            policy, config, lang_proj_state_dict = load_checkpoint(str(checkpoint_path), device)
            task_id = config['task_id']

            # Create evaluation environment
            vec_env = EvalVectorEnv(task_id, args.num_envs, device,
                                    max_steps=config.get('max_steps'),
                                    lang_dim=config.get('lang_dim', 32))

            # Load lang_proj weights if available
            if lang_proj_state_dict is not None:
                vec_env.load_lang_proj(lang_proj_state_dict)

            # Run evaluation
            metrics, routing_data = evaluate(policy, vec_env, args.num_episodes, device)

            # Build result row
            row = build_result_row(checkpoint_path, metrics, config, args.num_episodes, checkpoint_dir)
            new_rows.append(row)

            # Save routing cache (unless skipped)
            if not args.skip_routing:
                save_routing_data(routing_data, checkpoint_path, config,
                                  metrics, args.num_episodes, args.cache_dir)


            # Print summary
            seed_str = f"/seed_{config['seed']}" if config.get('seed') is not None else ""
            update_str = f"@{config['update']}" if config.get('update') is not None else ""
            tqdm.write(f"  {task_id}/trial_{config['trial']}{seed_str}{update_str}: "
                      f"success={metrics['success_rate']:.1%}, "
                      f"path_ratio={metrics['path_ratio']:.2f}, "
                      f"lpc={metrics['mean_lpc']:.1f}, "
                      f"bot_plan_failures={metrics.get('bot_plan_failures', 0)}")

        except Exception as e:
            tqdm.write(f"Error evaluating {checkpoint_path}: {e}")
            continue

    # Append new results and save
    if new_rows:
        new_df = pd.DataFrame(new_rows)

        if args.force:
            # Remove old entries for re-evaluated checkpoints
            evaluated_paths = set(new_df['checkpoint_path'])
            results_df = results_df[~results_df['checkpoint_path'].isin(evaluated_paths)]

        # Concatenate (handle empty DataFrame case)
        if results_df.empty:
            results_df = new_df
        else:
            results_df = pd.concat([results_df, new_df], ignore_index=True)

        results_df.to_csv(args.results_path, index=False)
        print(f"\nSaved {len(new_rows)} new result(s) to {args.results_path}")
        print(f"Total results: {len(results_df)} rows")

    # Print summary table
    print("\n" + "=" * 80)
    print("EVALUATION SUMMARY")
    print("=" * 80)
    summary_cols = ['task_id', 'trial', 'seed', 'update', 'success_rate', 'path_ratio', 'mean_lpc']
    if new_rows:
        summary_df = pd.DataFrame(new_rows)[summary_cols]
        print(summary_df.to_string(index=False))
    print("=" * 80)


if __name__ == "__main__":
    main()
