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
import torch
import pandas as pd
from pathlib import Path
from datetime import datetime
from tqdm import tqdm

from eval_mop import load_checkpoint, evaluate, EvalVectorEnv


def discover_checkpoints(checkpoint_dir: str, task_filter: str = None, trial_filter: int = None) -> list[Path]:
    """Scan checkpoint directory for all checkpoint_final.pt files."""
    checkpoints = []
    base = Path(checkpoint_dir)

    if not base.exists():
        print(f"Warning: Checkpoint directory '{checkpoint_dir}' does not exist.")
        return checkpoints

    for checkpoint_path in base.glob("**/checkpoint_final.pt"):
        # Extract task_id and trial from path structure: task_id/trial_N/checkpoint_final.pt
        task_id = checkpoint_path.parent.parent.name
        trial_str = checkpoint_path.parent.name  # "trial_0"

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

        checkpoints.append(checkpoint_path)

    return sorted(checkpoints)


def load_existing_results(results_path: str) -> pd.DataFrame:
    """Load existing CSV or return empty DataFrame with correct schema."""
    if Path(results_path).exists():
        return pd.read_csv(results_path)

    # Return empty DataFrame with expected columns
    return pd.DataFrame(columns=[
        'checkpoint_path', 'task_id', 'trial', 'num_episodes',
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


def build_result_row(checkpoint_path: Path, metrics: dict, config: dict, num_episodes: int, base_dir: Path) -> dict:
    """Build a result row dictionary from evaluation results and config."""
    return {
        'checkpoint_path': str(checkpoint_path.relative_to(base_dir.parent)),
        'task_id': config['task_id'],
        'trial': int(config['trial']),
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

    cache_path = Path(cache_dir) / task_id / f"trial_{trial}"
    cache_path.mkdir(parents=True, exist_ok=True)

    # Convert routing data to JSON-serializable format
    routing_json = []
    for pos, layer_routing, lpc, env_context in routing_data:
        # Handle both numpy arrays and regular lists
        layer_routing_json = {}
        for k, v in layer_routing.items():
            if hasattr(v, 'tolist'):
                layer_routing_json[k] = v.tolist()
            else:
                layer_routing_json[k] = list(v)

        routing_json.append({
            'position': [int(p) for p in pos],
            'layer_routing': layer_routing_json,
            'lpc': float(lpc),
            'env_context': env_context
        })

    cache_data = {
        'checkpoint_path': str(checkpoint_path),
        'num_episodes': int(num_episodes),
        'evaluated_at': datetime.now().isoformat(),
        'metrics': {
            'success_rate': float(metrics['success_rate']),
            'path_ratio': float(metrics['path_ratio']),
            'mean_lpc': float(metrics['mean_lpc']),
            'bot_plan_failures': int(metrics.get('bot_plan_failures', 0)),
        },
        'routing_data': routing_json
    }

    with open(cache_path / 'routing_data.json', 'w') as f:
        json.dump(cache_data, f, indent=2)


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
    checkpoints = discover_checkpoints(args.checkpoint_dir, args.task, args.trial)
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
            vec_env = EvalVectorEnv(task_id, args.num_envs, device, lang_dim=config.get('lang_dim', 32))

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
            tqdm.write(f"  {task_id}/trial_{config['trial']}: "
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
    summary_cols = ['task_id', 'trial', 'success_rate', 'path_ratio', 'mean_lpc']
    if new_rows:
        summary_df = pd.DataFrame(new_rows)[summary_cols]
        print(summary_df.to_string(index=False))
    print("=" * 80)


if __name__ == "__main__":
    main()
