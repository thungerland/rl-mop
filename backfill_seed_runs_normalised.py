import json
import re
import torch
from pathlib import Path
import pandas as pd


def read_cache_header(json_path):
    """Read only the first 2KB to extract metrics and checkpoint_path."""
    with open(json_path, 'rb') as f:
        chunk = f.read(2000).decode('utf-8', errors='replace')
    cp = re.search(r'"checkpoint_path":\s*"([^"]+)"', chunk)
    ne = re.search(r'"num_episodes":\s*(\d+)', chunk)
    ea = re.search(r'"evaluated_at":\s*"([^"]+)"', chunk)
    m = re.search(r'"metrics":\s*(\{[^}]+\})', chunk)
    if not (cp and ne and m):
        return None
    return {
        'checkpoint_path': cp.group(1),
        'num_episodes': int(ne.group(1)),
        'evaluated_at': ea.group(1) if ea else '',
        'metrics': json.loads(m.group(1)),
    }

task = 'BabyAI-UnlockPickup-v0'
base = Path('.')
cache_root = base / 'evaluation_cache_normalised' / task
ckpt_root = base / 'modal_checkpoints_normalised' / task
results_path = base / 'evaluation_results_normalised.csv'

TARGET_TRIALS = {20, 21, 22, 23, 26}

rows = []
for json_path in sorted(cache_root.glob('trial_*/seed_*/update_*/routing_data.json')):
    parts = json_path.parts  # [..., trial_N, seed_S, update_U, routing_data.json]
    trial = int(parts[-4].split('_')[1])
    seed = int(parts[-3].split('_')[1])
    update = int(parts[-2].split('_')[1])
    if trial not in TARGET_TRIALS:
        continue

    cache = read_cache_header(json_path)
    if cache is None:
        print(f'Could not parse header (skipping): {json_path}')
        continue
    metrics = cache['metrics']

    # Load config from matching checkpoint
    ckpt_file = ckpt_root / f'trial_{trial}' / f'seed_{seed}' / f'checkpoint_{update}.pt'
    if not ckpt_file.exists():
        ckpt_file = ckpt_root / f'trial_{trial}' / f'seed_{seed}' / 'checkpoint_final.pt'
    if not ckpt_file.exists():
        print(f'No checkpoint found for trial_{trial}/seed_{seed}/update_{update}, skipping')
        continue
    ckpt = torch.load(str(ckpt_file), map_location='cpu', weights_only=False)
    config = ckpt['config']

    cp = cache['checkpoint_path']
    rel_path = 'modal_checkpoints_normalised' + cp[len('/checkpoints_normalised'):] if cp.startswith('/checkpoints_normalised/') else cp

    rows.append({
        'checkpoint_path': rel_path,
        'task_id': task,
        'trial': trial,
        'seed': seed,
        'update': update,
        'num_episodes': cache['num_episodes'],
        'success_rate': metrics['success_rate'],
        'path_ratio': metrics.get('path_ratio', ''),
        'mean_lpc': metrics['mean_lpc'],
        'bot_plan_failures': metrics.get('bot_plan_failures', 0),
        'num_updates': config.get('num_updates', ''),
        'unroll_len': config.get('unroll_len', ''),
        'lr': config.get('lr', ''),
        'lpc_alpha': config.get('lpc_alpha', ''),
        'expert_hidden_sizes': json.dumps(config.get('expert_hidden_sizes', '')),
        'intermediate_dim': config.get('intermediate_dim', ''),
        'router_hidden_size': config.get('router_hidden_size', ''),
        'max_steps': config.get('max_steps', ''),
        'lang_dim': config.get('lang_dim', 32),
        'evaluated_at': cache['evaluated_at'],
    })
    print(f'trial_{trial}/seed_{seed}/update_{update}: success={metrics["success_rate"]:.3f}, lpc_alpha={config.get("lpc_alpha")}')

print(f'\nCollected {len(rows)} rows.')

if results_path.exists():
    existing = pd.read_csv(results_path)
else:
    existing = pd.DataFrame()

new_df = pd.DataFrame(rows)

for col in ['seed', 'update']:
    if col not in existing.columns:
        existing[col] = ''

combined = pd.concat([existing, new_df], ignore_index=True)
combined = combined.drop_duplicates(subset=['checkpoint_path', 'seed', 'update'])
combined.to_csv(results_path, index=False)
print(f'Added {len(rows)} rows. Total now: {len(combined)}')
