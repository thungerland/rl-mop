# Backfilling evaluation_results.csv from eval caches

Use this when you have new `evaluation_cache/` entries that aren't yet in `evaluation_results.csv` — e.g. after running evals on Modal and syncing the cache locally without running `batch_eval.py`.

## Prerequisites

- Eval caches exist under `evaluation_cache/<task_id>/<task_id>/trial_N/routing_data.json`
- Corresponding checkpoints exist under `modal_checkpoints/<task_id>/trial_N/checkpoint_final.pt`
- The conda env is active: `conda activate rl-mop`

## Script

Run the following, editing `tasks_to_process` to list the tasks you want to backfill:

```python
import json, torch, re
from pathlib import Path
import pandas as pd

base = Path('.')  # run from rl-mop root
cache_base = base / 'evaluation_cache'
modal_ckpt_base = base / 'modal_checkpoints'
results_path = base / 'evaluation_results.csv'

tasks_to_process = [
    'BabyAI-GoToLocalS6N4-v0',
    # add more tasks here
]

def load_cache_metadata(json_path):
    """Load cache JSON, falling back to partial read for corrupt files."""
    try:
        with open(json_path) as f:
            return json.load(f)
    except json.JSONDecodeError:
        with open(json_path, 'rb') as f:
            chunk = f.read(2000).decode('utf-8', errors='replace')
        m = re.match(r'\{.*?"metrics":\s*(\{[^}]+\})', chunk, re.DOTALL)
        if m:
            cp = re.search(r'"checkpoint_path":\s*"([^"]+)"', chunk)
            ne = re.search(r'"num_episodes":\s*(\d+)', chunk)
            ea = re.search(r'"evaluated_at":\s*"([^"]+)"', chunk)
            return {
                'checkpoint_path': cp.group(1) if cp else '',
                'num_episodes': int(ne.group(1)) if ne else 0,
                'evaluated_at': ea.group(1) if ea else '',
                'metrics': json.loads(m.group(1)),
            }
        return None

rows = []
for task in tasks_to_process:
    for json_path in sorted((cache_base / task).glob('**/routing_data.json')):
        trial_str = json_path.parent.name
        trial = int(trial_str.split('_')[1])

        cache = load_cache_metadata(json_path)
        if not cache:
            print(f'Could not parse {json_path}, skipping')
            continue

        metrics = cache['metrics']
        ckpt_path = modal_ckpt_base / task / trial_str / 'checkpoint_final.pt'
        if not ckpt_path.exists():
            print(f'No checkpoint for {task}/{trial_str}, skipping')
            continue
        ckpt = torch.load(str(ckpt_path), map_location='cpu', weights_only=False)
        config = ckpt['config']

        cp = cache['checkpoint_path']
        rel_path = 'modal_checkpoints' + cp[len('/checkpoints'):] if cp.startswith('/checkpoints/') else cp

        rows.append({
            'checkpoint_path': rel_path,
            'task_id': task,
            'trial': trial,
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
        print(f'{task}/trial_{trial}: success={metrics["success_rate"]:.3f}, lpc_alpha={config.get("lpc_alpha")}')

existing = pd.read_csv(results_path)
combined = pd.concat([existing, pd.DataFrame(rows)], ignore_index=True)
combined.to_csv(results_path, index=False)
print(f'\nAdded {len(rows)} rows. Total: {len(combined)}')
```

## Notes

- **Duplicate guard**: the script blindly appends — if you run it twice you'll get duplicate rows. Check `evaluation_results.csv` first and only list tasks/trials not already present, or deduplicate after with `combined.drop_duplicates(subset=['checkpoint_path'])`.
- **Corrupt cache files**: the `load_cache_metadata` fallback recovers `metrics` from the first 2 KB of the file. The routing data in that cache is still broken and won't be usable for visualization — re-run the eval to regenerate a clean cache if needed.
- **Cache path structure**: caches are nested as `evaluation_cache/<task_id>/<task_id>/trial_N/` (double task_id). The glob `**/routing_data.json` handles this automatically.
- **`/checkpoints/` paths**: Modal evals write checkpoint paths as `/checkpoints/<task_id>/...`. These are remapped to `modal_checkpoints/<task_id>/...` in the CSV.
