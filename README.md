# rl-mop

Adaptive imitation learning model investigating deep-learning routing patterns for complex task learning (BabyAI gridworld).

---

## Workflow overview

```
train → evaluate → analyze → plot
```

Each step writes output to a self-contained directory. Use the `_normalised` suffix convention to keep normalised-loss runs isolated from originals (see [Normalised loss runs](#normalised-loss-runs)).

---

## 1. Train

### Local

```bash
python train_mop.py --config config.yaml --task_id BabyAI-GoToObj-v0 --trial 0

# Increase max steps for multi-room tasks
python train_mop.py --task_id BabyAI-GoToImpUnlock-v0 --trial 0 --max_steps 1500

# With LPC regularization (paper eq. 1 — unnormalized)
python train_mop.py --task_id BabyAI-UnlockPickup-v0 --trial 20 --lpc_alpha 1e-4

# With normalized LPC regularization (paper eq. 2)
python train_mop.py --task_id BabyAI-UnlockPickup-v0 --trial 20 --lpc_alpha 1e-4 --normalize_lpc
```

### Modal (cloud)

**Single trial:**
```bash
modal run --detach modal_app.py::main \
  --task-ids "BabyAI-UnlockPickup-v0" --trials "20" --script train_mop.py \
  --extra-args "--lpc_alpha 1e-4 --num_updates 5000"
```

**Multi-seed (recommended for analysis):**
```bash
modal run --detach modal_app.py::main_seeds \
  --task-ids "BabyAI-UnlockPickup-v0" \
  --trials "20,21,22,23,26" \
  --seeds "0,1,2,3,4,5,6,7,8,9" \
  --checkpoint-interval 250 \
  --extra-args "--lpc_alpha 1e-4 --num_updates 5000"
```

Checkpoints are saved to the `rl-mop` Modal volume at:
```
/checkpoints/<task_id>/trial_<N>/seed_<S>/checkpoint_<U>.pt
/checkpoints/<task_id>/trial_<N>/seed_<S>/checkpoint_final.pt  ← alias
```

Download locally:
```bash
modal volume get rl-mop /BabyAI-UnlockPickup-v0/trial_20 ./modal_checkpoints/BabyAI-UnlockPickup-v0/trial_20
# or download everything:
modal volume get rl-mop / ./modal_checkpoints/
```

---

## 2. Evaluate

### On Modal (recommended)

`eval_parallel` discovers unevaluated checkpoints from a local folder and spawns one GPU job per missing checkpoint in parallel:

```bash
# Evaluate all missing checkpoints (1000 episodes each)
modal run modal_app.py::eval_parallel

# Filter by task/trial/seed/update
modal run modal_app.py::eval_parallel --task "BabyAI-UnlockPickup-v0" --trial 20
modal run modal_app.py::eval_parallel --task "BabyAI-UnlockPickup-v0" --trial 20 --seed 3
modal run modal_app.py::eval_parallel --task "BabyAI-UnlockPickup-v0" --trial 20 --update 1500

# Force re-evaluate
modal run modal_app.py::eval_parallel --force
```

Download results:
```bash
modal volume get rl-mop-eval /eval_output/evaluation_results.csv ./evaluation_results.csv
modal volume get rl-mop-eval /eval_output/evaluation_cache ./evaluation_cache

# Download only one task's cache (avoids overwriting others)
modal volume get rl-mop-eval /eval_output/evaluation_cache/BabyAI-UnlockPickup-v0 ./evaluation_cache/
```

### Local batch eval

```bash
python batch_eval.py                                      # all new checkpoints
python batch_eval.py --force                              # re-evaluate all
python batch_eval.py --task BabyAI-UnlockPickup-v0 --trial 20 --seed 0 --update 1500
python batch_eval.py --skip_routing                       # faster, no routing cache
```

Results go to `evaluation_results.csv`; routing caches go to `evaluation_cache/`.

### Single checkpoint

```bash
python eval_mop.py --checkpoint checkpoints/BabyAI-GoToObj-v0/trial_0/checkpoint_final.pt --num_episodes 100
```

---

## 3. Analyze & Plot

### Per-trial routing heatmaps

```bash
python analyze.py <task_id> <trial> [plot_type] [--seed S] [--update U]

# Examples
python analyze.py BabyAI-UnlockPickup-v0 20 overall --seed 3
python analyze.py BabyAI-UnlockPickup-v0 20 kl_heatmap --seed 3 --update 1500
python analyze.py BabyAI-UnlockPickup-v0 20 by_key_phase --seed 3
```

Available plot types: `overall`, `by_starting_room`, `by_door_location`, `by_door_and_box_row`, `by_carrying_phase`, `by_door_unlocked_phase`, `by_key_phase`, `by_agent_and_target_quadrant`, `action_frequency`, `action_frequency_carrying`, `across_episode_entropy_heatmap`, `kl_heatmap`, `kl_heatmap_global`, `cell_action_distribution`, and grouped variants of each.

Output: `plots/<task_id>/trial_<N>/[seed_<S>/][update_<U>/]<plot_type>.png`

### Spatial statistics

```bash
python stats.py <task_id> <trial> [group_by] [--seed S] [--update U]
```

Prints per-timestep and spatial correlation tables to stdout (no output files).

### Seed-aggregated correlation plots

Produces phase-subplot figures aggregating r-values across seeds. Each trial = one alpha value; seeds = unit of replication.

```bash
python seed_agg_plots.py BabyAI-UnlockPickup-v0 --trials 20,21,22,23,26 --update 5000 --save
```

Output: `corr_plots/<task_id>/<task_id>_seed_agg_phase_subplots.png`

### Metrics vs alpha / update plots

```bash
python metrics_vs_alpha_plots.py --csv eval_metrics_unlockpickup.csv --save
```

Output: `corr_plots/<task_id>/`

### Pooled scatter and regression plots

```bash
python analyze_unlockpickup_pooled.py --csv eval_metrics_unlockpickup.csv --out_dir plots/pooled --save
```

### Compute seed metrics CSV (on Modal)

Reads all routing caches for a task and computes per-seed correlation and complexity metrics into a compact CSV:

```bash
modal run modal_app.py::run_extract_seed_metrics --trials "20,21,22,23,26"
# Download:
modal volume get rl-mop-eval /eval_output/eval_metrics_unlockpickup.csv ./eval_metrics_unlockpickup.csv
```

---

## Normalised loss runs

The paper normalises the LPC routing cost by the task loss (eq. 2):

```
L = L_response + α · LPC / (L_response + ε)
```

Use `--normalize_lpc` to enable this. All outputs go to `_normalised`-suffixed directories to keep them isolated from original results.

### Train on Modal

```bash
modal run --detach modal_app.py::main_seeds \
  --task-ids "BabyAI-UnlockPickup-v0" \
  --trials "20,21,22,23,26" \
  --seeds "0,1,2,3,4,5,6,7,8,9" \
  --checkpoint-interval 250 \
  --checkpoint-dir "/checkpoints_normalised" \
  --extra-args "--normalize_lpc --lpc_alpha 1e-4 --num_updates 5000"
```

Checkpoints land in the separate `rl-mop-normalised` Modal volume. Download:
```bash
modal volume get rl-mop-normalised /checkpoints_normalised ./modal_checkpoints_normalised
```

### Evaluate

```bash
# On Modal
modal run modal_app.py::eval_parallel \
  --checkpoint-base-local "modal_checkpoints_normalised" \
  --checkpoint-dir "/checkpoints_normalised" \
  --cache-dir "/eval_output/evaluation_cache_normalised" \
  --results-path "/eval_output/evaluation_results_normalised.csv" \
  --results-path-local "evaluation_results_normalised.csv"

# Download
modal volume get rl-mop-eval /eval_output/evaluation_results_normalised.csv ./evaluation_results_normalised.csv
modal volume get rl-mop-eval /eval_output/evaluation_cache_normalised ./evaluation_cache_normalised

# Or locally
python batch_eval.py \
  --checkpoint_dir modal_checkpoints_normalised \
  --results_path evaluation_results_normalised.csv \
  --cache_dir evaluation_cache_normalised
```

### Analyze & plot

All analysis scripts accept directory flags that default to the original paths, so just pass the `_normalised` variants:

```bash
python analyze.py BabyAI-UnlockPickup-v0 20 overall --seed 3 \
  --cache_dir evaluation_cache_normalised --plots_dir plots_normalised

python stats.py BabyAI-UnlockPickup-v0 20 --seed 3 \
  --cache_dir evaluation_cache_normalised

python seed_agg_plots.py BabyAI-UnlockPickup-v0 --trials 20,21,22,23,26 --update 5000 --save \
  --cache_dir evaluation_cache_normalised \
  --results_path evaluation_results_normalised.csv \
  --output corr_plots_normalised/BabyAI-UnlockPickup-v0/BabyAI-UnlockPickup-v0_seed_agg_phase_subplots.png

python metrics_vs_alpha_plots.py --save \
  --csv eval_metrics_unlockpickup_normalised.csv \
  --output_dir corr_plots_normalised/BabyAI-UnlockPickup-v0/

python analyze_unlockpickup_pooled.py --save \
  --csv eval_metrics_unlockpickup_normalised.csv \
  --out_dir plots_normalised/pooled
```

Compute seed metrics CSV on Modal for normalised runs:
```bash
modal run modal_app.py::run_extract_seed_metrics \
  --trials "20,21,22,23,26" \
  --cache-dir "/eval_output/evaluation_cache_normalised" \
  --output-csv "/eval_output/eval_metrics_unlockpickup_normalised.csv"

modal volume get rl-mop-eval /eval_output/eval_metrics_unlockpickup_normalised.csv \
  ./eval_metrics_unlockpickup_normalised.csv
```

---

## Output directory reference

| Directory | Contents |
|-----------|----------|
| `checkpoints/` | Local training checkpoints |
| `modal_checkpoints/` | Downloaded from `rl-mop` Modal volume |
| `modal_checkpoints_normalised/` | Downloaded from `rl-mop-normalised` Modal volume |
| `evaluation_cache/` | Routing data JSON caches (original runs) |
| `evaluation_cache_normalised/` | Routing data JSON caches (normalised runs) |
| `evaluation_results.csv` | Eval metrics per checkpoint (original) |
| `evaluation_results_normalised.csv` | Eval metrics per checkpoint (normalised) |
| `eval_metrics_unlockpickup.csv` | Pre-computed seed metrics (original) |
| `eval_metrics_unlockpickup_normalised.csv` | Pre-computed seed metrics (normalised) |
| `plots/` | Per-trial routing/entropy/KL plots (original) |
| `plots_normalised/` | Per-trial plots (normalised) |
| `corr_plots/` | Seed-aggregated and alpha-sweep plots (original) |
| `corr_plots_normalised/` | Seed-aggregated and alpha-sweep plots (normalised) |

## Modal volumes

| Volume name | Mount path | Contents |
|-------------|-----------|----------|
| `rl-mop` | `/checkpoints` | Original training checkpoints |
| `rl-mop-normalised` | `/checkpoints_normalised` | Normalised-loss checkpoints |
| `rl-mop-eval` | `/eval_output` | Eval caches and CSVs (both variants) |
