# Project Reference

## Overview

Reinforcement learning research project training a **Mixture-of-Experts (MoE)** policy on BabyAI tasks. The pipeline covers training (locally or on Modal cloud), evaluation, caching, and interactive visualization via marimo.

---

## Directory Structure

```
rl-mop/
├── config.yaml                  # Base training config
├── mixture_of_experts.py        # MoE policy: Expert, Router, MixtureOfExpertsPolicy
├── train.py                     # Baseline GRU training
├── train_mop.py                 # MoE training
├── eval_mop.py                  # evaluate() + EvalVectorEnv
├── batch_eval.py                # Batch eval over all checkpoints → CSV + cache
├── modal_app.py                 # Cloud training (Modal.com)
├── plotting_utils.py            # Stateless, reusable plotting functions
├── routing_viz.py               # Marimo notebook: interactive visualization
├── evaluation_results.csv       # Summary metrics across all checkpoints
├── checkpoints/                 # Local checkpoints
├── modal_checkpoints/           # Cloud-trained checkpoints (20+ tasks)
├── evaluation_cache/            # Cached routing data (JSON) from batch_eval
└── plots/                       # Saved heatmap PNGs
```

---

## Data Flow Pipeline

```
train_mop.py / modal_app.py
        ↓
modal_checkpoints/<task_id>/trial_<N>/checkpoint_final.pt
        ↓
batch_eval.py  (or routing_viz.py for on-demand eval)
        ↓
evaluation_results.csv          ← summary metrics, one row per checkpoint
evaluation_cache/<task_id>/trial_<N>/routing_data.json  ← full routing data
        ↓
routing_viz.py  (marimo)
        ↓
plots/<task_id>/trial_<N>/routing_heatmap[_by_starting_room].png
```

---

## Key Data Formats

### Checkpoint (`.pt`)
```python
{
  "policy_state_dict": ...,      # MixtureOfExpertsPolicy weights
  "config": {                    # Full training config
      "task_id", "trial", "input_dim", "intermediate_dim",
      "expert_hidden_sizes",     # e.g. [[0,16,32],[0,16,32],[0,16,32]]
      "router_hidden_size", "num_actions", "lang_dim",
      "lr", "lpc_alpha", "num_updates", "unroll_len", "max_steps"
  },
  "lang_proj_state_dict": ...    # DistilBERT → lang_dim projection weights
}
```

### Routing Data Cache (`routing_data.json`)
```python
{
  "checkpoint_path": "modal_checkpoints/...",
  "num_episodes": 1000,
  "metrics": {"success_rate", "path_ratio", "mean_lpc", "bot_plan_failures"},
  "routing_data": [              # One entry per timestep across all episodes
    {
      "position": [x, y],
      "layer_routing": {
          "layer_0": [w0, w1, w2],   # Softmax router weights per expert
          "layer_1": [...],
          "layer_2": [...]
      },
      "lpc": float,              # Layered Parameter Cost for this step
      "env_context": {           # Captured at episode start
          "grid_size", "agent_start_pos", "agent_start_room",
          "room_grid_shape", "goals", "doors", "keys", "balls", "boxes"
      }
    }, ...
  ]
}
```

### `evaluation_results.csv` Columns
`checkpoint_path, task_id, trial, num_episodes, success_rate, path_ratio, mean_lpc, bot_plan_failures, num_updates, unroll_len, lr, lpc_alpha, expert_hidden_sizes, intermediate_dim, router_hidden_size, max_steps, lang_dim, evaluated_at`

---

## Model Architecture (`mixture_of_experts.py`)

- **3 MoE layers** (configurable via `expert_hidden_sizes`)
- **3 experts per layer**: hidden sizes 0 (identity/skip), 16, 32
- **Router**: GRU-based (`router_hidden_size=64`), outputs softmax weights per expert per layer
- Input → linear projection to `intermediate_dim=64` → MoE layers (weighted sum of expert GRU outputs)
- Language conditioning: DistilBERT (768-dim) → `lang_proj` (→ `lang_dim=32`) concatenated into routing

**LPC (Layered Parameter Cost):** `sum over layers of (router_weight_i × expert_hidden_size_i²)`. Used as a training regularizer (`lpc_alpha`) and evaluation metric.

---

## `plotting_utils.py` — Function Reference

| Function | Purpose |
|---|---|
| `compute_grid_bounds(positions)` | Returns `x_min/max, y_min/max, grid_width/height` from position list |
| `aggregate_routing_by_position(routing_data, filter_fn)` | Averages routing weights and LPC per `(x,y)` cell |
| `render_routing_heatmap(ax, avg_routing_by_pos, grid_info, layer_name)` | Renders one MoE layer's heatmap: dominant expert → hue, confidence → intensity |
| `render_lpc_heatmap(ax, avg_lpc_by_pos, grid_info)` | Renders LPC as viridis heatmap |
| `plot_overall_routing(routing_data, env_image, env_mission, filter_fn, title_suffix)` | Full multi-panel figure: env image + one heatmap per layer + LPC |
| `get_available_analyses(routing_data)` | Returns available grouping modes e.g. `['overall', 'by_starting_room']` |
| `group_routing_data(routing_data, group_by)` | Groups timesteps by an `env_context` field |
| `room_labels_for_groups(sorted_keys, room_grid_shape)` | Maps room coords to labels like `'Top-Left'` |
| `plot_grouped_routing(routing_data, group_by, ...)` | Multi-row figure, one row per group (e.g. per starting room) |

**Constants:** `EXPERT_CMAPS` (6 custom colormaps, one per expert), `UNVISITED_COLOR`

---

## `routing_viz.py` — Marimo Notebook Structure

| Cell | Variables produced | Purpose |
|---|---|---|
| 1 | — | Imports |
| 2 | `checkpoint_dropdown`, `use_cache_toggle`, `num_episodes_slider`, `num_envs_slider` | UI controls; scans `modal_checkpoints/` + `checkpoints/` |
| 3 | `routing_data` | Load from cache or run `evaluate()` fresh; displays metrics |
| 4 | `analysis_dropdown` | Selects analysis mode (overall / by_starting_room) |
| 5 | `room_env_images` | Renders one env screenshot per starting room |
| 6 | `fig_heatmap` | Calls `plot_overall_routing` or `plot_grouped_routing` |
| 7 | — | `mo.mpl.interactive(fig_heatmap)` |
| 8 | — | Save button → `plots/<task_id>/trial_<N>/routing_heatmap*.png` |

---

## Visualization Strategy

**Extend within the existing pattern — don't create new scripts per plot type.**

- New plot functions → add to `plotting_utils.py`
- New interactive views → add tabs in `routing_viz.py` via `mo.ui.tabs()`
- Cross-task comparisons → create a second marimo notebook (e.g. `cross_task_viz.py`) that reads `evaluation_results.csv` + multiple cache files
- Split `plotting_utils.py` into a `plotting/` package only when it exceeds ~1000 lines
- If a new plot needs data not currently cached, extend `evaluate()` in `eval_mop.py` and update the cache schema

**Plot categories available from cached data:**

| Category | What to plot | Data source |
|---|---|---|
| Spatial routing | Dominant expert per grid cell | `routing_data` positions + layer_routing |
| LPC spatial | Computational cost per grid cell | `routing_data` lpc + positions |
| Temporal | Expert weights over timesteps within an episode | `routing_data` ordered by episode |
| Distributions | Histograms of expert weights, routing entropy | `routing_data` layer_routing |
| Cross-task metrics | Success rate vs. LPC trade-offs | `evaluation_results.csv` |
| Spatial variants | Routing near doors/keys/goals | `routing_data` + env_context object positions |
| Expert specialization | Which expert "owns" which behaviour | `routing_data` + episode outcome |
