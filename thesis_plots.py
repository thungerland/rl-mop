"""
thesis_plots.py — Publication-ready final figures for the Master's thesis.

Usage:
    python thesis_plots.py --plot metrics_vs_alpha [--save]
    python thesis_plots.py --plot lpc_heatmaps [--save]
    python thesis_plots.py --plot entropy_heatmaps [--save]
    python thesis_plots.py --plot kl_local_heatmaps [--save]
    python thesis_plots.py --plot kl_global_heatmaps [--save]
    python thesis_plots.py --plot seed_agg_A [--save]           # α = 1e-6, 1e-5, 1e-4
    python thesis_plots.py --plot seed_agg_B [--save]           # α = 1e-3, 1e-2
    python thesis_plots.py --plot seed_agg_C [--save]           # α = 1e-6 … 1e-2
    python thesis_plots.py --plot corr_bar_A [--save]           # successful α bars
    python thesis_plots.py --plot corr_bar_B [--save]           # unsuccessful α bars
    python thesis_plots.py --plot corr_bar_C [--save]           # A and B side-by-side
    python thesis_plots.py --plot all [--save]

Add --normalised to any command to use the normalised CSV and save to plots/thesis/normalised/.

Plots:
  metrics_vs_alpha     — Dual-axis line chart: success rate and mean LPC vs α
                         (unnormalised and normalised side-by-side).

  lpc_heatmaps         — 4 phases × 4 α columns grid of spatial mean LPC heatmaps.
                         Seeds pooled per α column.

  entropy_heatmaps     — Same grid layout for empirical action entropy H(A|S).

  kl_local_heatmaps    — Same grid layout for KL divergence vs per-phase local P_a.

  kl_global_heatmaps   — Same grid layout for KL divergence vs global P_a
                         (pooled across all phases for each α).

  seed_agg_A/B/C       — Seed-aggregated Pearson r vs α line chart, one subplot per
                         phase. Three metric lines (H(A|S), KL local, KL global) with
                         Fisher z SEM bands, seed scatter, and success rate overlay.
                         Stars indicate significance (bold, inline).

  corr_bar_A           — Grouped bar chart of Fisher z aggregated r by phase and metric,
                         pooling α ∈ {1e-6, 1e-5, 1e-4} (successful models, //// hatch).

  corr_bar_B           — Same for α ∈ {1e-3, 1e-2} (unsuccessful models, .... hatch).

  corr_bar_C           — Versions A and B plotted side-by-side for each metric/phase.

Dependencies: numpy, pandas, matplotlib, seaborn, scipy
"""

import argparse
import json
import pathlib

import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats as scipy_stats

from analyze_unlockpickup_pooled import (
    _fisher_aggregate,
    _init_phases as _init_pool_phases,
    _sig_stars,
    load_data as _load_pooled_data,
)
from corr_plots import PHASE_LABELS, PHASE_LIST, TASK_PHASE_SYSTEM, load_alpha_map, sig_marker
from metrics_vs_alpha_plots import ALPHA_ORDER, ALPHA_LABELS, _alpha_label, aggregate_by_group
from plotting_utils import (
    UNVISITED_COLOR,
    _compute_group_across_episode_entropy,
    aggregate_routing_by_position,
    build_routing_data_tuples,
    compute_empirical_entropy,
    compute_grid_bounds,
    group_routing_data,
    key_phase_labels_for_groups,
    unlock_phase_labels_for_groups,
)
from seed_agg_plots import aggregate_seed_records, discover_seeded_caches, load_seed_records_from_csv


# ── Global font / style settings ──────────────────────────────────────────────

class _FontSizes:
    """Font size preset for publication figures.

    Two presets are defined below:
    - ``_FS_FULLWIDTH``  — for plots that span the full NeurIPS column width
    - ``_FS_HALFWIDTH``  — for plots that occupy only half the column width
      (labels must be larger so they remain legible when the figure is scaled down)
    """
    def __init__(
        self,
        base:       float = 9.0,
        axis_label: float = 9.0,
        axis_title: float = 9.0,
        tick:       float = 8.0,
        legend:     float = 7.5,
        annot:      float = 7.0,
    ) -> None:
        self.base       = base        # rcParams font.size
        self.axis_label = axis_label  # axes.labelsize / set_ylabel / set_xlabel
        self.axis_title = axis_title  # axes.titlesize / set_title
        self.tick       = tick        # xtick.labelsize / ytick.labelsize
        self.legend     = legend      # legend.fontsize
        self.annot      = annot       # significance stars and small annotations


# Preset for full-width figures (calibrated for NeurIPS 10 pt body, ~6.75 in column)
_FS_FULLWIDTH = _FontSizes(base=11, axis_label=11, axis_title=11, tick=10, legend=9.5, annot=9)

# Preset for half-width figures — bump sizes so labels remain legible at ~3.25 in
_FS_HALFWIDTH = _FontSizes(base=12, axis_label=12, axis_title=12, tick=11, legend=10.5, annot=10)


def _apply_pub_style(fs: _FontSizes = _FS_FULLWIDTH) -> None:
    """Apply publication-ready font and style settings (Arial, consistent sizes)."""
    plt.rcParams.update({
        'font.family':      'sans-serif',
        'font.sans-serif':  ['Arial', 'Helvetica', 'DejaVu Sans'],
        'font.size':        fs.base,
        'axes.labelsize':   fs.axis_label,
        'axes.titlesize':   fs.axis_title,
        'xtick.labelsize':  fs.tick,
        'ytick.labelsize':  fs.tick,
        'legend.fontsize':  fs.legend,
        'figure.dpi':       100,
    })
    sns.set_style('whitegrid')


# ── Constants ─────────────────────────────────────────────────────────────────

TASK_ID_DEFAULT = 'BabyAI-UnlockPickup-v0'

# Alpha values shown in heatmap columns (exclude 1e-3 where performance breaks down)
HEATMAP_ALPHA_ORDER = [0.0, 1e-6, 1e-5, 1e-4]

# Trials to include in heatmaps (subset of all trials in evaluation_results.csv)
HEATMAP_TRIALS = {20, 21, 22, 26}

# key_phase group keys match group_routing_data() output — integer 1-tuples
KEY_PHASES = [(0,), (1,), (2,), (3,)]

# Per-task configuration: CSV paths, output subdirectory, heatmap trials, and phase settings.
_TASK_CONFIGS = {
    'unlockpickup': {
        'task_id':       'BabyAI-UnlockPickup-v0',
        'csv_unnorm':    'eval_metrics_unlockpickup.csv',
        'csv_norm':      'eval_metrics_unlockpickup_normalised.csv',
        'out_subdir':    None,                      # saves directly to base out_dir
        'heatmap_trials': {20, 21, 22, 26},
        'phase_keys':    [(0,), (1,), (2,), (3,)],
        'phase_group_by': 'key_phase',
        'max_seed_mb':   None,                      # use all seeds
        'fig_width_bar': None,                      # use default (3.2 * n_phases)
        'bar_title_extra': None,                    # no task name in title
    },
    'opentwo_doors': {
        'task_id':       'BabyAI-OpenTwoDoors-v0',
        'csv_unnorm':    'eval_metrics_opentwodoorsv0.csv',
        'csv_norm':      None,                      # no normalised CSV exists
        'out_subdir':    'OpenTwoDoors',
        'heatmap_trials': {20, 21, 22, 23},
        'phase_keys':    [(0,), (1,)],
        'phase_group_by': 'unlock_phase',
        'max_seed_mb':   30,                        # skip seeds with cache > 30 MB
        'fig_width_bar': None,                      # use default (3.2 * n_phases)
        'bar_title_extra': 'BabyAI-OpenTwoDoors-v0',
    },
}

# Plasma colours: LPC = light end, success = dark end
_COLOR_SUCCESS = plt.cm.plasma(0.55)  # magenta-purple, clearly distinct from LPC
_COLOR_LPC     = plt.cm.plasma(0.20)  # light blue

# Plasma metric colors — consistent between seed-agg line chart and correlation bar chart.
# Chosen to be well-separated from each other and from _COLOR_SUCCESS (0.55).
_THESIS_METRIC_LINES = [
    {'corr': 'lpc_entropy',   'metric': 'entropy',   'label': 'H(A|S) vs LPC',      'color': plt.cm.plasma(0.05), 'ls': '-', 'lw': 2.0},
    {'corr': 'lpc_kl_local',  'metric': 'kl_local',  'label': 'KL (local) vs LPC',  'color': plt.cm.plasma(0.65), 'ls': '-', 'lw': 2.0},
    {'corr': 'lpc_kl_global', 'metric': 'kl_global', 'label': 'KL (global) vs LPC', 'color': plt.cm.plasma(0.40), 'ls': '-', 'lw': 2.0},
]

# Two-line phase x-tick labels for bar chart (matches heatmap style)
_PHASE_TICK_LABELS_2LINE = {
    'pre_key':              'Pre-key',
    'post_key_pre_unlock':  'With-key\n(pre-unlock)',
    'with_key_post_unlock': 'With-key\n(post-unlock)',
    'post_unlock_post_key': 'Post-unlock\n(post-key)',
    'pre_unlock':           'Pre-unlock',
    'post_unlock':          'Post-unlock',
}

# Alpha range groupings for the new thesis plots
_ALPHA_SUCCESSFUL   = [1e-6, 1e-5, 1e-4]
_ALPHA_UNSUCCESSFUL = [1e-3, 1e-2]   # 1e-2 silently skipped if absent in CSV
_ALPHA_COMBINED     = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2]

# Hatch patterns distinguishing successful vs unsuccessful alpha bars
_HATCH_SUCCESSFUL   = '////'
_HATCH_UNSUCCESSFUL = '....'


# ── Figure 1: Success rate + Mean LPC vs α ────────────────────────────────────

def _load_and_filter_csv(csv_path: str, task_id: str) -> pd.DataFrame:
    """Load CSV, filter to task_id and ALPHA_ORDER, keep final checkpoint per (trial, seed)."""
    path = pathlib.Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    df = pd.read_csv(path)
    if 'task_id' in df.columns:
        df = df[df['task_id'] == task_id].copy()
    if df.empty:
        raise ValueError(f"No rows for task_id={task_id!r} in {csv_path}")
    idx = df.groupby(['trial', 'seed'])['update'].transform('max')
    df = df[df['update'] == idx].copy()
    df = df[df['lpc_alpha'].isin(ALPHA_ORDER)].copy()
    return df


def _load_success_rate_by_alpha(
    csv_path: str,
    task_id: str,
    alpha_subset: list | None = None,
) -> dict:
    """Return {alpha: (mean, sem)} for success_rate at the final checkpoint per (trial, seed).

    Uses inline loading (not _load_and_filter_csv) to avoid being constrained to ALPHA_ORDER.
    """
    df = pd.read_csv(csv_path)
    if 'task_id' in df.columns:
        df = df[df['task_id'] == task_id].copy()
    if df.empty:
        return {}
    idx = df.groupby(['trial', 'seed'])['update'].transform('max')
    df = df[df['update'] == idx].copy()
    if alpha_subset is not None:
        df = df[df['lpc_alpha'].isin(alpha_subset)]
    result = {}
    for alpha, grp in df.groupby('lpc_alpha'):
        vals = grp['success_rate'].dropna()
        if len(vals) == 0:
            continue
        result[float(alpha)] = (float(vals.mean()), float(vals.sem()))
    return result


def _draw_dual_axis_panel(ax, df: pd.DataFrame, subplot_title: str) -> None:
    """Draw success rate (left axis) and mean LPC (right axis) on ax."""
    agg_df, raw_df = aggregate_by_group(df, 'lpc_alpha')
    agg_df['_order'] = agg_df['lpc_alpha'].map({a: i for i, a in enumerate(ALPHA_ORDER)})
    agg_df = agg_df.sort_values('_order').drop(columns='_order').reset_index(drop=True)

    present_alphas = list(agg_df['lpc_alpha'].values)
    x_positions = list(range(len(present_alphas)))
    x_labels = [_alpha_mathtext(a, include_symbol=False) for a in present_alphas]
    alpha_to_x = {a: i for i, a in enumerate(present_alphas)}

    rng = np.random.default_rng(42)

    ax_lpc = ax.twinx()

    for metric_col, color, target_ax in [
        ('success_rate', _COLOR_SUCCESS, ax),
        ('mean_lpc',     _COLOR_LPC,     ax_lpc),
    ]:
        xs, ys, y_lo, y_hi = [], [], [], []
        for _, row in agg_df.iterrows():
            a = row['lpc_alpha']
            x = alpha_to_x.get(a)
            if x is None:
                continue
            xs.append(x)
            ys.append(row[f'{metric_col}_mean'])
            y_lo.append(row[f'{metric_col}_lo'])
            y_hi.append(row[f'{metric_col}_hi'])

        valid = [i for i, y in enumerate(ys) if not np.isnan(y)]
        if valid:
            vx  = [xs[i]   for i in valid]
            vy  = [ys[i]   for i in valid]
            vlo = [y_lo[i] for i in valid]
            vhi = [y_hi[i] for i in valid]
            target_ax.plot(vx, vy, color=color, linewidth=2.0, marker='o',
                           markersize=5, zorder=4)
            if any(lo != hi for lo, hi in zip(vlo, vhi)):
                target_ax.fill_between(vx, vlo, vhi, color=color, alpha=0.22, zorder=3)

        # Individual seed scatter
        for _, srow in raw_df.iterrows():
            a = srow['lpc_alpha']
            x = alpha_to_x.get(a)
            if x is None:
                continue
            val = srow[metric_col]
            if np.isnan(val):
                continue
            target_ax.scatter(x + rng.uniform(-0.18, 0.18), val,
                              color=color, alpha=0.25, s=12, zorder=2, linewidths=0)

    # Axis styling
    ax.set_xlim(-0.5, len(x_positions) - 0.5)
    ax.set_xticks(x_positions)
    ax.set_xticklabels(x_labels, rotation=0, ha='center', fontsize=8)
    ax.set_xlabel('α (regularisation)', fontsize=9)
    ax.grid(False)  # remove all gridlines (whitegrid adds them)
    ax.set_ylim(-0.02, 1.05)
    ax.set_ylabel('Success rate', fontsize=9, color=_COLOR_SUCCESS)
    ax.tick_params(axis='y', labelcolor=_COLOR_SUCCESS, labelsize=8)

    # Force the right axis to use the same number of ticks as the left so
    # gridlines coincide. Gridlines are drawn by the left axis only.
    n_ticks = len(ax.get_yticks())
    lpc_lo, lpc_hi = ax_lpc.get_ylim()
    ax_lpc.set_yticks(np.linspace(lpc_lo, lpc_hi, n_ticks))
    ax_lpc.grid(False)

    ax_lpc.set_ylabel('Mean LPC', fontsize=9, color=_COLOR_LPC)
    ax_lpc.tick_params(axis='y', labelcolor=_COLOR_LPC, labelsize=8)

    ax.set_title(subplot_title, fontsize=9, pad=5)

    # Legend
    handles = [
        plt.Line2D([0], [0], color=_COLOR_SUCCESS, linewidth=2, marker='o',
                   markersize=5, label='Success rate'),
        plt.Line2D([0], [0], color=_COLOR_LPC, linewidth=2, marker='o',
                   markersize=5, label='Mean LPC'),
    ]
    ax.legend(handles=handles, fontsize=7.5, loc='upper right', frameon=True)


def plot_metrics_vs_alpha(
    csv_unnorm: str,
    csv_norm: str,
    task_id: str = TASK_ID_DEFAULT,
) -> plt.Figure:
    """Two-subplot figure: unnormalised (left) and normalised (right) success + LPC vs α."""
    _apply_pub_style()

    df_unnorm = _load_and_filter_csv(csv_unnorm, task_id)
    df_norm   = _load_and_filter_csv(csv_norm,   task_id)

    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(11, 4.5))

    _draw_dual_axis_panel(ax_left,  df_unnorm, 'Unnormalised')
    _draw_dual_axis_panel(ax_right, df_norm,   'Normalised')

    fig.tight_layout()
    fig.suptitle(
        r'Mean Learned Pathway Complexity (LPC) and Agent Success Rate vs Regularisation ($\alpha$)',
        fontsize=11, y=1.001,
    )
    return fig


# ── Figure 2: Per-phase heatmap grid ─────────────────────────────────────────

def _load_seeded_routing_data(
    task_id: str,
    alpha: float,
    update: int,
    alpha_map: dict,
    cache_dir: str,
    heatmap_trials: set | None = None,
    max_seed_mb: float | None = None,
) -> list:
    """Pool routing_data from seeds for a given alpha value.

    Pooling seeds before computing entropy preserves spatial contrast:
    more visits per position reduces the relative weight of Dirichlet smoothing,
    so rarely-visited corners show lower entropy vs frequently-visited centre cells.

    max_seed_mb: if set, skip any seed whose cache file exceeds this size in MB.
    """
    _trials = heatmap_trials if heatmap_trials is not None else HEATMAP_TRIALS
    trials_for_alpha = [t for t, a in alpha_map.items() if a == alpha and t in _trials]
    routing_data = []
    for trial in sorted(trials_for_alpha):
        seed_caches = discover_seeded_caches(task_id, trial, cache_dir, update)
        for seed_num, upd_num, path in seed_caches:
            if max_seed_mb is not None:
                size_mb = pathlib.Path(path).stat().st_size / 1e6
                if size_mb > max_seed_mb:
                    print(f'    [skip] seed_{seed_num} ({size_mb:.0f} MB > {max_seed_mb} MB limit)')
                    continue
            with open(path) as f:
                cache = json.load(f)
            routing_data.extend(build_routing_data_tuples(cache))
    return routing_data


def _render_heatmap(ax, grid: np.ndarray, grid_info: dict, vmin, vmax, cmap) -> object:
    """Render a single heatmap cell; return the image object for colorbar use."""
    grid_width = grid_info['grid_width']
    grid_height = grid_info['grid_height']

    im = ax.imshow(grid, origin='upper', cmap=cmap, vmin=vmin, vmax=vmax,
                   aspect='equal')

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xticks(np.arange(-0.5, grid_width, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, grid_height, 1), minor=True)
    ax.grid(which='minor', color='white', linestyle='-', linewidth=0.5)
    ax.tick_params(which='minor', size=0)
    return im


def _lpc_grid(avg_lpc_by_pos: dict, grid_info: dict) -> np.ndarray:
    """Build a 2-D LPC array from position dict."""
    g = np.full((grid_info['grid_height'], grid_info['grid_width']), np.nan)
    for (x, y), val in avg_lpc_by_pos.items():
        g[y - grid_info['y_min'], x - grid_info['x_min']] = val
    return g


def _entropy_grid(H_s: dict, grid_info: dict) -> np.ndarray:
    """Build a 2-D entropy array from position dict."""
    g = np.full((grid_info['grid_height'], grid_info['grid_width']), np.nan)
    for (x, y), val in H_s.items():
        g[y - grid_info['y_min'], x - grid_info['x_min']] = val
    return g


def _kl_grid(KL_s: dict, grid_info: dict) -> np.ndarray:
    """Build a 2-D KL divergence array from position dict."""
    g = np.full((grid_info['grid_height'], grid_info['grid_width']), np.nan)
    for (x, y), val in KL_s.items():
        g[y - grid_info['y_min'], x - grid_info['x_min']] = val
    return g


def _compute_group_kl_local(group_data: list, min_visits: int = 5) -> dict:
    """KL divergence per position using the per-group local marginal P_a as reference."""
    emp = compute_empirical_entropy(group_data, min_visits=min_visits)
    return {pos: val for pos, val in emp['KL_s'].items()
            if emp['include_mask'].get(pos, False)}


def _compute_group_kl_global(group_data: list, global_P_a: np.ndarray,
                              min_visits: int = 5) -> dict:
    """KL divergence per position using a pre-computed global marginal P_a as reference."""
    emp = compute_empirical_entropy(group_data, min_visits=min_visits, P_a=global_P_a)
    return {pos: val for pos, val in emp['KL_s'].items()
            if emp['include_mask'].get(pos, False)}


def _sample_env_image(task_id: str) -> np.ndarray | None:
    """Sample a single environment render. Returns None if gymnasium is unavailable."""
    try:
        import gymnasium as gym
        import minigrid  # noqa: F401
        env = gym.make(task_id, render_mode='rgb_array')
        env.reset()
        img = env.unwrapped.get_frame(tile_size=32, agent_pov=False, highlight=False)
        env.close()
        return img
    except Exception as e:
        print(f"[warn] Could not sample env image: {e}")
        return None


def _alpha_mathtext(a: float, include_symbol: bool = True) -> str:
    """Format alpha as mathtext.

    include_symbol=True  → '$\\alpha = 0$', '$\\alpha = 10^{-6}$'  (for column headers)
    include_symbol=False → '$0$', '$10^{-6}$'                       (for x-tick labels)
    """
    if a == 0.0:
        return r'$\alpha = 0$' if include_symbol else r'$0$'
    exp = int(round(np.log10(a)))
    if include_symbol:
        return rf'$\alpha = 10^{{{exp}}}$'
    return rf'$10^{{{exp}}}$'


def plot_phase_alpha_heatmaps(
    task_id: str = TASK_ID_DEFAULT,
    update: int = 5000,
    cache_dir: str = 'evaluation_cache',
    results_path: str = 'evaluation_results.csv',
    metric: str = 'lpc',  # 'lpc' or 'entropy'
    min_visits: int = 5,
    env_image: np.ndarray | None = None,
    heatmap_trials: set | None = None,
    phase_keys: list | None = None,
    phase_group_by: str = 'key_phase',
    max_seed_mb: float | None = None,
) -> plt.Figure:
    """4-row × 4-col heatmap grid: rows = key_phase, cols = alpha values.

    metric: 'lpc' → mean LPC per cell; 'entropy' → H(A|S) per cell.
    All columns share a single plasma colorbar.
    """
    assert metric in ('lpc', 'entropy', 'kl_local', 'kl_global'), \
        "metric must be 'lpc', 'entropy', 'kl_local', or 'kl_global'"

    alpha_map = load_alpha_map(task_id, results_path)
    if not alpha_map:
        raise RuntimeError(f"No alpha map found for {task_id} in {results_path}")

    cmap = plt.cm.plasma.copy()
    cmap.set_bad(color=UNVISITED_COLOR)

    effective_phase_keys = phase_keys if phase_keys is not None else KEY_PHASES
    effective_heatmap_trials = heatmap_trials if heatmap_trials is not None else HEATMAP_TRIALS

    n_phases = len(effective_phase_keys)
    n_alphas = len(HEATMAP_ALPHA_ORDER)

    # ── Pass 1: load all data, compute grids, find global bounds ──────────────
    print(f"[info] Loading routing data for metric='{metric}' ...")

    # We need global grid_info across all alpha×phase to keep a consistent
    # coordinate system. Collect all positions first.
    all_positions = []
    data_store = {}  # alpha -> pooled routing_data list

    for alpha in HEATMAP_ALPHA_ORDER:
        alpha_label = _alpha_label(alpha)
        print(f"  α = {alpha_label}")
        routing_data = _load_seeded_routing_data(
            task_id, alpha, update, alpha_map, cache_dir,
            heatmap_trials=effective_heatmap_trials,
            max_seed_mb=max_seed_mb,
        )
        if not routing_data:
            print(f"    [warn] No data for α={alpha_label}")
            data_store[alpha] = []
            continue

        all_positions.extend(s['position'] for s in routing_data)
        data_store[alpha] = routing_data

    if not all_positions:
        raise RuntimeError("No routing data loaded for any alpha value.")

    global_grid_info = compute_grid_bounds(all_positions)

    # For kl_global: compute one global P_a per alpha (pooled across all phases)
    alpha_global_P_a = {}
    if metric == 'kl_global':
        for alpha in HEATMAP_ALPHA_ORDER:
            rd = data_store.get(alpha, [])
            if rd:
                alpha_global_P_a[alpha] = compute_empirical_entropy(rd)['P_a']

    # ── Pass 2: build per-(alpha, phase) grids and collect all values ─────────
    grids = {}      # (alpha, phase) -> np.ndarray
    all_vals = []

    # Two-line phase labels to avoid vertical overlap between rows
    _key_phase_label_wrap = {
        (0,): 'Pre-key',
        (1,): 'With-key\n(pre-unlock)',
        (2,): 'With-key\n(post-unlock)',
        (3,): 'Post-unlock\n(post-key)',
    }
    if phase_group_by == 'unlock_phase':
        phase_labels_map = unlock_phase_labels_for_groups(effective_phase_keys)
    else:
        phase_labels_map = {k: _key_phase_label_wrap.get(k, v)
                            for k, v in key_phase_labels_for_groups(effective_phase_keys).items()}

    for alpha in HEATMAP_ALPHA_ORDER:
        routing_data = data_store.get(alpha, [])
        grouped = group_routing_data(routing_data, phase_group_by) if routing_data else {}

        for phase in effective_phase_keys:
            group_data = grouped.get(phase, [])
            if not group_data:
                grids[(alpha, phase)] = None
                continue

            if metric == 'lpc':
                _, avg_lpc, _, _ = aggregate_routing_by_position(group_data)
                g = _lpc_grid(avg_lpc, global_grid_info)
            elif metric == 'entropy':
                H_s = _compute_group_across_episode_entropy(group_data, min_visits=min_visits)
                g = _entropy_grid(H_s, global_grid_info)
            elif metric == 'kl_local':
                KL_s = _compute_group_kl_local(group_data, min_visits=min_visits)
                g = _kl_grid(KL_s, global_grid_info)
            else:  # kl_global
                KL_s = _compute_group_kl_global(
                    group_data, alpha_global_P_a[alpha], min_visits=min_visits
                )
                g = _kl_grid(KL_s, global_grid_info)

            grids[(alpha, phase)] = g
            all_vals.extend(v for v in g.ravel() if np.isfinite(v))

    vmin = float(np.nanmin(all_vals)) if all_vals else 0.0
    vmax = float(np.nanmax(all_vals)) if all_vals else 1.0

    # ── Pass 3: build figure ──────────────────────────────────────────────────
    has_env = env_image is not None

    # Derive env image aspect ratio (h/w) so heatmap cells match its rendered height.
    # The heatmap grid is (grid_height x grid_width) cells; we want the heatmap axes
    # to have the same aspect as the env image so all columns align.
    env_h, env_w = (env_image.shape[:2] if has_env else (1, 1))
    env_aspect = env_h / env_w  # height per unit width

    # The heatmap grid has grid_height rows and grid_width cols of equal-sized cells.
    # With aspect='equal', the axes height = axes_width * (grid_height / grid_width).
    grid_h = global_grid_info['grid_height']
    grid_w = global_grid_info['grid_width']
    heatmap_aspect = grid_h / grid_w  # height per unit width

    # We use a fixed column width (inches) for heatmap columns.
    # The env column width is scaled so that both columns render to the same height.
    hmap_col_w = 2.2       # inches for each heatmap column
    hmap_col_h = hmap_col_w * heatmap_aspect  # rendered height of each heatmap cell

    if has_env:
        env_col_w = hmap_col_h / env_aspect   # env column width so height = hmap_col_h
    else:
        env_col_w = 0.0

    cbar_col_w = 0.25      # inches for colorbar column
    label_margin = 1.4     # left margin for row labels (wider for two-line labels)
    right_margin = 0.15
    top_margin   = 0.75    # extra space between suptitle and first row
    bottom_margin = 0.10

    n_data_cols = n_alphas + (1 if has_env else 0)
    total_data_w = (env_col_w if has_env else 0.0) + hmap_col_w * n_alphas
    fig_w = label_margin + total_data_w + cbar_col_w + right_margin
    fig_h = bottom_margin + hmap_col_h * n_phases + top_margin

    fig = plt.figure(figsize=(fig_w, fig_h))

    # Lay out columns manually in figure-fraction coordinates.
    # We place each cell with fig.add_axes([left, bottom, width, height]) in fig coords.
    # Compute column left edges (in inches from left margin).
    col_widths = []
    if has_env:
        col_widths.append(env_col_w)
    col_widths.extend([hmap_col_w] * n_alphas)

    hgap = 0.08  # inches between columns
    vgap = 0.08  # inches between rows

    col_lefts_in = [label_margin]
    for w in col_widths[:-1]:
        col_lefts_in.append(col_lefts_in[-1] + w + hgap)

    # Row bottoms in inches from figure bottom (row 0 = top phase = largest bottom value)
    row_bottoms_in = []
    for row_idx in range(n_phases):
        # row 0 is at the top
        b = bottom_margin + (n_phases - 1 - row_idx) * (hmap_col_h + vgap)
        row_bottoms_in.append(b)

    def to_fig(x_in, y_in, w_in, h_in):
        """Convert inch coords to figure-fraction [left, bottom, width, height]."""
        return [x_in / fig_w, y_in / fig_h, w_in / fig_w, h_in / fig_h]

    axes_grid = {}  # (row_idx, col_idx) -> ax
    last_im = None

    for row_idx, phase in enumerate(effective_phase_keys):
        col_idx = 0
        b_in = row_bottoms_in[row_idx]

        # Environment image column
        if has_env:
            rect = to_fig(col_lefts_in[col_idx], b_in, env_col_w, hmap_col_h)
            ax_env = fig.add_axes(rect)
            ax_env.imshow(env_image, aspect='auto')
            ax_env.set_xticks([])
            ax_env.set_yticks([])
            for spine in ax_env.spines.values():
                spine.set_visible(False)
            if row_idx == 0:
                ax_env.set_title('Environment', fontsize=9, pad=4)
            label = phase_labels_map.get(phase, str(phase))
            ax_env.set_ylabel(label, fontsize=9, labelpad=4)
            axes_grid[(row_idx, col_idx)] = ax_env
            col_idx += 1

        for alpha in HEATMAP_ALPHA_ORDER:
            l_in = col_lefts_in[col_idx]
            rect = to_fig(l_in, b_in, hmap_col_w, hmap_col_h)
            ax = fig.add_axes(rect)

            g = grids.get((alpha, phase))
            if g is None:
                ax.set_visible(False)
                col_idx += 1
                continue

            im = _render_heatmap(ax, g, global_grid_info, vmin, vmax, cmap)
            last_im = im
            axes_grid[(row_idx, col_idx)] = ax

            if row_idx == 0:
                ax.set_title(_alpha_mathtext(alpha), fontsize=10, pad=8)

            if not has_env and col_idx == 0:
                label = phase_labels_map.get(phase, str(phase))
                ax.set_ylabel(label, fontsize=9, labelpad=4)

            col_idx += 1

    # Colorbar: span exactly from top edge of row-0 to bottom edge of last row
    if last_im is not None:
        cbar_top_in    = row_bottoms_in[0] + hmap_col_h           # top of first row
        cbar_bottom_in = row_bottoms_in[n_phases - 1]              # bottom of last row
        cbar_h_in      = cbar_top_in - cbar_bottom_in
        cbar_l_in      = col_lefts_in[-1] + hmap_col_w + hgap     # right of last hmap col
        cbar_w_in      = cbar_col_w

        cbar_rect = to_fig(cbar_l_in, cbar_bottom_in, cbar_w_in, cbar_h_in)
        cbar_ax = fig.add_axes(cbar_rect)
        cb = fig.colorbar(last_im, cax=cbar_ax)
        _cbar_labels = {
            'lpc':      'Mean LPC',
            'entropy':  'H(A|S) (bits)',
            'kl_local': 'KL local (bits)',
            'kl_global':'KL global (bits)',
        }
        cb.set_label(_cbar_labels[metric], fontsize=10)
        cb.ax.tick_params(labelsize=9)

    # Centre the title over the data columns (between the row-label margin and the colorbar)
    data_left_in  = label_margin
    data_right_in = col_lefts_in[-1] + hmap_col_w
    title_x = (data_left_in + data_right_in) / 2 / fig_w
    title_y = (row_bottoms_in[0] + hmap_col_h + 0.48) / fig_h

    _suptitles = {
        'lpc':      'Phase-state-dependent mean LPC',
        'entropy':  'Phase-state-dependent empirical action entropy H(A|S)',
        'kl_local': 'Phase-state-dependent KL divergence (local)',
        'kl_global':'Phase-state-dependent KL divergence (global)',
    }
    fig.suptitle(_suptitles[metric], fontsize=12, x=title_x, y=title_y, ha='center')

    return fig


# ── Figure 3: Seed-aggregated phase line plots ────────────────────────────────

def _aggregate_corr_stats(df: pd.DataFrame, phases: list) -> dict:
    """Fisher z aggregate r per (metric, phase) from a pooled DataFrame.

    Returns {metric: {'r': [...], 'ci_lo': [...], 'ci_hi': [...], 'p': [...], 'N': [...]}}
    with one entry per phase (in order).
    """
    data = {
        spec['metric']: {'r': [], 'ci_lo': [], 'ci_hi': [], 'p': [], 'N': []}
        for spec in _THESIS_METRIC_LINES
    }
    for phase in phases:
        lpc_col = f'mean_lpc_{phase}'
        x = df[lpc_col].values if lpc_col in df.columns else np.array([])
        N_phase = int(np.sum(np.isfinite(x)))
        for spec in _THESIS_METRIC_LINES:
            metric = spec['metric']
            r_col = f'r_{phase}_lpc_{metric}'
            r_vals = df[r_col].dropna().values if r_col in df.columns else np.array([])
            r_agg, p_agg, ci_lo, ci_hi = _fisher_aggregate(r_vals)
            data[metric]['r'].append(r_agg)
            data[metric]['ci_lo'].append(ci_lo)
            data[metric]['ci_hi'].append(ci_hi)
            data[metric]['p'].append(p_agg)
            data[metric]['N'].append(N_phase)
    return data


def plot_thesis_seed_agg(
    csv_path: str,
    task_id: str = TASK_ID_DEFAULT,
    alpha_subset: list | None = None,
    sig_level: float = 0.05,
) -> plt.Figure:
    """Seed-aggregated Pearson r vs alpha — one subplot per phase.

    Three metric lines (H(A|S), KL local, KL global) with Fisher z SEM bands and individual
    seed scatter. Success rate overlaid using _COLOR_SUCCESS.

    Three versions are produced by passing different alpha_subset values:
      A: _ALPHA_SUCCESSFUL   = [1e-6, 1e-5, 1e-4]
      B: _ALPHA_UNSUCCESSFUL = [1e-3, 1e-2]
      C: _ALPHA_COMBINED     = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2]

    Missing alpha values are silently skipped.
    """
    _apply_pub_style()

    phases = PHASE_LIST[TASK_PHASE_SYSTEM.get(task_id, 'key_phase')]
    if alpha_subset is None:
        alpha_subset = list(_ALPHA_SUCCESSFUL)

    seed_df = load_seed_records_from_csv(csv_path, task_id, phases, alpha_subset)
    if seed_df.empty:
        raise RuntimeError(f'No seed records found in {csv_path} for task_id={task_id!r}')

    agg_df = aggregate_seed_records(seed_df)
    sr_by_alpha = _load_success_rate_by_alpha(csv_path, task_id, alpha_subset)

    # Resolve x-axis: only alphas present in the aggregated data
    present_alphas = set(agg_df['lpc_alpha'].unique())
    alpha_subset = [a for a in alpha_subset if a in present_alphas or a in sr_by_alpha]
    alpha_subset = sorted(set(alpha_subset), key=lambda a: (a == 0.0, a))  # 0 first, then ascending

    alpha_to_x = {a: i for i, a in enumerate(alpha_subset)}
    x_positions = list(range(len(alpha_subset)))

    # y-range: fixed to [-1, 1] (valid correlation range) with small padding
    y_min, y_max = -1.05, 1.05

    n_phases = len(phases)
    fig, axes = plt.subplots(1, n_phases, figsize=(3.6 * n_phases, 4.6), sharey=True)
    if n_phases == 1:
        axes = [axes]

    rng = np.random.default_rng(42)

    for ax, phase in zip(axes, phases):
        phase_str = str(phase)
        phase_agg = agg_df[agg_df['phase'] == phase_str]
        phase_seed = seed_df[seed_df['phase'] == phase_str]

        # ── Metric lines ──
        sig_data_per_metric = []   # filled during loop, used for significance strips
        for line_spec in _THESIS_METRIC_LINES:
            corr_type = line_spec['corr']
            corr_agg = phase_agg[phase_agg['corr'] == corr_type]
            corr_seed = phase_seed[phase_seed['corr'] == corr_type]

            xs, ys, y_lo, y_hi, ps = [], [], [], [], []
            for a in alpha_subset:
                row_match = corr_agg[corr_agg['lpc_alpha'] == a]
                if row_match.empty:
                    xs.append(alpha_to_x[a]); ys.append(float('nan'))
                    y_lo.append(float('nan')); y_hi.append(float('nan'))
                    ps.append(float('nan'))
                    continue
                row = row_match.iloc[0]
                mz = float(row['mean_z'])
                sz = float(row['se_z']) if not np.isnan(row['se_z']) else float('nan')
                mr = float(row['mean_r'])
                xs.append(alpha_to_x[a]); ys.append(mr); ps.append(float(row['p_agg']))
                if not np.isnan(sz):
                    y_lo.append(float(np.tanh(mz - sz)))
                    y_hi.append(float(np.tanh(mz + sz)))
                else:
                    y_lo.append(mr); y_hi.append(mr)

            valid_idx = [i for i, y in enumerate(ys) if not np.isnan(y)]
            if valid_idx:
                vx  = [xs[i] for i in valid_idx]
                vy  = [ys[i] for i in valid_idx]
                vlo = [y_lo[i] for i in valid_idx]
                vhi = [y_hi[i] for i in valid_idx]
                vp  = [ps[i] for i in valid_idx]

                ax.plot(vx, vy, color=line_spec['color'], linestyle=line_spec['ls'],
                        linewidth=line_spec['lw'], marker='o', markersize=5,
                        label=line_spec['label'], zorder=4)
                if any(lo != hi for lo, hi in zip(vlo, vhi)):
                    ax.fill_between(vx, vlo, vhi, color=line_spec['color'],
                                    alpha=0.22, zorder=3)

            # Individual seed scatter
            for a in alpha_subset:
                seed_vals = corr_seed[corr_seed['lpc_alpha'] == a]['r'].dropna().values
                for r_val in seed_vals:
                    x_jitter = alpha_to_x[a] + rng.uniform(-0.18, 0.18)
                    ax.scatter(x_jitter, r_val, color=line_spec['color'],
                               alpha=0.22, s=12, zorder=2, linewidths=0)

            # Accumulate significance per alpha for this metric (used for inline stars)
            sig_data_per_metric.append({
                'color': line_spec['color'],
                'ps': ps,   # one p per alpha_subset position (may include nan)
                'ys': ys,   # corresponding mean_r values
            })

        # ── Success rate line ──
        sr_xs, sr_ys, sr_lo, sr_hi = [], [], [], []
        for a in alpha_subset:
            if a not in sr_by_alpha:
                continue
            sr_mean, sr_sem = sr_by_alpha[a]
            if np.isnan(sr_mean):
                continue
            sr_xs.append(alpha_to_x[a])
            sr_ys.append(sr_mean)
            sem = sr_sem if not np.isnan(sr_sem) else 0.0
            sr_lo.append(sr_mean - abs(sem))
            sr_hi.append(sr_mean + abs(sem))

        if sr_xs:
            ax.plot(sr_xs, sr_ys, color=_COLOR_SUCCESS, linestyle='--',
                    linewidth=2.0, marker='s', markersize=5,
                    label='Success rate', zorder=4)
            if any(lo != hi for lo, hi in zip(sr_lo, sr_hi)):
                ax.fill_between(sr_xs, sr_lo, sr_hi, color=_COLOR_SUCCESS,
                                alpha=0.22, zorder=3)

        # ── Significance stars (inline, bold, above/below each point) ──
        for sdata in sig_data_per_metric:
            for xi, r_val, p_val in zip(x_positions, sdata['ys'], sdata['ps']):
                mark = sig_marker(p_val, sig_level)
                if not mark or np.isnan(r_val):
                    continue
                offset = 9 if r_val >= 0 else -9
                va = 'bottom' if r_val >= 0 else 'top'
                ax.annotate(
                    mark,
                    xy=(xi, r_val), xycoords='data',
                    textcoords='offset points', xytext=(0, offset),
                    ha='center', va=va, fontsize=9,
                    color=sdata['color'],
                    fontproperties={'weight': 'bold'},
                )

        # ── Axis style ──
        ax.axhline(0, color='black', linewidth=0.8, linestyle=':', alpha=0.6, zorder=1)
        ax.set_xticks(x_positions)
        ax.set_xticklabels(
            [_alpha_mathtext(a, include_symbol=False) for a in alpha_subset],
            rotation=0, ha='center', fontsize=9,
        )
        ax.set_xlabel(r'$\alpha$ (regularisation)', fontsize=10)
        ax.set_ylim(y_min, y_max)
        ax.set_title(PHASE_LABELS.get(phase_str, str(phase)), fontsize=10, pad=4)
        ax.tick_params(axis='y', labelsize=9)
        ax.yaxis.grid(True, linestyle='--', alpha=0.5)
        ax.xaxis.grid(False)
        ax.set_axisbelow(True)

    axes[0].set_ylabel('Pearson r', fontsize=10)

    # Shared legend below all subplots
    legend_handles = [
        plt.Line2D([0], [0], color=spec['color'], linestyle=spec['ls'],
                   linewidth=spec['lw'], marker='o', markersize=5, label=spec['label'])
        for spec in _THESIS_METRIC_LINES
    ]
    legend_handles.append(
        plt.Line2D([0], [0], color=_COLOR_SUCCESS, linestyle='--', linewidth=2.0,
                   marker='s', markersize=5, label='Success rate')
    )
    fig.legend(legend_handles, [h.get_label() for h in legend_handles],
               loc='lower center', ncol=len(legend_handles),
               fontsize=9, frameon=True, bbox_to_anchor=(0.5, 0.01))
    fig.tight_layout(rect=[0, 0.08, 1, 1])
    return fig


# ── Figure 4: Correlation bar charts ─────────────────────────────────────────

def plot_thesis_corr_bar(
    csv_path: str,
    task_id: str = TASK_ID_DEFAULT,
    version: str = 'A',
    update: int = 5000,
    alpha_subset_A: list | None = None,
    alpha_subset_B: list | None = None,
    fig_width: float | None = None,
    title_task_name: str | None = None,
    fs: _FontSizes = _FS_FULLWIDTH,
) -> plt.Figure:
    """Grouped bar chart of Fisher z aggregated Pearson r, one subplot per phase.

    version='A': pool alpha_subset_A (successful), shaded fill
    version='B': pool alpha_subset_B (unsuccessful), hollow fill (outline only)
    version='C': shaded (A) and hollow (B) bars side-by-side per metric per phase subplot
    """
    _apply_pub_style(fs)

    _init_pool_phases(task_id)
    phases = PHASE_LIST[TASK_PHASE_SYSTEM.get(task_id, 'key_phase')]

    if alpha_subset_A is None:
        alpha_subset_A = list(_ALPHA_SUCCESSFUL)
    if alpha_subset_B is None:
        alpha_subset_B = list(_ALPHA_UNSUCCESSFUL)

    alpha_A_str = '{' + ', '.join(f'1e-{-int(round(np.log10(a)))}' for a in alpha_subset_A) + '}'
    alpha_B_str = '{' + ', '.join(f'1e-{-int(round(np.log10(a)))}' for a in alpha_subset_B) + '}'

    # Load data
    if version in ('A', 'B'):
        alpha_subset = alpha_subset_A if version == 'A' else alpha_subset_B
        df = _load_pooled_data(csv_path, task_id, update, alpha_subset)
        data = _aggregate_corr_stats(df, phases)
    else:  # C
        df_A = _load_pooled_data(csv_path, task_id, update, alpha_subset_A)
        df_B = _load_pooled_data(csv_path, task_id, update, alpha_subset_B)
        data_A = _aggregate_corr_stats(df_A, phases)
        data_B = _aggregate_corr_stats(df_B, phases)

    n_metrics = len(_THESIS_METRIC_LINES)
    n_phases  = len(phases)

    # ── Bar geometry ──
    bar_width = 0.14
    group_gap = 0.10   # gap between metric groups within a subplot

    if version in ('A', 'B'):
        # One bar per metric, centered in the group
        metric_offsets = (np.arange(n_metrics) - (n_metrics - 1) / 2.0) * (bar_width + group_gap)
        metric_x_centers = metric_offsets
    else:  # C: two bars (A and B) per metric, side by side
        pair_gap = 0.02
        metric_offsets = (np.arange(n_metrics) - (n_metrics - 1) / 2.0) * (2 * bar_width + pair_gap + group_gap)
        x_A_offsets = metric_offsets - (bar_width + pair_gap) / 2
        x_B_offsets = metric_offsets + (bar_width + pair_gap) / 2
        metric_x_centers = metric_offsets

    default_width = fig_width if fig_width is not None else 3.2 * n_phases
    fig, axes = plt.subplots(1, n_phases, figsize=(default_width, 4.5), sharey=True)
    if n_phases == 1:
        axes = [axes]

    all_ci_hi, all_ci_lo = [], []

    for ax_idx, (ax, phase) in enumerate(zip(axes, phases)):
        if version in ('A', 'B'):
            face_alpha = 0.22 if version == 'A' else 0.0
            for m_idx, spec in enumerate(_THESIS_METRIC_LINES):
                metric = spec['metric']
                color  = spec['color']
                r     = data[metric]['r'][ax_idx]
                ci_lo = data[metric]['ci_lo'][ax_idx]
                ci_hi = data[metric]['ci_hi'][ax_idx]
                p     = data[metric]['p'][ax_idx]
                if np.isnan(r):
                    continue
                xpos = metric_x_centers[m_idx]
                ax.bar(xpos, r, width=bar_width * 0.9,
                       facecolor=(*color[:3], face_alpha),
                       edgecolor=color, linewidth=1.2, zorder=3)
                ax.errorbar(xpos, r, yerr=[[r - ci_lo], [ci_hi - r]],
                            fmt='none', color='black', capsize=3, linewidth=1.2, zorder=4)
                star = _sig_stars(p)
                if star:
                    star_y = (ci_hi + 0.02) if r >= 0 else (ci_lo - 0.02)
                    va = 'bottom' if r >= 0 else 'top'
                    ax.text(xpos, star_y, star, ha='center', va=va, fontsize=fs.annot)
                all_ci_hi.append(ci_hi)
                all_ci_lo.append(ci_lo)

        else:  # version C
            for m_idx, spec in enumerate(_THESIS_METRIC_LINES):
                metric = spec['metric']
                color  = spec['color']
                for v_idx, (data_v, face_alpha) in enumerate([
                    (data_A, 0.22),   # successful: shaded
                    (data_B, 0.0),    # unsuccessful: hollow
                ]):
                    xpos  = x_A_offsets[m_idx] if v_idx == 0 else x_B_offsets[m_idx]
                    r     = data_v[metric]['r'][ax_idx]
                    ci_lo = data_v[metric]['ci_lo'][ax_idx]
                    ci_hi = data_v[metric]['ci_hi'][ax_idx]
                    p     = data_v[metric]['p'][ax_idx]
                    if np.isnan(r):
                        continue
                    ax.bar(xpos, r, width=bar_width * 0.9,
                           facecolor=(*color[:3], face_alpha),
                           edgecolor=color, linewidth=1.2, zorder=3)
                    ax.errorbar(xpos, r, yerr=[[r - ci_lo], [ci_hi - r]],
                                fmt='none', color='black', capsize=3, linewidth=1.2, zorder=4)
                    star = _sig_stars(p)
                    if star:
                        star_y = (ci_hi + 0.02) if r >= 0 else (ci_lo - 0.02)
                        va = 'bottom' if r >= 0 else 'top'
                        ax.text(xpos, star_y, star, ha='center', va=va, fontsize=fs.annot)
                    all_ci_hi.append(ci_hi)
                    all_ci_lo.append(ci_lo)

        # ── Per-subplot axis decoration ──
        ax.axhline(0, color='black', linewidth=0.8, linestyle='--', zorder=1)
        ax.grid(False)
        ax.set_xticks([])
        ax.tick_params(axis='y', labelsize=fs.tick)
        phase_str = str(phase)
        ax.set_title(PHASE_LABELS.get(phase_str, phase_str), fontsize=fs.axis_title, pad=2)

    # ── Shared y-axis limits and label ──
    y_min_val = min(0.0, min(all_ci_lo, default=0.0)) - 0.05
    y_max_val = max(all_ci_hi, default=0.0) + 0.20
    for ax in axes:
        ax.set_ylim(y_min_val, y_max_val)
    axes[0].set_ylabel('Pearson r  (vs Mean LPC)', fontsize=fs.axis_label)

    # ── Figure suptitle ──
    if version == 'A':
        suptitle = f'Correlation with Mean LPC  (successful: alpha in {alpha_A_str})'
    elif version == 'B':
        suptitle = f'Correlation with Mean LPC  (unsuccessful: alpha in {alpha_B_str})'
    else:
        suptitle = 'Correlation with Mean LPC — successful vs unsuccessful alpha'
    if title_task_name:
        suptitle = f'{suptitle}  |  {title_task_name}'
    fig.suptitle(suptitle, fontsize=fs.axis_title, y=0.98)

    # ── Shared legend at bottom ──
    metric_handles = [
        mpatches.Patch(facecolor=(*spec['color'][:3], 0.22),
                       edgecolor=spec['color'], linewidth=1.2, label=spec['label'])
        for spec in _THESIS_METRIC_LINES
    ]
    if version == 'C':
        shade_handles = [
            mpatches.Patch(facecolor=(0.5, 0.5, 0.5, 0.22), edgecolor='grey',
                           linewidth=1.2, label=f'alpha in {alpha_A_str}'),
            mpatches.Patch(facecolor=(0.5, 0.5, 0.5, 0.0), edgecolor='grey',
                           linewidth=1.2, label=f'alpha in {alpha_B_str}'),
        ]
        all_handles = metric_handles + shade_handles
        ncol = n_metrics + 2
    else:
        all_handles = metric_handles
        ncol = n_metrics

    fig.legend(handles=all_handles, loc='lower center', bbox_to_anchor=(0.5, 0.0),
               ncol=ncol, fontsize=fs.legend, frameon=True)
    fig.tight_layout(rect=[0, 0.10, 1, 0.97])
    fig.subplots_adjust(top=0.88, bottom=0.14)
    return fig


# ── Saving helper ─────────────────────────────────────────────────────────────

def _save_or_prompt(fig: plt.Figure, out_path: pathlib.Path, auto_save: bool) -> None:
    if auto_save:
        fig.savefig(out_path, dpi=150, bbox_inches='tight')
        print(f'[info] Saved → {out_path}')
    else:
        plt.figure(fig.number)
        plt.show(block=False)
        ans = input(f'Save to {out_path}? [y/N] ').strip().lower()
        if ans == 'y':
            fig.savefig(out_path, dpi=150, bbox_inches='tight')
            print(f'[info] Saved → {out_path}')
        else:
            print('[info] Not saved.')


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description='Thesis final publication figures.'
    )
    parser.add_argument(
        '--plot',
        choices=[
            'metrics_vs_alpha', 'lpc_heatmaps', 'entropy_heatmaps',
            'kl_local_heatmaps', 'kl_global_heatmaps',
            'seed_agg_A', 'seed_agg_B', 'seed_agg_C',
            'corr_bar_A', 'corr_bar_B', 'corr_bar_C',
            'all',
        ],
        default='all',
    )
    parser.add_argument(
        '--task',
        choices=list(_TASK_CONFIGS.keys()),
        default='unlockpickup',
        help='Task to plot: unlockpickup (default) or opentwo_doors.',
    )
    parser.add_argument(
        '--csv_unnorm',
        default=None,
        help='Override path to unnormalised eval metrics CSV.',
    )
    parser.add_argument(
        '--csv_norm',
        default=None,
        help='Override path to normalised eval metrics CSV.',
    )
    parser.add_argument(
        '--update', type=int, default=5000,
        help='Training checkpoint to use for heatmaps (default: 5000).',
    )
    parser.add_argument(
        '--cache_dir', default='evaluation_cache',
        help='Root directory for routing_data.json caches.',
    )
    parser.add_argument(
        '--results_path', default='evaluation_results.csv',
        help='Path to evaluation_results.csv for alpha→trial mapping.',
    )
    parser.add_argument(
        '--out_dir', default='plots/thesis',
        help='Output directory for saved figures.',
    )
    parser.add_argument('--save', action='store_true')
    parser.add_argument(
        '--normalised', action='store_true',
        help='Use normalised eval metrics CSV (unlockpickup only).',
    )
    args = parser.parse_args()

    _apply_pub_style()

    # ── Resolve task config ────────────────────────────────────────────────────
    _tcfg = _TASK_CONFIGS[args.task]
    task_id = _tcfg['task_id']
    safe_task = task_id.replace('/', '_')

    if args.normalised:
        if _tcfg['csv_norm'] is None:
            parser.error(f"--normalised is not supported for --task {args.task!r} (no normalised CSV)")

    csv_unnorm = args.csv_unnorm or _tcfg['csv_unnorm']
    csv_norm   = args.csv_norm   or _tcfg['csv_norm'] or csv_unnorm
    csv_path   = csv_norm if args.normalised else csv_unnorm

    base_out_dir = pathlib.Path(args.out_dir)
    if _tcfg['out_subdir']:
        base_out_dir = base_out_dir / _tcfg['out_subdir']
    out_dir = base_out_dir / 'normalised' if args.normalised else base_out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    norm_suffix = ''

    # ── Determine which plots are supported for this task ─────────────────────
    _supported_plots = (
        {'lpc_heatmaps', 'entropy_heatmaps', 'kl_local_heatmaps', 'kl_global_heatmaps', 'corr_bar_A'}
        if args.task == 'opentwo_doors'
        else {
            'metrics_vs_alpha', 'lpc_heatmaps', 'entropy_heatmaps',
            'kl_local_heatmaps', 'kl_global_heatmaps',
            'seed_agg_A', 'seed_agg_B', 'seed_agg_C',
            'corr_bar_A', 'corr_bar_B', 'corr_bar_C',
        }
    )
    if args.plot != 'all' and args.plot not in _supported_plots:
        parser.error(f"--plot {args.plot!r} is not supported for --task {args.task!r}")

    if args.plot in ('metrics_vs_alpha', 'all') and 'metrics_vs_alpha' in _supported_plots:
        print('\n[info] Generating metrics vs alpha figure ...')
        fig = plot_metrics_vs_alpha(csv_unnorm, csv_norm, task_id)
        _save_or_prompt(fig, out_dir / f'{safe_task}_metrics_vs_alpha.png', args.save)

    _heatmap_plots = {'lpc_heatmaps', 'entropy_heatmaps', 'kl_local_heatmaps', 'kl_global_heatmaps'}
    if (args.plot in _heatmap_plots or args.plot == 'all') and _heatmap_plots & _supported_plots:
        print('\n[info] Sampling environment image ...')
        env_image = _sample_env_image(task_id)

    if args.plot in ('lpc_heatmaps', 'all') and 'lpc_heatmaps' in _supported_plots:
        print('\n[info] Generating LPC phase-alpha heatmap grid ...')
        fig = plot_phase_alpha_heatmaps(
            task_id=task_id,
            update=args.update,
            cache_dir=args.cache_dir,
            results_path=args.results_path,
            metric='lpc',
            env_image=env_image,
            heatmap_trials=_tcfg['heatmap_trials'],
            phase_keys=_tcfg['phase_keys'],
            phase_group_by=_tcfg['phase_group_by'],
            max_seed_mb=_tcfg['max_seed_mb'],
        )
        _save_or_prompt(fig, out_dir / f'{safe_task}_lpc_phase_alpha_heatmaps.png', args.save)

    if args.plot in ('entropy_heatmaps', 'all') and 'entropy_heatmaps' in _supported_plots:
        print('\n[info] Generating entropy phase-alpha heatmap grid ...')
        fig = plot_phase_alpha_heatmaps(
            task_id=task_id,
            update=args.update,
            cache_dir=args.cache_dir,
            results_path=args.results_path,
            metric='entropy',
            env_image=env_image,
            heatmap_trials=_tcfg['heatmap_trials'],
            phase_keys=_tcfg['phase_keys'],
            phase_group_by=_tcfg['phase_group_by'],
            max_seed_mb=_tcfg['max_seed_mb'],
        )
        _save_or_prompt(fig, out_dir / f'{safe_task}_entropy_phase_alpha_heatmaps.png', args.save)

    if args.plot in ('kl_local_heatmaps', 'all') and 'kl_local_heatmaps' in _supported_plots:
        print('\n[info] Generating KL local phase-alpha heatmap grid ...')
        fig = plot_phase_alpha_heatmaps(
            task_id=task_id,
            update=args.update,
            cache_dir=args.cache_dir,
            results_path=args.results_path,
            metric='kl_local',
            env_image=env_image,
            heatmap_trials=_tcfg['heatmap_trials'],
            phase_keys=_tcfg['phase_keys'],
            phase_group_by=_tcfg['phase_group_by'],
            max_seed_mb=_tcfg['max_seed_mb'],
        )
        _save_or_prompt(fig, out_dir / f'{safe_task}_kl_local_phase_alpha_heatmaps.png', args.save)

    if args.plot in ('kl_global_heatmaps', 'all') and 'kl_global_heatmaps' in _supported_plots:
        print('\n[info] Generating KL global phase-alpha heatmap grid ...')
        fig = plot_phase_alpha_heatmaps(
            task_id=task_id,
            update=args.update,
            cache_dir=args.cache_dir,
            results_path=args.results_path,
            metric='kl_global',
            env_image=env_image,
            heatmap_trials=_tcfg['heatmap_trials'],
            phase_keys=_tcfg['phase_keys'],
            phase_group_by=_tcfg['phase_group_by'],
            max_seed_mb=_tcfg['max_seed_mb'],
        )
        _save_or_prompt(fig, out_dir / f'{safe_task}_kl_global_phase_alpha_heatmaps.png', args.save)

    if args.plot in ('seed_agg_A', 'all') and 'seed_agg_A' in _supported_plots:
        print('\n[info] Generating seed-aggregated phase plot (successful α: 1e-6, 1e-5, 1e-4) ...')
        fig = plot_thesis_seed_agg(csv_path, task_id, list(_ALPHA_SUCCESSFUL))
        _save_or_prompt(fig, out_dir / f'{safe_task}_seed_agg_A{norm_suffix}.png', args.save)

    if args.plot in ('seed_agg_B', 'all') and 'seed_agg_B' in _supported_plots:
        print('\n[info] Generating seed-aggregated phase plot (unsuccessful α: 1e-3, 1e-2) ...')
        fig = plot_thesis_seed_agg(csv_path, task_id, list(_ALPHA_UNSUCCESSFUL))
        _save_or_prompt(fig, out_dir / f'{safe_task}_seed_agg_B{norm_suffix}.png', args.save)

    if args.plot in ('seed_agg_C', 'all') and 'seed_agg_C' in _supported_plots:
        print('\n[info] Generating seed-aggregated phase plot (combined α: 1e-6 to 1e-2) ...')
        fig = plot_thesis_seed_agg(csv_path, task_id, list(_ALPHA_COMBINED))
        _save_or_prompt(fig, out_dir / f'{safe_task}_seed_agg_C{norm_suffix}.png', args.save)

    if args.plot in ('corr_bar_A', 'all') and 'corr_bar_A' in _supported_plots:
        print('\n[info] Generating correlation bar chart version A (successful α) ...')
        # OpenTwoDoors corr_bar_A is placed at half page width in the thesis —
        # use _FS_HALFWIDTH so labels remain legible when scaled down.
        _corr_bar_A_fs = _FS_HALFWIDTH if task_id == 'BabyAI-OpenTwoDoors-v0' else _FS_FULLWIDTH
        fig = plot_thesis_corr_bar(csv_path, task_id, version='A', update=args.update,
                                   fig_width=_tcfg['fig_width_bar'],
                                   title_task_name=_tcfg['bar_title_extra'],
                                   fs=_corr_bar_A_fs)
        _save_or_prompt(fig, out_dir / f'{safe_task}_corr_bar_A{norm_suffix}.png', args.save)

    if args.plot in ('corr_bar_B', 'all') and 'corr_bar_B' in _supported_plots:
        print('\n[info] Generating correlation bar chart version B (unsuccessful α) ...')
        fig = plot_thesis_corr_bar(csv_path, task_id, version='B', update=args.update,
                                   fig_width=_tcfg['fig_width_bar'],
                                   title_task_name=_tcfg['bar_title_extra'])
        _save_or_prompt(fig, out_dir / f'{safe_task}_corr_bar_B{norm_suffix}.png', args.save)

    if args.plot in ('corr_bar_C', 'all') and 'corr_bar_C' in _supported_plots:
        print('\n[info] Generating correlation bar chart version C (successful vs unsuccessful) ...')
        fig = plot_thesis_corr_bar(csv_path, task_id, version='C', update=args.update,
                                   fig_width=_tcfg['fig_width_bar'],
                                   title_task_name=_tcfg['bar_title_extra'])
        _save_or_prompt(fig, out_dir / f'{safe_task}_corr_bar_C{norm_suffix}.png', args.save)

    if not args.save:
        input('\nPress Enter to close all figures ...')
    plt.close('all')


if __name__ == '__main__':
    main()