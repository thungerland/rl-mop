"""
corr_plots.py — Correlation plots across time phases or regularisation parameters.

Plots Pearson r between:
  lpc_entropy : per-cell empirical H(A|S) vs per-cell mean LPC  (spatial correlation,
                computed on phase-filtered data)
  lpc_dist    : per-timestep LPC vs distance to subgoal
  entropy_dist: per-timestep empirical H(A|S) vs distance to subgoal
  kl_dist     : per-timestep KL(pi_hat||P_a) vs distance to subgoal

Two plot modes:
  alpha : x-axis = lpc_alpha (regularisation strength), one line per time phase
  phase : x-axis = time phase, one line per lpc_alpha value

Statistical significance is indicated with asterisks above data points.

Usage:
    python corr_plots.py <task_id>
        [--mode alpha|phase]
        [--corr lpc_entropy|lpc_dist|entropy_dist|kl_dist]
        [--dist dist_to_door|dist_to_key|dist_to_target]  # only needed with --phase_system none
        [--phase_system key_phase|unlock_phase|none]
        [--trials all|20,21,22]
        [--results_path evaluation_results.csv]
        [--cache_dir evaluation_cache]
        [--output path/to/out.png]
        [--sig_level 0.05]

Examples:
    python corr_plots.py BabyAI-UnlockPickup-v0 --mode alpha --corr lpc_entropy
    python corr_plots.py BabyAI-UnlockPickup-v0 --mode phase --corr lpc_dist
    python corr_plots.py BabyAI-OpenTwoDoors-v0 --mode alpha --corr kl_dist
"""

import argparse
import json
import pathlib

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from plotting_utils import build_routing_data_tuples, compute_empirical_entropy
from stats import (
    _filter_by_phase,
    per_timestep_lpc_dist_correlation,
    per_timestep_entropy_dist_correlation,
    per_timestep_kl_dist_correlation,
    spatial_entropy_lpc_correlation,
    spatial_kl_lpc_correlation,
)

# ── Phase system definitions ──────────────────────────────────────────────────

TASK_PHASE_SYSTEM = {
    'BabyAI-UnlockPickup-v0': 'key_phase',
    'BabyAI-OpenTwoDoors-v0': 'unlock_phase',
}

PHASE_LIST = {
    'key_phase': [
        'pre_key',
        'post_key_pre_unlock',
        'with_key_post_unlock',
        'post_unlock_post_key',
    ],
    'unlock_phase': [
        'pre_unlock',
        'post_unlock',
    ],
    'none': [None],
}

PHASE_LABELS = {
    'pre_key':              'Pre-key',
    'post_key_pre_unlock':  'With-key (pre-unlock)',
    'with_key_post_unlock': 'With-key (post-unlock)',
    'post_unlock_post_key': 'Post-unlock (post-key)',
    'pre_unlock':           'Pre-unlock',
    'post_unlock':          'Post-unlock',
    None:                   'All timesteps',
}

# Implied distance field per phase. For dist-based correlations the distance used
# is always the one semantically relevant to the current subgoal.
PHASE_DIST_MAP = {
    # UnlockPickup (key_phase)
    'pre_key':              'dist_to_key',     # agent navigating to key
    'post_key_pre_unlock':  'dist_to_door',    # carrying key, navigating to door
    'with_key_post_unlock': 'dist_to_target',  # door open, navigating to box
    'post_unlock_post_key': 'dist_to_target',  # dropped key, navigating to box
    # OpenTwoDoors (unlock_phase) — dist_to_door tracks next locked door in both phases
    'pre_unlock':           'dist_to_door',    # navigating to first door
    'post_unlock':          'dist_to_door',    # navigating to second door
    # Single-phase fallback — caller must supply dist explicitly
    None:                   None,
}


# ── Data loading ──────────────────────────────────────────────────────────────

_VALID_ALPHAS = {0.0, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0}


def _is_valid_alpha(alpha: float) -> bool:
    """Accept 0.0 or powers of 10 in [1e-6, 1.0]."""
    if alpha == 0.0:
        return True
    return any(abs(alpha - v) / max(v, 1e-12) < 1e-6 for v in _VALID_ALPHAS)


def load_alpha_map(task_id: str, results_path: str = 'evaluation_results.csv') -> dict:
    """Load trial -> lpc_alpha mapping from evaluation_results.csv.

    Only includes trials whose lpc_alpha is a power of 10 in [1e-6, 1.0].
    """
    df = pd.read_csv(results_path)
    subset = df[df['task_id'] == task_id][['trial', 'lpc_alpha']].dropna()
    subset = subset.drop_duplicates(subset=['trial'])
    result = {}
    for _, row in subset.iterrows():
        alpha = float(row['lpc_alpha'])
        if _is_valid_alpha(alpha):
            result[int(row['trial'])] = alpha
    return result


def load_trial_data(task_id: str, trial: int, cache_dir: str = 'evaluation_cache') -> list:
    """Load and return routing_data for a single trial."""
    cache_path = pathlib.Path(cache_dir) / task_id / f'trial_{trial}' / 'routing_data.json'
    if not cache_path.exists():
        cache_path = pathlib.Path(cache_dir) / task_id / task_id / f'trial_{trial}' / 'routing_data.json'
    if not cache_path.exists():
        raise FileNotFoundError(f"Cache not found: {cache_path}")
    with open(cache_path) as f:
        cache = json.load(f)
    return build_routing_data_tuples(cache)


# ── Correlation computation ───────────────────────────────────────────────────

def compute_corr(
    routing_data: list,
    corr_type: str,
    phase,
    dist_field: str | None,
    global_P_a=None,
) -> dict:
    """Compute one correlation value using phase-filtered data throughout.

    All quantities (H_s, KL_s, mean LPC, dist pairings) are derived from the
    phase-filtered subset only, so distributions reflect within-phase behaviour.

    Args:
        global_P_a: marginal action distribution computed over all timesteps (not
            phase-filtered). Used only for corr_type='lpc_kl_global'. If None and
            corr_type is 'lpc_kl_global', falls back to local (phase) marginal.

    Returns dict with 'r', 'p', 'n'.
    """
    phase_data = list(_filter_by_phase(routing_data, phase))

    if corr_type == 'lpc_entropy':
        result = spatial_entropy_lpc_correlation(phase_data)
        return {'r': result['r'], 'p': result['p'], 'n': result.get('n_cells', 0)}

    if corr_type == 'lpc_kl_local':
        result = spatial_kl_lpc_correlation(phase_data, P_a=None)
        return {'r': result['r'], 'p': result['p'], 'n': result.get('n_cells', 0)}

    if corr_type == 'lpc_kl_global':
        result = spatial_kl_lpc_correlation(phase_data, P_a=global_P_a)
        return {'r': result['r'], 'p': result['p'], 'n': result.get('n_cells', 0)}

    if dist_field is None:
        raise ValueError(f"dist_field must be specified for corr_type={corr_type} with phase={phase}")

    # Empirical entropy computed on phase-filtered data — phase-specific H_s and KL_s.
    emp = compute_empirical_entropy(phase_data)
    H_s_masked  = {pos: v for pos, v in emp['H_s'].items()  if emp['include_mask'][pos]}
    KL_s_masked = {pos: v for pos, v in emp['KL_s'].items() if emp['include_mask'][pos]}

    if corr_type == 'lpc_dist':
        return per_timestep_lpc_dist_correlation(phase_data, dist_field)
    elif corr_type == 'entropy_dist':
        return per_timestep_entropy_dist_correlation(phase_data, H_s_masked, dist_field)
    elif corr_type == 'kl_dist':
        return per_timestep_kl_dist_correlation(phase_data, KL_s_masked, dist_field)
    else:
        raise ValueError(f"Unknown corr_type: {corr_type}")


def collect_records(
    task_id: str,
    trials: list,
    alpha_map: dict,
    phases: list,
    corr_type: str,
    fallback_dist: str | None,
    cache_dir: str,
) -> pd.DataFrame:
    """Build a DataFrame of correlation results across trials and phases.

    For dist-based corr types, the distance field is determined per phase via
    PHASE_DIST_MAP. fallback_dist is used only for the None phase (phase_system='none').

    Columns: trial, lpc_alpha, phase, phase_label, dist_field, r, p, n
    """
    records = []
    for trial in trials:
        alpha = alpha_map.get(trial)
        if alpha is None:
            print(f"  [skip] trial {trial}: no lpc_alpha in evaluation_results.csv")
            continue
        print(f"  Loading trial {trial} (lpc_alpha={alpha:.2e})...", end=' ', flush=True)
        try:
            routing_data = load_trial_data(task_id, trial, cache_dir)
        except FileNotFoundError as e:
            print(f"not found — {e}")
            continue
        print(f"{len(routing_data)} timesteps")

        # Compute global marginal P_a once per trial (over all timesteps).
        global_emp = compute_empirical_entropy(routing_data)
        global_P_a = global_emp['P_a']

        for phase in phases:
            dist_field = PHASE_DIST_MAP.get(phase, None)
            if dist_field is None and corr_type not in ('lpc_entropy', 'lpc_kl_local', 'lpc_kl_global'):
                dist_field = fallback_dist  # use explicit --dist for none-phase system
            try:
                res = compute_corr(routing_data, corr_type, phase, dist_field, global_P_a=global_P_a)
            except Exception as e:
                print(f"    [warn] trial={trial} phase={phase}: {e}")
                res = {'r': float('nan'), 'p': float('nan'), 'n': 0}

            records.append({
                'trial':       trial,
                'lpc_alpha':   alpha,
                'phase':       str(phase),
                'phase_label': PHASE_LABELS.get(phase, str(phase)),
                'dist_field':  dist_field or '',
                'r':           res['r'],
                'p':           res['p'],
                'n':           res.get('n', res.get('n_cells', 0)),
            })

    return pd.DataFrame(records)


# ── Statistical significance ──────────────────────────────────────────────────

def sig_marker(p: float, level: float = 0.05) -> str:
    if np.isnan(p) or p >= level:
        return ''
    if p < 0.001:
        return '***'
    if p < 0.01:
        return '**'
    return '*'


# ── Plotting ──────────────────────────────────────────────────────────────────

def _build_title(task_id: str, mode: str, corr_type: str) -> str:
    corr_label = {
        'lpc_entropy':  'LPC vs H(A|S)',
        'lpc_dist':     'LPC vs dist-to-subgoal',
        'entropy_dist': 'H(A|S) vs dist-to-subgoal',
        'kl_dist':      'KL vs dist-to-subgoal',
    }.get(corr_type, corr_type)
    mode_label = 'by regularisation' if mode == 'alpha' else 'by phase'
    return f"{task_id}\nPearson r: {corr_label}  [{mode_label}]"


def plot_corr_vs_alpha(
    df: pd.DataFrame,
    sig_level: float,
    title: str,
) -> plt.Figure:
    """Plot correlation vs lpc_alpha with one line per phase."""
    fig, ax = plt.subplots(figsize=(8, 5))

    phase_labels = df['phase_label'].unique().tolist()
    palette = sns.color_palette('tab10', n_colors=len(phase_labels))
    color_map = dict(zip(phase_labels, palette))

    # Aggregate mean r per (lpc_alpha, phase_label) over trials for the main line
    agg = df.groupby(['lpc_alpha', 'phase_label'])['r'].mean().reset_index()
    sns.lineplot(
        data=agg,
        x='lpc_alpha',
        y='r',
        hue='phase_label',
        palette=color_map,
        ax=ax,
        legend='full',
        markers=False,
    )

    # Scatter individual trial points; marker encodes significance
    for _, row in df.iterrows():
        if np.isnan(row['r']):
            continue
        color = color_map.get(row['phase_label'], 'grey')
        significant = not np.isnan(row['p']) and row['p'] < sig_level
        marker = 'o' if significant else 'x'
        ax.scatter(row['lpc_alpha'], row['r'], color=color, marker=marker,
                   s=40, zorder=5, linewidths=1.2)
        mark = sig_marker(row['p'], sig_level)
        if mark:
            ax.annotate(mark, (row['lpc_alpha'], row['r']),
                        textcoords='offset points', xytext=(0, 5),
                        ha='center', fontsize=7, color=color)

    ax.axhline(0, color='black', linewidth=0.8, linestyle='--', alpha=0.5)
    ax.set_xscale('log')
    alphas_sorted = sorted(df['lpc_alpha'].unique())
    ax.set_xlim(alphas_sorted[0] * 0.5, alphas_sorted[-1] * 2)
    ax.set_xlabel('lpc_alpha (regularisation strength)')
    ax.set_ylabel('Pearson r')
    ax.set_title(title)
    ax.legend(title='Phase', bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=9)
    fig.tight_layout()
    return fig


def plot_corr_vs_phase(
    df: pd.DataFrame,
    sig_level: float,
    title: str,
    phase_order: list,
) -> plt.Figure:
    """Plot correlation vs phase with one line per lpc_alpha value."""
    fig, ax = plt.subplots(figsize=(8, 5))

    alphas = sorted(df['lpc_alpha'].unique().tolist())
    palette = sns.color_palette('viridis', n_colors=max(len(alphas), 2))
    color_map = dict(zip(alphas, palette))

    # Build ordered phase labels
    label_order = [PHASE_LABELS.get(p, str(p)) for p in phase_order]

    for alpha in alphas:
        subset = df[df['lpc_alpha'] == alpha]
        mean_r = subset.groupby('phase_label')['r'].mean().reindex(label_order)
        mean_p = subset.groupby('phase_label')['p'].mean().reindex(label_order)
        color = color_map[alpha]
        ax.plot(label_order, mean_r.values, marker='o', color=color,
                label=f'α={alpha:.2e}')

        for x_idx, (plabel, r_val, p_val) in enumerate(
                zip(label_order, mean_r.values, mean_p.values)):
            if np.isnan(r_val):
                continue
            mark = sig_marker(p_val, sig_level)
            if mark:
                ax.annotate(mark, (x_idx, r_val),
                            textcoords='offset points', xytext=(0, 5),
                            ha='center', fontsize=7, color=color)

    ax.axhline(0, color='black', linewidth=0.8, linestyle='--', alpha=0.5)
    ax.set_xticks(range(len(label_order)))
    ax.set_xticklabels(label_order, rotation=20, ha='right')
    ax.set_xlabel('Time phase')
    ax.set_ylabel('Pearson r')
    ax.set_title(title)
    ax.legend(title='lpc_alpha', bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=9)
    fig.tight_layout()
    return fig


# ── Phase-subplots figure ─────────────────────────────────────────────────────

# Ordered alpha values and their tick labels for the phase_subplots figure.
_SUBPLOT_ALPHA_ORDER = [0.0, 1e-6, 1e-5, 1e-4]
_SUBPLOT_ALPHA_LABELS = ['0', '1e-6', '1e-5', '1e-4']

# Line styles / colours for the three metrics shown in phase_subplots mode.
_SUBPLOT_LINES = [
    {'corr': 'lpc_entropy',   'label': 'H(A|S) vs LPC',    'color': '#2a9d8f', 'ls': '-',  'alpha': 1.0, 'lw': 2.0},
    {'corr': 'lpc_kl_local',  'label': 'KL local vs LPC',  'color': '#e76f51', 'ls': '-',  'alpha': 1.0, 'lw': 2.0},
    {'corr': 'lpc_kl_global', 'label': 'KL global vs LPC', 'color': '#6c8ebf', 'ls': '--', 'alpha': 0.6, 'lw': 1.4},
]


def _collect_phase_subplot_records(
    task_id: str,
    trials: list,
    alpha_map: dict,
    phases: list,
    cache_dir: str,
    alpha_subset: list | None = None,
) -> pd.DataFrame:
    """Collect correlation records for all three subplot metrics simultaneously.

    Args:
        alpha_subset: if provided, only load trials whose lpc_alpha is in this set.

    Returns a DataFrame with columns:
        trial, lpc_alpha, phase, phase_label, corr, r, p, n
    """
    alpha_subset_set = set(alpha_subset) if alpha_subset is not None else None
    records = []
    for trial in trials:
        alpha = alpha_map.get(trial)
        if alpha is None:
            print(f"  [skip] trial {trial}: no lpc_alpha in evaluation_results.csv")
            continue
        if alpha_subset_set is not None and alpha not in alpha_subset_set:
            continue
        alpha_label = f'{alpha:.0e}' if alpha > 0 else '0'
        print(f"  Loading trial {trial} (lpc_alpha={alpha_label})...", end=' ', flush=True)
        try:
            routing_data = load_trial_data(task_id, trial, cache_dir)
        except FileNotFoundError as e:
            print(f"not found — {e}")
            continue
        print(f"{len(routing_data)} timesteps")

        global_emp = compute_empirical_entropy(routing_data)
        global_P_a = global_emp['P_a']

        for phase in phases:
            for line_spec in _SUBPLOT_LINES:
                corr_type = line_spec['corr']
                try:
                    res = compute_corr(routing_data, corr_type, phase, dist_field=None,
                                       global_P_a=global_P_a)
                except Exception as e:
                    print(f"    [warn] trial={trial} phase={phase} corr={corr_type}: {e}")
                    res = {'r': float('nan'), 'p': float('nan'), 'n': 0}
                records.append({
                    'trial':       trial,
                    'lpc_alpha':   alpha,
                    'phase':       str(phase),
                    'phase_label': PHASE_LABELS.get(phase, str(phase)),
                    'corr':        corr_type,
                    'r':           res['r'],
                    'p':           res['p'],
                    'n':           res.get('n', res.get('n_cells', 0)),
                })

    return pd.DataFrame(records)


def plot_phase_subplots_vs_alpha(
    df: pd.DataFrame,
    phases: list,
    alpha_subset: list,
    sig_level: float,
    title: str,
) -> plt.Figure:
    """Publication-quality figure: one subplot per phase, x=alpha (categorical), y=Pearson r.

    Three lines per subplot: H(A|S) vs LPC, KL-local vs LPC, KL-global vs LPC (faint).
    Shared y-axis range across all subplots. Categorical x-axis (no log scale).
    """
    # Validate expected alphas are present.
    present_alphas = set(df['lpc_alpha'].unique())
    for a in alpha_subset:
        if a not in present_alphas:
            raise KeyError(
                f"Alpha {a} not found in data. Available: {sorted(present_alphas)}. "
                "Check evaluation_results.csv and --alpha_values."
            )

    # Validate expected phases are present.
    present_phases = set(df['phase'].unique())
    for p in phases:
        if str(p) not in present_phases:
            raise KeyError(
                f"Phase '{p}' not found in data. Available: {present_phases}."
            )

    # Build categorical x positions.
    alpha_to_x = {a: i for i, a in enumerate(alpha_subset)}
    x_labels = []
    for a in alpha_subset:
        if a == 0.0:
            x_labels.append('0')
        else:
            x_labels.append(f'{a:.0e}')
    x_positions = list(range(len(alpha_subset)))

    n_phases = len(phases)
    fig, axes = plt.subplots(
        1, n_phases,
        figsize=(3.8 * n_phases, 4.2),
        sharey=True,
    )
    if n_phases == 1:
        axes = [axes]

    sns.set_style('whitegrid')
    plt.rcParams.update({'font.size': 10, 'axes.titlesize': 9, 'axes.labelsize': 9})

    # Determine shared y-axis range.
    r_vals = df[df['lpc_alpha'].isin(alpha_subset) & df['phase'].isin([str(p) for p in phases])]['r'].dropna()
    if len(r_vals) == 0:
        y_min, y_max = -1.0, 1.0
    else:
        pad = max(0.08, (r_vals.max() - r_vals.min()) * 0.15)
        y_min = r_vals.min() - pad
        y_max = r_vals.max() + pad

    for ax, phase in zip(axes, phases):
        phase_str = str(phase)
        phase_df = df[df['phase'] == phase_str]

        for line_spec in _SUBPLOT_LINES:
            corr_type = line_spec['corr']
            corr_df = phase_df[phase_df['corr'] == corr_type]

            # Aggregate over trials per alpha (mean ± sem for error bars).
            xs, ys, yerrs, ps = [], [], [], []
            for a in alpha_subset:
                subset = corr_df[corr_df['lpc_alpha'] == a]['r'].dropna()
                p_subset = corr_df[corr_df['lpc_alpha'] == a]['p'].dropna()
                if len(subset) == 0:
                    xs.append(alpha_to_x[a])
                    ys.append(float('nan'))
                    yerrs.append(0.0)
                    ps.append(float('nan'))
                else:
                    xs.append(alpha_to_x[a])
                    ys.append(float(subset.mean()))
                    yerrs.append(float(subset.sem()) if len(subset) > 1 else 0.0)
                    ps.append(float(p_subset.mean()))

            valid = [(x, y, e, p) for x, y, e, p in zip(xs, ys, yerrs, ps) if not np.isnan(y)]
            if not valid:
                continue
            vx, vy, ve, vp = zip(*valid)

            has_error = any(e > 0 for e in ve)
            ax.plot(
                vx, vy,
                color=line_spec['color'],
                linestyle=line_spec['ls'],
                linewidth=line_spec['lw'],
                alpha=line_spec['alpha'],
                marker='o',
                markersize=5,
                label=line_spec['label'],
                zorder=3,
            )
            if has_error:
                ax.fill_between(
                    vx,
                    [y - e for y, e in zip(vy, ve)],
                    [y + e for y, e in zip(vy, ve)],
                    color=line_spec['color'],
                    alpha=0.12 * line_spec['alpha'],
                    zorder=2,
                )

            # Significance markers.
            for x_pos, r_val, p_val in zip(vx, vy, vp):
                mark = sig_marker(p_val, sig_level)
                if mark:
                    ax.annotate(
                        mark, (x_pos, r_val),
                        textcoords='offset points', xytext=(0, 6),
                        ha='center', fontsize=7,
                        color=line_spec['color'],
                        alpha=line_spec['alpha'],
                    )

        ax.axhline(0, color='black', linewidth=0.8, linestyle=':', alpha=0.6, zorder=1)
        ax.set_xticks(x_positions)
        ax.set_xticklabels(x_labels, rotation=30, ha='right', fontsize=8)
        ax.set_xlabel('α (regularisation)', fontsize=9)
        ax.set_ylim(y_min, y_max)
        ax.set_title(PHASE_LABELS.get(phase, str(phase)), fontsize=9, pad=4)
        ax.tick_params(axis='y', labelsize=8)

    axes[0].set_ylabel('Pearson r', fontsize=9)

    # Single legend outside the rightmost subplot.
    handles, labels = axes[-1].get_legend_handles_labels()
    # Deduplicate (same label from different axes).
    seen = {}
    for h, l in zip(handles, labels):
        if l not in seen:
            seen[l] = h
    fig.legend(
        seen.values(), seen.keys(),
        loc='lower center',
        ncol=len(_SUBPLOT_LINES),
        fontsize=8,
        frameon=True,
        bbox_to_anchor=(0.5, -0.02),
    )

    fig.suptitle(title, fontsize=10, y=1.02)
    fig.tight_layout(rect=[0, 0.08, 1, 1])
    return fig


# ── Scatter panel figure ──────────────────────────────────────────────────────

# 4-stop plasma ramp: perceptually ordered, print-safe.
_SCATTER_ALPHA_COLORS = [
    plt.cm.plasma(0.15),  # α = 0      (light yellow)
    plt.cm.plasma(0.40),  # α = 1e-6
    plt.cm.plasma(0.65),  # α = 1e-5
    plt.cm.plasma(0.88),  # α = 1e-4   (deep purple)
]
_SCATTER_PHASE_MARKERS = ['o', 's', '^', 'D']


def plot_scatter_panel(
    df: pd.DataFrame,
    phases: list,
    alpha_subset: list,
    title: str,
) -> plt.Figure:
    """2D summary scatter: x = r(H(A|S), LPC), y = r(KL_local, LPC).

    Each point = one (phase, alpha) combination. Color encodes alpha (plasma ramp),
    marker encodes phase. Zero reference lines and phase-label annotations included.
    """
    # Validate expected alphas and phases are present.
    present_alphas = set(df['lpc_alpha'].unique())
    for a in alpha_subset:
        if a not in present_alphas:
            raise KeyError(
                f"Alpha {a} not found in data. Available: {sorted(present_alphas)}."
            )
    present_phases = set(df['phase'].unique())
    for p in phases:
        if str(p) not in present_phases:
            raise KeyError(f"Phase '{p}' not found in data. Available: {present_phases}.")

    # Pivot: mean r per (alpha, phase, corr) then unstack corr into columns.
    pivot = (
        df[df['corr'].isin(['lpc_entropy', 'lpc_kl_local'])]
        .groupby(['lpc_alpha', 'phase', 'phase_label', 'corr'])['r']
        .mean()
        .unstack('corr')
        .reset_index()
    )
    # Ensure both columns exist (guard against missing data).
    for col in ('lpc_entropy', 'lpc_kl_local'):
        if col not in pivot.columns:
            raise KeyError(f"Column '{col}' missing after pivot — no data for that corr type.")

    alpha_color = {a: c for a, c in zip(alpha_subset, _SCATTER_ALPHA_COLORS)}
    phase_marker = {str(p): m for p, m in zip(phases, _SCATTER_PHASE_MARKERS)}

    sns.set_style('whitegrid')
    fig, ax = plt.subplots(figsize=(8.5, 5.2))

    # Zero reference lines.
    ax.axvline(0, color='#aaaaaa', linewidth=0.9, linestyle='--', zorder=1)
    ax.axhline(0, color='#aaaaaa', linewidth=0.9, linestyle='--', zorder=1)

    # Scatter points.
    for _, row in pivot.iterrows():
        alpha = row['lpc_alpha']
        if alpha not in alpha_subset:
            continue
        phase_str = row['phase']
        x = row.get('lpc_entropy', float('nan'))
        y = row.get('lpc_kl_local', float('nan'))
        if np.isnan(x) or np.isnan(y):
            continue

        color = alpha_color.get(alpha, '#888888')
        marker = phase_marker.get(phase_str, 'o')

        ax.scatter(
            x, y,
            color=color,
            marker=marker,
            s=90,
            zorder=3,
            edgecolors='white',
            linewidths=0.6,
        )

        # Phase-label annotation, slightly offset above-right.
        label = row['phase_label']
        ax.annotate(
            label,
            (x, y),
            textcoords='offset points',
            xytext=(5, 4),
            fontsize=6.5,
            color=color,
            alpha=0.85,
            zorder=4,
        )

    ax.set_xlabel('r  [H(A|S)  vs  LPC]', fontsize=10)
    ax.set_ylabel('r  [KL local  vs  LPC]', fontsize=10)
    ax.tick_params(labelsize=8)

    # ── Alpha legend (color patches) ──
    alpha_labels = ['0' if a == 0.0 else f'{a:.0e}' for a in alpha_subset]
    alpha_handles = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=c,
                   markersize=8, label=f'α = {lbl}')
        for c, lbl in zip(_SCATTER_ALPHA_COLORS, alpha_labels)
    ]
    # ── Phase legend (marker shapes) ──
    phase_handles = [
        plt.Line2D([0], [0], marker=m, color='w', markerfacecolor='#555555',
                   markersize=8, label=PHASE_LABELS.get(p, str(p)))
        for p, m in zip(phases, _SCATTER_PHASE_MARKERS)
    ]

    # Combine into two groups with a blank separator.
    legend1 = ax.legend(
        handles=alpha_handles,
        title='Regularisation α',
        title_fontsize=8,
        fontsize=7.5,
        loc='upper left',
        bbox_to_anchor=(1.02, 1.0),
        frameon=True,
    )
    ax.add_artist(legend1)
    ax.legend(
        handles=phase_handles,
        title='Phase',
        title_fontsize=8,
        fontsize=7.5,
        loc='upper left',
        bbox_to_anchor=(1.02, 0.52),
        frameon=True,
    )

    ax.set_title(title, fontsize=10, pad=6)
    fig.tight_layout(rect=[0, 0, 0.78, 1])
    return fig


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Correlation plots across phases or regularisation'
    )
    parser.add_argument('task_id', type=str)
    parser.add_argument('--mode', choices=['alpha', 'phase', 'phase_subplots', 'scatter_panel'], default='alpha')
    parser.add_argument(
        '--corr',
        choices=['lpc_entropy', 'lpc_dist', 'entropy_dist', 'kl_dist'],
        default='lpc_entropy',
        help='Correlation type for alpha/phase modes. Ignored in phase_subplots mode.',
    )
    parser.add_argument(
        '--alpha_values',
        type=str,
        default=None,
        help='Comma-separated alpha values to include in phase_subplots mode, e.g. "0,1e-6,1e-5,1e-4".',
    )
    parser.add_argument(
        '--dist',
        choices=['dist_to_door', 'dist_to_key', 'dist_to_target'],
        default=None,
        help='Distance field to use. Only needed when --phase_system none; otherwise '
             'the distance is implied by each phase (see PHASE_DIST_MAP).',
    )
    parser.add_argument(
        '--phase_system',
        choices=['key_phase', 'unlock_phase', 'none'],
        default=None,
        help='Phase system to use. Inferred from task_id if not specified.',
    )
    parser.add_argument(
        '--trials',
        type=str,
        default='all',
        help='Comma-separated trial numbers, or "all".',
    )
    parser.add_argument('--results_path', type=str, default='evaluation_results.csv')
    parser.add_argument('--cache_dir', type=str, default='evaluation_cache')
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output path. Default: corr_plots/{task_id}/{task_id}_{mode}_{corr}.png',
    )
    parser.add_argument('--sig_level', type=float, default=0.05)
    args = parser.parse_args()

    # Resolve phase system
    phase_system = args.phase_system
    if phase_system is None:
        phase_system = TASK_PHASE_SYSTEM.get(args.task_id, 'none')
        print(f"Phase system: {phase_system}  (inferred from task_id)")
    phases = PHASE_LIST[phase_system]

    # Load alpha map
    print(f"Loading alpha map from {args.results_path}...")
    alpha_map = load_alpha_map(args.task_id, args.results_path)
    if not alpha_map:
        print(f"No entries found for task_id={args.task_id} in {args.results_path}")
        return

    # Resolve trials
    if args.trials == 'all':
        trials = sorted(alpha_map.keys())
    else:
        trials = [int(t.strip()) for t in args.trials.split(',')]

    if args.corr != 'lpc_entropy' and phase_system == 'none' and args.dist is None \
            and args.mode not in ('phase_subplots', 'scatter_panel'):
        parser.error("--dist is required when --phase_system none and corr type is not lpc_entropy")

    safe_task = args.task_id.replace('/', '_')
    out_dir = pathlib.Path('corr_plots') / safe_task
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.mode == 'phase_subplots':
        # Parse requested alpha subset.
        if args.alpha_values is not None:
            alpha_subset = [float(a.strip()) for a in args.alpha_values.split(',')]
        else:
            alpha_subset = _SUBPLOT_ALPHA_ORDER

        print(f"Task: {args.task_id}  Trials: {trials}  Phases: {phases}")
        print(f"Mode: phase_subplots  Alpha subset: {alpha_subset}")
        print()

        df = _collect_phase_subplot_records(
            task_id=args.task_id,
            trials=trials,
            alpha_map=alpha_map,
            phases=phases,
            cache_dir=args.cache_dir,
            alpha_subset=alpha_subset,
        )

        if df.empty:
            print("No data collected — cannot plot.")
            return

        title = (
            f"{args.task_id}\n"
            f"Phase-wise Pearson r vs regularisation strength α"
        )
        fig = plot_phase_subplots_vs_alpha(
            df,
            phases=phases,
            alpha_subset=alpha_subset,
            sig_level=args.sig_level,
            title=title,
        )

        out_path = args.output
        if out_path is None:
            out_path = str(out_dir / f'{safe_task}_phase_subplots_alpha.png')

    elif args.mode == 'scatter_panel':
        if args.alpha_values is not None:
            alpha_subset = [float(a.strip()) for a in args.alpha_values.split(',')]
        else:
            alpha_subset = _SUBPLOT_ALPHA_ORDER

        print(f"Task: {args.task_id}  Trials: {trials}  Phases: {phases}")
        print(f"Mode: scatter_panel  Alpha subset: {alpha_subset}")
        print()

        df = _collect_phase_subplot_records(
            task_id=args.task_id,
            trials=trials,
            alpha_map=alpha_map,
            phases=phases,
            cache_dir=args.cache_dir,
            alpha_subset=alpha_subset,
        )

        if df.empty:
            print("No data collected — cannot plot.")
            return

        title = (
            f"{args.task_id}\n"
            f"Entropy vs KL-local correlation with LPC — by phase and α"
        )
        fig = plot_scatter_panel(
            df,
            phases=phases,
            alpha_subset=alpha_subset,
            title=title,
        )

        out_path = args.output
        if out_path is None:
            out_path = str(out_dir / f'{safe_task}_scatter_panel_alpha.png')

    else:
        print(f"Task: {args.task_id}  Trials: {trials}  Phases: {phases}")
        print(f"Corr: {args.corr}  Mode: {args.mode}"
              + (f"  Dist (fallback): {args.dist}" if args.dist else "  Dist: implied by phase"))
        print()

        df = collect_records(
            task_id=args.task_id,
            trials=trials,
            alpha_map=alpha_map,
            phases=phases,
            corr_type=args.corr,
            fallback_dist=args.dist,
            cache_dir=args.cache_dir,
        )

        if df.empty:
            print("No data collected — cannot plot.")
            return

        title = _build_title(args.task_id, args.mode, args.corr)

        if args.mode == 'alpha':
            fig = plot_corr_vs_alpha(df, sig_level=args.sig_level, title=title)
        else:
            fig = plot_corr_vs_phase(df, sig_level=args.sig_level, title=title,
                                      phase_order=phases)

        out_path = args.output
        if out_path is None:
            out_path = str(out_dir / f'{safe_task}_{args.mode}_{args.corr}.png')

    plt.show(block=False)
    answer = input(f"Save to {out_path}? [y/N] ").strip().lower()
    if answer == 'y':
        fig.savefig(out_path, dpi=150, bbox_inches='tight')
        print(f"Saved → {out_path}")
    else:
        print("Not saved.")


if __name__ == '__main__':
    main()
