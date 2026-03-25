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

_VALID_ALPHAS = {1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0}


def _is_valid_alpha(alpha: float) -> bool:
    """Accept only powers of 10 in [1e-6, 1.0]."""
    return any(abs(alpha - v) / max(v, 1e-12) < 1e-6 for v in _VALID_ALPHAS)


def load_alpha_map(task_id: str, results_path: str = 'evaluation_results.csv') -> dict:
    """Load trial -> lpc_alpha mapping from evaluation_results.csv.

    Only includes trials whose lpc_alpha is a power of 10 in [1e-6, 1.0].
    """
    df = pd.read_csv(results_path)
    subset = df[df['task_id'] == task_id][['trial', 'lpc_alpha']].dropna()
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

def compute_corr(routing_data: list, corr_type: str, phase, dist_field: str | None) -> dict:
    """Compute one correlation value using phase-filtered data throughout.

    All quantities (H_s, KL_s, mean LPC, dist pairings) are derived from the
    phase-filtered subset only, so distributions reflect within-phase behaviour.

    Returns dict with 'r', 'p', 'n'.
    """
    phase_data = list(_filter_by_phase(routing_data, phase))

    if corr_type == 'lpc_entropy':
        result = spatial_entropy_lpc_correlation(phase_data)
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

        for phase in phases:
            dist_field = PHASE_DIST_MAP.get(phase, None)
            if dist_field is None and corr_type != 'lpc_entropy':
                dist_field = fallback_dist  # use explicit --dist for none-phase system
            try:
                res = compute_corr(routing_data, corr_type, phase, dist_field)
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


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Correlation plots across phases or regularisation'
    )
    parser.add_argument('task_id', type=str)
    parser.add_argument('--mode', choices=['alpha', 'phase'], default='alpha')
    parser.add_argument(
        '--corr',
        choices=['lpc_entropy', 'lpc_dist', 'entropy_dist', 'kl_dist'],
        default='lpc_entropy',
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
        help='Output path. Default: plots/corr_plots/{task_id}_{mode}_{corr}.png',
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

    if args.corr != 'lpc_entropy' and phase_system == 'none' and args.dist is None:
        parser.error("--dist is required when --phase_system none and corr type is not lpc_entropy")

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

    # Default output path
    out_path = args.output
    if out_path is None:
        out_dir = pathlib.Path('plots') / 'corr_plots'
        out_dir.mkdir(parents=True, exist_ok=True)
        safe_task = args.task_id.replace('/', '_')
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
