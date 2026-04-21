"""
analyze_unlockpickup_pooled.py — Pooled publication-quality analysis for BabyAI-UnlockPickup-v0.

Produces three figures, pooling all evaluation datapoints
(seed × alpha × checkpoint) for alpha ∈ {1e-6, 1e-5, 1e-4}.
(alpha=0.0 and alpha=1e-3 are excluded.)

  Figure 1 — pooled_scatter_entropy_by_phase.png
    1×4 row: x = mean LPC, y = H(A|S). One panel per phase.
    OLS regression + 95% CI. Annotated with r, p, N.

  Figure 2 — pooled_scatter_pc_by_phase.png
    1×4 row: x = mean LPC, y = policy complexity I(S;A). One panel per phase.
    OLS regression + 95% CI. Annotated with r, p, N.

  Figure 3 — correlation_summary_by_phase.png
    Combined grouped bar chart: 3 metrics × 4 phases.
    r values aggregated via Fisher z-transform across seeds × alpha.
    p-values from one-sample t-test (H0: r = 0) on raw per-seed r values.

Usage:
    python analyze_unlockpickup_pooled.py [--csv PATH] [--out_dir DIR] [--save] [--task_id ID]

Dependencies: numpy, pandas, matplotlib, seaborn, scipy  (no statsmodels)
"""

import argparse
import pathlib
import sys

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats as scipy_stats


# ── Constants ─────────────────────────────────────────────────────────────────

# alpha=1e-3 excluded: success breaks down at that regularisation strength.
ALPHA_ORDER = [1e-6, 1e-5, 1e-4]

# Phases, labels, and colors are resolved at runtime from the task_id.
# These module-level names are populated by _init_phases() called in main().
PHASES = []
PHASE_LABELS = {}
PHASE_COLORS = []

# Full palette — first N entries are used depending on number of phases
_ALL_PHASE_COLORS = ['#264653', '#2a9d8f', '#e76f51', '#e9c46a']

# Import phase metadata from corr_plots
from corr_plots import PHASE_LIST, PHASE_LABELS as _CORR_PHASE_LABELS, TASK_PHASE_SYSTEM


def _init_phases(task_id: str) -> None:
    """Populate module-level PHASES, PHASE_LABELS, PHASE_COLORS for the given task."""
    global PHASES, PHASE_LABELS, PHASE_COLORS
    phase_system = TASK_PHASE_SYSTEM.get(task_id, 'key_phase')
    PHASES = PHASE_LIST[phase_system]
    PHASE_LABELS = {ph: _CORR_PHASE_LABELS[ph] for ph in PHASES}
    PHASE_COLORS = _ALL_PHASE_COLORS[:len(PHASES)]

# Viridis samples for coloring scatter points by lpc_alpha value
ALPHA_POINT_COLORS = [plt.cm.viridis(v) for v in np.linspace(0.0, 0.85, len(ALPHA_ORDER))]

# Per-metric colors and labels for the combined correlation summary (Fig 3)
METRICS = ['entropy', 'kl_local', 'kl_global']
METRIC_COLORS = {
    'entropy':  '#9b59b6',
    'pc':       '#e76f51',
    'kl_local': '#2a9d8f',
    'kl_global': '#264653',
}
METRIC_LABELS = {
    'entropy':  'H(A|S)',
    'pc':       'Policy complexity',
    'kl_local': 'KL (local)',
    'kl_global': 'KL (global)',
}


def _alpha_label(a: float) -> str:
    """Human-readable alpha label (mirrors metrics_vs_alpha_plots.py)."""
    return '0' if a == 0.0 else f'{a:.0e}'


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Pooled scatter/correlation/regression analysis for BabyAI-UnlockPickup-v0.',
    )
    parser.add_argument(
        '--csv', default='eval_metrics_unlockpickup.csv',
        help='Path to eval metrics CSV (default: eval_metrics_unlockpickup.csv)',
    )
    parser.add_argument(
        '--out_dir', default='plots/pooled',
        help='Output directory for figures and stats CSV (default: plots/pooled)',
    )
    parser.add_argument(
        '--save', action='store_true',
        help='Auto-save figures without interactive prompt',
    )
    parser.add_argument(
        '--task_id', default='BabyAI-UnlockPickup-v0',
        help='Task ID to filter from CSV (default: BabyAI-UnlockPickup-v0)',
    )
    parser.add_argument(
        '--update', type=int, default=5000,
        help='Only use rows from this training checkpoint (default: 5000)',
    )
    parser.add_argument(
        '--min_update', type=int, default=None,
        help='Include all checkpoints with update >= this value instead of filtering to --update. '
             'Overrides --update when set.',
    )
    parser.add_argument(
        '--alpha_values', nargs='+', type=float,
        default=[1e-6, 1e-5, 1e-4],
        help='Alpha values to include (default: 1e-6 1e-5 1e-4)',
    )
    return parser.parse_args()


# ── Data loading ───────────────────────────────────────────────────────────────

def load_data(csv_path: str, task_id: str, update: int, alpha_values: list,
              min_update: int = None) -> pd.DataFrame:
    """Load, filter to task_id, restrict to specified alpha values and update checkpoint(s).

    If min_update is set, includes all rows with update >= min_update (overrides update).
    Otherwise filters to the single update value.
    """
    path = pathlib.Path(csv_path)
    if not path.exists():
        sys.exit(f'[error] CSV not found: {csv_path}')

    df = pd.read_csv(path)

    if 'task_id' in df.columns:
        df = df[df['task_id'] == task_id].copy()

    if df.empty:
        sys.exit(f'[error] No rows found for task_id={task_id!r} in {csv_path}')

    if 'lpc_alpha' in df.columns:
        df = df[df['lpc_alpha'].isin(alpha_values)].copy()

    if df.empty:
        sys.exit(f'[error] No rows remain after filtering to alpha ∈ {alpha_values}')

    if 'update' in df.columns:
        if min_update is not None:
            df = df[df['update'] >= min_update].copy()
            update_label = f'update>={min_update}'
        else:
            df = df[df['update'] == update].copy()
            update_label = f'update={update}'
    else:
        update_label = 'update=unknown'

    if df.empty:
        sys.exit(f'[error] No rows remain after filtering to {update_label}')

    n_seeds = df['seed'].nunique() if 'seed' in df.columns else '?'
    n_alphas = df['lpc_alpha'].nunique() if 'lpc_alpha' in df.columns else '?'
    print(
        f'[info] Loaded {len(df)} rows for {task_id} '
        f'({update_label}, {n_seeds} seeds, {n_alphas} alpha values)'
    )
    print(f'[info] Alpha values used: {[_alpha_label(a) for a in sorted(alpha_values)]}')
    return df


# ── Figure saving ──────────────────────────────────────────────────────────────

def save_figure(fig: plt.Figure, out_dir: pathlib.Path, stem: str, auto_save: bool) -> None:
    """Save figure as PNG with optional interactive Y/N prompt."""
    out_path_png = out_dir / f'{stem}.png'

    if auto_save:
        fig.savefig(out_path_png, dpi=150, bbox_inches='tight')
        print(f'[info] Saved -> {out_path_png}')
    else:
        plt.figure(fig.number)
        plt.show(block=False)
        answer = input(f'Save to {out_path_png}? [y/N] ').strip().lower()
        if answer == 'y':
            fig.savefig(out_path_png, dpi=150, bbox_inches='tight')
            print(f'[info] Saved -> {out_path_png}')
        else:
            print('[info] Not saved.')


# ── Statistical helpers ────────────────────────────────────────────────────────

def compute_pearsonr_with_ci(x: np.ndarray, y: np.ndarray, alpha: float = 0.05) -> tuple:
    """Pearson r with Fisher z-transform 95% CI.

    Returns (N, r, p, r_ci_low, r_ci_high). All floats NaN if N < 4.
    """
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    N = len(x)

    nan5 = (N, float('nan'), float('nan'), float('nan'), float('nan'))
    if N < 4:
        return nan5

    r, p = scipy_stats.pearsonr(x, y)

    z = np.arctanh(np.clip(r, -0.9999, 0.9999))
    se_z = 1.0 / np.sqrt(N - 3)
    z_crit = scipy_stats.norm.ppf(1.0 - alpha / 2.0)
    r_ci_low = float(np.tanh(z - z_crit * se_z))
    r_ci_high = float(np.tanh(z + z_crit * se_z))

    return N, float(r), float(p), r_ci_low, r_ci_high


def _r_ci_from_precomputed(r: float, N: int, alpha: float = 0.05) -> tuple:
    """Fisher z CI for a pre-computed r value given sample size N."""
    if np.isnan(r) or N < 4:
        return float('nan'), float('nan')
    z = np.arctanh(np.clip(r, -0.9999, 0.9999))
    se_z = 1.0 / np.sqrt(N - 3)
    z_crit = scipy_stats.norm.ppf(1.0 - alpha / 2.0)
    return float(np.tanh(z - z_crit * se_z)), float(np.tanh(z + z_crit * se_z))


def _fisher_aggregate(r_values) -> tuple:
    """Aggregate per-seed Pearson r values via Fisher z-transform.

    Returns (r_agg, p_value, ci_lo, ci_hi).
    - r_agg  : back-transformed mean of Fisher z scores
    - p_value: one-sample t-test of raw r values against H0: mean = 0
    - ci_lo/hi: 95% CI derived from the spread of Fisher z scores
    """
    r_arr = np.asarray(r_values, dtype=float)
    r_arr = r_arr[np.isfinite(r_arr)]
    n = len(r_arr)
    if n < 2:
        return float('nan'), float('nan'), float('nan'), float('nan')

    z_arr = np.arctanh(np.clip(r_arr, -0.9999, 0.9999))
    z_mean = z_arr.mean()
    r_agg = float(np.tanh(z_mean))

    # t-test on raw r values against H0: mean = 0
    _, p_val = scipy_stats.ttest_1samp(r_arr, popmean=0.0)

    # 95% CI: SE of the mean z score across the n estimates
    se_z = z_arr.std(ddof=1) / np.sqrt(n)
    z_crit = scipy_stats.norm.ppf(0.975)
    ci_lo = float(np.tanh(z_mean - z_crit * se_z))
    ci_hi = float(np.tanh(z_mean + z_crit * se_z))

    return r_agg, float(p_val), ci_lo, ci_hi


def format_p(p: float) -> str:
    if np.isnan(p):
        return 'p = n/a'
    if p < 0.001:
        return 'p < 0.001'
    return f'p = {p:.3f}'


def _sig_stars(p: float) -> str:
    if np.isnan(p):
        return ''
    if p < 0.001:
        return '***'
    if p < 0.01:
        return '**'
    if p < 0.05:
        return '*'
    return 'ns'


# ── Shared scatter helper (Figures 1 & 2) ─────────────────────────────────────

def _plot_scatter_row(df: pd.DataFrame, y_col_prefix: str,
                      y_label: str, title: str) -> plt.Figure:
    """1×4 scatter of mean LPC vs a phase-local metric.

    y_col_prefix: 'mean_entropy' or 'policy_complexity'
    """
    n_phases = len(PHASES)
    fig, axes = plt.subplots(1, n_phases, figsize=(4.5 * n_phases, 4.5), sharey=False)
    if n_phases == 1:
        axes = [axes]
    alpha_to_color = {a: ALPHA_POINT_COLORS[i] for i, a in enumerate(ALPHA_ORDER)}

    for ax, phase, phase_color in zip(axes, PHASES, PHASE_COLORS):
        lpc_col = f'mean_lpc_{phase}'
        y_col = f'{y_col_prefix}_{phase}'

        if lpc_col not in df.columns or y_col not in df.columns:
            ax.set_visible(False)
            continue

        sub = df[[lpc_col, y_col, 'lpc_alpha']].dropna()
        N = len(sub)

        if N == 0:
            ax.set_title(PHASE_LABELS[phase], fontsize=11, pad=6)
            continue

        point_colors = [alpha_to_color.get(a, ALPHA_POINT_COLORS[0]) for a in sub['lpc_alpha']]
        ax.scatter(sub[lpc_col], sub[y_col], c=point_colors, alpha=0.65, s=22,
                   zorder=3, linewidths=0)

        x_vals = sub[lpc_col].values
        y_vals = sub[y_col].values
        slope, intercept, r, p, se_slope = scipy_stats.linregress(x_vals, y_vals)

        x_line = np.linspace(x_vals.min(), x_vals.max(), 200)
        y_line = intercept + slope * x_line

        x_bar = x_vals.mean()
        ss_xx = np.sum((x_vals - x_bar) ** 2)
        t_crit = scipy_stats.t.ppf(0.975, df=N - 2)
        se_mean = se_slope * np.sqrt(ss_xx) * np.sqrt(
            1.0 / N + (x_line - x_bar) ** 2 / ss_xx
        )
        ax.plot(x_line, y_line, color=phase_color, linewidth=2.0, zorder=4)
        ax.fill_between(x_line, y_line - t_crit * se_mean, y_line + t_crit * se_mean,
                        alpha=0.18, color=phase_color, zorder=2)

        ax.text(0.05, 0.95, f'r = {r:.2f}\n{format_p(p)}\nN = {N}',
                transform=ax.transAxes, va='top', ha='left', fontsize=9,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.75, linewidth=0))

        ax.set_xlabel('Mean LPC', fontsize=10)
        ax.set_ylabel(y_label, fontsize=10)
        ax.set_title(PHASE_LABELS[phase], fontsize=11, pad=6)
        ax.tick_params(labelsize=9)

    legend_patches = [
        mpatches.Patch(facecolor=ALPHA_POINT_COLORS[i], label=f'α = {_alpha_label(a)}', alpha=0.9)
        for i, a in enumerate(ALPHA_ORDER)
    ]
    fig.legend(handles=legend_patches, title='lpc_alpha', loc='lower center',
               ncol=len(ALPHA_ORDER), fontsize=9, title_fontsize=9, frameon=True,
               bbox_to_anchor=(0.5, -0.10))
    fig.suptitle(title, fontsize=12, y=1.03)
    fig.tight_layout()
    return fig


# ── Figure 3: Combined correlation summary ─────────────────────────────────────

def plot_correlation_summary(df: pd.DataFrame, task_id: str) -> tuple:
    """Combined grouped bar chart: 3 metrics × 4 phases.

    Aggregates per-seed pre-computed r values (r_<phase>_lpc_<metric>) via Fisher z-transform.
    P-values from a one-sample t-test on the raw r values (H0: mean r = 0).

    Returns fig.
    """
    n_metrics = len(METRICS)
    bar_width = 0.18
    group_gap = 0.12
    x_positions = np.arange(len(PHASES)) * (n_metrics * bar_width + group_gap)

    # Collect stats per metric per phase
    data = {m: {'r': [], 'ci_lo': [], 'ci_hi': [], 'N': [], 'p': []} for m in METRICS}
    phase_N = {}

    _metric_col_suffix = {
        'entropy':  'entropy',
        'kl_local': 'kl_local',
        'kl_global': 'kl_global',
    }

    for phase in PHASES:
        lpc_col = f'mean_lpc_{phase}'
        x = df[lpc_col].values if lpc_col in df.columns else np.array([])
        N_phase = int(np.sum(np.isfinite(x)))
        phase_N[phase] = N_phase

        for metric in METRICS:
            r_col = f'r_{phase}_lpc_{_metric_col_suffix[metric]}'
            r_vals = df[r_col].dropna().values if r_col in df.columns else np.array([])
            r_agg, p_agg, ci_lo, ci_hi = _fisher_aggregate(r_vals)
            data[metric]['r'].append(r_agg)
            data[metric]['ci_lo'].append(ci_lo)
            data[metric]['ci_hi'].append(ci_hi)
            data[metric]['N'].append(N_phase)
            data[metric]['p'].append(p_agg)

    # Plot
    fig, ax = plt.subplots(figsize=(13, 5))
    all_ci_hi, all_ci_lo = [], []

    for m_idx, metric in enumerate(METRICS):
        offsets = (m_idx - n_metrics / 2.0 + 0.5) * bar_width
        color = METRIC_COLORS[metric]

        for i, phase in enumerate(PHASES):
            r = data[metric]['r'][i]
            r_lo = data[metric]['ci_lo'][i]
            r_hi = data[metric]['ci_hi'][i]
            p = data[metric]['p'][i]

            if np.isnan(r):
                continue

            xpos = x_positions[i] + offsets
            ax.bar(xpos, r, width=bar_width * 0.9, color=color, alpha=0.85, zorder=3,
                   label=METRIC_LABELS[metric] if i == 0 else '_nolegend_')
            ax.errorbar(xpos, r, yerr=[[r - r_lo], [r_hi - r]],
                        fmt='none', color='black', capsize=3, linewidth=1.2, zorder=4)

            if not np.isnan(p):
                star = _sig_stars(p)
                if r >= 0:
                    star_y, va = r_hi + 0.02, 'bottom'
                else:
                    star_y, va = r_lo - 0.02, 'top'
                ax.text(xpos, star_y, star, ha='center', va=va, fontsize=7)

            all_ci_hi.append(r_hi)
            all_ci_lo.append(r_lo)

    ax.axhline(0, color='black', linewidth=0.8, linestyle='--', zorder=1)

    # x-ticks at centre of each phase group
    ax.set_xticks(x_positions)
    ax.set_xticklabels(
        [f'{PHASE_LABELS[ph]}\n(N={phase_N[ph]})' for ph in PHASES],
        fontsize=9,
    )
    ax.tick_params(axis='y', labelsize=9)

    y_min = min(0.0, min(all_ci_lo, default=0.0)) - 0.05
    y_max = max(all_ci_hi, default=0.0) + 0.18
    ax.set_ylim(y_min, y_max)

    ax.set_ylabel('Pearson r  (vs Mean LPC)', fontsize=11)
    ax.set_title(
        r'Correlation with Mean LPC by phase and metric'
        '\n'
        r'(Fisher z aggregation across seeds $\times$ alpha; t-test p-values vs $H_0$: $r = 0$)',
        fontsize=11,
    )

    # Combined legend outside the plot area to the right
    metric_handles = [
        mpatches.Patch(color=METRIC_COLORS[m], label=METRIC_LABELS[m])
        for m in METRICS
    ]
    sig_handles = [
        mpatches.Patch(color='none', label='— Significance —'),
        mpatches.Patch(color='none', label='***  p < 0.001'),
        mpatches.Patch(color='none', label='**    p < 0.01'),
        mpatches.Patch(color='none', label='*      p < 0.05'),
        mpatches.Patch(color='none', label='ns    p ≥ 0.05'),
    ]
    ax.legend(
        handles=metric_handles + sig_handles,
        loc='upper left', bbox_to_anchor=(1.01, 1), borderaxespad=0,
        fontsize=8, frameon=True,
    )
    fig.suptitle(task_id, fontsize=13)
    fig.tight_layout(rect=[0, 0, 0.82, 1])

    return fig


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    _init_phases(args.task_id)

    sns.set_style('whitegrid')
    plt.rcParams.update({
        'font.size': 11,
        'axes.labelsize': 11,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'figure.dpi': 100,
    })

    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_data(args.csv, args.task_id, args.update, args.alpha_values, args.min_update)

    # Build a filename suffix encoding the selected alphas and update range
    update_label = f'gte{args.min_update}' if args.min_update is not None else str(args.update)
    update_desc  = f'update>={args.min_update}' if args.min_update is not None else f'update={args.update}'
    alpha_suffix = 'alpha_' + '_'.join(_alpha_label(a) for a in sorted(args.alpha_values))
    alpha_suffix += f'_update_{update_label}'

    # Figure 1: LPC vs H(A|S)
    print('[info] Generating Figure 1: pooled scatter LPC vs H(A|S) ...')
    fig1 = _plot_scatter_row(
        df, 'mean_entropy', 'Mean H(A|S) (bits)',
        f'{args.task_id} — Mean LPC vs H(A|S), pooled across seeds × alpha ({update_desc})',
    )
    save_figure(fig1, out_dir, f'pooled_scatter_entropy_by_phase_{alpha_suffix}', args.save)

    # Figure 2: LPC vs policy complexity
    print('[info] Generating Figure 2: pooled scatter LPC vs policy complexity ...')
    fig2 = _plot_scatter_row(
        df, 'policy_complexity', 'Policy complexity I(S;A)',
        f'{args.task_id} — Mean LPC vs policy complexity, pooled across seeds × alpha ({update_desc})',
    )
    save_figure(fig2, out_dir, f'pooled_scatter_pc_by_phase_{alpha_suffix}', args.save)

    # Figure 3: Combined correlation summary
    print('[info] Generating Figure 3: combined correlation summary ...')
    fig3 = plot_correlation_summary(df, args.task_id)
    save_figure(fig3, out_dir, f'correlation_summary_by_phase_{alpha_suffix}', args.save)

    if not args.save:
        input('Press Enter to close all figures...')
    plt.close('all')


if __name__ == '__main__':
    main()
