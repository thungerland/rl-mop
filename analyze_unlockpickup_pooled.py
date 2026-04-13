"""
analyze_unlockpickup_pooled.py — Pooled publication-quality analysis for BabyAI-UnlockPickup-v0.

Produces five figures and a stats CSV, pooling all evaluation datapoints
(seed × alpha × checkpoint) for alpha ∈ {0, 1e-6, 1e-5, 1e-4}.
(alpha=1e-3 is excluded because success breaks down at that regularisation strength.)

  Figure 1 — pooled_scatter_entropy_by_phase.png
    1×4 row: x = mean LPC, y = H(A|S). One panel per phase.
    OLS regression + 95% CI. Annotated with r, p, N.

  Figure 2 — pooled_scatter_pc_by_phase.png
    1×4 row: x = mean LPC, y = policy complexity I(S;A). One panel per phase.
    OLS regression + 95% CI. Annotated with r, p, N.

  Figure 3 — correlation_summary_by_phase.png
    Combined grouped bar chart: 4 metrics × 4 phases.
    H(A|S) and policy complexity: r recomputed from raw data, Fisher z CI, significance stars.
    KL-local and KL-global: mean of pre-computed r columns, Fisher z CI (no p-value available).

  Figure 4 — regression_coefficients_entropy_by_phase.png
    OLS coefficient of LPC → H(A|S) per phase, controlling for alpha and seed.

  Figure 5 — regression_coefficients_pc_by_phase.png
    OLS coefficient of LPC → policy complexity per phase, controlling for alpha and seed.

  stats_summary.csv
    Phase-level table for all metrics.

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
ALPHA_ORDER = [0.0, 1e-6, 1e-5, 1e-4]

PHASES = [
    'pre_key',
    'post_key_pre_unlock',
    'with_key_post_unlock',
    'post_unlock_post_key',
]

PHASE_LABELS = {
    'pre_key':              'Pre-key',
    'post_key_pre_unlock':  'With-key\n(pre-unlock)',
    'with_key_post_unlock': 'With-key\n(post-unlock)',
    'post_unlock_post_key': 'Post-unlock\n(post-key)',
}

# Same palette as metrics_vs_alpha_plots.py
PHASE_COLORS = ['#264653', '#2a9d8f', '#e76f51', '#e9c46a']

# Viridis samples for coloring scatter points by lpc_alpha value
ALPHA_POINT_COLORS = [plt.cm.viridis(v) for v in np.linspace(0.0, 0.85, len(ALPHA_ORDER))]

# Per-metric colors and labels for the combined correlation summary (Fig 3)
METRICS = ['entropy', 'pc', 'kl_local', 'kl_global']
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
        '--alpha_values', nargs='+', type=float,
        default=[0.0, 1e-6, 1e-5, 1e-4],
        help='Alpha values to include (default: 0 1e-6 1e-5 1e-4)',
    )
    return parser.parse_args()


# ── Data loading ───────────────────────────────────────────────────────────────

def load_data(csv_path: str, task_id: str, update: int, alpha_values: list) -> pd.DataFrame:
    """Load, filter to task_id, restrict to specified alpha values and a single update checkpoint."""
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

    # Restrict to the specified training checkpoint
    if 'update' in df.columns:
        df = df[df['update'] == update].copy()

    if df.empty:
        sys.exit(f'[error] No rows remain after filtering to update={update}')

    n_seeds = df['seed'].nunique() if 'seed' in df.columns else '?'
    n_alphas = df['lpc_alpha'].nunique() if 'lpc_alpha' in df.columns else '?'
    print(
        f'[info] Loaded {len(df)} rows for {task_id} '
        f'(update={update}, {n_seeds} seeds, {n_alphas} alpha values)'
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


def fit_ols_with_controls(sub: pd.DataFrame, lpc_col: str, y_col: str) -> tuple:
    """OLS: y ~ lpc + C(lpc_alpha) + C(seed), numpy-only.

    Returns (coef, ci_low, ci_high, p_val) for the LPC predictor. All NaN if underdetermined.
    """
    nan4 = (float('nan'),) * 4

    sub = sub[[lpc_col, y_col, 'lpc_alpha', 'seed']].dropna()
    N = len(sub)
    if N < 5:
        return nan4

    intercept = np.ones((N, 1))
    lpc_vals = sub[lpc_col].values.reshape(-1, 1)
    alpha_dummies = pd.get_dummies(sub['lpc_alpha'], drop_first=True).values.astype(float)
    seed_dummies = pd.get_dummies(sub['seed'], drop_first=True).values.astype(float)

    X = np.hstack([intercept, lpc_vals, alpha_dummies, seed_dummies])
    y = sub[y_col].values.astype(float)

    n_params = X.shape[1]
    if N <= n_params:
        return nan4

    beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    residuals = y - X @ beta
    sigma2 = float(residuals @ residuals) / (N - n_params)
    cov = sigma2 * np.linalg.pinv(X.T @ X)

    coef = float(beta[1])
    se = float(np.sqrt(max(cov[1, 1], 0.0)))
    if se == 0.0:
        return nan4

    t_stat = coef / se
    p_val = float(2.0 * scipy_stats.t.sf(abs(t_stat), df=N - n_params))
    return coef, coef - 1.96 * se, coef + 1.96 * se, p_val


# ── Shared scatter helper (Figures 1 & 2) ─────────────────────────────────────

def _plot_scatter_row(df: pd.DataFrame, y_col_prefix: str,
                      y_label: str, title: str) -> plt.Figure:
    """1×4 scatter of mean LPC vs a phase-local metric.

    y_col_prefix: 'mean_entropy' or 'policy_complexity'
    """
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.5), sharey=False)
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


# ── Shared regression coefficient helper (Figures 4 & 5) ──────────────────────

def _plot_regression_coefficients(df: pd.DataFrame, y_col_prefix: str,
                                   y_label: str, title: str) -> tuple:
    """OLS coefficient of LPC per phase for a given metric.

    Returns (fig, results) where results is a list of (phase, coef, ci_lo, ci_hi, p).
    """
    results = []
    for phase in PHASES:
        lpc_col = f'mean_lpc_{phase}'
        y_col = f'{y_col_prefix}_{phase}'
        sub_cols = [c for c in [lpc_col, y_col, 'lpc_alpha', 'seed'] if c in df.columns]
        sub = df[sub_cols].dropna()
        if 'lpc_alpha' not in sub.columns or 'seed' not in sub.columns:
            results.append((phase, float('nan'), float('nan'), float('nan'), float('nan')))
        else:
            coef, ci_lo, ci_hi, p = fit_ols_with_controls(sub, lpc_col, y_col)
            results.append((phase, coef, ci_lo, ci_hi, p))

    fig, ax = plt.subplots(figsize=(8, 5))
    x_positions = np.arange(len(PHASES))

    for i, (phase, coef, ci_lo, ci_hi, p) in enumerate(results):
        if np.isnan(coef):
            continue
        ax.scatter(x_positions[i], coef, color=PHASE_COLORS[i], s=80, zorder=4)
        ax.errorbar(x_positions[i], coef,
                    yerr=[[coef - ci_lo], [ci_hi - coef]],
                    fmt='none', color=PHASE_COLORS[i], capsize=6, linewidth=2.0, zorder=3)
        star = _sig_stars(p)
        offset = (ci_hi - ci_lo) * 0.12 + 1e-10
        star_y = ci_hi + offset if coef >= 0 else ci_lo - offset
        va = 'bottom' if coef >= 0 else 'top'
        ax.text(x_positions[i], star_y, star, ha='center', va=va, fontsize=10)

    ax.axhline(0, color='black', linewidth=1.2, linestyle='--', alpha=0.7, zorder=1)
    ax.set_xticks(x_positions)
    ax.set_xticklabels([PHASE_LABELS[ph] for ph in PHASES], fontsize=9)
    ax.tick_params(axis='y', labelsize=9)
    ax.set_ylabel(y_label, fontsize=11)
    ax.set_title(title, fontsize=11, pad=16)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    return fig, results


# ── Figure 3: Combined correlation summary ─────────────────────────────────────

def plot_correlation_summary(df: pd.DataFrame, task_id: str) -> tuple:
    """Combined grouped bar chart: 4 metrics × 4 phases.

    Entropy, KL-local, KL-global: use pre-computed r_<phase>_lpc_* and p_<phase>_lpc_* columns
    directly (mean r across runs, mean p for significance stars).
    Policy complexity: computed from raw mean_lpc_<phase> vs policy_complexity_<phase> columns
    (no pre-computed r available).

    Returns (fig, stats_rows) where stats_rows is a list of dicts, one per phase.
    """
    n_metrics = len(METRICS)
    bar_width = 0.18
    group_gap = 0.12
    x_positions = np.arange(len(PHASES)) * (n_metrics * bar_width + group_gap)

    # Collect stats per metric per phase
    data = {m: {'r': [], 'ci_lo': [], 'ci_hi': [], 'N': [], 'p': []} for m in METRICS}
    phase_N = {}

    for phase in PHASES:
        lpc_col = f'mean_lpc_{phase}'
        x = df[lpc_col].values if lpc_col in df.columns else np.array([])
        N_phase = int(np.sum(np.isfinite(x)))
        phase_N[phase] = N_phase

        # H(A|S): use pre-computed r and p columns, averaged across runs
        r_ent_col = f'r_{phase}_lpc_entropy'
        p_ent_col = f'p_{phase}_lpc_entropy'
        r_ent = float(df[r_ent_col].dropna().mean()) if r_ent_col in df.columns else float('nan')
        p_ent = float(df[p_ent_col].dropna().mean()) if p_ent_col in df.columns else float('nan')
        lo_ent, hi_ent = _r_ci_from_precomputed(r_ent, N_phase)
        data['entropy']['r'].append(r_ent)
        data['entropy']['ci_lo'].append(lo_ent)
        data['entropy']['ci_hi'].append(hi_ent)
        data['entropy']['N'].append(N_phase)
        data['entropy']['p'].append(p_ent)

        # Policy complexity: computed from raw means (no pre-computed r available)
        pc_col = f'policy_complexity_{phase}'
        y_pc = df[pc_col].values if pc_col in df.columns else np.array([])
        N_pc, r_pc, p_pc, lo_pc, hi_pc = compute_pearsonr_with_ci(x, y_pc)
        data['pc']['r'].append(r_pc)
        data['pc']['ci_lo'].append(lo_pc)
        data['pc']['ci_hi'].append(hi_pc)
        data['pc']['N'].append(N_pc)
        data['pc']['p'].append(p_pc)

        # KL-local: use pre-computed r and p columns, averaged across runs
        r_kl_loc_col = f'r_{phase}_lpc_kl_local'
        p_kl_loc_col = f'p_{phase}_lpc_kl_local'
        r_kl_loc = float(df[r_kl_loc_col].dropna().mean()) if r_kl_loc_col in df.columns else float('nan')
        p_kl_loc = float(df[p_kl_loc_col].dropna().mean()) if p_kl_loc_col in df.columns else float('nan')
        lo_kl_loc, hi_kl_loc = _r_ci_from_precomputed(r_kl_loc, N_phase)
        data['kl_local']['r'].append(r_kl_loc)
        data['kl_local']['ci_lo'].append(lo_kl_loc)
        data['kl_local']['ci_hi'].append(hi_kl_loc)
        data['kl_local']['N'].append(N_phase)
        data['kl_local']['p'].append(p_kl_loc)

        # KL-global: use pre-computed r and p columns, averaged across runs
        r_kl_glob_col = f'r_{phase}_lpc_kl_global'
        p_kl_glob_col = f'p_{phase}_lpc_kl_global'
        r_kl_glob = float(df[r_kl_glob_col].dropna().mean()) if r_kl_glob_col in df.columns else float('nan')
        p_kl_glob = float(df[p_kl_glob_col].dropna().mean()) if p_kl_glob_col in df.columns else float('nan')
        lo_kl_glob, hi_kl_glob = _r_ci_from_precomputed(r_kl_glob, N_phase)
        data['kl_global']['r'].append(r_kl_glob)
        data['kl_global']['ci_lo'].append(lo_kl_glob)
        data['kl_global']['ci_hi'].append(hi_kl_glob)
        data['kl_global']['N'].append(N_phase)
        data['kl_global']['p'].append(p_kl_glob)

    # Build stats_rows (one per phase)
    stats_rows = []
    for i, phase in enumerate(PHASES):
        stats_rows.append({
            'phase': phase,
            'N_entropy': data['entropy']['N'][i],
            'r_entropy': data['entropy']['r'][i],
            'p_entropy': data['entropy']['p'][i],
            'r_entropy_ci_low': data['entropy']['ci_lo'][i],
            'r_entropy_ci_high': data['entropy']['ci_hi'][i],
            'N_pc': data['pc']['N'][i],
            'r_pc': data['pc']['r'][i],
            'p_pc': data['pc']['p'][i],
            'r_pc_ci_low': data['pc']['ci_lo'][i],
            'r_pc_ci_high': data['pc']['ci_hi'][i],
            'r_kl_local': data['kl_local']['r'][i],
            'r_kl_local_ci_low': data['kl_local']['ci_lo'][i],
            'r_kl_local_ci_high': data['kl_local']['ci_hi'][i],
            'r_kl_global': data['kl_global']['r'][i],
            'r_kl_global_ci_low': data['kl_global']['ci_lo'][i],
            'r_kl_global_ci_high': data['kl_global']['ci_hi'][i],
            'reg_coef_entropy': float('nan'),
            'reg_coef_entropy_ci_low': float('nan'),
            'reg_coef_entropy_ci_high': float('nan'),
            'reg_coef_entropy_p': float('nan'),
            'reg_coef_pc': float('nan'),
            'reg_coef_pc_ci_low': float('nan'),
            'reg_coef_pc_ci_high': float('nan'),
            'reg_coef_pc_p': float('nan'),
        })

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
                star_y = r_hi + 0.02 if r >= 0 else r_lo - 0.04
                ax.text(xpos, star_y, star, ha='center', va='bottom', fontsize=7)

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
        'Correlation with Mean LPC by phase and metric\n'
        '(pooled, 95% CI via Fisher z-transform; KL from pre-computed per-run r)',
        fontsize=11,
    )
    ax.legend(loc='upper right', fontsize=9, frameon=True)
    fig.suptitle(task_id, fontsize=13)
    fig.tight_layout()

    return fig, stats_rows


# ── Stats CSV ──────────────────────────────────────────────────────────────────

def write_stats_csv(stats_rows: list, out_dir: pathlib.Path, alpha_suffix: str = '') -> None:
    """Write stats_summary.csv and print a concise table to stdout."""
    stats_df = pd.DataFrame(stats_rows)
    suffix = f'_{alpha_suffix}' if alpha_suffix else ''
    out_path = out_dir / f'stats_summary{suffix}.csv'
    stats_df.to_csv(out_path, index=False, float_format='%.6f')
    print(f'\n[info] Stats summary saved -> {out_path}')

    display_cols = ['phase', 'N_entropy', 'r_entropy', 'p_entropy',
                    'r_pc', 'p_pc', 'r_kl_local', 'r_kl_global']
    display_cols = [c for c in display_cols if c in stats_df.columns]
    print('\n' + stats_df[display_cols].to_string(
        index=False, float_format=lambda x: f'{x:.4f}'
    ) + '\n')


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()

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

    df = load_data(args.csv, args.task_id, args.update, args.alpha_values)

    # Build a filename suffix encoding the selected alphas, e.g. "alpha_0_1e-06_1e-05_1e-04"
    alpha_suffix = 'alpha_' + '_'.join(_alpha_label(a) for a in sorted(args.alpha_values))

    # Figure 1: LPC vs H(A|S)
    print('[info] Generating Figure 1: pooled scatter LPC vs H(A|S) ...')
    fig1 = _plot_scatter_row(
        df, 'mean_entropy', 'Mean H(A|S) (bits)',
        f'{args.task_id} — Mean LPC vs H(A|S), pooled across seeds × alpha (update={args.update})',
    )
    save_figure(fig1, out_dir, f'pooled_scatter_entropy_by_phase_{alpha_suffix}', args.save)

    # Figure 2: LPC vs policy complexity
    print('[info] Generating Figure 2: pooled scatter LPC vs policy complexity ...')
    fig2 = _plot_scatter_row(
        df, 'policy_complexity', 'Policy complexity I(S;A)',
        f'{args.task_id} — Mean LPC vs policy complexity, pooled across seeds × alpha (update={args.update})',
    )
    save_figure(fig2, out_dir, f'pooled_scatter_pc_by_phase_{alpha_suffix}', args.save)

    # Figure 3: Combined correlation summary
    print('[info] Generating Figure 3: combined correlation summary ...')
    fig3, stats_rows = plot_correlation_summary(df, args.task_id)
    save_figure(fig3, out_dir, f'correlation_summary_by_phase_{alpha_suffix}', args.save)

    # Figure 4: Regression coefficients for H(A|S)
    print('[info] Generating Figure 4: regression coefficients LPC → H(A|S) ...')
    fig4, reg_entropy = _plot_regression_coefficients(
        df, 'mean_entropy',
        'OLS coefficient of LPC\n(controlling for α and seed)',
        'LPC → H(A|S) regression coefficient by phase\n'
        '(OLS: entropy ~ LPC + α dummies + seed dummies)',
    )
    fig4.suptitle(args.task_id, fontsize=13)
    save_figure(fig4, out_dir, f'regression_coefficients_entropy_by_phase_{alpha_suffix}', args.save)

    # Figure 5: Regression coefficients for policy complexity
    print('[info] Generating Figure 5: regression coefficients LPC → policy complexity ...')
    fig5, reg_pc = _plot_regression_coefficients(
        df, 'policy_complexity',
        'OLS coefficient of LPC\n(controlling for α and seed)',
        'LPC → policy complexity regression coefficient by phase\n'
        '(OLS: PC ~ LPC + α dummies + seed dummies)',
    )
    fig5.suptitle(args.task_id, fontsize=13)
    save_figure(fig5, out_dir, f'regression_coefficients_pc_by_phase_{alpha_suffix}', args.save)

    # Merge regression results into stats_rows
    for i, (phase, coef, ci_lo, ci_hi, p) in enumerate(reg_entropy):
        stats_rows[i].update({
            'reg_coef_entropy': coef,
            'reg_coef_entropy_ci_low': ci_lo,
            'reg_coef_entropy_ci_high': ci_hi,
            'reg_coef_entropy_p': p,
        })
    for i, (phase, coef, ci_lo, ci_hi, p) in enumerate(reg_pc):
        stats_rows[i].update({
            'reg_coef_pc': coef,
            'reg_coef_pc_ci_low': ci_lo,
            'reg_coef_pc_ci_high': ci_hi,
            'reg_coef_pc_p': p,
        })

    write_stats_csv(stats_rows, out_dir, alpha_suffix)

    if not args.save:
        input('Press Enter to close all figures...')
    plt.close('all')


if __name__ == '__main__':
    main()
