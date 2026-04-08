"""
metrics_vs_alpha_plots.py — Seed-aggregated metric plots from eval_metrics_unlockpickup.csv.

Produces two publication-quality single-panel figures:

  Plot 1 (--plot alpha):  4 metrics vs regularisation strength α
    - x-axis: categorical alpha values (0, 1e-6, 1e-5, 1e-4, 1e-3)
    - Data: final checkpoints (max update per seed) for trials 20/21/22/23/26
    - Each metric normalised to [0, 1] across the plotted alpha range
    - Mean line + SEM band + individual seed scatter

  Plot 2 (--plot updates): 4 metrics vs training update (trial 20 only)
    - x-axis: categorical update checkpoints (250, 500, ..., 5000)
    - Same normalisation, shading, and scatter style

Requires: eval_metrics_unlockpickup.csv produced by modal_app.py::run_extract_seed_metrics

Usage:
    python metrics_vs_alpha_plots.py --csv eval_metrics_unlockpickup.csv --plot alpha --save
    python metrics_vs_alpha_plots.py --csv eval_metrics_unlockpickup.csv --plot updates --save
    python metrics_vs_alpha_plots.py --csv eval_metrics_unlockpickup.csv --plot both --save
"""

import argparse
import pathlib

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


# ── Constants ─────────────────────────────────────────────────────────────────

ALPHA_ORDER = [0.0, 1e-6, 1e-5, 1e-4, 1e-3]
ALPHA_LABELS = ['0', '1e-6', '1e-5', '1e-4', '1e-3']

METRICS = [
    {'col': 'success_rate',      'label': 'Success rate',       'color': '#264653'},
    {'col': 'mean_lpc',          'label': 'Mean LPC',           'color': '#2a9d8f'},
    {'col': 'policy_complexity', 'label': 'Policy complexity',  'color': '#e76f51'},
    {'col': 'path_ratio',        'label': 'Path ratio',         'color': '#e9c46a'},
]

PHASES = [
    'pre_key',
    'post_key_pre_unlock',
    'with_key_post_unlock',
    'post_unlock_post_key',
]

PHASE_LABELS = {
    'pre_key':              'Pre-key',
    'post_key_pre_unlock':  'With-key (pre-unlock)',
    'with_key_post_unlock': 'With-key (post-unlock)',
    'post_unlock_post_key': 'Post-unlock (post-key)',
}

PHASE_METRIC_LINES = [
    {'col_prefix': 'policy_complexity', 'label': 'I(S;A)',   'color': '#e76f51', 'ls': '-',  'lw': 2.0, 'alpha': 1.0},
    {'col_prefix': 'mean_entropy',      'label': 'H\u0305(A|S)', 'color': '#9b59b6', 'ls': '-',  'lw': 2.0, 'alpha': 1.0},
    {'col_prefix': 'mean_lpc',          'label': 'Mean LPC', 'color': '#2a9d8f', 'ls': '--', 'lw': 1.4, 'alpha': 0.8},
]


def _alpha_label(a: float) -> str:
    return '0' if a == 0.0 else f'{a:.0e}'


# ── Aggregation ───────────────────────────────────────────────────────────────

def aggregate_by_group(df: pd.DataFrame, group_col: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute mean ± SEM for each metric, grouped by group_col.

    Returns:
        agg_df: one row per group value, columns: <group_col>, <metric>_mean,
                <metric>_sem, <metric>_lo, <metric>_hi  (lo/hi = mean ± SEM)
        raw_df: original df filtered to rows present in agg_df
    """
    metric_cols = [m['col'] for m in METRICS]
    groups = sorted(df[group_col].dropna().unique())

    rows = []
    for g in groups:
        sub = df[df[group_col] == g]
        row = {group_col: g}
        for col in metric_cols:
            vals = sub[col].dropna().values
            n = len(vals)
            if n == 0:
                mean, sem = float('nan'), float('nan')
            elif n == 1:
                mean, sem = float(vals[0]), float('nan')
            else:
                mean = float(np.mean(vals))
                sem = float(np.std(vals, ddof=1) / np.sqrt(n))
            row[f'{col}_mean'] = mean
            row[f'{col}_sem'] = sem
            row[f'{col}_lo'] = mean - sem if not np.isnan(sem) else mean
            row[f'{col}_hi'] = mean + sem if not np.isnan(sem) else mean
        rows.append(row)

    return pd.DataFrame(rows), df


# ── Plotting ──────────────────────────────────────────────────────────────────

def _plot_metrics_panel(
    agg_df: pd.DataFrame,
    raw_df: pd.DataFrame,
    group_col: str,
    x_positions: list,
    x_labels: list,
    title: str,
    xlabel: str,
) -> plt.Figure:
    """One subplot per metric, shared x-axis. Raw values, each metric's own y-axis."""
    sns.set_style('whitegrid')
    plt.rcParams.update({'font.size': 10, 'axes.titlesize': 9, 'axes.labelsize': 9})

    n = len(METRICS)
    fig, axes = plt.subplots(1, n, figsize=(3.8 * n, 4.6), sharey=False)
    rng = np.random.default_rng(42)

    x_to_pos = {g: i for i, g in zip(x_positions, agg_df[group_col].values)}

    for ax, metric in zip(axes, METRICS):
        col = metric['col']
        color = metric['color']

        xs, ys, y_lo, y_hi = [], [], [], []
        for _, row in agg_df.iterrows():
            g = row[group_col]
            x = x_to_pos.get(g, None)
            if x is None:
                continue
            xs.append(x)
            ys.append(row[f'{col}_mean'])
            y_lo.append(row[f'{col}_lo'])
            y_hi.append(row[f'{col}_hi'])

        valid = [i for i, y in enumerate(ys) if not np.isnan(y)]
        if valid:
            vx  = [xs[i]   for i in valid]
            vy  = [ys[i]   for i in valid]
            vlo = [y_lo[i] for i in valid]
            vhi = [y_hi[i] for i in valid]

            ax.plot(vx, vy, color=color, linewidth=2.0, marker='o', markersize=5,
                    zorder=4)

            if any(lo != hi for lo, hi in zip(vlo, vhi)):
                ax.fill_between(vx, vlo, vhi, color=color, alpha=0.25, zorder=3)

        # Individual seed scatter
        for _, srow in raw_df.iterrows():
            g = srow[group_col]
            x = x_to_pos.get(g, None)
            if x is None:
                continue
            val = srow[col]
            if np.isnan(val):
                continue
            x_jitter = x + rng.uniform(-0.18, 0.18)
            ax.scatter(x_jitter, val, color=color, alpha=0.25, s=12, zorder=2,
                       linewidths=0)

        ax.set_xticks(x_positions)
        ax.set_xticklabels(x_labels, rotation=30, ha='right', fontsize=8)
        ax.set_xlabel(xlabel, fontsize=9)
        ax.set_title(metric['label'], fontsize=9, pad=4)
        ax.tick_params(axis='y', labelsize=8)

    axes[0].set_ylabel('Value', fontsize=9)
    fig.suptitle(title, fontsize=10, y=1.02)
    fig.tight_layout()
    return fig


def plot_vs_alpha(df: pd.DataFrame, task_id: str) -> plt.Figure:
    """Plot metrics vs regularisation strength α (final checkpoints, all trials)."""
    # Use final checkpoint per (trial, seed)
    idx = df.groupby(['trial', 'seed'])['update'].transform('max')
    final_df = df[df['update'] == idx].copy()

    # Sort by lpc_alpha, keep only the expected alpha values
    final_df = final_df[final_df['lpc_alpha'].isin(ALPHA_ORDER)].copy()

    agg_df, raw_df = aggregate_by_group(final_df, 'lpc_alpha')
    # Reorder rows to match ALPHA_ORDER
    agg_df['_order'] = agg_df['lpc_alpha'].map({a: i for i, a in enumerate(ALPHA_ORDER)})
    agg_df = agg_df.sort_values('_order').drop(columns='_order').reset_index(drop=True)

    present_alphas = list(agg_df['lpc_alpha'].values)
    x_positions = list(range(len(present_alphas)))
    x_labels = [_alpha_label(a) for a in present_alphas]

    return _plot_metrics_panel(
        agg_df, raw_df,
        group_col='lpc_alpha',
        x_positions=x_positions,
        x_labels=x_labels,
        title=f'{task_id}\nMetrics vs α (mean ± SEM across seeds)',
        xlabel='α (regularisation)',
    )


def plot_vs_updates(df: pd.DataFrame, task_id: str, trial: int = 20) -> plt.Figure:
    """Plot metrics vs training update for a single trial."""
    sub = df[df['trial'] == trial].copy()
    if sub.empty:
        raise ValueError(f"No data found for trial {trial}")

    agg_df, raw_df = aggregate_by_group(sub, 'update')
    agg_df = agg_df.sort_values('update').reset_index(drop=True)

    present_updates = list(agg_df['update'].astype(int).values)
    x_positions = list(range(len(present_updates)))
    x_labels = [str(u) for u in present_updates]

    return _plot_metrics_panel(
        agg_df, raw_df,
        group_col='update',
        x_positions=x_positions,
        x_labels=x_labels,
        title=f'{task_id}  (trial {trial})\nMetrics vs training update (mean ± SEM across seeds)',
        xlabel='Training update',
    )


def _normalise_lines(line_data: list) -> list:
    """Normalise each line independently to [0, 1] based on its own mean range.

    Each entry in line_data is a dict with keys: ys, y_lo, y_hi, scatter_vals.
    The same linear transform is applied to lo/hi bounds and scatter so SEM
    bands and seed scatter remain correctly positioned relative to the mean.
    Returns a new list of dicts with the same keys, values linearly rescaled.
    """
    result = []
    for ld in line_data:
        means = [y for y in ld['ys'] if not np.isnan(y)]
        if not means:
            result.append(ld)
            continue
        vmin = min(means)
        vmax = max(means)
        span = vmax - vmin if (vmax - vmin) > 1e-12 else 1.0

        def _scale(v, vmin=vmin, span=span):
            return (v - vmin) / span if not np.isnan(v) else float('nan')

        result.append({
            'ys':           [_scale(v) for v in ld['ys']],
            'y_lo':         [_scale(v) for v in ld['y_lo']],
            'y_hi':         [_scale(v) for v in ld['y_hi']],
            'scatter_vals': [_scale(v) for v in ld['scatter_vals']],
        })
    return result


def _ylim_from_bands(norm_lines: list, pad: float = 0.08) -> tuple:
    """Compute y-axis limits from the SEM band extremes of normalised lines.

    Ignores scatter outliers — limits are based solely on the shaded band range,
    with a small relative padding applied.
    """
    lo_vals = [v for ld in norm_lines for v in ld['y_lo'] if not np.isnan(v)]
    hi_vals = [v for ld in norm_lines for v in ld['y_hi'] if not np.isnan(v)]
    if not lo_vals or not hi_vals:
        return (0.0, 1.0)
    y_min = min(lo_vals)
    y_max = max(hi_vals)
    span = max(y_max - y_min, 1e-6)
    return (y_min - pad * span, y_max + pad * span)


def plot_vs_alpha_by_phase(df: pd.DataFrame, task_id: str) -> plt.Figure:
    """One subplot per phase, 3 metric lines each (I(S;A), H̄(A|S), mean LPC) vs α.

    Mirrors seed_agg_phase_subplots layout: 1 row × 4 phases, shared y=False,
    SEM bands + individual seed scatter, single legend below.
    """
    # Final checkpoints only
    idx = df.groupby(['trial', 'seed'])['update'].transform('max')
    final_df = df[df['update'] == idx].copy()
    final_df = final_df[final_df['lpc_alpha'].isin(ALPHA_ORDER)].copy()

    sns.set_style('whitegrid')
    plt.rcParams.update({'font.size': 10, 'axes.titlesize': 9, 'axes.labelsize': 9})

    n_phases = len(PHASES)
    fig, axes = plt.subplots(1, n_phases, figsize=(3.8 * n_phases, 4.6), sharey=False)
    rng = np.random.default_rng(42)

    present_alphas = [a for a in ALPHA_ORDER if a in final_df['lpc_alpha'].values]
    x_positions = list(range(len(present_alphas)))
    x_labels = [_alpha_label(a) for a in present_alphas]
    alpha_to_x = {a: i for i, a in enumerate(present_alphas)}

    for ax, phase in zip(axes, PHASES):
        # Collect raw line data for all metrics, then normalise together per subplot
        raw_lines = []
        for line_spec in PHASE_METRIC_LINES:
            col = f"{line_spec['col_prefix']}_{phase}"
            xs, ys, y_lo, y_hi, scatter_xs, scatter_vals = [], [], [], [], [], []
            if col not in final_df.columns:
                raw_lines.append(None)
                continue
            for a in present_alphas:
                vals = final_df[final_df['lpc_alpha'] == a][col].dropna().values
                x = alpha_to_x[a]
                if len(vals) == 0:
                    xs.append(x)
                    ys.append(float('nan'))
                    y_lo.append(float('nan'))
                    y_hi.append(float('nan'))
                    continue
                mean = float(np.mean(vals))
                sem = float(np.std(vals, ddof=1) / np.sqrt(len(vals))) if len(vals) > 1 else 0.0
                xs.append(x)
                ys.append(mean)
                y_lo.append(mean - sem)
                y_hi.append(mean + sem)
                for v in vals:
                    scatter_xs.append(x + rng.uniform(-0.18, 0.18))
                    scatter_vals.append(float(v))
            raw_lines.append({'xs': xs, 'ys': ys, 'y_lo': y_lo, 'y_hi': y_hi,
                               'scatter_xs': scatter_xs, 'scatter_vals': scatter_vals})

        norm_lines = _normalise_lines([ld for ld in raw_lines if ld is not None])
        norm_iter = iter(norm_lines)

        for line_spec, raw in zip(PHASE_METRIC_LINES, raw_lines):
            if raw is None:
                continue
            nd = next(norm_iter)
            xs = raw['xs']

            valid = [i for i, y in enumerate(nd['ys']) if not np.isnan(y)]
            if valid:
                vx  = [xs[i]      for i in valid]
                vy  = [nd['ys'][i]  for i in valid]
                vlo = [nd['y_lo'][i] for i in valid]
                vhi = [nd['y_hi'][i] for i in valid]

                ax.plot(vx, vy,
                        color=line_spec['color'], linestyle=line_spec['ls'],
                        linewidth=line_spec['lw'], alpha=line_spec['alpha'],
                        marker='o', markersize=5, zorder=4,
                        label=line_spec['label'])

                if any(lo != hi for lo, hi in zip(vlo, vhi)):
                    ax.fill_between(vx, vlo, vhi,
                                    color=line_spec['color'], alpha=0.25, zorder=3)

            for sx, sv in zip(raw['scatter_xs'], nd['scatter_vals']):
                if not np.isnan(sv):
                    ax.scatter(sx, sv, color=line_spec['color'], alpha=0.25,
                               s=12, zorder=2, linewidths=0)

        y_min, y_max = _ylim_from_bands(norm_lines)
        ax.set_ylim(y_min, y_max)
        ax.axhline(0, color='black', linewidth=0.8, linestyle=':', alpha=0.6, zorder=1)
        ax.set_xticks(x_positions)
        ax.set_xticklabels(x_labels, rotation=30, ha='right', fontsize=8)
        ax.set_xlabel('α (regularisation)', fontsize=9)
        ax.set_title(PHASE_LABELS[phase], fontsize=9, pad=4)
        ax.tick_params(axis='y', labelsize=8)

    axes[0].set_ylabel('Normalised value', fontsize=9)

    handles, labels = [], []
    for line_spec in PHASE_METRIC_LINES:
        handles.append(plt.Line2D(
            [0], [0],
            color=line_spec['color'], linestyle=line_spec['ls'],
            linewidth=line_spec['lw'], alpha=line_spec['alpha'],
        ))
        labels.append(line_spec['label'])

    fig.legend(handles, labels, loc='lower center', ncol=len(PHASE_METRIC_LINES),
               fontsize=8, frameon=True, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle(f'{task_id}\nPer-phase metrics vs α (mean ± SEM across seeds)',
                 fontsize=10, y=1.02)
    fig.tight_layout(rect=[0, 0.08, 1, 1])
    return fig


def plot_vs_updates_by_phase(df: pd.DataFrame, task_id: str, trial: int = 20) -> plt.Figure:
    """One subplot per phase, 3 metric lines each (I(S;A), H̄(A|S), mean LPC) vs training update.

    Same layout as plot_vs_alpha_by_phase but x-axis = update checkpoints for a single trial.
    """
    sub = df[df['trial'] == trial].copy()
    if sub.empty:
        raise ValueError(f"No data found for trial {trial}")

    present_updates = sorted(sub['update'].dropna().unique().astype(int))

    sns.set_style('whitegrid')
    plt.rcParams.update({'font.size': 10, 'axes.titlesize': 9, 'axes.labelsize': 9})

    n_phases = len(PHASES)
    fig, axes = plt.subplots(1, n_phases, figsize=(3.8 * n_phases, 4.6), sharey=False)
    rng = np.random.default_rng(42)

    x_positions = list(range(len(present_updates)))
    x_labels = [str(u) for u in present_updates]
    update_to_x = {u: i for i, u in enumerate(present_updates)}

    for ax, phase in zip(axes, PHASES):
        raw_lines = []
        for line_spec in PHASE_METRIC_LINES:
            col = f"{line_spec['col_prefix']}_{phase}"
            xs, ys, y_lo, y_hi, scatter_xs, scatter_vals = [], [], [], [], [], []
            if col not in sub.columns:
                raw_lines.append(None)
                continue
            for u in present_updates:
                vals = sub[sub['update'] == u][col].dropna().values
                x = update_to_x[u]
                if len(vals) == 0:
                    xs.append(x)
                    ys.append(float('nan'))
                    y_lo.append(float('nan'))
                    y_hi.append(float('nan'))
                    continue
                mean = float(np.mean(vals))
                sem = float(np.std(vals, ddof=1) / np.sqrt(len(vals))) if len(vals) > 1 else 0.0
                xs.append(x)
                ys.append(mean)
                y_lo.append(mean - sem)
                y_hi.append(mean + sem)
                for v in vals:
                    scatter_xs.append(x + rng.uniform(-0.18, 0.18))
                    scatter_vals.append(float(v))
            raw_lines.append({'xs': xs, 'ys': ys, 'y_lo': y_lo, 'y_hi': y_hi,
                               'scatter_xs': scatter_xs, 'scatter_vals': scatter_vals})

        norm_lines = _normalise_lines([ld for ld in raw_lines if ld is not None])
        norm_iter = iter(norm_lines)

        for line_spec, raw in zip(PHASE_METRIC_LINES, raw_lines):
            if raw is None:
                continue
            nd = next(norm_iter)
            xs = raw['xs']

            valid = [i for i, y in enumerate(nd['ys']) if not np.isnan(y)]
            if valid:
                vx  = [xs[i]        for i in valid]
                vy  = [nd['ys'][i]  for i in valid]
                vlo = [nd['y_lo'][i] for i in valid]
                vhi = [nd['y_hi'][i] for i in valid]

                ax.plot(vx, vy,
                        color=line_spec['color'], linestyle=line_spec['ls'],
                        linewidth=line_spec['lw'], alpha=line_spec['alpha'],
                        marker='o', markersize=5, zorder=4,
                        label=line_spec['label'])

                if any(lo != hi for lo, hi in zip(vlo, vhi)):
                    ax.fill_between(vx, vlo, vhi,
                                    color=line_spec['color'], alpha=0.25, zorder=3)

            for sx, sv in zip(raw['scatter_xs'], nd['scatter_vals']):
                if not np.isnan(sv):
                    ax.scatter(sx, sv, color=line_spec['color'], alpha=0.25,
                               s=12, zorder=2, linewidths=0)

        y_min, y_max = _ylim_from_bands(norm_lines)
        ax.set_ylim(y_min, y_max)
        ax.axhline(0, color='black', linewidth=0.8, linestyle=':', alpha=0.6, zorder=1)
        ax.set_xticks(x_positions)
        ax.set_xticklabels(x_labels, rotation=30, ha='right', fontsize=8)
        ax.set_xlabel('Training update', fontsize=9)
        ax.set_title(PHASE_LABELS[phase], fontsize=9, pad=4)
        ax.tick_params(axis='y', labelsize=8)

    axes[0].set_ylabel('Normalised value', fontsize=9)

    handles, labels = [], []
    for line_spec in PHASE_METRIC_LINES:
        handles.append(plt.Line2D(
            [0], [0],
            color=line_spec['color'], linestyle=line_spec['ls'],
            linewidth=line_spec['lw'], alpha=line_spec['alpha'],
        ))
        labels.append(line_spec['label'])

    fig.legend(handles, labels, loc='lower center', ncol=len(PHASE_METRIC_LINES),
               fontsize=8, frameon=True, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle(f'{task_id}  (trial {trial})\nPer-phase metrics vs training update (mean ± SEM across seeds)',
                 fontsize=10, y=1.02)
    fig.tight_layout(rect=[0, 0.08, 1, 1])
    return fig


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Seed-aggregated metric plots (LPC, policy complexity, success rate, path ratio).'
    )
    parser.add_argument(
        '--csv',
        type=str,
        default='eval_metrics_unlockpickup.csv',
        help='Path to eval_metrics_unlockpickup.csv produced by run_extract_seed_metrics.',
    )
    parser.add_argument(
        '--plot',
        choices=['alpha', 'updates', 'both', 'phase', 'phase_updates'],
        default='both',
        help='Which plot(s) to generate.',
    )
    parser.add_argument(
        '--task_id',
        type=str,
        default='BabyAI-UnlockPickup-v0',
    )
    parser.add_argument(
        '--trial',
        type=int,
        default=20,
        help='Trial number for the across-updates plot (default: 20).',
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default=None,
        help='Output directory. Default: corr_plots/<task_id>/',
    )
    parser.add_argument('--save', action='store_true', help='Save figures without prompting.')
    args = parser.parse_args()

    # Load CSV
    csv_path = pathlib.Path(args.csv)
    if not csv_path.exists():
        print(f"[error] CSV not found: {csv_path}")
        print("Run 'modal run modal_app.py::run_extract_seed_metrics' first.")
        return

    df = pd.read_csv(csv_path)
    df = df[df['task_id'] == args.task_id].copy()
    if df.empty:
        print(f"[error] No rows for task_id={args.task_id} in {csv_path}")
        return

    print(f"Loaded {len(df)} rows for {args.task_id}")
    print(f"  Trials:  {sorted(df['trial'].unique())}")
    print(f"  Updates: {sorted(df['update'].dropna().unique().astype(int))}")
    print(f"  Alphas:  {sorted(df['lpc_alpha'].unique())}")

    # Output dir
    safe_task = args.task_id.replace('/', '_')
    out_dir = pathlib.Path(args.output_dir or f'corr_plots/{safe_task}')
    out_dir.mkdir(parents=True, exist_ok=True)

    figures = []

    if args.plot in ('alpha', 'both'):
        print("\nGenerating metrics-vs-alpha plot...")
        fig_alpha = plot_vs_alpha(df, args.task_id)
        out_path = out_dir / f'{safe_task}_metrics_vs_alpha.png'
        figures.append((fig_alpha, out_path, 'alpha'))

    if args.plot in ('updates', 'both'):
        print(f"Generating metrics-vs-updates plot (trial {args.trial})...")
        fig_upd = plot_vs_updates(df, args.task_id, trial=args.trial)
        out_path = out_dir / f'{safe_task}_trial{args.trial}_metrics_vs_updates.png'
        figures.append((fig_upd, out_path, 'updates'))

    if args.plot == 'phase':
        print("\nGenerating per-phase metrics-vs-alpha plot...")
        fig_phase = plot_vs_alpha_by_phase(df, args.task_id)
        out_path = out_dir / f'{safe_task}_metrics_by_phase.png'
        figures.append((fig_phase, out_path, 'phase'))

    if args.plot == 'phase_updates':
        print(f"\nGenerating per-phase metrics-vs-updates plot (trial {args.trial})...")
        fig_phase_upd = plot_vs_updates_by_phase(df, args.task_id, trial=args.trial)
        out_path = out_dir / f'{safe_task}_trial{args.trial}_metrics_by_phase_vs_updates.png'
        figures.append((fig_phase_upd, out_path, 'phase_updates'))

    for fig, out_path, name in figures:
        if args.save:
            fig.savefig(out_path, dpi=150, bbox_inches='tight')
            print(f"Saved {name} plot -> {out_path}")
        else:
            plt.figure(fig.number)
            plt.show(block=False)
            answer = input(f"Save {name} plot to {out_path}? [y/N] ").strip().lower()
            if answer == 'y':
                fig.savefig(out_path, dpi=150, bbox_inches='tight')
                print(f"Saved -> {out_path}")
            else:
                print("Not saved.")


if __name__ == '__main__':
    main()
