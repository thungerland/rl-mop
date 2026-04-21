"""
seed_agg_plots.py — Seed-aggregated correlation plots across regularisation strengths.

Aggregates per-seed Pearson r values using Fisher z-transform and produces a
publication-quality figure with one subplot per phase, x-axis = alpha (categorical),
y-axis = mean Pearson r across seeds, with SEM bands and individual seed scatter.

Statistical method:
  - Each seed is one independent estimate (episodes are NOT pooled across seeds).
  - Fisher z-transform: z = arctanh(r)
  - Aggregation: mean_z, se_z = std(z)/sqrt(n_seeds)
  - Significance: one-sample t-test on z-values against 0 (scipy.stats.ttest_1samp)
  - Back-transform: mean_r = tanh(mean_z)
  - SEM band: [tanh(mean_z - se_z), tanh(mean_z + se_z)] (asymmetric in r-space)

Usage:
    python seed_agg_plots.py BabyAI-UnlockPickup-v0 --trials 23 --update 5000 --save
    python seed_agg_plots.py BabyAI-UnlockPickup-v0 --alpha_values "0,1e-6,1e-5,1e-4"
    python seed_agg_plots.py BabyAI-OpenTwoDoors-v0 --trials all --save
"""

import argparse
import json
import pathlib

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats as scipy_stats

from corr_plots import (
    _SUBPLOT_ALPHA_ORDER,
    _SUBPLOT_LINES,
    PHASE_LABELS,
    PHASE_LIST,
    TASK_PHASE_SYSTEM,
    compute_corr,
    load_alpha_map,
    sig_marker,
)
from plotting_utils import build_routing_data_tuples, compute_empirical_entropy


# ── Helpers ───────────────────────────────────────────────────────────────────

# Default alpha values for seed aggregation (extends corr_plots._SUBPLOT_ALPHA_ORDER).
_DEFAULT_ALPHA_ORDER = [0.0, 1e-6, 1e-5, 1e-4, 1e-3]


def _alpha_label(a: float) -> str:
    return '0' if a == 0.0 else f'{a:.0e}'


# ── Data loading ──────────────────────────────────────────────────────────────

def discover_seeded_caches(
    task_id: str,
    trial: int,
    cache_dir: str = 'evaluation_cache',
    update: int | None = None,
) -> list[tuple[int, int, pathlib.Path]]:
    """Discover seeded routing_data.json caches for a single trial.

    Looks for: cache_dir/task_id/trial_N/seed_S/update_U/routing_data.json

    Args:
        update: if given, only include caches for exactly this update number.
                If None, picks the largest update number available per seed.

    Returns:
        List of (seed, update_number, path) tuples, sorted by seed.
    """
    base = pathlib.Path(cache_dir) / task_id / f'trial_{trial}'
    if not base.exists():
        return []

    results = []
    for seed_dir in sorted(base.glob('seed_*')):
        if not seed_dir.is_dir():
            continue
        try:
            seed_num = int(seed_dir.name.split('_')[1])
        except (IndexError, ValueError):
            print(f"  [warn] Could not parse seed number from {seed_dir.name}, skipping.")
            continue

        if update is not None:
            candidate = seed_dir / f'update_{update}' / 'routing_data.json'
            if candidate.exists():
                results.append((seed_num, update, candidate))
            else:
                print(f"  [warn] trial={trial} seed={seed_num}: "
                      f"update_{update}/routing_data.json not found, skipping.")
        else:
            # Find all update dirs and pick the largest update number.
            update_dirs = []
            for upd_dir in seed_dir.glob('update_*'):
                if not upd_dir.is_dir():
                    continue
                try:
                    upd_num = int(upd_dir.name.split('_')[1])
                except (IndexError, ValueError):
                    continue
                candidate = upd_dir / 'routing_data.json'
                if candidate.exists():
                    update_dirs.append((upd_num, candidate))
            if not update_dirs:
                print(f"  [warn] trial={trial} seed={seed_num}: "
                      f"no valid update dirs found, skipping.")
                continue
            update_dirs.sort(key=lambda x: x[0], reverse=True)
            best_upd, best_path = update_dirs[0]
            results.append((seed_num, best_upd, best_path))

    results.sort(key=lambda x: x[0])
    return results


def collect_seed_records(
    task_id: str,
    trials: list[int],
    alpha_map: dict[int, float],
    phases: list,
    cache_dir: str = 'evaluation_cache',
    alpha_subset: list[float] | None = None,
    update: int | None = None,
) -> pd.DataFrame:
    """Load routing data and compute per-seed correlations for all three metrics.

    Mirrors _collect_phase_subplot_records in corr_plots.py, but iterates over
    (trial, seed) pairs instead of just trials. Each (trial, seed) is one
    independent estimate.

    Args:
        alpha_subset: if provided, only load trials whose lpc_alpha is in this set.
        update: passed to discover_seeded_caches; None = latest per seed.

    Returns:
        DataFrame with columns: trial, seed, lpc_alpha, phase, phase_label, corr, r, p, n
    """
    alpha_subset_set = set(alpha_subset) if alpha_subset is not None else None
    records = []

    for trial in trials:
        alpha = alpha_map.get(trial)
        if alpha is None:
            print(f"  [skip] trial {trial}: no lpc_alpha in evaluation_results.csv")
            continue
        if alpha_subset_set is not None and alpha not in alpha_subset_set:
            print(f"  [skip] trial {trial}: alpha={_alpha_label(alpha)} not in requested subset "
                  f"{[_alpha_label(a) for a in (alpha_subset or [])]}")
            continue

        seed_cache_list = discover_seeded_caches(task_id, trial, cache_dir, update)
        if not seed_cache_list:
            print(f"  [warn] trial {trial} (alpha={_alpha_label(alpha)}): "
                  f"no seeded caches found — skipping")
            continue

        for seed_num, upd_num, path in seed_cache_list:
            print(
                f"  Loading trial {trial} seed {seed_num} update {upd_num} "
                f"(alpha={_alpha_label(alpha)})...",
                end=' ', flush=True,
            )
            try:
                with open(path) as f:
                    cache = json.load(f)
                routing_data = build_routing_data_tuples(cache)
            except Exception as e:
                print(f"ERROR loading: {e}")
                continue
            print(f"{len(routing_data)} timesteps")

            # Compute global marginal P_a once per (trial, seed) for lpc_kl_global.
            global_emp = compute_empirical_entropy(routing_data)
            global_P_a = global_emp['P_a']

            for phase in phases:
                for line_spec in _SUBPLOT_LINES:
                    corr_type = line_spec['corr']
                    try:
                        res = compute_corr(
                            routing_data, corr_type, phase,
                            dist_field=None, global_P_a=global_P_a,
                        )
                    except Exception as e:
                        print(f"    [warn] trial={trial} seed={seed_num} "
                              f"phase={phase} corr={corr_type}: {e}")
                        res = {'r': float('nan'), 'p': float('nan'), 'n': 0}

                    records.append({
                        'trial':       trial,
                        'seed':        seed_num,
                        'lpc_alpha':   alpha,
                        'phase':       str(phase),
                        'phase_label': PHASE_LABELS.get(phase, str(phase)),
                        'corr':        corr_type,
                        'r':           res['r'],
                        'p':           res['p'],
                        'n':           res.get('n', res.get('n_cells', 0)),
                    })

    return pd.DataFrame(records) if records else pd.DataFrame(
        columns=['trial', 'seed', 'lpc_alpha', 'phase', 'phase_label', 'corr', 'r', 'p', 'n']
    )


# ── Aggregation ───────────────────────────────────────────────────────────────

def aggregate_seed_records(seed_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per-seed correlations using Fisher z-transform.

    For each (lpc_alpha, phase, corr) group:
      - Convert r to z = arctanh(r)  [clip r to ±0.9999 to avoid ±inf]
      - Compute mean_z, se_z = std(z, ddof=1) / sqrt(n_seeds)
      - One-sample t-test of z against 0 (H0: population r = 0)
      - Back-transform: mean_r = tanh(mean_z)
      - SEM band endpoints: [tanh(mean_z - se_z), tanh(mean_z + se_z)]
      - pct_positive: fraction of seeds with r > 0

    Single-seed groups: report r as-is, inference fields = NaN.

    Returns DataFrame with columns:
        lpc_alpha, phase, phase_label, corr,
        mean_r, sem_r, mean_z, se_z,
        p_agg, n_seeds, pct_positive
    """
    rows = []

    for (alpha, phase, corr), grp in seed_df.groupby(
        ['lpc_alpha', 'phase', 'corr'], sort=False
    ):
        r_vals = grp['r'].dropna().values
        n_seeds = len(r_vals)
        phase_label = grp['phase_label'].iloc[0] if len(grp) > 0 else str(phase)

        if n_seeds == 0:
            continue

        # Clip to avoid arctanh blowup at ±1.
        n_clipped = int(np.sum(np.abs(r_vals) >= 0.9999))
        if n_clipped > 0:
            print(f"  [warn] {n_clipped} r-value(s) clipped to ±0.9999 "
                  f"(alpha={_alpha_label(alpha)}, phase={phase}, corr={corr})")
        r_clipped = np.clip(r_vals, -0.9999, 0.9999)
        z_vals = np.arctanh(r_clipped)

        mean_z = float(np.mean(z_vals))
        mean_r = float(np.tanh(mean_z))
        pct_positive = float(np.mean(r_vals > 0)) * 100.0

        if n_seeds >= 2:
            se_z = float(np.std(z_vals, ddof=1) / np.sqrt(n_seeds))
            # Asymmetric SEM half-width in r-space (upper side).
            sem_r = float(np.tanh(mean_z + se_z) - mean_r)
            _, p_agg = scipy_stats.ttest_1samp(r_vals, popmean=0.0)
            p_agg = float(p_agg)
        else:
            se_z = float('nan')
            sem_r = float('nan')
            p_agg = float('nan')

        rows.append({
            'lpc_alpha':    alpha,
            'phase':        phase,
            'phase_label':  phase_label,
            'corr':         corr,
            'mean_r':       mean_r,
            'sem_r':        sem_r,
            'mean_z':       mean_z,
            'se_z':         se_z,
            'p_agg':        p_agg,
            'n_seeds':      n_seeds,
            'pct_positive': pct_positive,
        })

    return pd.DataFrame(rows) if rows else pd.DataFrame(columns=[
        'lpc_alpha', 'phase', 'phase_label', 'corr',
        'mean_r', 'sem_r', 'mean_z', 'se_z', 'p_agg', 'n_seeds', 'pct_positive',
    ])


# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_seed_agg_phase_subplots(
    seed_df: pd.DataFrame,
    agg_df: pd.DataFrame,
    phases: list,
    alpha_subset: list[float],
    sig_level: float = 0.05,
    title: str = '',
) -> plt.Figure:
    """Publication-quality figure: one subplot per phase, x=alpha (categorical), y=Pearson r.

    Three lines per subplot: H(A|S) vs LPC, KL-local vs LPC, KL-global vs LPC.
    Error bands show SEM across seeds (asymmetric Fisher-z bands).
    Individual seed r-values are shown as faint jittered scatter.
    Shared y-axis range across all subplots.

    Args:
        seed_df: per-seed records from collect_seed_records
        agg_df: aggregated records from aggregate_seed_records
        phases: list of phase values (in display order)
        alpha_subset: ordered list of alpha values for x-axis
        sig_level: significance threshold for markers
        title: figure suptitle
    """
    # Validate.
    present_alphas = set(agg_df['lpc_alpha'].unique())
    missing_a = [a for a in alpha_subset if a not in present_alphas]
    if missing_a:
        print(f"  [warn] Alphas not found in aggregated data (will be skipped): {missing_a}")
        alpha_subset = [a for a in alpha_subset if a in present_alphas]
    if not alpha_subset:
        raise ValueError("No valid alphas remain after filtering — cannot plot.")

    present_phases = set(agg_df['phase'].unique())
    missing_p = [str(p) for p in phases if str(p) not in present_phases]
    if missing_p:
        print(f"  [warn] Phases not found in aggregated data (will be skipped): {missing_p}")
        phases = [p for p in phases if str(p) in present_phases]
    if not phases:
        raise ValueError("No valid phases remain after filtering — cannot plot.")

    alpha_to_x = {a: i for i, a in enumerate(alpha_subset)}
    x_positions = list(range(len(alpha_subset)))
    x_labels = [_alpha_label(a) for a in alpha_subset]

    n_phases = len(phases)
    sns.set_style('whitegrid')
    plt.rcParams.update({'font.size': 10, 'axes.titlesize': 9, 'axes.labelsize': 9})

    fig, axes = plt.subplots(
        1, n_phases,
        figsize=(3.8 * n_phases, 4.6),
        sharey=True,
    )
    if n_phases == 1:
        axes = [axes]

    # Compute shared y-axis range from aggregated mean ± sem only (not raw seed scatter).
    r_lo_vals, r_hi_vals = [], []
    for _, row in agg_df[agg_df['lpc_alpha'].isin(alpha_subset)].iterrows():
        if np.isnan(row['mean_r']):
            continue
        sem = row['sem_r'] if not np.isnan(row['sem_r']) else 0.0
        r_lo_vals.append(row['mean_r'] - abs(sem))
        r_hi_vals.append(row['mean_r'] + abs(sem))

    if r_lo_vals and r_hi_vals:
        span = max(r_hi_vals) - min(r_lo_vals)
        pad = max(0.08, span * 0.18)
        y_min = min(r_lo_vals) - pad
        y_max = max(r_hi_vals) + pad
    else:
        y_min, y_max = -1.0, 1.0

    rng = np.random.default_rng(42)

    for ax, phase in zip(axes, phases):
        phase_str = str(phase)
        phase_agg = agg_df[agg_df['phase'] == phase_str]
        phase_seed = seed_df[seed_df['phase'] == phase_str]

        for line_spec in _SUBPLOT_LINES:
            corr_type = line_spec['corr']
            corr_agg = phase_agg[phase_agg['corr'] == corr_type]
            corr_seed = phase_seed[phase_seed['corr'] == corr_type]

            # ── Mean line and SEM band ──
            xs, ys, y_lo, y_hi, ps = [], [], [], [], []
            for a in alpha_subset:
                row_match = corr_agg[corr_agg['lpc_alpha'] == a]
                if row_match.empty:
                    xs.append(alpha_to_x[a])
                    ys.append(float('nan'))
                    y_lo.append(float('nan'))
                    y_hi.append(float('nan'))
                    ps.append(float('nan'))
                    continue
                row = row_match.iloc[0]
                mz = row['mean_z']
                sz = row['se_z']
                mr = float(row['mean_r'])
                xs.append(alpha_to_x[a])
                ys.append(mr)
                ps.append(float(row['p_agg']))
                if not np.isnan(sz) and not np.isnan(mz):
                    y_lo.append(float(np.tanh(mz - sz)))
                    y_hi.append(float(np.tanh(mz + sz)))
                else:
                    y_lo.append(mr)
                    y_hi.append(mr)

            valid_idx = [i for i, y in enumerate(ys) if not np.isnan(y)]
            if valid_idx:
                vx = [xs[i] for i in valid_idx]
                vy = [ys[i] for i in valid_idx]
                vlo = [y_lo[i] for i in valid_idx]
                vhi = [y_hi[i] for i in valid_idx]
                vp = [ps[i] for i in valid_idx]

                ax.plot(
                    vx, vy,
                    color=line_spec['color'],
                    linestyle=line_spec['ls'],
                    linewidth=line_spec['lw'],
                    alpha=line_spec['alpha'],
                    marker='o',
                    markersize=5,
                    label=line_spec['label'],
                    zorder=4,
                )
                has_band = any(lo != hi for lo, hi in zip(vlo, vhi))
                if has_band:
                    ax.fill_between(
                        vx, vlo, vhi,
                        color=line_spec['color'],
                        alpha=0.25,
                        zorder=3,
                    )

                # Significance markers from across-seed test.
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

            # ── Individual seed scatter ──
            for a in alpha_subset:
                seed_vals = corr_seed[corr_seed['lpc_alpha'] == a]['r'].dropna().values
                for r_val in seed_vals:
                    x_jitter = alpha_to_x[a] + rng.uniform(-0.18, 0.18)
                    ax.scatter(
                        x_jitter, r_val,
                        color=line_spec['color'],
                        alpha=0.25,
                        s=12,
                        zorder=2,
                        linewidths=0,
                    )

        ax.axhline(0, color='black', linewidth=0.8, linestyle=':', alpha=0.6, zorder=1)
        ax.set_xticks(x_positions)
        ax.set_xticklabels(x_labels, rotation=30, ha='right', fontsize=8)
        ax.set_xlabel('α (regularisation)', fontsize=9)
        ax.set_ylim(y_min, y_max)
        ax.set_title(PHASE_LABELS.get(phase, str(phase)), fontsize=9, pad=4)
        ax.tick_params(axis='y', labelsize=8)

    axes[0].set_ylabel('Pearson r', fontsize=9)

    # Single legend below the subplots.
    handles, labels = [], []
    for line_spec in _SUBPLOT_LINES:
        handles.append(plt.Line2D(
            [0], [0],
            color=line_spec['color'],
            linestyle=line_spec['ls'],
            linewidth=line_spec['lw'],
            alpha=line_spec['alpha'],
        ))
        labels.append(line_spec['label'])

    fig.legend(
        handles, labels,
        loc='lower center',
        ncol=len(_SUBPLOT_LINES),
        fontsize=8,
        frameon=True,
        bbox_to_anchor=(0.5, -0.02),
    )

    fig.suptitle(title, fontsize=10, y=1.02)
    fig.tight_layout(rect=[0, 0.08, 1, 1])
    return fig


# ── Summary reporting ─────────────────────────────────────────────────────────

def print_seed_summary(
    agg_df: pd.DataFrame,
    task_id: str,
    phases: list,
    alpha_subset: list[float],
) -> None:
    """Print a compact tabular summary of seed-aggregated correlations.

    Format:
        Task: <task_id>
        Phase: <phase_label> | Metric: <corr_label>
          alpha    mean_r   sem_r   p_agg      n_seeds  pct_pos
          0        +0.452   0.031   0.0023 **  5        80.0%
          ...
    """
    corr_labels = {spec['corr']: spec['label'] for spec in _SUBPLOT_LINES}
    corr_order = [spec['corr'] for spec in _SUBPLOT_LINES]

    print(f"\nTask: {task_id}")
    print("=" * 72)

    for phase in phases:
        phase_str = str(phase)
        phase_label = PHASE_LABELS.get(phase, phase_str)
        phase_data = agg_df[agg_df['phase'] == phase_str]

        for corr_type in corr_order:
            corr_data = phase_data[phase_data['corr'] == corr_type]
            if corr_data.empty:
                continue

            label = corr_labels.get(corr_type, corr_type)
            print(f"\nPhase: {phase_label}  |  Metric: {label}")
            print(f"  {'alpha':<8}  {'mean_r':>8}  {'sem_r':>7}  {'p_agg':<12}  "
                  f"{'n_seeds':>7}  {'pct_pos':>7}")
            print("  " + "-" * 62)

            for a in alpha_subset:
                row_match = corr_data[corr_data['lpc_alpha'] == a]
                if row_match.empty:
                    print(f"  {_alpha_label(a):<8}  {'—':>8}  {'—':>7}  {'—':<12}  "
                          f"{'—':>7}  {'—':>7}")
                    continue
                row = row_match.iloc[0]
                mean_r = row['mean_r']
                sem_r = row['sem_r']
                p_agg = row['p_agg']
                n_seeds = int(row['n_seeds'])
                pct_pos = row['pct_positive']

                r_str = f"{mean_r:+.4f}" if not np.isnan(mean_r) else "  NaN "
                sem_str = f"{sem_r:.4f}" if not np.isnan(sem_r) else "  NaN "
                n_note = " (single)" if n_seeds == 1 else ""

                if np.isnan(p_agg):
                    p_str = f"{'NaN':<12}"
                else:
                    mark = sig_marker(p_agg, 0.05)
                    p_str = f"{p_agg:.4f} {mark:<3}"

                pct_str = f"{pct_pos:.0f}%"
                print(f"  {_alpha_label(a):<8}  {r_str:>8}  {sem_str:>7}  "
                      f"{p_str:<12}  {n_seeds:>7}{n_note}  {pct_str:>7}")

    print()


# ── CSV-backed loading ────────────────────────────────────────────────────────

def load_seed_records_from_csv(
    csv_path: str,
    task_id: str,
    phases: list,
    alpha_subset: list | None = None,
) -> pd.DataFrame:
    """Build a seed_df from eval_metrics_unlockpickup.csv without loading any JSON caches.

    Reads the pre-computed r_<phase>_<corr> and p_<phase>_<corr> columns and
    melts them into the same shape as collect_seed_records output:
      trial, seed, lpc_alpha, phase, phase_label, corr, r, p, n

    Args:
        csv_path: path to eval_metrics_unlockpickup.csv
        task_id: filter to this task
        phases: list of phase strings to include
        alpha_subset: if provided, only include rows whose lpc_alpha is in this set
    """
    df = pd.read_csv(csv_path)
    df = df[df['task_id'] == task_id].copy()
    if df.empty:
        return pd.DataFrame(columns=['trial', 'seed', 'lpc_alpha', 'phase',
                                     'phase_label', 'corr', 'r', 'p', 'n'])

    # Use final checkpoint per (trial, seed)
    idx = df.groupby(['trial', 'seed'])['update'].transform('max')
    df = df[df['update'] == idx].copy()

    if alpha_subset is not None:
        df = df[df['lpc_alpha'].isin(alpha_subset)]

    corr_types = [spec['corr'] for spec in _SUBPLOT_LINES]
    records = []
    for _, row in df.iterrows():
        for phase in phases:
            for corr_type in corr_types:
                r_col = f'r_{phase}_{corr_type}'
                p_col = f'p_{phase}_{corr_type}'
                if r_col not in df.columns:
                    continue
                records.append({
                    'trial':       int(row['trial']),
                    'seed':        int(row['seed']),
                    'lpc_alpha':   float(row['lpc_alpha']),
                    'phase':       str(phase),
                    'phase_label': PHASE_LABELS.get(phase, str(phase)),
                    'corr':        corr_type,
                    'r':           float(row[r_col]),
                    'p':           float(row[p_col]) if p_col in df.columns else float('nan'),
                    'n':           0,
                })

    return pd.DataFrame(records) if records else pd.DataFrame(
        columns=['trial', 'seed', 'lpc_alpha', 'phase', 'phase_label', 'corr', 'r', 'p', 'n']
    )


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Seed-aggregated correlation plots across regularisation strengths.'
    )
    parser.add_argument('task_id', type=str)
    parser.add_argument(
        '--trials',
        type=str,
        default='all',
        help='Comma-separated trial numbers, or "all".',
    )
    parser.add_argument(
        '--alpha_values',
        type=str,
        default=None,
        help='Comma-separated alpha values, e.g. "0,1e-6,1e-5,1e-4,1e-3". '
             f'Default: {_DEFAULT_ALPHA_ORDER}',
    )
    parser.add_argument(
        '--update',
        type=int,
        default=None,
        help='Specific training update checkpoint to use (e.g. 5000). '
             'Default: latest update per seed.',
    )
    parser.add_argument(
        '--phase_system',
        choices=['key_phase', 'unlock_phase', 'none'],
        default=None,
        help='Phase system. Inferred from task_id if not specified.',
    )
    parser.add_argument(
        '--from_csv',
        type=str,
        default=None,
        help='Path to eval_metrics_unlockpickup.csv. If provided, skips JSON cache '
             'loading entirely and reads pre-computed r/p values from the CSV.',
    )
    parser.add_argument('--results_path', type=str, default='evaluation_results.csv')
    parser.add_argument('--cache_dir', type=str, default='evaluation_cache')
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output path. Default: corr_plots/{task_id}/{task_id}_seed_agg_phase_subplots.png',
    )
    parser.add_argument('--sig_level', type=float, default=0.05)
    parser.add_argument(
        '--save',
        action='store_true',
        help='Save figure without prompting.',
    )
    args = parser.parse_args()

    # ── Resolve phase system ──
    phase_system = args.phase_system or TASK_PHASE_SYSTEM.get(args.task_id, 'none')
    print(f"Phase system: {phase_system}  (task: {args.task_id})")
    phases = PHASE_LIST[phase_system]

    # ── Load alpha map ──
    print(f"Loading alpha map from {args.results_path}...")
    alpha_map = load_alpha_map(args.task_id, args.results_path)
    if not alpha_map:
        print(f"No entries found for task_id={args.task_id} in {args.results_path}")
        return

    # ── Resolve trials ──
    if args.trials == 'all':
        trials = sorted(alpha_map.keys())
    else:
        trials = [int(t.strip()) for t in args.trials.split(',')]

    # ── Resolve alpha subset ──
    if args.alpha_values is not None:
        alpha_subset = [float(a.strip()) for a in args.alpha_values.split(',')]
    else:
        alpha_subset = list(_DEFAULT_ALPHA_ORDER)

    print(f"Trials: {trials}")
    print(f"Alpha subset: {[_alpha_label(a) for a in alpha_subset]}")
    print(f"Update: {'latest' if args.update is None else args.update}")
    print()

    # ── Collect per-seed records ──
    if args.from_csv:
        print(f"Loading pre-computed r/p values from {args.from_csv}...")
        seed_df = load_seed_records_from_csv(
            csv_path=args.from_csv,
            task_id=args.task_id,
            phases=phases,
            alpha_subset=alpha_subset,
        )
        if seed_df.empty:
            print(f"No data found for task_id={args.task_id} in {args.from_csv}")
            return
    else:
        seed_df = collect_seed_records(
            task_id=args.task_id,
            trials=trials,
            alpha_map=alpha_map,
            phases=phases,
            cache_dir=args.cache_dir,
            alpha_subset=alpha_subset,
            update=args.update,
        )

    if seed_df.empty:
        print("No data collected — check that seeded caches exist under "
              f"{args.cache_dir}/{args.task_id}/trial_N/seed_S/update_U/routing_data.json")
        return

    n_seeds_total = seed_df.groupby(['trial', 'seed']).ngroups
    print(f"\nCollected {len(seed_df)} records from {n_seeds_total} (trial, seed) pairs.")

    # ── Aggregate with Fisher z ──
    print("Aggregating across seeds (Fisher z-transform)...")
    agg_df = aggregate_seed_records(seed_df)

    if agg_df.empty:
        print("Aggregation produced no results — exiting.")
        return

    # ── Print summary ──
    print_seed_summary(agg_df, args.task_id, phases, alpha_subset)

    # ── Filter alpha_subset to what's actually present ──
    present_alphas = set(agg_df['lpc_alpha'].unique())
    missing = [a for a in alpha_subset if a not in present_alphas]
    if missing:
        print(f"[warn] Alphas not found in data: {[_alpha_label(a) for a in missing]}")
        alpha_subset = [a for a in alpha_subset if a in present_alphas]
    if not alpha_subset:
        print("No valid alphas for plotting — exiting.")
        return

    # ── Build figure ──
    title = (
        f"{args.task_id}\n"
        f"Phase-wise Pearson r vs α — aggregated over seeds (Fisher z ± SEM)"
    )
    print("Generating figure...")
    fig = plot_seed_agg_phase_subplots(
        seed_df=seed_df,
        agg_df=agg_df,
        phases=phases,
        alpha_subset=alpha_subset,
        sig_level=args.sig_level,
        title=title,
    )

    # ── Resolve output path ──
    safe_task = args.task_id.replace('/', '_')
    out_dir = pathlib.Path('corr_plots') / safe_task
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.output or str(
        out_dir / f'{safe_task}_seed_agg_phase_subplots.png'
    )

    # ── Save or prompt ──
    if args.save:
        fig.savefig(out_path, dpi=150, bbox_inches='tight')
        print(f"Saved → {out_path}")
    else:
        plt.show(block=False)
        answer = input(f"Save to {out_path}? [y/N] ").strip().lower()
        if answer == 'y':
            fig.savefig(out_path, dpi=150, bbox_inches='tight')
            print(f"Saved → {out_path}")
        else:
            print("Not saved.")


if __name__ == '__main__':
    main()
