"""
stats.py — Statistical analyses on routing cache data.

Always printed:
  - Spatial Pearson r (per-cell empirical H(A|S) vs per-cell mean LPC)
  - Distance correlations (requires new-format cache with t_step/t_unlocked/distances):
      2-phase (pre/post unlock):
        lpc/entropy/kl vs dist_to_door   [pre-unlock]
        lpc/entropy/kl vs dist_to_target [post-unlock]
      4-phase (pre-key / with-key pre-unlock / with-key post-unlock / post-unlock post-key, requires t_pick/t_drop):
        lpc/entropy/kl vs dist_to_key    [pre-key]
        lpc/entropy/kl vs dist_to_door   [with-key/pre-unlock]
        lpc/entropy/kl vs dist_to_target [with-key/post-unlock]
        lpc/entropy/kl vs dist_to_target [post-unlock/post-key]

Optionally printed (pass group_by as third argument):
  - Grouped spatial correlation: spatial Pearson r computed independently per group

Usage:
    python stats.py <task_id> <trial> [group_by]

group_by options:
    door_location             — group by door position
    door_and_box_row          — group by door+box configuration
    carrying_phase            — group by carrying phase (not carrying / carrying)
    unlock_phase              — group by episode timeline (pre-unlock / post-unlock)
    key_phase                 — group by episode timeline (pre-key / with-key pre-unlock / with-key post-unlock / post-unlock post-key)
    agent_and_target_quadrant — group by agent & target start quadrant

Examples:
    python stats.py BabyAI-UnlockPickup-v0 10
    python stats.py BabyAI-UnlockPickup-v0 10 unlock_phase
    python stats.py BabyAI-UnlockPickup-v0 10 key_phase
    python stats.py BabyAI-UnlockPickup-v0 10 door_location
    python stats.py BabyAI-UnlockPickup-v0 10 door_and_box_row
    python stats.py BabyAI-UnlockPickup-v0 10 carrying_phase
"""

import json
import sys
import pathlib
from collections import defaultdict

import numpy as np
from scipy import stats


def _filter_by_phase(routing_data, phase):
    """Yield samples matching the given phase.

    Args:
        phase: one of:
            'pre_unlock'           — t_step < t_unlocked (or t_unlocked is None)
            'post_unlock'          — t_step >= t_unlocked
            'pre_key'              — t_step < t_pick (or t_pick is None)
            'post_key_pre_unlock'  — t_pick <= t_step < t_unlocked
            'with_key_post_unlock' — t_unlocked <= t_step < t_drop (or t_drop is None)
            'post_unlock_post_key' — t_step >= t_drop
            None                   — all samples
    """
    for s in routing_data:
        if phase is None:
            yield s
            continue
        t_step = s.get('t_step')
        t_unlocked = s.get('t_unlocked')
        if t_step is None:
            continue
        if phase == 'pre_unlock':
            if t_unlocked is None or t_step < t_unlocked:
                yield s
        elif phase == 'post_unlock':
            if t_unlocked is not None and t_step >= t_unlocked:
                yield s
        elif phase == 'pre_key':
            t_pick = s.get('t_pick')
            if t_pick is None or t_step < t_pick:
                yield s
        elif phase == 'post_key_pre_unlock':
            t_pick = s.get('t_pick')
            if t_pick is not None and t_step >= t_pick:
                if t_unlocked is None or t_step < t_unlocked:
                    yield s
        elif phase == 'with_key_post_unlock':
            t_drop = s.get('t_drop')
            if t_unlocked is not None and t_step >= t_unlocked:
                if t_drop is None or t_step < t_drop:
                    yield s
        elif phase == 'post_unlock_post_key':
            t_drop = s.get('t_drop')
            if t_drop is not None and t_step >= t_drop:
                yield s


def per_timestep_lpc_dist_correlation(routing_data: list, dist_field: str, phase: str = None) -> dict:
    """
    Pearson correlation between LPC and a distance field across timesteps.

    Args:
        routing_data: List of dicts with 'lpc' and the specified dist_field.
        dist_field: 'dist_to_door' or 'dist_to_target'.
        phase: 'pre_unlock', 'post_unlock', 'pre_key', 'post_key_pre_unlock',
               'with_key_post_unlock', 'post_unlock_post_key', or None (all timesteps).

    Returns:
        dict with keys 'r', 'p', 'n'.
    """
    pairs = [
        (s['lpc'], s[dist_field])
        for s in _filter_by_phase(routing_data, phase)
        if s.get('lpc') is not None and s.get(dist_field) is not None
    ]
    if len(pairs) < 3:
        return {'r': float('nan'), 'p': float('nan'), 'n': len(pairs)}
    lpc, d = zip(*pairs)
    r, p = stats.pearsonr(lpc, d)
    return {'r': float(r), 'p': float(p), 'n': len(pairs)}


def per_timestep_entropy_dist_correlation(
    routing_data: list,
    H_s: dict,
    dist_field: str,
    phase: str = None,
) -> dict:
    """
    Pearson r between per-position empirical H(A|S=s) and a distance field per timestep.

    For each timestep, looks up H_s[position] and pairs with dist_field at that timestep.
    Positions absent from H_s (fewer than min_visits) are skipped automatically.

    Args:
        routing_data: List of sample dicts.
        H_s: dict mapping position -> H(A|S=s) in bits (from compute_empirical_entropy, masked).
        dist_field: 'dist_to_door', 'dist_to_key', or 'dist_to_target'.
        phase: 'pre_unlock', 'post_unlock', 'pre_key', 'post_key_pre_unlock',
               'with_key_post_unlock', 'post_unlock_post_key', or None.

    Returns:
        dict with keys 'r', 'p', 'n'.
    """
    pairs = [
        (H_s[s['position']], s[dist_field])
        for s in _filter_by_phase(routing_data, phase)
        if s['position'] in H_s and s.get(dist_field) is not None
    ]
    if len(pairs) < 3:
        return {'r': float('nan'), 'p': float('nan'), 'n': len(pairs)}
    h, d = zip(*pairs)
    r, p = stats.pearsonr(h, d)
    return {'r': float(r), 'p': float(p), 'n': len(pairs)}


def per_timestep_kl_dist_correlation(
    routing_data: list,
    KL_s: dict,
    dist_field: str,
    phase: str = None,
) -> dict:
    """
    Pearson r between per-position KL(pi_hat(.|s) || P(a)) and a distance field per timestep.

    For each timestep, looks up KL_s[position] and pairs with dist_field at that timestep.
    Positions absent from KL_s (fewer than min_visits) are skipped automatically.

    Args:
        routing_data: List of sample dicts.
        KL_s: dict mapping position -> KL divergence in bits (from compute_empirical_entropy, masked).
        dist_field: 'dist_to_door', 'dist_to_key', or 'dist_to_target'.
        phase: 'pre_unlock', 'post_unlock', 'pre_key', 'post_key_pre_unlock',
               'with_key_post_unlock', 'post_unlock_post_key', or None.

    Returns:
        dict with keys 'r', 'p', 'n'.
    """
    pairs = [
        (KL_s[s['position']], s[dist_field])
        for s in _filter_by_phase(routing_data, phase)
        if s['position'] in KL_s and s.get(dist_field) is not None
    ]
    if len(pairs) < 3:
        return {'r': float('nan'), 'p': float('nan'), 'n': len(pairs)}
    kl, d = zip(*pairs)
    r, p = stats.pearsonr(kl, d)
    return {'r': float(r), 'p': float(p), 'n': len(pairs)}


def per_timestep_entropy_lpc_correlation(
    routing_data: list,
    H_s: dict,
    phase: str = None,
) -> dict:
    """
    Pearson r between per-position H(A|S=s) and LPC per timestep.

    For each timestep, looks up H_s[position] and pairs with lpc at that timestep.
    Positions absent from H_s (fewer than min_visits) are skipped automatically.

    Args:
        routing_data: List of sample dicts.
        H_s: dict mapping position -> H(A|S=s) in bits (from compute_empirical_entropy, masked).
        phase: phase filter or None.

    Returns:
        dict with keys 'r', 'p', 'n'.
    """
    pairs = [
        (H_s[s['position']], s['lpc'])
        for s in _filter_by_phase(routing_data, phase)
        if s['position'] in H_s and s.get('lpc') is not None
    ]
    if len(pairs) < 3:
        return {'r': float('nan'), 'p': float('nan'), 'n': len(pairs)}
    h, lpc = zip(*pairs)
    r, p = stats.pearsonr(h, lpc)
    return {'r': float(r), 'p': float(p), 'n': len(pairs)}


def per_timestep_kl_lpc_correlation(
    routing_data: list,
    KL_s: dict,
    phase: str = None,
) -> dict:
    """
    Pearson r between per-position KL(pi_hat(.|s) || P_a) and LPC per timestep.

    For each timestep, looks up KL_s[position] and pairs with lpc at that timestep.
    Positions absent from KL_s (fewer than min_visits) are skipped automatically.

    Args:
        routing_data: List of sample dicts.
        KL_s: dict mapping position -> KL divergence in bits (from compute_empirical_entropy, masked).
        phase: phase filter or None.

    Returns:
        dict with keys 'r', 'p', 'n'.
    """
    pairs = [
        (KL_s[s['position']], s['lpc'])
        for s in _filter_by_phase(routing_data, phase)
        if s['position'] in KL_s and s.get('lpc') is not None
    ]
    if len(pairs) < 3:
        return {'r': float('nan'), 'p': float('nan'), 'n': len(pairs)}
    kl, lpc = zip(*pairs)
    r, p = stats.pearsonr(kl, lpc)
    return {'r': float(r), 'p': float(p), 'n': len(pairs)}


def spatial_entropy_lpc_correlation(routing_data: list, min_visits: int = 5) -> dict:
    """
    Pearson correlation between per-cell mean LPC and per-cell empirical H(A|S=s).

    Per-cell entropy: H(A|S=s) in bits from Dirichlet-smoothed empirical action counts.
    Per-cell LPC: mean LPC across all visits.
    Only includes positions with at least min_visits visits.

    Args:
        routing_data: List of dicts with keys 'position', 'action', 'lpc'.
        min_visits: Minimum visits to include a position.

    Returns:
        dict with keys 'r', 'p', 'n_cells'.
    """
    from plotting_utils import compute_empirical_entropy
    result = compute_empirical_entropy(routing_data, min_visits=min_visits)
    H_s = result['H_s']
    include_mask = result['include_mask']

    position_lpc = defaultdict(list)
    for s in routing_data:
        if s.get('lpc') is not None:
            position_lpc[s['position']].append(s['lpc'])
    lpc_by_pos = {pos: float(np.mean(vals)) for pos, vals in position_lpc.items()}

    shared_positions = sorted(
        pos for pos in set(H_s) & set(lpc_by_pos)
        if include_mask.get(pos, False)
    )
    if len(shared_positions) < 3:
        return {'r': float('nan'), 'p': float('nan'), 'n_cells': len(shared_positions)}

    entropy_vec = [H_s[pos] for pos in shared_positions]
    lpc_vec = [lpc_by_pos[pos] for pos in shared_positions]

    r, p = stats.pearsonr(entropy_vec, lpc_vec)
    return {'r': float(r), 'p': float(p), 'n_cells': len(shared_positions)}


def spatial_kl_lpc_correlation(routing_data: list, min_visits: int = 5, P_a=None) -> dict:
    """
    Pearson correlation between per-cell KL(pi_hat(.|s) || P_a_ref) and per-cell mean LPC.

    Args:
        routing_data: List of dicts with keys 'position', 'action', 'lpc'.
        min_visits: Minimum visits to include a position.
        P_a: Optional reference marginal action distribution (shape (n_actions,)).
             If None, computed from routing_data (local/phase marginal).
             Pass a pre-computed global P_a for KL_global.

    Returns:
        dict with keys 'r', 'p', 'n_cells'.
    """
    from plotting_utils import compute_empirical_entropy
    result = compute_empirical_entropy(routing_data, min_visits=min_visits, P_a=P_a)
    KL_s = result['KL_s']
    include_mask = result['include_mask']

    position_lpc = defaultdict(list)
    for s in routing_data:
        if s.get('lpc') is not None:
            position_lpc[s['position']].append(s['lpc'])
    lpc_by_pos = {pos: float(np.mean(vals)) for pos, vals in position_lpc.items()}

    shared_positions = sorted(
        pos for pos in set(KL_s) & set(lpc_by_pos)
        if include_mask.get(pos, False)
    )
    if len(shared_positions) < 3:
        return {'r': float('nan'), 'p': float('nan'), 'n_cells': len(shared_positions)}

    kl_vec  = [KL_s[pos]       for pos in shared_positions]
    lpc_vec = [lpc_by_pos[pos] for pos in shared_positions]
    r, p = stats.pearsonr(kl_vec, lpc_vec)
    return {'r': float(r), 'p': float(p), 'n_cells': len(shared_positions)}


def grouped_spatial_entropy_lpc_correlation(routing_data: list, group_by: str) -> dict:
    """
    Spatial entropy-LPC correlation computed independently per group.

    Partitions routing_data by group_by, then runs spatial_entropy_lpc_correlation
    within each group.

    Args:
        routing_data: List of sample dicts.
        group_by: Field to group by (e.g. 'door_location', 'door_and_box_row').

    Returns:
        Dict mapping group_key -> {'r', 'p', 'n_cells'}.
    """
    from plotting_utils import group_routing_data
    groups = group_routing_data(routing_data, group_by)
    return {key: spatial_entropy_lpc_correlation(group_data) for key, group_data in groups.items()}


if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("Usage: python stats.py <task_id> <trial> [group_by]")
        sys.exit(1)

    task_id = sys.argv[1]
    trial = int(sys.argv[2])
    group_by = sys.argv[3] if len(sys.argv) > 3 else None

    cache_path = pathlib.Path('evaluation_cache') / task_id / f'trial_{trial}' / 'routing_data.json'
    if not cache_path.exists():
        cache_path = pathlib.Path('evaluation_cache') / task_id / task_id / f'trial_{trial}' / 'routing_data.json'
    if not cache_path.exists():
        print(f"Cache not found: {cache_path}")
        sys.exit(1)

    with open(cache_path) as f:
        cache = json.load(f)

    from plotting_utils import (
        build_routing_data_tuples,
        group_routing_data,
        door_location_labels_for_groups,
        door_and_box_row_labels_for_groups,
        carrying_phase_labels_for_groups,
        agent_and_target_quadrant_labels_for_groups,
        room_labels_for_groups,
        unlock_phase_labels_for_groups,
        key_phase_labels_for_groups,
    )
    routing_data = build_routing_data_tuples(cache)

    print(f"Task: {task_id}  Trial: {trial}  Timesteps: {len(routing_data)}")
    print()

    from plotting_utils import compute_empirical_entropy
    _global_emp = compute_empirical_entropy(routing_data)
    P_a_global  = _global_emp['P_a']

    _fmt_sp = lambda res: f"r={res['r']:+.4f}  p={res['p']:.4e}  n_cells={res['n_cells']}"

    sp_H  = spatial_entropy_lpc_correlation(routing_data)
    sp_kl = spatial_kl_lpc_correlation(routing_data)
    print("Spatial correlation (per-cell metric vs mean LPC per grid cell)")
    print(f"  entropy  {_fmt_sp(sp_H)}")
    print(f"  kl       {_fmt_sp(sp_kl)}")

    if group_by is not None:
        print()
        from plotting_utils import group_routing_data
        groups = group_routing_data(routing_data, group_by)
        sorted_keys = sorted(groups.keys())

        if group_by == 'door_location':
            labels = door_location_labels_for_groups(sorted_keys)
        elif group_by == 'door_and_box_row':
            labels = door_and_box_row_labels_for_groups(sorted_keys)
        elif group_by == 'carrying_phase':
            labels = carrying_phase_labels_for_groups(sorted_keys)
        elif group_by == 'agent_and_target_quadrant':
            labels = agent_and_target_quadrant_labels_for_groups(sorted_keys)
        elif group_by == 'agent_start_room':
            first_ctx = routing_data[0]['env_context']
            labels = room_labels_for_groups(sorted_keys, first_ctx.get('room_grid_shape'))
        elif group_by == 'unlock_phase':
            labels = unlock_phase_labels_for_groups(sorted_keys)
        elif group_by == 'key_phase':
            labels = key_phase_labels_for_groups(sorted_keys)
        else:
            labels = {k: f"{group_by}={k}" for k in sorted_keys}

        max_label_len = max(len(labels[k]) for k in sorted_keys)
        cell_w = len('r=+0.0000  p=0.00e+00  n_cells=9999')
        cols = ['entropy', 'kl [local]', 'kl [global]']
        print(f"Grouped spatial correlation by {group_by} (per-cell metric vs mean LPC per cell)")
        print('  ' + ' ' * max_label_len + '  ' + '  '.join(c.ljust(cell_w) for c in cols))
        for key in sorted_keys:
            group_data = groups[key]
            res_H       = spatial_entropy_lpc_correlation(group_data)
            res_kl_loc  = spatial_kl_lpc_correlation(group_data)
            res_kl_glob = spatial_kl_lpc_correlation(group_data, P_a=P_a_global)
            label = labels[key].ljust(max_label_len)
            cells = [_fmt_sp(res_H), _fmt_sp(res_kl_loc), _fmt_sp(res_kl_glob)]
            print('  ' + label + '  ' + '  '.join(c.ljust(cell_w) for c in cells))

    has_new_fields = any(s.get('t_step') is not None for s in routing_data[:10])
    if has_new_fields:
        def _phase_emp(phase):
            """KL_local: P_a computed from phase data only."""
            pd = list(_filter_by_phase(routing_data, phase))
            emp = compute_empirical_entropy(pd)
            H  = {pos: v for pos, v in emp['H_s'].items()  if emp['include_mask'][pos]}
            KL = {pos: v for pos, v in emp['KL_s'].items() if emp['include_mask'][pos]}
            return pd, H, KL

        def _phase_emp_global(phase):
            """KL_global: P_a is global (phase-independent) marginal."""
            pd = list(_filter_by_phase(routing_data, phase))
            emp = compute_empirical_entropy(pd, P_a=P_a_global)
            KL = {pos: v for pos, v in emp['KL_s'].items() if emp['include_mask'][pos]}
            return KL

        def _fmt_cell(res):
            return f"r={res['r']:+.4f} p={res['p']:.2e} n={res['n']}"

        def _print_table(rows, header):
            cols = ['lpc', 'entropy', 'kl [local]', 'kl [global]']
            cell_w = len('r=+0.0000 p=0.00e+00 n=99999')
            label_w = max(len(r[0]) for r in rows)
            print(header)
            print('  ' + ' ' * label_w + '  ' + '  '.join(c.ljust(cell_w) for c in cols))
            for row_label, dist_field, phase in rows:
                pd, H, KL_local = _phase_emp(phase)
                KL_global = _phase_emp_global(phase)
                cells = [
                    _fmt_cell(per_timestep_lpc_dist_correlation(pd, dist_field)),
                    _fmt_cell(per_timestep_entropy_dist_correlation(pd, H, dist_field)),
                    _fmt_cell(per_timestep_kl_dist_correlation(pd, KL_local, dist_field)),
                    _fmt_cell(per_timestep_kl_dist_correlation(pd, KL_global, dist_field)),
                ]
                print('  ' + row_label.ljust(label_w) + '  ' + '  '.join(c.ljust(cell_w) for c in cells))

        def _print_lpc_table(rows, header):
            cols = ['entropy', 'kl [local]', 'kl [global]']
            cell_w = len('r=+0.0000 p=0.00e+00 n=99999')
            label_w = max(len(r[0]) for r in rows)
            print(header)
            print('  ' + ' ' * label_w + '  ' + '  '.join(c.ljust(cell_w) for c in cols))
            for row_label, phase in rows:
                pd, H, KL_local = _phase_emp(phase)
                KL_global = _phase_emp_global(phase)
                cells = [
                    _fmt_cell(per_timestep_entropy_lpc_correlation(pd, H)),
                    _fmt_cell(per_timestep_kl_lpc_correlation(pd, KL_local)),
                    _fmt_cell(per_timestep_kl_lpc_correlation(pd, KL_global)),
                ]
                print('  ' + row_label.ljust(label_w) + '  ' + '  '.join(c.ljust(cell_w) for c in cells))

        has_t_pick = any(s.get('dist_to_key') is not None for s in routing_data[:10])

        if group_by is not None:
            lpc_rows_2phase = [
                ('pre-unlock ', 'pre_unlock'),
                ('post-unlock', 'post_unlock'),
            ]
            print()
            _print_lpc_table(lpc_rows_2phase, "Per-timestep spatial correlation vs LPC — 2-phase")

            if has_t_pick:
                lpc_rows_4phase = [
                    ('pre-key             ', 'pre_key'),
                    ('w-key/pre-unlock    ', 'post_key_pre_unlock'),
                    ('w-key/post-unlock   ', 'with_key_post_unlock'),
                    ('post-unlock/post-key', 'post_unlock_post_key'),
                ]
                print()
                _print_lpc_table(lpc_rows_4phase, "Per-timestep spatial correlation vs LPC — 4-phase")

        rows_2phase = [
            ('dist_to_door   pre-unlock ', 'dist_to_door',   'pre_unlock'),
            ('dist_to_target post-unlock', 'dist_to_target', 'post_unlock'),
        ]

        print()
        _print_table(rows_2phase, "Distance correlations — 2-phase (pre/post unlock)")

    has_t_pick = any(s.get('dist_to_key') is not None for s in routing_data[:10])
    if has_new_fields and has_t_pick:
        rows_4phase = [
            ('dist_to_key    pre-key          ', 'dist_to_key',    'pre_key'),
            ('dist_to_door   w-key/pre-unlock ', 'dist_to_door',   'post_key_pre_unlock'),
            ('dist_to_target w-key/post-unlock', 'dist_to_target', 'with_key_post_unlock'),
            ('dist_to_target post-unlock/post-key', 'dist_to_target', 'post_unlock_post_key'),
        ]
        print()
        _print_table(rows_4phase, "Distance correlations — 4-phase (pre-key / w-key pre-unlock / w-key post-unlock / post-unlock w-key)")
