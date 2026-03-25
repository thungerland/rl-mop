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

    sp = spatial_entropy_lpc_correlation(routing_data)
    print(f"Spatial correlation (empirical H(A|S) vs mean LPC per grid cell)")
    print(f"  r = {sp['r']:.4f}  p = {sp['p']:.4e}  n_cells = {sp['n_cells']}")

    if group_by is not None:
        print()
        grouped = grouped_spatial_entropy_lpc_correlation(routing_data, group_by)
        sorted_keys = sorted(grouped.keys())

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
        print(f"Grouped spatial correlation by {group_by} (entropy vs mean LPC per cell)")
        for key in sorted_keys:
            res = grouped[key]
            label = labels[key].ljust(max_label_len)
            print(f"  {label}  r = {res['r']:+.4f}  p = {res['p']:.4e}  n_cells = {res['n_cells']}")

    has_new_fields = any(s.get('t_step') is not None for s in routing_data[:10])
    if has_new_fields:
        from plotting_utils import compute_empirical_entropy

        def _phase_emp(phase):
            """Compute H_s/KL_s masked dicts from phase-filtered data."""
            pd = list(_filter_by_phase(routing_data, phase))
            emp = compute_empirical_entropy(pd)
            H  = {pos: v for pos, v in emp['H_s'].items()  if emp['include_mask'][pos]}
            KL = {pos: v for pos, v in emp['KL_s'].items() if emp['include_mask'][pos]}
            return pd, H, KL

        print()
        print("Distance correlations [2-phase: pre/post unlock]")
        for dist_field, phase, label in [
            ('dist_to_door',   'pre_unlock',  'lpc     vs dist_to_door   [pre-unlock ]'),
            ('dist_to_door',   'pre_unlock',  'entropy vs dist_to_door   [pre-unlock ]'),
            ('dist_to_door',   'pre_unlock',  'kl      vs dist_to_door   [pre-unlock ]'),
            ('dist_to_target', 'post_unlock', 'lpc     vs dist_to_target [post-unlock]'),
            ('dist_to_target', 'post_unlock', 'entropy vs dist_to_target [post-unlock]'),
            ('dist_to_target', 'post_unlock', 'kl      vs dist_to_target [post-unlock]'),
        ]:
            pd, H, KL = _phase_emp(phase)
            if label.startswith('lpc'):
                res = per_timestep_lpc_dist_correlation(pd, dist_field)
            elif label.startswith('entropy'):
                res = per_timestep_entropy_dist_correlation(pd, H, dist_field)
            else:
                res = per_timestep_kl_dist_correlation(pd, KL, dist_field)
            print(f"  {label}  r={res['r']:+.4f}  p={res['p']:.4e}  n={res['n']}")

    has_t_pick = any(s.get('dist_to_key') is not None for s in routing_data[:10])
    if has_new_fields and has_t_pick:
        print()
        print("Distance correlations [4-phase: pre-key / with-key pre-unlock / with-key post-unlock / post-unlock post-key]")
        for dist_field, phase, label in [
            ('dist_to_key',    'pre_key',             'lpc     vs dist_to_key    [pre-key                ]'),
            ('dist_to_key',    'pre_key',             'entropy vs dist_to_key    [pre-key                ]'),
            ('dist_to_key',    'pre_key',             'kl      vs dist_to_key    [pre-key                ]'),
            ('dist_to_door',   'post_key_pre_unlock', 'lpc     vs dist_to_door   [with-key/pre-unlock    ]'),
            ('dist_to_door',   'post_key_pre_unlock', 'entropy vs dist_to_door   [with-key/pre-unlock    ]'),
            ('dist_to_door',   'post_key_pre_unlock', 'kl      vs dist_to_door   [with-key/pre-unlock    ]'),
            ('dist_to_target', 'with_key_post_unlock','lpc     vs dist_to_target [with-key/post-unlock   ]'),
            ('dist_to_target', 'with_key_post_unlock','entropy vs dist_to_target [with-key/post-unlock   ]'),
            ('dist_to_target', 'with_key_post_unlock','kl      vs dist_to_target [with-key/post-unlock   ]'),
            ('dist_to_target', 'post_unlock_post_key','lpc     vs dist_to_target [post-unlock/post-key   ]'),
            ('dist_to_target', 'post_unlock_post_key','entropy vs dist_to_target [post-unlock/post-key   ]'),
            ('dist_to_target', 'post_unlock_post_key','kl      vs dist_to_target [post-unlock/post-key   ]'),
        ]:
            pd, H, KL = _phase_emp(phase)
            if label.startswith('lpc'):
                res = per_timestep_lpc_dist_correlation(pd, dist_field)
            elif label.startswith('entropy'):
                res = per_timestep_entropy_dist_correlation(pd, H, dist_field)
            else:
                res = per_timestep_kl_dist_correlation(pd, KL, dist_field)
            print(f"  {label}  r={res['r']:+.4f}  p={res['p']:.4e}  n={res['n']}")
