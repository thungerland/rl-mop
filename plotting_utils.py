"""
Plotting utilities for routing visualization.

Provides reusable functions for aggregating routing data and creating heatmaps.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.gridspec as gridspec
from collections import defaultdict


def build_routing_data_tuples(cache: dict) -> list:
    """Convert a loaded routing_data.json cache dict into the list-of-dicts format
    consumed by all plotting functions.

    Supports v1 (per-timestep env_context), v2/v3 (deduplicated episodes list).
    If action_logits are absent (old cache), emits a one-time warning and sets
    'action_logits' to None per record.
    """
    episodes = cache.get('episodes')
    raw = cache['routing_data']

    has_logits = 'action_logits' in raw[0] if raw else False
    if not has_logits:
        import warnings
        warnings.warn(
            "Cache has no 'action_logits' (old format). "
            "Logit-based plots will not work — re-run evaluation to populate them.",
            stacklevel=2,
        )

    return [
        {
            'position': tuple(r['position']),
            'layer_routing': {k: np.array(v) for k, v in r['layer_routing'].items()},
            'lpc': r['lpc'],
            'env_context': episodes[r['episode']] if episodes is not None else r.get('env_context', {}),
            'carrying': r.get('carrying', 0),
            'door_unlocked': r.get('door_unlocked', 0),
            'action_logits': np.array(r['action_logits'], dtype=np.float32) if 'action_logits' in r else None,
            'action': (
                int(r['action']) if 'action' in r
                else int(np.argmax(r['action_logits'])) if r.get('action_logits') is not None
                else None
            ),
            'entropy': r['entropy'] if 'entropy' in r else None,
            't_step': r.get('t_step'),
            't_unlocked': r.get('t_unlocked'),
            't_pick': r.get('t_pick'),
            't_drop': r.get('t_drop'),
            'dist_to_door': r.get('dist_to_door'),
            'dist_to_key': r.get('dist_to_key'),
            'dist_to_target': r.get('dist_to_target'),
        }
        for r in raw
    ]


def compute_empirical_entropy(
    routing_data: list,
    n_actions: int = 7,
    alpha: float = 0.5,
    min_visits: int = 5,
) -> dict:
    """
    Compute empirical action entropy per grid position from actual actions taken.

    Extracts (position, action) pairs, builds count tables, applies Dirichlet
    smoothing (alpha=0.5, Jeffreys prior), then computes:
      - pi_hat(a|s): smoothed empirical distribution
      - H(A|S=s) in bits (log2)
      - KL(pi_hat(.|s) || P(a)) in bits
      - P(s): visitation frequency
      - include_mask: positions with n_visits >= min_visits

    Args:
        routing_data: List of sample dicts with 'position' and 'action' keys.
        n_actions: Number of discrete actions (7 for BabyAI).
        alpha: Dirichlet smoothing concentration (Jeffreys prior = 0.5).
        min_visits: Minimum visits to include a position in analysis.

    Returns:
        dict with keys: 'pi_hat', 'H_s', 'KL_s', 'P_s', 'include_mask', 'P_a'
    """
    valid = [s for s in routing_data if s.get('action') is not None]
    if not valid:
        return {
            'pi_hat': {}, 'H_s': {}, 'KL_s': {},
            'P_s': {}, 'include_mask': {}, 'P_a': np.ones(n_actions) / n_actions,
        }

    counts = defaultdict(lambda: np.zeros(n_actions, dtype=np.float64))
    for s in valid:
        counts[s['position']][s['action']] += 1

    total_visits = sum(c.sum() for c in counts.values())

    # Marginal action distribution (Dirichlet-smoothed)
    global_counts = np.zeros(n_actions, dtype=np.float64)
    for c in counts.values():
        global_counts += c
    P_a = (global_counts + alpha) / (global_counts.sum() + n_actions * alpha)

    pi_hat, H_s, KL_s, P_s, include_mask = {}, {}, {}, {}, {}
    for pos, c in counts.items():
        n = c.sum()
        smoothed = (c + alpha) / (n + n_actions * alpha)
        pi_hat[pos] = smoothed
        H_s[pos] = float(-np.sum(smoothed * np.log2(smoothed + 1e-12)))
        KL_s[pos] = float(np.sum(smoothed * np.log2((smoothed + 1e-12) / (P_a + 1e-12))))
        P_s[pos] = n / total_visits
        include_mask[pos] = bool(n >= min_visits)

    return {
        'pi_hat': pi_hat,
        'H_s': H_s,
        'KL_s': KL_s,
        'P_s': P_s,
        'include_mask': include_mask,
        'P_a': P_a,
    }


def compute_global_mutual_information(KL_s: dict, P_s: dict) -> float:
    """
    Compute I(S; A) = sum_s P(s) * KL(pi_hat(.|s) || P(a)) in bits.
    """
    shared = set(KL_s) & set(P_s)
    return float(sum(P_s[s] * KL_s[s] for s in shared))


def _make_expert_cmap(light_color, dark_color, name):
    """Create a colormap from a light to dark shade for expert confidence gradients."""
    return mcolors.LinearSegmentedColormap.from_list(name, [light_color, dark_color], N=256)


# Expert colormaps: intensity reflects expert complexity.
# Expert 0 (identity/skip) is mildest, higher experts are progressively more vivid.
# Each colormap provides a gradient from light (low confidence) to dark (high confidence).
EXPERT_CMAPS = [
    _make_expert_cmap('#d4e6f1', '#7fb3d8', 'expert0_light_blue'),   # Expert 0: soft pastel blue
    _make_expert_cmap('#c3a6d6', '#8e44ad', 'expert1_purple'),       # Expert 1: medium purple
    _make_expert_cmap('#f1948a', '#c0392b', 'expert2_red'),          # Expert 2: strong red
    _make_expert_cmap('#f0b27a', '#e67e22', 'expert3_orange'),       # Expert 3: orange
    _make_expert_cmap('#82e0aa', '#27ae60', 'expert4_green'),        # Expert 4: green
    _make_expert_cmap('#aab7b8', '#515a5a', 'expert5_grey'),         # Expert 5: grey
]

# Color for unvisited cells
UNVISITED_COLOR = (1.0, 1.0, 1.0)  # White for unvisited/no-data cells


def compute_grid_bounds(positions: list[tuple]) -> dict:
    """
    Compute grid bounds from a list of positions.

    Args:
        positions: List of (x, y) tuples

    Returns:
        dict with x_min, x_max, y_min, y_max, grid_width, grid_height
    """
    x_coords = [p[0] for p in positions]
    y_coords = [p[1] for p in positions]
    x_min, x_max = min(x_coords), max(x_coords)
    y_min, y_max = min(y_coords), max(y_coords)

    return {
        'x_min': x_min,
        'x_max': x_max,
        'y_min': y_min,
        'y_max': y_max,
        'grid_width': x_max - x_min + 1,
        'grid_height': y_max - y_min + 1,
    }


def pos_to_quadrant(x: int, y: int, room_bounds: tuple) -> str:
    """Map a grid position to a room quadrant label.

    Args:
        x: Grid x coordinate
        y: Grid y coordinate
        room_bounds: (x_min, y_min, x_max, y_max) inclusive interior floor bounds

    Returns:
        One of 'TL', 'TR', 'BL', 'BR'.
        MiniGrid convention: y increases downward, so lower y = Top.
    """
    x_min, y_min, x_max, y_max = room_bounds
    mid_x = (x_min + x_max) / 2
    mid_y = (y_min + y_max) / 2
    top = y < mid_y
    left = x < mid_x
    if top and left:
        return "TL"
    elif top:
        return "TR"
    elif left:
        return "BL"
    else:
        return "BR"


def aggregate_routing_by_position(
    routing_data: list,
    layer_expert_sizes: list = None,
) -> tuple[dict, dict, dict, list]:
    """
    Aggregate routing weights, LPC, and per-layer LPC by position.

    Args:
        routing_data: List of dicts with keys position, layer_routing, lpc, env_context
        layer_expert_sizes: Optional list of per-layer expert size lists, e.g.
            [[0, 16, 32], [0, 16, 32], [0, 16, 32]]. When provided, computes
            per-layer LPC = mean_t(sum_k(w_k * s_k^2)) per position per layer.

    Returns:
        avg_routing_by_pos: dict mapping position -> {layer_name: avg_weights}
        avg_lpc_by_pos: dict mapping position -> avg_lpc
        avg_layer_lpc_by_pos: dict mapping position -> {layer_name: avg_layer_lpc}
            (empty dict if layer_expert_sizes is None)
        layer_names: sorted list of layer names
    """
    position_routing = defaultdict(list)
    position_lpc = defaultdict(list)
    position_layer_lpc = defaultdict(lambda: defaultdict(list))

    for sample in routing_data:
        pos = sample['position']
        layer_routing = sample['layer_routing']
        lpc = sample['lpc']

        position_routing[pos].append(layer_routing)
        position_lpc[pos].append(lpc)

        if layer_expert_sizes is not None:
            for layer_idx, expert_sizes in enumerate(layer_expert_sizes):
                layer_key = f'layer_{layer_idx}'
                weights = layer_routing.get(layer_key)
                if weights is not None:
                    sizes_sq = np.array([s ** 2 for s in expert_sizes], dtype=np.float64)
                    position_layer_lpc[pos][layer_key].append(float(np.dot(weights, sizes_sq)))

    if not position_routing:
        return {}, {}, {}, []

    # Compute averages
    avg_routing_by_pos = {}
    for pos, routings in position_routing.items():
        avg_routing_by_pos[pos] = {
            layer: np.mean([r[layer] for r in routings], axis=0)
            for layer in routings[0].keys()
        }

    avg_lpc_by_pos = {pos: np.mean(lpcs) for pos, lpcs in position_lpc.items()}

    avg_layer_lpc_by_pos = {
        pos: {layer: np.mean(vals) for layer, vals in layers.items()}
        for pos, layers in position_layer_lpc.items()
    }

    # Get layer names from first sample
    layer_names = sorted(list(position_routing.values())[0][0].keys())

    return avg_routing_by_pos, avg_lpc_by_pos, avg_layer_lpc_by_pos, layer_names


def render_routing_heatmap(
    ax,
    avg_routing_by_pos: dict,
    grid_info: dict,
    layer_name: str,
    expert_cmaps: list = None
):
    """
    Render a routing heatmap for a single layer on the given axes.

    Shows dominant expert (by color) and confidence (by intensity).
    Unvisited cells are rendered as dark gray.

    Args:
        ax: Matplotlib axes to render on
        avg_routing_by_pos: dict mapping position -> {layer_name: avg_weights}
        grid_info: dict from compute_grid_bounds()
        layer_name: Name of the layer to visualize
        expert_cmaps: List of colormaps for each expert (defaults to EXPERT_CMAPS)
    """
    if expert_cmaps is None:
        expert_cmaps = EXPERT_CMAPS

    grid_width = grid_info['grid_width']
    grid_height = grid_info['grid_height']
    x_min = grid_info['x_min']
    y_min = grid_info['y_min']

    # Create grids for dominant expert and confidence
    dominant_grid = np.full((grid_height, grid_width), -1, dtype=int)
    confidence_grid = np.full((grid_height, grid_width), np.nan)

    for pos, routing in avg_routing_by_pos.items():
        x, y = pos
        gx, gy = x - x_min, y - y_min

        weights = routing[layer_name]
        dominant_expert = np.argmax(weights)
        confidence = weights[dominant_expert]

        dominant_grid[gy, gx] = dominant_expert
        confidence_grid[gy, gx] = confidence

    # Create RGB image - unvisited cells are dark gray
    rgb_image = np.ones((grid_height, grid_width, 3)) * np.array(UNVISITED_COLOR)

    for gy in range(grid_height):
        for gx in range(grid_width):
            expert_idx = dominant_grid[gy, gx]
            if expert_idx >= 0:
                cmap = expert_cmaps[expert_idx % len(expert_cmaps)]
                # Map confidence (0.33 to 1.0 typical range) to color intensity (0.3 to 1.0)
                intensity = 0.3 + 0.7 * confidence_grid[gy, gx]
                color = cmap(intensity)
                rgb_image[gy, gx, :] = color[:3]

    ax.imshow(rgb_image, origin='upper')
    ax.set_title(f"{layer_name}\n(color=expert, intensity=confidence)")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")

    # Add grid lines
    ax.set_xticks(np.arange(-0.5, grid_width, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, grid_height, 1), minor=True)
    ax.grid(which='minor', color='white', linestyle='-', linewidth=0.5)
    ax.tick_params(which='minor', size=0)


def render_lpc_heatmap(ax, avg_lpc_by_pos: dict, grid_info: dict, vmin=None, vmax=None):
    """
    Render an LPC heatmap on the given axes.

    Unvisited cells are rendered as dark gray (NaN in colormap).

    Args:
        ax: Matplotlib axes to render on
        avg_lpc_by_pos: dict mapping position -> avg_lpc
        grid_info: dict from compute_grid_bounds()
        vmin, vmax: Optional color scale bounds (for shared scaling across columns)
    """
    grid_width = grid_info['grid_width']
    grid_height = grid_info['grid_height']
    x_min = grid_info['x_min']
    y_min = grid_info['y_min']

    # Create LPC grid - NaN for unvisited
    lpc_grid = np.full((grid_height, grid_width), np.nan)

    for pos, lpc in avg_lpc_by_pos.items():
        x, y = pos
        gx, gy = x - x_min, y - y_min
        lpc_grid[gy, gx] = lpc

    # Create colormap that shows NaN as dark gray
    cmap = plt.cm.viridis.copy()
    cmap.set_bad(color=UNVISITED_COLOR)

    im = ax.imshow(lpc_grid, origin='upper', cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_title("Mean LPC")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")

    # Add grid lines
    ax.set_xticks(np.arange(-0.5, grid_width, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, grid_height, 1), minor=True)
    ax.grid(which='minor', color='white', linestyle='-', linewidth=0.5)
    ax.tick_params(which='minor', size=0)

    return im


def render_layer_lpc_heatmap(
    ax,
    avg_layer_lpc_by_pos: dict,
    grid_info: dict,
    layer_name: str,
    vmin=None,
    vmax=None,
):
    """
    Render a per-layer LPC heatmap: mean_t(sum_k(w_k * s_k^2)) per cell for one layer.

    Unvisited cells are rendered as dark gray (NaN in colormap).

    Args:
        ax: Matplotlib axes to render on
        avg_layer_lpc_by_pos: dict mapping position -> {layer_name: avg_layer_lpc}
        grid_info: dict from compute_grid_bounds()
        layer_name: which layer to render (e.g. 'layer_0')
        vmin, vmax: Optional color scale bounds (for shared scaling across columns)
    """
    grid_width = grid_info['grid_width']
    grid_height = grid_info['grid_height']
    x_min = grid_info['x_min']
    y_min = grid_info['y_min']

    lpc_grid = np.full((grid_height, grid_width), np.nan)

    for pos, layer_lpcs in avg_layer_lpc_by_pos.items():
        val = layer_lpcs.get(layer_name)
        if val is not None:
            x, y = pos
            gx, gy = x - x_min, y - y_min
            lpc_grid[gy, gx] = val

    cmap = plt.cm.viridis.copy()
    cmap.set_bad(color=UNVISITED_COLOR)

    im = ax.imshow(lpc_grid, origin='upper', cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_title(f"{layer_name} LPC")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")

    ax.set_xticks(np.arange(-0.5, grid_width, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, grid_height, 1), minor=True)
    ax.grid(which='minor', color='white', linestyle='-', linewidth=0.5)
    ax.tick_params(which='minor', size=0)

    return im


def plot_overall_routing(
    routing_data: list,
    env_image: np.ndarray = None,
    env_mission: str = "",
    layer_expert_sizes: list = None,
) -> plt.Figure:
    """
    Create a complete routing visualization figure.

    Shows environment layout (if provided), mean LPC heatmap, and per-layer LPC
    heatmaps. Columns go from general (mean LPC) to specific (per-layer LPC).
    All LPC columns share a single viridis colorbar with global vmin/vmax.

    Args:
        routing_data: List of dicts with keys position, layer_routing, lpc, env_context
        env_image: Optional environment render to show
        env_mission: Optional language instruction to show under the environment image
        layer_expert_sizes: Optional list of per-layer expert size lists, e.g.
            [[0, 16, 32], [0, 16, 32], [0, 16, 32]]. Required for per-layer LPC columns.

    Returns:
        matplotlib Figure object
    """
    # Aggregate data
    avg_routing_by_pos, avg_lpc_by_pos, avg_layer_lpc_by_pos, layer_names = (
        aggregate_routing_by_position(routing_data, layer_expert_sizes=layer_expert_sizes)
    )

    if not avg_routing_by_pos:
        fig, ax = plt.subplots(1, 1, figsize=(6, 4))
        ax.text(0.5, 0.5, "No data after filtering", ha='center', va='center', fontsize=14)
        ax.axis('off')
        return fig

    # Compute grid bounds
    positions = list(avg_routing_by_pos.keys())
    grid_info = compute_grid_bounds(positions)

    has_layer_lpc = bool(avg_layer_lpc_by_pos)

    # Column layout: [env?] | [mean LPC] | [layer_0 LPC] | ... | [colorbar]
    num_lpc_cols = 1 + (len(layer_names) if has_layer_lpc else 0)  # mean + per-layer
    num_main_cols = num_lpc_cols + (1 if env_image is not None else 0)
    num_cols = num_main_cols + 1  # +1 for shared colorbar column

    fig = plt.figure(figsize=(5 * num_main_cols, 7))
    gs = gridspec.GridSpec(
        1, num_cols,
        width_ratios=[1] * num_main_cols + [0.05],
        hspace=0.08, wspace=0.35,
    )

    plot_idx = 0

    # Environment render (if provided)
    if env_image is not None:
        ax = fig.add_subplot(gs[0, plot_idx])
        ax.imshow(env_image)
        ax.set_title("Environment Layout")
        if env_mission:
            ax.set_xlabel(env_mission, fontsize=9)
            ax.set_xticks([])
            ax.set_yticks([])
        else:
            ax.axis('off')
        plot_idx += 1

    # Compute shared color scale across all LPC values
    all_lpc_vals = list(avg_lpc_by_pos.values())
    if has_layer_lpc:
        for layer_lpcs in avg_layer_lpc_by_pos.values():
            all_lpc_vals.extend(layer_lpcs.values())
    global_vmin = float(np.nanmin(all_lpc_vals)) if all_lpc_vals else None
    global_vmax = float(np.nanmax(all_lpc_vals)) if all_lpc_vals else None

    # Mean LPC heatmap (general, leftmost data column)
    ax = fig.add_subplot(gs[0, plot_idx])
    lpc_im = render_lpc_heatmap(ax, avg_lpc_by_pos, grid_info, vmin=global_vmin, vmax=global_vmax)
    plot_idx += 1

    # Per-layer LPC heatmaps (specific, right of mean LPC)
    if has_layer_lpc:
        for layer_name in layer_names:
            ax = fig.add_subplot(gs[0, plot_idx])
            render_layer_lpc_heatmap(
                ax, avg_layer_lpc_by_pos, grid_info, layer_name,
                vmin=global_vmin, vmax=global_vmax,
            )
            plot_idx += 1

    # Shared colorbar
    cbar_ax = fig.add_subplot(gs[0, num_main_cols])
    fig.colorbar(lpc_im, cax=cbar_ax).set_label('LPC')

    return fig


def get_available_analyses(routing_data: list) -> list[str]:
    """
    Determine which analysis types are available based on the routing data.

    Args:
        routing_data: List of (position, layer_routing, lpc, env_context) tuples

    Returns:
        List of available analysis names (e.g., ['overall', 'by_starting_room'])
    """
    available = ['overall']

    if not routing_data:
        return available

    # Check first sample's env_context for available fields
    env_context = routing_data[0]['env_context']

    if not env_context:
        return available

    # Check for room-based grouping
    if env_context.get('agent_start_room') is not None:
        available.append('by_starting_room')

    # Door-location grouping: only offer if distinct door-position-sets <= 9.
    # Sample up to 200 timesteps — env_context repeats within episodes so a
    # small probe is sufficient to discover the distinct door configurations.
    if env_context.get('doors') is not None:
        distinct_door_sets = set()
        for sample in routing_data[:200]:
            ctx = sample['env_context']
            doors = ctx.get('doors')
            if doors:
                door_pos_key = tuple(sorted((d[0], d[1]) for d in doors))
                distinct_door_sets.add(door_pos_key)
        if 1 <= len(distinct_door_sets) <= 9:
            available.append('by_door_location')

    # Combined door+box-row grouping: only offer if both doors and boxes exist
    # and distinct combined keys <= 16.
    if env_context.get('doors') is not None and env_context.get('boxes') is not None:
        distinct_combined = set()
        for sample in routing_data[:200]:
            ctx = sample['env_context']
            doors = ctx.get('doors')
            boxes = ctx.get('boxes')
            if doors and boxes:
                door_key = tuple(sorted((d[0], d[1]) for d in doors))
                box_row_key = tuple(sorted(b[1] for b in boxes))
                distinct_combined.add((door_key, box_row_key))
        if 1 <= len(distinct_combined) <= 16:
            available.append('by_door_and_box_row')

    # Carrying-phase split: only offer if both carrying=0 and carrying=1 timesteps exist.
    if len(routing_data) > 0 and 'carrying' in routing_data[0]:
        carrying_values = set(sample['carrying'] for sample in routing_data)
        if carrying_values == {0, 1}:
            available.append('by_carrying_phase')

    # Agent+target quadrant grouping: requires agent_start_pos, target_pos, room_bounds.
    # Only offered when there is meaningful variation in agent start quadrant (>= 2 distinct)
    # and the total number of combinations is within reason (<= 16, i.e. 4x4).
    if (env_context.get('agent_start_pos') is not None
            and env_context.get('target_pos') is not None
            and env_context.get('room_bounds') is not None):
        distinct_quad_combos = set()
        distinct_agent_quads = set()
        for sample in routing_data[:200]:
            ctx = sample['env_context']
            agent_pos = ctx.get('agent_start_pos')
            target_pos = ctx.get('target_pos')
            room_bounds = ctx.get('room_bounds')
            if agent_pos is None or target_pos is None or room_bounds is None:
                continue
            aq = pos_to_quadrant(agent_pos[0], agent_pos[1], room_bounds)
            tq = pos_to_quadrant(target_pos[0], target_pos[1], room_bounds)
            distinct_quad_combos.add((aq, tq))
            distinct_agent_quads.add(aq)
        if len(distinct_agent_quads) >= 2 and len(distinct_quad_combos) <= 16:
            available.append('by_agent_and_target_quadrant')

    return available


def group_routing_data(routing_data: list, group_by: str) -> dict[tuple, list]:
    """
    Group routing data by a field in env_context.

    Args:
        routing_data: List of (position, layer_routing, lpc, env_context) tuples
        group_by: Field name in env_context to group by (e.g., 'agent_start_room')

    Returns:
        Dict mapping group_key -> list of samples in that group.
        Keys are tuples for hashability.
    """
    groups = defaultdict(list)
    for sample in routing_data:
        env_context = sample['env_context']

        if group_by == 'carrying_phase':
            key = (sample.get('carrying', 0),)
        elif group_by == 'door_unlocked_phase':
            t_step = sample.get('t_step')
            t_unlocked = sample.get('t_unlocked')
            if t_step is None:
                continue
            if t_unlocked is None or t_step < t_unlocked:
                key = (0,)   # pre-unlock / pre-open
            else:
                key = (1,)   # post-unlock / post-open
        elif group_by == 'door_location':
            doors = env_context.get('doors')
            if not doors:
                continue
            # sorted tuple of (x, y) pairs — order-independent, ignores color/state
            key = tuple(sorted((d[0], d[1]) for d in doors))
        elif group_by == 'door_and_box_row':
            doors = env_context.get('doors')
            boxes = env_context.get('boxes')
            if not doors or not boxes:
                continue
            door_key = tuple(sorted((d[0], d[1]) for d in doors))
            box_row_key = tuple(sorted(b[1] for b in boxes))
            key = (door_key, box_row_key)
        elif group_by == 'agent_and_target_quadrant':
            agent_pos = env_context.get('agent_start_pos')
            target_pos = env_context.get('target_pos')
            room_bounds = env_context.get('room_bounds')
            if agent_pos is None or target_pos is None or room_bounds is None:
                continue
            aq = pos_to_quadrant(agent_pos[0], agent_pos[1], room_bounds)
            tq = pos_to_quadrant(target_pos[0], target_pos[1], room_bounds)
            key = (aq, tq)
        elif group_by == 'unlock_phase':
            t_step = sample.get('t_step')
            t_unlocked = sample.get('t_unlocked')
            if t_step is None:
                continue
            if t_unlocked is None or t_step < t_unlocked:
                key = (0,)   # pre-unlock
            else:
                key = (1,)   # post-unlock
        elif group_by == 'key_phase':
            t_step = sample.get('t_step')
            t_pick = sample.get('t_pick')
            t_unlocked = sample.get('t_unlocked')
            t_drop = sample.get('t_drop')
            if t_step is None:
                continue
            if t_pick is None or t_step < t_pick:
                key = (0,)   # pre-key
            elif t_unlocked is None or t_step < t_unlocked:
                key = (1,)   # with-key, pre-unlock
            elif t_drop is None or t_step < t_drop:
                key = (2,)   # with-key, post-unlock
            else:
                key = (3,)   # post-unlock, post-key (key dropped)
        else:
            key = env_context.get(group_by)
            if key is None:
                continue
            if isinstance(key, list):
                key = tuple(key)
            elif not isinstance(key, tuple):
                key = (key,)

        groups[key].append(sample)
    return dict(groups)


def _positional_names(n: int, axis: str) -> list[str]:
    """Generate positional names for a grid axis.

    Args:
        n: Number of positions along this axis
        axis: 'row' for Top/Bottom, 'col' for Left/Right
    """
    if axis == 'row':
        ends = ('Top', 'Bottom')
    else:
        ends = ('Left', 'Right')
    if n == 1:
        return ['Center']
    if n == 2:
        return list(ends)
    if n == 3:
        return [ends[0], 'Center', ends[1]]
    return [ends[0]] + [f'{axis.title()} {i}' for i in range(1, n - 1)] + [ends[1]]


def room_labels_for_groups(sorted_keys: list, room_grid_shape: tuple = None) -> dict:
    """Map room top-left keys to human-readable labels like 'Top-Left', 'Center'.

    Args:
        sorted_keys: List of room_top tuples, sorted
        room_grid_shape: (num_cols, num_rows) if known

    Returns:
        Dict mapping room_key -> label string
    """
    if room_grid_shape is None or not sorted_keys:
        return {k: f"Room at {k}" for k in sorted_keys}

    num_cols, num_rows = room_grid_shape
    xs = sorted(set(k[0] for k in sorted_keys))
    ys = sorted(set(k[1] for k in sorted_keys))
    x_to_col = {x: i for i, x in enumerate(xs)}
    y_to_row = {y: j for j, y in enumerate(ys)}

    row_names = _positional_names(num_rows, 'row')
    col_names = _positional_names(num_cols, 'col')

    labels = {}
    for key in sorted_keys:
        col_idx = x_to_col[key[0]]
        row_idx = y_to_row[key[1]]
        r = row_names[row_idx] if row_idx < len(row_names) else f"Row {row_idx}"
        c = col_names[col_idx] if col_idx < len(col_names) else f"Col {col_idx}"
        if r == 'Center' and c == 'Center':
            labels[key] = 'Center'
        else:
            labels[key] = f"{r}-{c}"
    return labels


def door_location_labels_for_groups(sorted_keys: list) -> dict:
    """Map door-location group keys to human-readable labels.

    Args:
        sorted_keys: List of group keys, each a tuple of (x, y) tuples,
                     e.g. ((5, 3),) for a single door.

    Returns:
        Dict mapping group_key -> label string
    """
    labels = {}
    for key in sorted_keys:
        if len(key) == 1:
            labels[key] = f"Door at {key[0][0]},{key[0][1]}"
        else:
            coords = ",".join(f"({x},{y})" for x, y in key)
            labels[key] = f"Doors at {coords}"
    return labels


def door_and_box_row_labels_for_groups(sorted_keys: list) -> dict:
    """Map door+box-row group keys to human-readable labels.

    Args:
        sorted_keys: List of group keys, each a tuple of
                     (door_pos_tuple, box_row_tuple), e.g.
                     (((5, 3),), (2,)) meaning door at (5,3), box on row 2.

    Returns:
        Dict mapping group_key -> label string
    """
    labels = {}
    for key in sorted_keys:
        door_part, box_row_part = key
        if len(door_part) == 1:
            door_str = f"Door {door_part[0][0]},{door_part[0][1]}"
        else:
            door_str = "Doors " + ",".join(f"({x},{y})" for x, y in door_part)
        if len(box_row_part) == 1:
            box_str = f"Box row {box_row_part[0]}"
        else:
            box_str = "Box rows " + ",".join(str(r) for r in box_row_part)
        labels[key] = f"{door_str} / {box_str}"
    return labels


def agent_and_target_quadrant_labels_for_groups(sorted_keys: list) -> dict:
    """Map (agent_quadrant, target_quadrant) group keys to human-readable labels.

    Args:
        sorted_keys: List of group keys, each a 2-tuple of quadrant strings
                     e.g. ('TL', 'BR')

    Returns:
        Dict mapping group_key -> label string like 'Agent: TL / Target: BR'
    """
    return {key: f"Agent: {key[0]} / Target: {key[1]}" for key in sorted_keys}


ACTION_NAMES = ['left', 'right', 'forward', 'pickup', 'drop', 'toggle', 'done']


def carrying_phase_labels_for_groups(sorted_keys: list) -> dict:
    """Map carrying-phase group keys to human-readable labels.

    Args:
        sorted_keys: List of group keys, each a 1-tuple (carrying,)
                     where carrying is 0 (not carrying) or 1 (carrying object).

    Returns:
        Dict mapping group_key -> label string
    """
    labels = {(0,): "Not carrying", (1,): "Carrying object"}
    return {k: labels.get(k, f"Phase {k[0]}") for k in sorted_keys}


def door_unlocked_phase_labels_for_groups(sorted_keys: list) -> dict:
    """Map door-unlocked-phase group keys to human-readable labels.

    Args:
        sorted_keys: List of group keys, each a 1-tuple (door_unlocked,)
                     where door_unlocked is 0 (door still locked) or 1 (door unlocked).

    Returns:
        Dict mapping group_key -> label string
    """
    labels = {(0,): "Door locked", (1,): "Door unlocked"}
    return {k: labels.get(k, f"Phase {k[0]}") for k in sorted_keys}


def unlock_phase_labels_for_groups(sorted_keys: list) -> dict:
    """Map unlock-phase group keys to human-readable labels.

    Args:
        sorted_keys: List of group keys, each a 1-tuple (phase,)
                     where 0 = pre-unlock (t_step < t_unlocked or never unlocked)
                     and 1 = post-unlock (t_step >= t_unlocked).

    Returns:
        Dict mapping group_key -> label string
    """
    labels = {(0,): "Pre-unlock", (1,): "Post-unlock"}
    return {k: labels.get(k, f"Phase {k[0]}") for k in sorted_keys}


def key_phase_labels_for_groups(sorted_keys: list) -> dict:
    """Map key-phase group keys to human-readable labels.

    Args:
        sorted_keys: List of group keys, each a 1-tuple (phase,)
                     where 0 = pre-key, 1 = with-key/pre-unlock, 2 = with-key/post-unlock,
                     3 = post-unlock/post-key (key dropped).

    Returns:
        Dict mapping group_key -> label string
    """
    labels = {
        (0,): "Pre-key",
        (1,): "With-key (pre-unlock)",
        (2,): "With-key (post-unlock)",
        (3,): "Post-unlock (post-key)",
    }
    return {k: labels.get(k, f"Phase {k[0]}") for k in sorted_keys}


def plot_grouped_routing(
    routing_data: list,
    group_by: str,
    env_image: np.ndarray = None,
    env_mission: str = "",
    room_env_images: dict = None,
    max_groups: int = 25,
    layer_expert_sizes: list = None,
) -> plt.Figure:
    """
    Create a multi-row figure with one row of LPC heatmaps per group.

    Each row shows: [env_image?] | [mean LPC] | [layer_0 LPC] | [layer_1 LPC] | ...
    Columns go from general (mean LPC) to specific (per-layer LPC).
    All LPC columns share a single viridis colorbar with global vmin/vmax.

    Args:
        routing_data: List of (position, layer_routing, lpc, env_context) tuples
        group_by: Field in env_context to group by (e.g., 'agent_start_room')
        env_image: Optional fallback environment render
        env_mission: Optional fallback mission string
        room_env_images: Optional dict mapping group_key -> (image, mission) per group
        max_groups: Maximum number of groups to display
        layer_expert_sizes: Optional list of per-layer expert size lists for per-layer LPC

    Returns:
        matplotlib Figure object
    """
    groups = group_routing_data(routing_data, group_by)

    if not groups:
        fig, ax = plt.subplots(1, 1, figsize=(6, 4))
        ax.text(0.5, 0.5, f"No data for grouping by '{group_by}'",
                ha='center', va='center', fontsize=14)
        ax.axis('off')
        return fig

    # Sort groups by key for consistent ordering
    sorted_keys = sorted(groups.keys())[:max_groups]
    num_groups = len(sorted_keys)

    # Get room_grid_shape from first sample (for room labels)
    first_ctx = routing_data[0]['env_context']
    room_grid_shape = first_ctx.get('room_grid_shape')

    # Aggregate first group to get layer names
    first_avg, _, _, layer_names = aggregate_routing_by_position(
        groups[sorted_keys[0]], layer_expert_sizes=layer_expert_sizes
    )
    if not first_avg:
        fig, ax = plt.subplots(1, 1, figsize=(6, 4))
        ax.text(0.5, 0.5, "No visited positions in first group",
                ha='center', va='center', fontsize=14)
        ax.axis('off')
        return fig

    # Pre-aggregate all groups to compute global vmin/vmax across all LPC values
    group_aggregates = {}
    all_lpc_vals = []
    for group_key in sorted_keys:
        avg_routing, avg_lpc, avg_layer_lpc, _ = aggregate_routing_by_position(
            groups[group_key], layer_expert_sizes=layer_expert_sizes
        )
        if not avg_routing:
            continue
        group_aggregates[group_key] = (avg_routing, avg_lpc, avg_layer_lpc)
        all_lpc_vals.extend(avg_lpc.values())
        for layer_lpcs in avg_layer_lpc.values():
            all_lpc_vals.extend(layer_lpcs.values())

    global_vmin = float(np.nanmin(all_lpc_vals)) if all_lpc_vals else None
    global_vmax = float(np.nanmax(all_lpc_vals)) if all_lpc_vals else None

    has_layer_lpc = any(agg[2] for agg in group_aggregates.values())

    # Column layout: [env?] | [mean LPC] | [layer_0 LPC] | ... | [colorbar]
    num_lpc_cols = 1 + (len(layer_names) if has_layer_lpc else 0)
    has_env_image = env_image is not None
    num_main_cols = num_lpc_cols + (1 if has_env_image else 0)
    num_cols = num_main_cols + 1  # +1 for shared colorbar column

    # Figure dimensions
    col_width = 4.5
    row_height = 4.5
    fig = plt.figure(figsize=(col_width * num_main_cols, row_height * num_groups + 0.5))

    width_ratios = [1] * num_main_cols + [0.05]
    gs = gridspec.GridSpec(
        num_groups, num_cols,
        height_ratios=[1] * num_groups,
        width_ratios=width_ratios,
        hspace=0.35, wspace=0.35,
    )

    # Compute grid bounds from ALL data (so all rows share the same coordinate system)
    all_positions = [sample['position'] for sample in routing_data]
    global_grid_info = compute_grid_bounds(all_positions)

    lpc_im = None  # Track last rendered image for colorbar

    # Compute human-readable labels for all groups
    if group_by == 'agent_start_room':
        group_labels = room_labels_for_groups(sorted_keys, room_grid_shape)
    elif group_by == 'door_location':
        group_labels = door_location_labels_for_groups(sorted_keys)
    elif group_by == 'door_and_box_row':
        group_labels = door_and_box_row_labels_for_groups(sorted_keys)
    elif group_by == 'carrying_phase':
        group_labels = carrying_phase_labels_for_groups(sorted_keys)
    elif group_by == 'door_unlocked_phase':
        group_labels = door_unlocked_phase_labels_for_groups(sorted_keys)
    elif group_by == 'key_phase':
        group_labels = key_phase_labels_for_groups(sorted_keys)
    elif group_by == 'agent_and_target_quadrant':
        group_labels = agent_and_target_quadrant_labels_for_groups(sorted_keys)
    else:
        group_labels = {k: f"{group_by}={k}" for k in sorted_keys}

    for row_idx, group_key in enumerate(sorted_keys):
        if group_key not in group_aggregates:
            continue

        avg_routing, avg_lpc, avg_layer_lpc = group_aggregates[group_key]
        group_data = groups[group_key]
        col_idx = 0
        label = group_labels[group_key]

        # Environment image (per-group if available, otherwise fallback)
        if has_env_image:
            if room_env_images and group_key in room_env_images:
                row_image, row_mission = room_env_images[group_key]
            else:
                row_image, row_mission = env_image, env_mission

            ax = fig.add_subplot(gs[row_idx, col_idx])
            ax.imshow(row_image)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_ylabel(f"{label}\n({len(group_data)} samples)", fontsize=9)
            ax.set_xlabel(row_mission, fontsize=8)
            if row_idx == 0:
                ax.set_title("Environment Layout")
            col_idx += 1

        # Mean LPC heatmap (general)
        ax = fig.add_subplot(gs[row_idx, col_idx])
        if not has_env_image and col_idx == 0:
            ax.set_ylabel(f"{label}\n({len(group_data)} samples)", fontsize=9)
        lpc_im = render_lpc_heatmap(
            ax, avg_lpc, global_grid_info, vmin=global_vmin, vmax=global_vmax
        )
        if row_idx == 0:
            ax.set_title("Mean LPC")
        else:
            ax.set_title("")
        col_idx += 1

        # Per-layer LPC heatmaps (specific)
        if has_layer_lpc:
            for layer_name in layer_names:
                ax = fig.add_subplot(gs[row_idx, col_idx])
                render_layer_lpc_heatmap(
                    ax, avg_layer_lpc, global_grid_info, layer_name,
                    vmin=global_vmin, vmax=global_vmax,
                )
                if row_idx == 0:
                    ax.set_title(f"{layer_name} LPC")
                else:
                    ax.set_title("")
                col_idx += 1

    # Shared colorbar spanning all rows
    if lpc_im is not None:
        cbar_ax = fig.add_subplot(gs[:, num_main_cols])
        fig.colorbar(lpc_im, cax=cbar_ax).set_label('LPC')

    return fig


def plot_action_frequency(routing_data: list, group_by: str = None) -> plt.Figure:
    """
    Bar chart of how often each action is chosen (argmax of logits).

    Args:
        routing_data: List of dicts with keys position, layer_routing, lpc, env_context,
                      carrying, action_logits
        group_by: If 'carrying', shows side-by-side bars for carrying=0 vs carrying=1.
                  If None, shows a single bar per action.

    Returns:
        matplotlib Figure object
    """
    valid = [s for s in routing_data if s.get('action_logits') is not None]
    if not valid:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.text(0.5, 0.5, "No action_logits in data", ha='center', va='center', fontsize=14)
        ax.axis('off')
        return fig

    n_actions = len(ACTION_NAMES)
    x = np.arange(n_actions)

    fig, ax = plt.subplots(figsize=(9, 5))

    if group_by == 'carrying':
        groups = {0: [], 1: []}
        for s in valid:
            groups[s.get('carrying', 0)].append(np.argmax(s['action_logits']))

        width = 0.35
        colors = ['#5b9bd5', '#ed7d31']
        labels = ['Not carrying', 'Carrying object']
        for i, (carrying_val, label, color) in enumerate(zip([0, 1], labels, colors)):
            counts = np.array(groups[carrying_val])
            if len(counts) == 0:
                freqs = np.zeros(n_actions)
            else:
                freqs = np.bincount(counts, minlength=n_actions)
            offset = (i - 0.5) * width
            bars = ax.bar(x + offset, freqs, width, label=label, color=color, alpha=0.85)
            for bar, val in zip(bars, freqs):
                if val > 0:
                    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                            str(int(val)), ha='center', va='bottom', fontsize=7)
        ax.legend()
    else:
        chosen = np.array([np.argmax(s['action_logits']) for s in valid])
        freqs = np.bincount(chosen, minlength=n_actions)
        bars = ax.bar(x, freqs, color='#5b9bd5', alpha=0.85)
        for bar, val in zip(bars, freqs):
            if val > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                        str(int(val)), ha='center', va='bottom', fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(ACTION_NAMES)
    ax.set_xlabel("Action")
    ax.set_ylabel("Timesteps")
    title = "Action Frequency"
    if group_by == 'carrying':
        title += " by Carrying Phase"
    ax.set_title(title)
    fig.tight_layout()
    return fig


def plot_cell_action_distribution(
    routing_data: list,
    phase: str = "post_unlock",
    position_mode: str = "pre_door",
    pos_filter: tuple = None,
) -> plt.Figure:
    """
    Bar chart of empirical action distribution at a specific cell, filtered by phase.

    Useful for diagnosing high-entropy cells (e.g. just before the door post-unlock).

    Args:
        routing_data: List of dicts with keys position, action, t_step, t_unlocked, env_context.
        phase: "post_unlock" (t_step >= t_unlocked), "pre_unlock", or "all".
        position_mode: "pre_door" to target (door_x-1, door_y); ignored if pos_filter is set.
        pos_filter: Explicit (x, y) absolute position to filter to. Overrides position_mode.

    Returns:
        matplotlib Figure object.
    """
    # Phase filter
    if phase == "post_unlock":
        phase_data = [
            s for s in routing_data
            if s.get('t_unlocked') is not None and s.get('t_step', 0) >= s['t_unlocked']
        ]
    elif phase == "pre_unlock":
        phase_data = [
            s for s in routing_data
            if s.get('t_unlocked') is None or s.get('t_step', 0) < s['t_unlocked']
        ]
    else:
        phase_data = list(routing_data)

    # Position filter
    if pos_filter is not None:
        cell_data = [s for s in phase_data if tuple(s['position']) == tuple(pos_filter)]
        cell_label = str(pos_filter)
    else:
        # pre_door: position is (door_x - 1, door_y) per sample's own door location
        cell_data = []
        cell_positions = set()
        for s in phase_data:
            doors = (s.get('env_context') or {}).get('doors', [])
            if not doors:
                continue
            door_x, door_y = doors[0][0], doors[0][1]
            target = (door_x - 1, door_y)
            if tuple(s['position']) == target:
                cell_data.append(s)
                cell_positions.add(target)
        cell_label = ", ".join(str(p) for p in sorted(cell_positions)) if cell_positions else "pre-door"

    if not cell_data:
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.text(0.5, 0.5, f"No data at {cell_label} for phase='{phase}'",
                ha='center', va='center', fontsize=13)
        ax.axis('off')
        return fig

    def _get_action(s):
        a = s.get('action')
        if a is None and s.get('action_logits') is not None:
            a = int(np.argmax(s['action_logits']))
        return int(a) if a is not None else None

    n_actions = len(ACTION_NAMES)
    x = np.arange(n_actions)

    # Split by y coordinate when in pre_door mode (one sub-bar per row)
    if pos_filter is None:
        y_vals = sorted({s['position'][1] for s in cell_data})
    else:
        y_vals = None

    if y_vals is not None and len(y_vals) > 1:
        # Grouped bars: one colour per y row
        colors = ['#5b9bd5', '#ed7d31', '#70ad47', '#ffc000', '#7030a0']
        n_groups = len(y_vals)
        width = 0.8 / n_groups
        fig, ax = plt.subplots(figsize=(max(9, 5 + n_groups), 5))
        total_n = 0
        for i, y_val in enumerate(y_vals):
            group = [s for s in cell_data if s['position'][1] == y_val]
            actions = [a for s in group for a in [_get_action(s)] if a is not None]
            total_n += len(actions)
            freqs = np.bincount(actions, minlength=n_actions) if actions else np.zeros(n_actions)
            offset = (i - (n_groups - 1) / 2) * width
            color = colors[i % len(colors)]
            bars = ax.bar(x + offset, freqs, width, label=f"y={y_val}  (N={len(actions)})",
                          color=color, alpha=0.85)
            for bar, val in zip(bars, freqs):
                if val > 0:
                    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                            str(int(val)), ha='center', va='bottom', fontsize=7)
        ax.legend(title="Row (y)")
        title_pos = cell_label
        total_label = f"N={total_n}"
    else:
        # Single-bar fallback (pos_filter or only one y value)
        actions = [a for s in cell_data for a in [_get_action(s)] if a is not None]
        if not actions:
            fig, ax = plt.subplots(figsize=(9, 5))
            ax.text(0.5, 0.5, "No action data available", ha='center', va='center', fontsize=13)
            ax.axis('off')
            return fig
        freqs = np.bincount(actions, minlength=n_actions)
        fig, ax = plt.subplots(figsize=(9, 5))
        bars = ax.bar(x, freqs, color='#5b9bd5', alpha=0.85)
        for bar, val in zip(bars, freqs):
            if val > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                        str(int(val)), ha='center', va='bottom', fontsize=9)
        title_pos = cell_label
        total_label = f"N={len(actions)}"

    ax.set_xticks(x)
    ax.set_xticklabels(ACTION_NAMES)
    ax.set_xlabel("Action")
    ax.set_ylabel("Count")
    ax.set_title(f"Action Distribution at {title_pos} — {phase} phase  ({total_label})")
    fig.tight_layout()
    return fig


def plot_across_episode_entropy_heatmap(
    routing_data: list,
    env_image: np.ndarray = None,
    env_mission: str = "",
    min_visits: int = 5,
) -> plt.Figure:
    """
    Empirical action entropy heatmap: H(A|S=s) in bits per grid cell.

    Uses Dirichlet-smoothed empirical action counts from the actions actually taken.
    Positions with fewer than min_visits visits are masked (NaN / unvisited color).

    Args:
        routing_data: List of dicts with 'position' and 'action' keys.
        env_image: Optional environment render to show alongside.
        env_mission: Optional mission string.
        min_visits: Minimum number of visits to include a position.

    Returns:
        matplotlib Figure object
    """
    result = compute_empirical_entropy(routing_data, min_visits=min_visits)
    H_s = result['H_s']
    include_mask = result['include_mask']

    if not H_s:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.text(0.5, 0.5, "No action data in routing_data", ha='center', va='center', fontsize=14)
        ax.axis('off')
        return fig

    positions = list(H_s.keys())
    grid_info = compute_grid_bounds(positions)
    grid_width = grid_info['grid_width']
    grid_height = grid_info['grid_height']
    x_min = grid_info['x_min']
    y_min = grid_info['y_min']

    entropy_grid = np.full((grid_height, grid_width), np.nan)
    for pos, val in H_s.items():
        if include_mask.get(pos, False):
            gx, gy = pos[0] - x_min, pos[1] - y_min
            entropy_grid[gy, gx] = val

    num_plots = 2 if env_image is not None else 1
    fig, axes = plt.subplots(1, num_plots, figsize=(5 * num_plots, 5))
    if num_plots == 1:
        axes = [axes]

    plot_idx = 0
    if env_image is not None:
        axes[plot_idx].imshow(env_image)
        axes[plot_idx].set_title("Environment Layout")
        if env_mission:
            axes[plot_idx].set_xlabel(env_mission, fontsize=9)
        axes[plot_idx].set_xticks([])
        axes[plot_idx].set_yticks([])
        plot_idx += 1

    cmap = plt.cm.plasma.copy()
    cmap.set_bad(color=UNVISITED_COLOR)
    im = axes[plot_idx].imshow(entropy_grid, origin='upper', cmap=cmap)
    axes[plot_idx].set_title(f"Empirical Action Entropy H(A|S) [bits]\n(min_visits={min_visits})")
    axes[plot_idx].set_xlabel("X")
    axes[plot_idx].set_ylabel("Y")
    axes[plot_idx].set_xticks(np.arange(-0.5, grid_width, 1), minor=True)
    axes[plot_idx].set_yticks(np.arange(-0.5, grid_height, 1), minor=True)
    axes[plot_idx].grid(which='minor', color='white', linestyle='-', linewidth=0.5)
    axes[plot_idx].tick_params(which='minor', size=0)
    fig.colorbar(im, ax=axes[plot_idx], label='Entropy (bits)')

    fig.tight_layout()
    return fig


def _compute_group_across_episode_entropy(
    group_data: list,
    min_visits: int = 5,
) -> dict:
    """Compute H(A|S=s) per position using empirical action counts (Dirichlet smoothed)."""
    result = compute_empirical_entropy(group_data, min_visits=min_visits)
    H_s = result['H_s']
    include_mask = result['include_mask']
    return {pos: val for pos, val in H_s.items() if include_mask.get(pos, False)}


def _plot_grouped_entropy_heatmap(
    routing_data: list,
    compute_entropy_fn,
    title: str,
    group_by: str,
    env_image: np.ndarray = None,
    env_mission: str = "",
    room_env_images: dict = None,
    max_groups: int = 25,
    cmap_name: str = 'plasma',
    colorbar_label: str = 'Entropy (bits)',
    min_visits: int = 5,
) -> plt.Figure:
    """Shared implementation for grouped entropy/KL heatmaps."""
    groups = group_routing_data(routing_data, group_by)

    if not groups:
        fig, ax = plt.subplots(1, 1, figsize=(6, 4))
        ax.text(0.5, 0.5, f"No data for grouping by '{group_by}'",
                ha='center', va='center', fontsize=14)
        ax.axis('off')
        return fig

    sorted_keys = sorted(groups.keys())[:max_groups]
    num_groups = len(sorted_keys)

    # Global grid bounds so all rows share the same coordinate system
    all_positions = [s['position'] for s in routing_data]
    global_grid_info = compute_grid_bounds(all_positions)
    grid_width = global_grid_info['grid_width']
    grid_height = global_grid_info['grid_height']
    x_min = global_grid_info['x_min']
    y_min = global_grid_info['y_min']

    # Group labels
    if group_by == 'agent_start_room':
        first_ctx = routing_data[0]['env_context']
        group_labels = room_labels_for_groups(sorted_keys, first_ctx.get('room_grid_shape'))
    elif group_by == 'door_location':
        group_labels = door_location_labels_for_groups(sorted_keys)
    elif group_by == 'door_and_box_row':
        group_labels = door_and_box_row_labels_for_groups(sorted_keys)
    elif group_by == 'carrying_phase':
        group_labels = carrying_phase_labels_for_groups(sorted_keys)
    elif group_by == 'door_unlocked_phase':
        group_labels = door_unlocked_phase_labels_for_groups(sorted_keys)
    elif group_by == 'key_phase':
        group_labels = key_phase_labels_for_groups(sorted_keys)
    elif group_by == 'agent_and_target_quadrant':
        group_labels = agent_and_target_quadrant_labels_for_groups(sorted_keys)
    else:
        group_labels = {k: f"{group_by}={k}" for k in sorted_keys}

    has_env_image = env_image is not None
    num_main_cols = (1 if has_env_image else 0) + 1  # [env_image?] + [entropy heatmap]
    num_cols = num_main_cols + 1  # +1 narrow colorbar column

    col_width = 4.5
    row_height = 4.5
    fig = plt.figure(figsize=(col_width * num_main_cols, row_height * num_groups))

    width_ratios = [1] * num_main_cols + [0.05]
    gs = gridspec.GridSpec(
        num_groups, num_cols,
        height_ratios=[1] * num_groups,
        width_ratios=width_ratios,
        hspace=0.35, wspace=0.35,
    )

    cmap = plt.cm.get_cmap(cmap_name).copy()
    cmap.set_bad(color=UNVISITED_COLOR)

    entropy_im = None

    for row_idx, group_key in enumerate(sorted_keys):
        group_data = groups[group_key]
        avg_entropy_by_pos = compute_entropy_fn(group_data, min_visits=min_visits)

        col_idx = 0
        label = group_labels[group_key]

        if has_env_image:
            if room_env_images and group_key in room_env_images:
                row_image, row_mission = room_env_images[group_key]
            else:
                row_image, row_mission = env_image, env_mission
            ax = fig.add_subplot(gs[row_idx, col_idx])
            ax.imshow(row_image)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_ylabel(f"{label}\n({len(group_data)} samples)", fontsize=9)
            ax.set_xlabel(row_mission, fontsize=8)
            if row_idx == 0:
                ax.set_title("Environment Layout")
            col_idx += 1

        entropy_grid = np.full((grid_height, grid_width), np.nan)
        for pos, val in avg_entropy_by_pos.items():
            gx, gy = pos[0] - x_min, pos[1] - y_min
            entropy_grid[gy, gx] = val

        ax = fig.add_subplot(gs[row_idx, col_idx])
        entropy_im = ax.imshow(entropy_grid, origin='upper', cmap=cmap)
        if not has_env_image:
            ax.set_ylabel(f"{label}\n({len(group_data)} samples)", fontsize=9)
        ax.set_xlabel("X")
        ax.set_xticks(np.arange(-0.5, grid_width, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, grid_height, 1), minor=True)
        ax.grid(which='minor', color='white', linestyle='-', linewidth=0.5)
        ax.tick_params(which='minor', size=0)
        if row_idx == 0:
            ax.set_title(title)

    if entropy_im is not None:
        cbar_ax = fig.add_subplot(gs[:, num_main_cols])
        fig.colorbar(entropy_im, cax=cbar_ax).set_label(colorbar_label)

    return fig


def plot_grouped_across_episode_entropy_heatmap(
    routing_data: list,
    group_by: str,
    env_image: np.ndarray = None,
    env_mission: str = "",
    room_env_images: dict = None,
    max_groups: int = 25,
    min_visits: int = 5,
) -> plt.Figure:
    """
    Grouped empirical action entropy heatmap: one row per group.

    Each row shows H(A|S=s) in bits computed from empirical action counts,
    independently within each group.

    Args:
        routing_data: List of sample dicts with 'position' and 'action' keys.
        group_by: Field to group by (e.g. 'door_location', 'door_and_box_row').
        env_image: Optional fallback environment render.
        env_mission: Optional fallback mission string.
        room_env_images: Optional dict mapping group_key -> (image, mission).
        max_groups: Maximum number of groups to display.
        min_visits: Minimum visits to include a position.

    Returns:
        matplotlib Figure object
    """
    return _plot_grouped_entropy_heatmap(
        routing_data,
        compute_entropy_fn=_compute_group_across_episode_entropy,
        title="Empirical Action Entropy H(A|S) [bits]",
        group_by=group_by,
        env_image=env_image,
        env_mission=env_mission,
        room_env_images=room_env_images,
        max_groups=max_groups,
        cmap_name='plasma',
        colorbar_label='Entropy (bits)',
        min_visits=min_visits,
    )


def _compute_group_kl(
    group_data: list,
    min_visits: int = 5,
) -> dict:
    """Compute KL(pi_hat(.|s) || P(a)) per position using empirical action counts."""
    result = compute_empirical_entropy(group_data, min_visits=min_visits)
    KL_s = result['KL_s']
    include_mask = result['include_mask']
    return {pos: val for pos, val in KL_s.items() if include_mask.get(pos, False)}


def plot_kl_heatmap(
    routing_data: list,
    env_image: np.ndarray = None,
    env_mission: str = "",
    min_visits: int = 5,
) -> plt.Figure:
    """
    KL divergence heatmap: KL(pi_hat(.|s) || P(a)) in bits per grid cell.

    Shows how informative each state is for distinguishing actions relative to
    the global marginal action distribution. Uses inferno colormap.
    Positions with fewer than min_visits visits are masked.

    Args:
        routing_data: List of dicts with 'position' and 'action' keys.
        env_image: Optional environment render to show alongside.
        env_mission: Optional mission string.
        min_visits: Minimum number of visits to include a position.

    Returns:
        matplotlib Figure object
    """
    result = compute_empirical_entropy(routing_data, min_visits=min_visits)
    KL_s = result['KL_s']
    include_mask = result['include_mask']

    if not KL_s:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.text(0.5, 0.5, "No action data in routing_data", ha='center', va='center', fontsize=14)
        ax.axis('off')
        return fig

    positions = list(KL_s.keys())
    grid_info = compute_grid_bounds(positions)
    grid_width = grid_info['grid_width']
    grid_height = grid_info['grid_height']
    x_min = grid_info['x_min']
    y_min = grid_info['y_min']

    kl_grid = np.full((grid_height, grid_width), np.nan)
    for pos, val in KL_s.items():
        if include_mask.get(pos, False):
            gx, gy = pos[0] - x_min, pos[1] - y_min
            kl_grid[gy, gx] = val

    num_plots = 2 if env_image is not None else 1
    fig, axes = plt.subplots(1, num_plots, figsize=(5 * num_plots, 5))
    if num_plots == 1:
        axes = [axes]

    plot_idx = 0
    if env_image is not None:
        axes[plot_idx].imshow(env_image)
        axes[plot_idx].set_title("Environment Layout")
        if env_mission:
            axes[plot_idx].set_xlabel(env_mission, fontsize=9)
        axes[plot_idx].set_xticks([])
        axes[plot_idx].set_yticks([])
        plot_idx += 1

    cmap = plt.cm.inferno.copy()
    cmap.set_bad(color=UNVISITED_COLOR)
    im = axes[plot_idx].imshow(kl_grid, origin='upper', cmap=cmap)
    axes[plot_idx].set_title(f"KL Divergence KL(π̂(·|s) ∥ P(a)) [bits]\n(min_visits={min_visits})")
    axes[plot_idx].set_xlabel("X")
    axes[plot_idx].set_ylabel("Y")
    axes[plot_idx].set_xticks(np.arange(-0.5, grid_width, 1), minor=True)
    axes[plot_idx].set_yticks(np.arange(-0.5, grid_height, 1), minor=True)
    axes[plot_idx].grid(which='minor', color='white', linestyle='-', linewidth=0.5)
    axes[plot_idx].tick_params(which='minor', size=0)
    fig.colorbar(im, ax=axes[plot_idx], label='KL divergence (bits)')

    fig.tight_layout()
    return fig


def plot_grouped_kl_heatmap(
    routing_data: list,
    group_by: str,
    env_image: np.ndarray = None,
    env_mission: str = "",
    room_env_images: dict = None,
    max_groups: int = 25,
    min_visits: int = 5,
) -> plt.Figure:
    """
    Grouped KL divergence heatmap: one row per group.

    Each row shows KL(pi_hat(.|s) || P(a)) in bits per position,
    computed independently within each group. Uses inferno colormap.

    Args:
        routing_data: List of sample dicts with 'position' and 'action' keys.
        group_by: Field to group by (e.g. 'door_location', 'door_and_box_row').
        env_image: Optional fallback environment render.
        env_mission: Optional fallback mission string.
        room_env_images: Optional dict mapping group_key -> (image, mission).
        max_groups: Maximum number of groups to display.
        min_visits: Minimum visits to include a position.

    Returns:
        matplotlib Figure object
    """
    return _plot_grouped_entropy_heatmap(
        routing_data,
        compute_entropy_fn=_compute_group_kl,
        title="KL Divergence KL(π̂(·|s) ∥ P(a)) [bits]",
        group_by=group_by,
        env_image=env_image,
        env_mission=env_mission,
        room_env_images=room_env_images,
        max_groups=max_groups,
        cmap_name='inferno',
        colorbar_label='KL divergence (bits)',
        min_visits=min_visits,
    )


