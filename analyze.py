"""
analyze.py — Focused single-task analysis script.

Usage:
    python analyze.py <task_id> <trial> [plot_type]

plot_type options (default: overall):
    overall                      — routing heatmap, all data combined
    by_starting_room             — routing heatmap grouped by agent starting room
    by_door_location             — routing heatmap grouped by door position
    by_door_and_box_row          — routing heatmap grouped by door+box configuration
    by_carrying_phase            — routing heatmap grouped by carrying phase
    by_door_unlocked_phase       — routing heatmap grouped by door locked/unlocked phase
    by_agent_and_target_quadrant — routing heatmap grouped by agent & target quadrant
    action_frequency             — bar chart of action frequencies
    action_frequency_carrying    — action frequency split by carrying phase
    across_episode_entropy_heatmap               — empirical H(A|S) per grid cell
    by_door_location_across_episode_entropy      — empirical entropy grouped by door location
    by_door_and_box_row_across_episode_entropy   — empirical entropy grouped by door+box row
    by_door_unlocked_phase_across_episode_entropy — empirical entropy grouped by door locked/unlocked phase
    by_key_phase                                 — routing heatmap grouped by key phase (pre-key / with-key pre-unlock / with-key post-unlock / post-unlock post-key)
    by_key_phase_across_episode_entropy          — empirical entropy grouped by key phase
    kl_heatmap                                   — KL_local(pi_hat || P_a^local) per grid cell
    by_door_location_kl                          — KL_local heatmap grouped by door location
    by_door_and_box_row_kl                       — KL_local heatmap grouped by door+box row
    by_door_unlocked_phase_kl                    — KL_local heatmap grouped by door locked/unlocked phase
    by_key_phase_kl                              — KL_local heatmap grouped by key phase
    kl_heatmap_global                            — KL_global(pi_hat || P_a^global) per grid cell
    by_door_location_kl_global                   — KL_global heatmap grouped by door location
    by_door_and_box_row_kl_global                — KL_global heatmap grouped by door+box row
    by_door_unlocked_phase_kl_global             — KL_global heatmap grouped by door locked/unlocked phase
    by_key_phase_kl_global                       — KL_global heatmap grouped by key phase

Examples:
    python analyze.py BabyAI-GoToDoor-v0 0
    python analyze.py BabyAI-GoToDoor-v0 0 by_carrying_phase
    python analyze.py BabyAI-GoToDoor-v0 0 action_frequency

Extending: add a new function to plotting_utils.py, import it here, add its
name to ALL_TYPES, add a branch in section 4, and add its filename in section 5.
"""

import sys
import json
import pathlib
import matplotlib.pyplot as plt

from plotting_utils import (
    build_routing_data_tuples,
    plot_overall_routing,
    plot_grouped_routing,
    plot_action_frequency,
    plot_across_episode_entropy_heatmap,
    plot_grouped_across_episode_entropy_heatmap,
    plot_kl_heatmap,
    plot_grouped_kl_heatmap,
    plot_cell_action_distribution,
    group_routing_data,
    pos_to_quadrant,
    compute_empirical_entropy,
)
from eval_mop import _first_target_pos

# ── 1. Parse args ─────────────────────────────────────────────────────────────
import argparse as _argparse
_parser = _argparse.ArgumentParser(
    description='Visualize routing data for a single checkpoint.',
    usage='python analyze.py <task_id> <trial> [plot_type] [--seed S] [--update U]'
)
_parser.add_argument('task_id')
_parser.add_argument('trial', type=int)
_parser.add_argument('plot_type', nargs='?', default='overall')
_parser.add_argument('--seed', type=int, default=None, help='Seed number (for seeded checkpoints)')
_parser.add_argument('--update', type=int, default=None, help='Training update step (e.g. 1500)')
_args = _parser.parse_args()

task_id = _args.task_id
trial = _args.trial
plot_type = _args.plot_type

GROUPED_ROUTING_TYPES = {
    "by_starting_room",
    "by_door_location",
    "by_door_and_box_row",
    "by_carrying_phase",
    "by_door_unlocked_phase",
    "by_key_phase",
    "by_agent_and_target_quadrant",
}

ALL_TYPES = {"overall"} | GROUPED_ROUTING_TYPES | {
    "action_frequency",
    "action_frequency_carrying",
    "across_episode_entropy_heatmap",
    "by_door_location_across_episode_entropy",
    "by_door_and_box_row_across_episode_entropy",
    "by_door_unlocked_phase_across_episode_entropy",
    "by_key_phase_across_episode_entropy",
    "kl_heatmap",
    "by_door_location_kl",
    "by_door_and_box_row_kl",
    "by_door_unlocked_phase_kl",
    "by_key_phase_kl",
    "kl_heatmap_global",
    "by_door_location_kl_global",
    "by_door_and_box_row_kl_global",
    "by_door_unlocked_phase_kl_global",
    "by_key_phase_kl_global",
    "cell_action_distribution",
}

if plot_type not in ALL_TYPES:
    print(f"Unknown plot_type '{plot_type}'. Options: {', '.join(sorted(ALL_TYPES))}")
    sys.exit(1)

# ── 2. Load cache ─────────────────────────────────────────────────────────────
_base = pathlib.Path('evaluation_cache') / task_id / f'trial_{trial}'
if _args.seed is not None:
    _base = _base / f'seed_{_args.seed}'
if _args.update is not None:
    _base = _base / f'update_{_args.update}'
cache_path = _base / 'routing_data.json'

# Legacy fallback: old flat layout without seed/update dirs
if not cache_path.exists() and _args.seed is None and _args.update is None:
    cache_path = pathlib.Path(f"evaluation_cache/{task_id}/{task_id}/trial_{trial}/routing_data.json")
if not cache_path.exists():
    print(f"Cache not found: {cache_path}")
    sys.exit(1)

with open(cache_path) as f:
    cache = json.load(f)

routing_data = build_routing_data_tuples(cache)
_P_a_global = compute_empirical_entropy(routing_data)['P_a']

# Load expert_hidden_sizes for per-layer LPC computation.
# Fallback: config.yaml (reliable default for all runs).
expert_hidden_sizes = cache.get('expert_hidden_sizes')
if expert_hidden_sizes is None:
    import yaml
    config_path = pathlib.Path('config.yaml')
    if config_path.exists():
        with open(config_path) as _f:
            _cfg = yaml.safe_load(_f)
        expert_hidden_sizes = _cfg.get('expert_hidden_sizes')

metrics = cache["metrics"]
print(
    f"Loaded {len(routing_data)} timesteps | "
    f"success={metrics['success_rate']:.1%} | "
    f"path_ratio={metrics['path_ratio']:.2f} | "
    f"mean_lpc={metrics['mean_lpc']:.2f}"
)

# ── 3. Sample environment images ──────────────────────────────────────────────
# Runs up to 1000 env resets to collect one render per group (same logic as
# routing_viz.py). Provides env_image to all plot functions and per-group images
# to grouped routing plots.

import gymnasium as gym
import minigrid  # noqa: F401  registers BabyAI envs with gymnasium

sample_env = gym.make(task_id, render_mode="rgb_array")
uw = sample_env.unwrapped

env_image = None
env_mission = ""
room_env_images = {}
door_env_images = {}
door_and_box_env_images = {}
quadrant_env_images = {}

target_rooms = set(group_routing_data(routing_data, "agent_start_room").keys())
target_door_locations = set(group_routing_data(routing_data, "door_location").keys())
target_door_box_locations = set(group_routing_data(routing_data, "door_and_box_row").keys())
target_quadrant_combos = set(group_routing_data(routing_data, "agent_and_target_quadrant").keys())


def all_targets_found():
    return (
        len(room_env_images) >= len(target_rooms)
        and len(door_env_images) >= len(target_door_locations)
        and len(door_and_box_env_images) >= len(target_door_box_locations)
        and len(quadrant_env_images) >= len(target_quadrant_combos)
    )


print("Sampling environment images...", end="", flush=True)
for _ in range(1000):
    obs, _ = sample_env.reset()
    env_image = uw.get_frame(tile_size=32, agent_pov=False, highlight=False)
    env_mission = obs.get("mission", "") if isinstance(obs, dict) else ""

    if hasattr(uw, "room_from_pos"):
        try:
            room = uw.room_from_pos(*uw.agent_pos)
            room_key = tuple(int(c) for c in room.top)
            if room_key in target_rooms and room_key not in room_env_images:
                room_env_images[room_key] = (env_image, env_mission)
        except Exception:
            pass

    if target_door_locations or target_door_box_locations:
        door_positions = []
        box_positions_y = []
        for j in range(uw.grid.height):
            for i in range(uw.grid.width):
                cell = uw.grid.get(i, j)
                if cell is not None:
                    if cell.type == "door":
                        door_positions.append((i, j))
                    elif cell.type == "box":
                        box_positions_y.append(j)
        door_key = tuple(sorted(door_positions))
        if door_key in target_door_locations and door_key not in door_env_images:
            door_env_images[door_key] = (env_image, env_mission)
        if target_door_box_locations and box_positions_y:
            box_row_key = tuple(sorted(box_positions_y))
            combined_key = (door_key, box_row_key)
            if combined_key in target_door_box_locations and combined_key not in door_and_box_env_images:
                door_and_box_env_images[combined_key] = (env_image, env_mission)

    if target_quadrant_combos:
        try:
            instr = getattr(uw, "instrs", None)
            target_pos = _first_target_pos(instr) if instr is not None else None
            if target_pos is not None:
                if hasattr(uw, "room_from_pos"):
                    room = uw.room_from_pos(*uw.agent_pos)
                    rx, ry = room.top
                    rw, rh = room.size
                    room_bounds = (rx + 1, ry + 1, rx + rw - 2, ry + rh - 2)
                else:
                    w, h = uw.grid.width, uw.grid.height
                    room_bounds = (1, 1, w - 2, h - 2)
                aq = pos_to_quadrant(uw.agent_pos[0], uw.agent_pos[1], room_bounds)
                tq = pos_to_quadrant(target_pos[0], target_pos[1], room_bounds)
                quad_key = (aq, tq)
                if quad_key in target_quadrant_combos and quad_key not in quadrant_env_images:
                    quadrant_env_images[quad_key] = (env_image, env_mission)
        except Exception:
            pass

    if all_targets_found():
        break

sample_env.close()
print(" done.")

# ── 4. Generate plot ──────────────────────────────────────────────────────────
group_by_map = {
    "by_starting_room":             ("agent_start_room",          room_env_images),
    "by_door_location":             ("door_location",             door_env_images),
    "by_door_and_box_row":          ("door_and_box_row",          door_and_box_env_images),
    "by_carrying_phase":            ("carrying_phase",            None),
    "by_door_unlocked_phase":       ("door_unlocked_phase",       None),
    "by_key_phase":                 ("key_phase",                 None),
    "by_agent_and_target_quadrant": ("agent_and_target_quadrant", quadrant_env_images),
}

if plot_type == "overall":
    fig = plot_overall_routing(
        routing_data, env_image=env_image, env_mission=env_mission,
        layer_expert_sizes=expert_hidden_sizes,
    )

elif plot_type in GROUPED_ROUTING_TYPES:
    group_by, per_group_images = group_by_map[plot_type]
    fig = plot_grouped_routing(
        routing_data,
        group_by=group_by,
        env_image=env_image,
        env_mission=env_mission,
        room_env_images=per_group_images,
        layer_expert_sizes=expert_hidden_sizes,
    )

elif plot_type == "action_frequency":
    fig = plot_action_frequency(routing_data)

elif plot_type == "action_frequency_carrying":
    fig = plot_action_frequency(routing_data, group_by="carrying")

elif plot_type == "across_episode_entropy_heatmap":
    fig = plot_across_episode_entropy_heatmap(routing_data, env_image=env_image, env_mission=env_mission)

elif plot_type == "by_door_location_across_episode_entropy":
    fig = plot_grouped_across_episode_entropy_heatmap(
        routing_data, group_by="door_location",
        env_image=env_image, env_mission=env_mission, room_env_images=door_env_images,
    )

elif plot_type == "by_door_and_box_row_across_episode_entropy":
    fig = plot_grouped_across_episode_entropy_heatmap(
        routing_data, group_by="door_and_box_row",
        env_image=env_image, env_mission=env_mission, room_env_images=door_and_box_env_images,
    )

elif plot_type == "by_door_unlocked_phase_across_episode_entropy":
    fig = plot_grouped_across_episode_entropy_heatmap(
        routing_data, group_by="door_unlocked_phase",
        env_image=env_image, env_mission=env_mission,
    )

elif plot_type == "by_key_phase_across_episode_entropy":
    fig = plot_grouped_across_episode_entropy_heatmap(
        routing_data, group_by="key_phase",
        env_image=env_image, env_mission=env_mission,
    )

elif plot_type == "kl_heatmap":
    fig = plot_kl_heatmap(routing_data, env_image=env_image, env_mission=env_mission)

elif plot_type == "by_door_location_kl":
    fig = plot_grouped_kl_heatmap(
        routing_data, group_by="door_location",
        env_image=env_image, env_mission=env_mission, room_env_images=door_env_images,
    )

elif plot_type == "by_door_and_box_row_kl":
    fig = plot_grouped_kl_heatmap(
        routing_data, group_by="door_and_box_row",
        env_image=env_image, env_mission=env_mission, room_env_images=door_and_box_env_images,
    )

elif plot_type == "by_door_unlocked_phase_kl":
    fig = plot_grouped_kl_heatmap(
        routing_data, group_by="door_unlocked_phase",
        env_image=env_image, env_mission=env_mission,
    )

elif plot_type == "by_key_phase_kl":
    fig = plot_grouped_kl_heatmap(
        routing_data, group_by="key_phase",
        env_image=env_image, env_mission=env_mission,
    )

elif plot_type == "kl_heatmap_global":
    fig = plot_kl_heatmap(routing_data, env_image=env_image, env_mission=env_mission, P_a=_P_a_global)

elif plot_type == "by_door_location_kl_global":
    fig = plot_grouped_kl_heatmap(
        routing_data, group_by="door_location",
        env_image=env_image, env_mission=env_mission, room_env_images=door_env_images,
        P_a=_P_a_global,
    )

elif plot_type == "by_door_and_box_row_kl_global":
    fig = plot_grouped_kl_heatmap(
        routing_data, group_by="door_and_box_row",
        env_image=env_image, env_mission=env_mission, room_env_images=door_and_box_env_images,
        P_a=_P_a_global,
    )

elif plot_type == "by_door_unlocked_phase_kl_global":
    fig = plot_grouped_kl_heatmap(
        routing_data, group_by="door_unlocked_phase",
        env_image=env_image, env_mission=env_mission,
        P_a=_P_a_global,
    )

elif plot_type == "by_key_phase_kl_global":
    fig = plot_grouped_kl_heatmap(
        routing_data, group_by="key_phase",
        env_image=env_image, env_mission=env_mission,
        P_a=_P_a_global,
    )

elif plot_type == "cell_action_distribution":
    fig = plot_cell_action_distribution(routing_data)

# ── 5. Preview & optionally save ──────────────────────────────────────────────
out_dir = pathlib.Path("plots") / task_id / f"trial_{trial}"
if _args.seed is not None:
    out_dir = out_dir / f"seed_{_args.seed}"
if _args.update is not None:
    out_dir = out_dir / f"update_{_args.update}"

filename_map = {
    "overall":                       "routing_heatmap.png",
    "by_starting_room":              "routing_heatmap_by_starting_room.png",
    "by_door_location":              "routing_heatmap_by_door_location.png",
    "by_door_and_box_row":           "routing_heatmap_by_door_and_box_row.png",
    "by_carrying_phase":             "routing_heatmap_by_carrying_phase.png",
    "by_door_unlocked_phase":        "routing_heatmap_by_door_unlocked_phase.png",
    "by_agent_and_target_quadrant":  "routing_heatmap_by_agent_and_target_quadrant.png",
    "action_frequency":              "logit_action_frequency.png",
    "action_frequency_carrying":     "logit_action_frequency_carrying.png",
    "across_episode_entropy_heatmap":              "entropy_heatmap.png",
    "by_door_location_across_episode_entropy":     "entropy_by_door_location.png",
    "by_door_and_box_row_across_episode_entropy":  "entropy_by_door_and_box_row.png",
    "by_door_unlocked_phase_across_episode_entropy": "entropy_by_door_unlocked_phase.png",
    "by_key_phase":                                "routing_heatmap_by_key_phase.png",
    "by_key_phase_across_episode_entropy":         "entropy_by_key_phase.png",
    "kl_heatmap":                                  "kl_local_heatmap.png",
    "by_door_location_kl":                         "kl_local_by_door_location.png",
    "by_door_and_box_row_kl":                      "kl_local_by_door_and_box_row.png",
    "by_door_unlocked_phase_kl":                   "kl_local_by_door_unlocked_phase.png",
    "by_key_phase_kl":                             "kl_local_by_key_phase.png",
    "kl_heatmap_global":                           "kl_global_heatmap.png",
    "by_door_location_kl_global":                  "kl_global_by_door_location.png",
    "by_door_and_box_row_kl_global":               "kl_global_by_door_and_box_row.png",
    "by_door_unlocked_phase_kl_global":            "kl_global_by_door_unlocked_phase.png",
    "by_key_phase_kl_global":                      "kl_global_by_key_phase.png",
    "cell_action_distribution":                    "cell_action_distribution.png",
}

out_path = out_dir / filename_map[plot_type]

plt.show(block=False)
plt.pause(0.1)

answer = input(f"Save to {out_path}? [y/N] ").strip().lower()
if answer == "y":
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved → {out_path}")
else:
    print("Not saved.")

plt.close(fig)
