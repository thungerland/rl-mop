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
    by_agent_and_target_quadrant — routing heatmap grouped by agent & target quadrant
    action_frequency             — bar chart of action frequencies
    action_frequency_carrying    — action frequency split by carrying phase
    entropy_heatmap              — spatial entropy heatmap of action logits

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
    plot_action_entropy_heatmap,
    group_routing_data,
    pos_to_quadrant,
)
from eval_mop import _first_target_pos

# ── 1. Parse args ─────────────────────────────────────────────────────────────
if len(sys.argv) < 3:
    print("Usage: python analyze.py <task_id> <trial> [plot_type]")
    sys.exit(1)

task_id = sys.argv[1]
trial = int(sys.argv[2])
plot_type = sys.argv[3] if len(sys.argv) > 3 else "overall"

GROUPED_ROUTING_TYPES = {
    "by_starting_room",
    "by_door_location",
    "by_door_and_box_row",
    "by_carrying_phase",
    "by_agent_and_target_quadrant",
}

ALL_TYPES = {"overall"} | GROUPED_ROUTING_TYPES | {
    "action_frequency",
    "action_frequency_carrying",
    "entropy_heatmap",
}

if plot_type not in ALL_TYPES:
    print(f"Unknown plot_type '{plot_type}'. Options: {', '.join(sorted(ALL_TYPES))}")
    sys.exit(1)

# ── 2. Load cache ─────────────────────────────────────────────────────────────
cache_path = pathlib.Path(f"evaluation_cache/{task_id}/trial_{trial}/routing_data.json")
if not cache_path.exists():
    print(f"Cache not found: {cache_path}")
    sys.exit(1)

with open(cache_path) as f:
    cache = json.load(f)

routing_data = build_routing_data_tuples(cache)

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
    "by_agent_and_target_quadrant": ("agent_and_target_quadrant", quadrant_env_images),
}

if plot_type == "overall":
    fig = plot_overall_routing(routing_data, env_image=env_image, env_mission=env_mission)

elif plot_type in GROUPED_ROUTING_TYPES:
    group_by, per_group_images = group_by_map[plot_type]
    fig = plot_grouped_routing(
        routing_data,
        group_by=group_by,
        env_image=env_image,
        env_mission=env_mission,
        room_env_images=per_group_images,
    )

elif plot_type == "action_frequency":
    fig = plot_action_frequency(routing_data)

elif plot_type == "action_frequency_carrying":
    fig = plot_action_frequency(routing_data, group_by="carrying")

elif plot_type == "entropy_heatmap":
    fig = plot_action_entropy_heatmap(routing_data, env_image=env_image, env_mission=env_mission)

# ── 5. Preview & optionally save ──────────────────────────────────────────────
out_dir = pathlib.Path("plots") / task_id / f"trial_{trial}"

filename_map = {
    "overall":                       "routing_heatmap.png",
    "by_starting_room":              "routing_heatmap_by_starting_room.png",
    "by_door_location":              "routing_heatmap_by_door_location.png",
    "by_door_and_box_row":           "routing_heatmap_by_door_and_box_row.png",
    "by_carrying_phase":             "routing_heatmap_by_carrying_phase.png",
    "by_agent_and_target_quadrant":  "routing_heatmap_by_agent_and_target_quadrant.png",
    "action_frequency":              "logit_action_frequency.png",
    "action_frequency_carrying":     "logit_action_frequency_carrying.png",
    "entropy_heatmap":               "logit_entropy_heatmap.png",
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
