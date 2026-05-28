#!/usr/bin/env python3
"""
Generate multi-robot 7×7 eval worlds for the robot-scalability experiment.

Configs produced (all 7×7 arena, 10 samples each):
  4r / 32t   → reuse existing eval_sample{N}_7x7.wbt (no new files)
               run with: --num_robots 4 --num_tags 32
  8r / 64t   → eval_sample{N}_7x7_8r.wbt
  12r / 128t → eval_sample{N}_7x7_12r.wbt
  16r / 208t → eval_sample{N}_7x7_16r.wbt  (+80 extra clustered tags)

Extra robot positions (rings around nest at origin):
  Ring 2 (robots 5-8):   r=0.30m, 45°/135°/225°/315°
  Ring 3 (robots 9-12):  r=0.75m, 0°/90°/180°/270°
  Ring 4 (robots 13-16): r=0.60m, 45°/135°/225°/315°

Extra tags for 16r (80 tags, 5 clusters × 16 tags each):
  Placed randomly (seeded per sample) avoiding existing cluster positions.

Usage:
    cd "/home/sara/Documents/Centralized Learning/RL-FL-Foraging"
    python3 generate_multirobot_worlds.py
"""

import math
import os
import random
import re

WORLDS_DIR  = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'worlds')
SAMPLES     = list(range(1, 11))
SPACING     = 0.0775   # tag grid spacing (m)
TAG_HALF_Z  = 0.01375  # z-offset for tag centre
TAG_SIZE    = 0.0275
TAG_TEXTURE = "textures/real_tag36h11_id0.png"

# ── Extra robot positions (beyond robots 1-4) ─────────────────────────────────
# Each entry: robot_number → (x, y) in metres
def _polar(r, deg):
    a = math.radians(deg)
    return (round(r * math.cos(a), 4), round(r * math.sin(a), 4))

EXTRA_ROBOT_POSITIONS = {
    5:  _polar(0.30, 45),
    6:  _polar(0.30, 135),
    7:  _polar(0.30, 225),
    8:  _polar(0.30, 315),
    9:  _polar(0.75, 0),
    10: _polar(0.75, 90),
    11: _polar(0.75, 180),
    12: _polar(0.75, 270),
    13: _polar(0.60, 45),
    14: _polar(0.60, 135),
    15: _polar(0.60, 225),
    16: _polar(0.60, 315),
}


# ── World-file text helpers ───────────────────────────────────────────────────

def _epuck_block(robot_num):
    x, y = EXTRA_ROBOT_POSITIONS[robot_num]
    ch   = robot_num
    return (
        f'\nDEF ROBOT{ch} E-puck {{\n'
        f'  translation {x} {y} 0\n'
        f'  name "robot{ch}"\n'
        f'  controller "epuck_decentralized_eval"\n'
        f'  emitter_channel {ch}\n'
        f'  receiver_channel {ch}\n'
        f'  camera_fieldOfView 1\n'
        f'  camera_width 512\n'
        f'  camera_height 512\n'
        f'  turretSlot [\n'
        f'    GPS {{\n'
        f'      name "gps"\n'
        f'      accuracy 0\n'
        f'    }}\n'
        f'    InertialUnit {{\n'
        f'      name "inertial_unit"\n'
        f'    }}\n'
        f'    Emitter {{\n'
        f'      name "phero_emitter"\n'
        f'      range 0.5\n'
        f'      channel 10\n'
        f'    }}\n'
        f'    Receiver {{\n'
        f'      name "phero_receiver"\n'
        f'      channel 10\n'
        f'      bufferSize 1024\n'
        f'    }}\n'
        f'  ]\n'
        f'}}\n'
    )


def _sup_comms(robot_nums):
    """Emitter+Receiver lines for the supervisor children block."""
    lines = []
    for ch in robot_nums:
        lines.append(f'    Emitter {{ name "emitter{ch}" channel {ch} }}\n')
        lines.append(f'    Receiver {{ name "receiver{ch}" channel {ch} }}\n')
    return ''.join(lines)


# ── Tag-cluster helpers ───────────────────────────────────────────────────────

def _tag_line(idx, x, y):
    return (
        f'DEF APRILTAG_{idx} Solid {{ '
        f'translation {x:.4f} {y:.4f} {TAG_HALF_Z} '
        f'children [ Shape {{ appearance PBRAppearance {{ '
        f'baseColorMap ImageTexture {{ url [ "{TAG_TEXTURE}" ] }} '
        f'roughness 1 metalness 0 }} '
        f'geometry Box {{ size {TAG_SIZE} {TAG_SIZE} {TAG_SIZE} }} }} ] '
        f'name "apriltag_{idx}" }}\n'
    )


def _cluster_4x4(start_idx, cx, cy):
    """Return (lines_list, next_start_idx) for a 4×4=16 tag cluster."""
    lines, idx = [], start_idx
    for row in range(4):
        for col in range(4):
            x = cx + (-1.5 + col) * SPACING
            y = cy + (-1.5 + row) * SPACING
            lines.append(_tag_line(idx, x, y))
            idx += 1
    return lines, idx


def _get_existing_centers(world_text):
    """Parse cluster centres from comment lines like '# C1 ... center=(x,y)'."""
    pattern = r'center=\(([^)]+)\)'
    centers = []
    for m in re.finditer(pattern, world_text):
        vals = m.group(1).split(',')
        try:
            centers.append((float(vals[0]), float(vals[1])))
        except ValueError:
            pass
    return centers


def _generate_extra_centers(existing, n_new, seed,
                             arena_safe=2.85, min_sep=0.60):
    """
    Place n_new cluster centres that keep ≥min_sep from all existing and new
    centres, and ≥(arena_half - arena_safe) from the arena wall.
    Uses a seeded RNG so each sample gets reproducible but different positions.
    """
    rng = random.Random(seed)
    new = []
    for _ in range(n_new):
        for _attempt in range(100_000):
            x = rng.uniform(-arena_safe, arena_safe)
            y = rng.uniform(-arena_safe, arena_safe)
            if all(math.sqrt((x - cx)**2 + (y - cy)**2) >= min_sep
                   for cx, cy in existing + new):
                new.append((x, y))
                break
        else:
            raise RuntimeError(
                f"Could not place extra cluster {_+1} for seed={seed}. "
                f"Existing={len(existing)} clusters, need {n_new} new ones.")
    return new


# ── World-file modification ───────────────────────────────────────────────────

def _insert_robots(text, robot_nums):
    """Insert E-puck blocks before the supervisor Robot { node."""
    new_blocks = ''.join(_epuck_block(n) for n in robot_nums)
    # Supervisor node starts with 'Robot {\n  translation 0 0 0'
    marker = '\nRobot {\n  translation 0 0 0'
    idx = text.rfind(marker)   # rfind — last occurrence = supervisor
    if idx == -1:
        raise ValueError("Could not find supervisor Robot node marker")
    return text[:idx] + new_blocks + text[idx:]


def _insert_sup_comms(text, robot_nums):
    """Insert emitter/receiver pairs inside supervisor children block."""
    # Anchor: last receiver4 line in supervisor children
    anchor = '    Receiver { name "receiver4" channel 4 }\n'
    idx = text.rfind(anchor)
    if idx == -1:
        raise ValueError("Could not find 'receiver4' anchor in supervisor block")
    insert_pos = idx + len(anchor)
    new_lines = _sup_comms(robot_nums)
    return text[:insert_pos] + new_lines + text[insert_pos:]


def _insert_extra_tags(text, extra_lines):
    """Insert extra tag definitions immediately before the first robot (DEF ROBOT1)."""
    marker = '\nDEF ROBOT1 E-puck {'
    idx = text.find(marker)
    if idx == -1:
        raise ValueError("Could not find 'DEF ROBOT1 E-puck {' marker")
    tag_block = '\n# Extra tags for 16-robot experiment\n' + ''.join(extra_lines)
    return text[:idx] + tag_block + text[idx:]


# ── Per-config world generation ───────────────────────────────────────────────

def generate_world(sample_num, num_robots):
    base_path = os.path.join(WORLDS_DIR, f'eval_sample{sample_num}_7x7.wbt')
    if not os.path.exists(base_path):
        print(f'  [SKIP] Base world not found: {base_path}')
        return

    with open(base_path, 'r') as f:
        text = f.read()

    extra_robot_nums = list(range(5, num_robots + 1))   # [] for 4r

    # --- 8r and 12r: just add robots + supervisor comms ----------------------
    if num_robots in (8, 12):
        text = _insert_robots(text, extra_robot_nums)
        text = _insert_sup_comms(text, extra_robot_nums)

    # --- 16r: add robots + supervisor comms + 80 extra tags ------------------
    elif num_robots == 16:
        existing_centers = _get_existing_centers(text)
        # 5 extra clusters × 16 tags = 80 tags (tags 129–208)
        seed = sample_num * 7777
        extra_centers = _generate_extra_centers(existing_centers, 5, seed)

        extra_tag_lines = []
        tag_idx = 129
        for ci, (cx, cy) in enumerate(extra_centers):
            lines, tag_idx = _cluster_4x4(tag_idx, cx, cy)
            extra_tag_lines.extend([
                f'# C{12+ci} (4x4=16 tags)  EXTRA  center=({cx:.4f},{cy:.4f})'
                f' - Tags {tag_idx-16}-{tag_idx-1}\n'
            ] + lines)

        text = _insert_extra_tags(text, extra_tag_lines)
        text = _insert_robots(text, extra_robot_nums)
        text = _insert_sup_comms(text, extra_robot_nums)

        # Update world title to reflect 16r
        text = text.replace(
            f'title "Multi-Agent AprilTag Foraging - EVAL 7x7 Sample {sample_num}"',
            f'title "Multi-Agent AprilTag Foraging - EVAL 7x7 {num_robots}r Sample {sample_num}"'
        )

    else:
        print(f'  [SKIP] num_robots={num_robots} — only 8/12/16 generate new files')
        return

    # Update title for 8r/12r too
    if num_robots in (8, 12):
        text = text.replace(
            f'title "Multi-Agent AprilTag Foraging - EVAL 7x7 Sample {sample_num}"',
            f'title "Multi-Agent AprilTag Foraging - EVAL 7x7 {num_robots}r Sample {sample_num}"'
        )

    out_path = os.path.join(WORLDS_DIR,
                             f'eval_sample{sample_num}_7x7_{num_robots}r.wbt')
    with open(out_path, 'w') as f:
        f.write(text)
    print(f'  [OK] {os.path.basename(out_path)}')


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("Generating multi-robot eval worlds for 7×7 arena ...\n")

    for num_robots in (8, 12, 16):
        print(f"--- {num_robots} robots ---")
        for sample in SAMPLES:
            generate_world(sample, num_robots)
        print()

    print("Done.")
    print()
    print("World files created:")
    for num_robots in (8, 12, 16):
        for s in SAMPLES:
            p = os.path.join(WORLDS_DIR, f'eval_sample{s}_7x7_{num_robots}r.wbt')
            exists = "✓" if os.path.exists(p) else "✗"
            print(f"  {exists} eval_sample{s}_7x7_{num_robots}r.wbt")

    print()
    print("Run the experiment with run_batch_eval.py, e.g.:")
    print("  python3 run_batch_eval.py --arena_size 7x7 --num_robots 4  --num_tags 32  --run_name decentralized_indep_v9")
    print("  python3 run_batch_eval.py --arena_size 7x7 --num_robots 8  --num_tags 64  --run_name decentralized_indep_v9")
    print("  python3 run_batch_eval.py --arena_size 7x7 --num_robots 12 --num_tags 128 --run_name decentralized_indep_v9")
    print("  python3 run_batch_eval.py --arena_size 7x7 --num_robots 16 --num_tags 208 --run_name decentralized_indep_v9")


if __name__ == '__main__':
    main()
