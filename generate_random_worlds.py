#!/usr/bin/env python3
"""
Generate random distribution world files for 5x5 arena.

64 tags placed uniformly at random, avoiding walls and nest.

Pass --start and --end to control the sample range (default 1-10).
  python generate_random_worlds.py --start 11 --end 20
"""

import argparse
import math
import random
import re
import os

TEMPLATE   = os.path.join(os.path.dirname(__file__), "worlds", "eval_random1_5x5.wbt")
OUT_DIR    = os.path.join(os.path.dirname(__file__), "worlds")
NUM_TAGS   = 64
ARENA_HALF = 2.1     # safe placement bound (2.5m - 0.4m margin)
NEST_RADIUS = 0.55   # min distance from nest (0,0)
MIN_SEP    = 0.08    # min tag-to-tag separation


def place_tags(rng, max_tries=50_000):
    positions = []
    for _ in range(NUM_TAGS):
        for _ in range(max_tries):
            x = rng.uniform(-ARENA_HALF, ARENA_HALF)
            y = rng.uniform(-ARENA_HALF, ARENA_HALF)
            if math.hypot(x, y) < NEST_RADIUS:
                continue
            if any(math.hypot(x - px, y - py) < MIN_SEP for px, py in positions):
                continue
            positions.append((x, y))
            break
        else:
            raise RuntimeError("Could not place all tags after max tries")
    return positions


def build_apriltag_lines(positions):
    tmpl = (
        'DEF APRILTAG_{idx} Solid {{ translation {x:.4f} {y:.4f} 0.01375 '
        'children [ Shape {{ appearance PBRAppearance {{ baseColorMap ImageTexture '
        '{{ url [ "textures/real_tag36h11_id0.png" ] }} roughness 1 metalness 0 }} '
        'geometry Box {{ size 0.0275 0.0275 0.0275 }} }} ] name "apriltag_{idx}" }}'
    )
    return [tmpl.format(idx=i + 1, x=x, y=y) for i, (x, y) in enumerate(positions)]


def generate_world(sample_idx, template_content, rng):
    positions = place_tags(rng)
    new_tag_lines = build_apriltag_lines(positions)

    lines = template_content.split('\n')
    first_idx = last_idx = None
    for i, line in enumerate(lines):
        if re.match(r'\s*DEF APRILTAG_\d+', line):
            if first_idx is None:
                first_idx = i
            last_idx = i

    if first_idx is None:
        raise RuntimeError("No APRILTAG lines found in template")

    new_lines = lines[:first_idx] + new_tag_lines + lines[last_idx + 1:]
    new_content = '\n'.join(new_lines)

    new_content = re.sub(r'(Sample\s*)\d+',
                         lambda m: m.group(1) + str(sample_idx),
                         new_content)
    return new_content


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=int, default=1)
    parser.add_argument("--end",   type=int, default=10)
    args = parser.parse_args()

    with open(TEMPLATE) as f:
        template_content = f.read()

    for sample in range(args.start, args.end + 1):
        rng = random.Random(sample * 137 + 13)
        content = generate_world(sample, template_content, rng)

        out_path = os.path.join(OUT_DIR, f"eval_random{sample}_5x5.wbt")
        with open(out_path, "w") as f:
            f.write(content)

        print(f"Sample {sample:2d}: 64 tags placed uniformly  → {out_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
