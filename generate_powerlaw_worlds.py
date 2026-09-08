#!/usr/bin/env python3
"""
Generate power-law distribution world files for 5x5 arena.

Matches paper figure (b) Powerlaw pattern:
  1 large cluster  (~20 tags)
  1 medium cluster (~10 tags)
  2-4 small clusters (~3-6 tags each)
  ~15-20 individual singletons scattered throughout arena

Pass --start and --end to control the sample range (default 1-10).
  python generate_powerlaw_worlds.py --start 11 --end 20
"""

import argparse
import math
import random
import re
import os

TEMPLATE    = os.path.join(os.path.dirname(__file__), "worlds", "eval_powerlaw1_5x5.wbt")
OUT_DIR     = os.path.join(os.path.dirname(__file__), "worlds")
NUM_TAGS    = 64
ARENA_HALF  = 2.1      # safe placement bounds (2.5m arena - 0.4m margin)
NEST_RADIUS = 0.55     # min distance from nest (0,0) for any placed item
TAG_SPACING = 0.0775   # grid pitch within a cluster (matches existing worlds)


def powerlaw_sizes(n_total, rng):
    """
    Generate cluster sizes matching paper figure (b):
      1 large (~n/3), 1 medium (~n/6), 2-4 small (3-6 each), rest as singletons.
    """
    large  = n_total // 3                           # ~21
    medium = n_total // 6                           # ~10
    n_sm   = rng.randint(2, 4)
    smalls = [rng.randint(3, 6) for _ in range(n_sm)]

    clustered = [large, medium] + smalls
    # Trim if overshoot (unlikely but safe)
    while sum(clustered) > n_total:
        clustered[-1] -= 1
        if clustered[-1] <= 0:
            clustered.pop()

    remaining = n_total - sum(clustered)
    return clustered + [1] * remaining   # singletons fill the rest


def cluster_positions(cx, cy, n):
    """Lay n tags in a compact square-ish grid centred at (cx, cy)."""
    cols = math.ceil(math.sqrt(n))
    rows = math.ceil(n / cols)
    positions = []
    ox = cx - (cols - 1) * TAG_SPACING / 2
    oy = cy - (rows - 1) * TAG_SPACING / 2
    for r in range(rows):
        for c in range(cols):
            if len(positions) >= n:
                break
            positions.append((ox + c * TAG_SPACING, oy + r * TAG_SPACING))
    return positions


def min_sep(size_a, size_b):
    """Minimum centre-to-centre separation depending on cluster sizes."""
    if size_a == 1 and size_b == 1:
        return 0.25   # singletons just need to be visually distinct
    elif size_a == 1 or size_b == 1:
        return 0.40   # singleton near a cluster
    else:
        return 0.90   # two proper clusters need clear space


def place_items(sizes, rng, max_tries=8000):
    """
    Place all cluster/singleton centres avoiding walls, nest, and each other.
    Returns list of (cx, cy, size).
    """
    placed = []
    for size in sizes:
        cols   = math.ceil(math.sqrt(size))
        rows   = math.ceil(size / cols)
        radius = math.sqrt((cols * TAG_SPACING / 2) ** 2 +
                           (rows * TAG_SPACING / 2) ** 2) + 0.04

        for _ in range(max_tries):
            cx = rng.uniform(-ARENA_HALF + radius, ARENA_HALF - radius)
            cy = rng.uniform(-ARENA_HALF + radius, ARENA_HALF - radius)

            if math.sqrt(cx**2 + cy**2) < NEST_RADIUS + radius:
                continue

            ok = all(
                math.sqrt((cx - px)**2 + (cy - py)**2) >= min_sep(size, ps)
                for px, py, ps in placed
            )
            if ok:
                placed.append((cx, cy, size))
                break
        else:
            raise RuntimeError(
                f"Could not place cluster of size {size} after {max_tries} tries")
    return placed


def build_apriltag_lines(all_positions):
    """Build wbt lines for all APRILTAG nodes."""
    tmpl = (
        'DEF APRILTAG_{idx} Solid {{ translation {x:.4f} {y:.4f} 0.01375 '
        'children [ Shape {{ appearance PBRAppearance {{ baseColorMap ImageTexture '
        '{{ url [ "textures/real_tag36h11_id0.png" ] }} roughness 1 metalness 0 }} '
        'geometry Box {{ size 0.0275 0.0275 0.0275 }} }} ] name "apriltag_{idx}" }}'
    )
    return [tmpl.format(idx=i + 1, x=x, y=y)
            for i, (x, y) in enumerate(all_positions)]


def generate_world(sample_idx, template_content, rng):
    sizes  = powerlaw_sizes(NUM_TAGS, rng)
    placed = place_items(sizes, rng)

    all_positions = []
    for cx, cy, size in placed:
        all_positions.extend(cluster_positions(cx, cy, size))

    assert len(all_positions) == NUM_TAGS, \
        f"Expected {NUM_TAGS} tags, got {len(all_positions)}"

    new_tag_lines = build_apriltag_lines(all_positions)

    # Replace contiguous APRILTAG block in the template
    lines = template_content.split('\n')
    first_idx = last_idx = None
    for i, line in enumerate(lines):
        if re.match(r'\s*DEF APRILTAG_\d+', line):
            if first_idx is None:
                first_idx = i
            last_idx = i

    if first_idx is None:
        raise RuntimeError("No APRILTAG lines found in template")

    new_lines   = lines[:first_idx] + new_tag_lines + lines[last_idx + 1:]
    new_content = '\n'.join(new_lines)

    new_content = re.sub(r'(Sample\s*)\d+',
                         lambda m: m.group(1) + str(sample_idx),
                         new_content)
    return new_content, sizes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=int, default=1)
    parser.add_argument("--end",   type=int, default=10)
    args = parser.parse_args()

    with open(TEMPLATE) as f:
        template_content = f.read()

    for sample in range(args.start, args.end + 1):
        rng = random.Random(sample * 42 + 7)
        content, sizes = generate_world(sample, template_content, rng)

        clusters   = [s for s in sizes if s > 1]
        singletons = sizes.count(1)

        out_path = os.path.join(OUT_DIR, f"eval_powerlaw{sample}_5x5.wbt")
        with open(out_path, "w") as f:
            f.write(content)

        print(f"Sample {sample:2d}: clusters={clusters}  singletons={singletons}  "
              f"total={sum(sizes)}  → {out_path}")

    print("\nDone. Open a world in Webots to verify the distribution visually.")


if __name__ == "__main__":
    main()
