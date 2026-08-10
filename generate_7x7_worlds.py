#!/usr/bin/env python3
"""
Generate 7x7 Exp-2 world files for 4/8/12/16-robot configurations.

Config    Robots  Tags   Density (7x7=49m²)
4r         4       64    1.31 /m²
8r         8      128    2.61 /m²
12r       12      192    3.92 /m²
16r       16      256    5.22 /m²

Each configuration: 20 samples, clustered tag layout (4x4=16 tags per cluster).
Overwrites existing worlds/eval_sampleN_7x7{suffix}.wbt files.
"""

import math
import random
from pathlib import Path

WORLDS_DIR = Path(__file__).parent / "worlds"

# ── robot positions (used for first N of 16) ─────────────────────────────────
ROBOT_XY = [
    (-0.5000,  0.0000),   # 1  — cardinal ring r=0.50
    ( 0.5000,  0.0000),   # 2
    ( 0.0000,  0.5000),   # 3
    ( 0.0000, -0.5000),   # 4
    ( 0.4619,  0.1913),   # 5  — offset ring r=0.50
    (-0.1913,  0.4619),   # 6
    (-0.4619, -0.1913),   # 7
    ( 0.1913, -0.4619),   # 8
    ( 0.5303,  0.5303),   # 9  — diagonal ring r=0.75
    (-0.5303,  0.5303),   # 10
    (-0.5303, -0.5303),   # 11
    ( 0.5303, -0.5303),   # 12
    ( 0.2296,  0.5543),   # 13 — offset ring r=0.60
    (-0.5543,  0.2296),   # 14
    (-0.2296, -0.5543),   # 15
    ( 0.5543, -0.2296),   # 16
]

# ── tag geometry ──────────────────────────────────────────────────────────────
TAG_SIZE     = 0.0275
TAG_Z        = TAG_SIZE / 2          # 0.01375
TAG_SPACING  = 0.0775                # center-to-center within cluster
CLUSTER_ROWS = 4
CLUSTER_COLS = 4
TAGS_PER_CLUSTER = CLUSTER_ROWS * CLUSTER_COLS   # 16

# ── placement constraints ─────────────────────────────────────────────────────
ARENA_HALF  = 3.5
CTR_BOUND   = 3.0     # cluster centers stay within ±3.0 m
NEST_EXCL   = 0.70    # min dist from (0,0) for cluster center
CLUSTER_SEP = 0.80    # min center-to-center distance between clusters
MAX_TRIES   = 100_000

# ── config table ─────────────────────────────────────────────────────────────
CONFIGS = [
    # (key,   suffix,   n_robots, n_tags, phero_slots)
    ("4r",   "",       4,        64,     False),
    ("8r",   "_8r",    8,        128,    True ),
    ("12r",  "_12r",   12,       192,    True ),
    ("16r",  "_16r",   16,       256,    True ),
]


# ── cluster placement ─────────────────────────────────────────────────────────
def place_clusters(n, rng):
    centers = []
    for idx in range(n):
        for _ in range(MAX_TRIES):
            x = rng.uniform(-CTR_BOUND, CTR_BOUND)
            y = rng.uniform(-CTR_BOUND, CTR_BOUND)
            if math.hypot(x, y) < NEST_EXCL:
                continue
            if any(math.hypot(x - cx, y - cy) < CLUSTER_SEP
                   for cx, cy in centers):
                continue
            centers.append((x, y))
            break
        else:
            raise RuntimeError(
                f"Could not place cluster {idx+1}/{n} after {MAX_TRIES} tries")
    return centers


# ── WBT building blocks ───────────────────────────────────────────────────────
def _tag_line(num, x, y):
    return (
        f'DEF APRILTAG_{num} Solid {{'
        f' translation {x:.4f} {y:.4f} {TAG_Z}'
        f' children [ Shape {{ appearance PBRAppearance {{'
        f' baseColorMap ImageTexture {{ url [ "textures/real_tag36h11_id0.png" ] }}'
        f' roughness 1 metalness 0 }}'
        f' geometry Box {{ size {TAG_SIZE} {TAG_SIZE} {TAG_SIZE} }} }} ]'
        f' name "apriltag_{num}" }}'
    )


def _robot_block(n, x, y, phero):
    lines = [
        f'DEF ROBOT{n} E-puck {{',
        f'  translation {x} {y} 0',
        f'  name "robot{n}"',
        f'  controller "epuck_driver"',
        f'  emitter_channel {n}',
        f'  receiver_channel {n}',
        f'  camera_fieldOfView 1',
        f'  camera_width 512',
        f'  camera_height 512',
    ]
    if phero:
        lines += [
            '  turretSlot [',
            '    GPS {',
            '      name "gps"',
            '      accuracy 0',
            '    }',
            '    InertialUnit {',
            '      name "inertial_unit"',
            '    }',
            '    Emitter {',
            '      name "phero_emitter"',
            '      range 0.5',
            '      channel 10',
            '    }',
            '    Receiver {',
            '      name "phero_receiver"',
            '      channel 10',
            '      bufferSize 1024',
            '    }',
            '  ]',
        ]
    lines.append('}')
    return '\n'.join(lines)


def _supervisor_block(n_robots):
    lines = ['Robot {', '  translation 0 0 0', '  children [']
    for i in range(1, n_robots + 1):
        lines.append(f'    Emitter {{ name "emitter{i}" channel {i} }}')
        lines.append(f'    Receiver {{ name "receiver{i}" channel {i} }}')
    lines += [
        '  ]',
        '  name "supervisor"',
        '  controller "<extern>"',
        '  supervisor TRUE',
        '}',
    ]
    return '\n'.join(lines)


def _header(sample, n_robots, n_tags):
    label = f"EVAL 7x7 {n_robots}R Sample {sample}"
    return f"""\
#VRML_SIM R2023b utf8

EXTERNPROTO "https://raw.githubusercontent.com/cyberbotics/webots/R2023b/projects/objects/backgrounds/protos/TexturedBackground.proto"
EXTERNPROTO "https://raw.githubusercontent.com/cyberbotics/webots/R2023b/projects/objects/backgrounds/protos/TexturedBackgroundLight.proto"
EXTERNPROTO "https://raw.githubusercontent.com/cyberbotics/webots/R2023b/projects/robots/gctronic/e-puck/protos/E-puck.proto"

WorldInfo {{
  title "Multi-Agent AprilTag Foraging - {label}"
  basicTimeStep 32
}}
Viewpoint {{
  orientation -0.5773502691896258 0.5773502691896258 0.5773502691896258 2.0944
  position 0 0 14
}}
TexturedBackground {{
}}
TexturedBackgroundLight {{
  castShadows FALSE
}}
DirectionalLight {{
  ambientIntensity 0.5
  direction -1 0 0
  intensity 0.8
}}
DirectionalLight {{
  ambientIntensity 0.5
  direction 1 0 0
  intensity 0.8
}}
DirectionalLight {{
  ambientIntensity 0.5
  direction 0 -1 0
  intensity 0.8
}}
DirectionalLight {{
  ambientIntensity 0.5
  direction 0 1 0
  intensity 0.8
}}
Solid {{
  children [
    Shape {{
      appearance PBRAppearance {{
        baseColor 0.85 0.85 0.85
        roughness 1
        metalness 0
      }}
      geometry Box {{
        size 7 7 0.001
      }}
    }}
  ]
  name "floor"
  boundingObject Box {{
    size 7 7 0.001
  }}
}}

Solid {{
  translation 0 3.5 0.05
  children [
    Shape {{
      appearance PBRAppearance {{ baseColor 0.3 0.3 0.3 roughness 1 metalness 0 }}
      geometry Box {{ size 7.1 0.04 0.1 }}
    }}
  ]
  name "wall_north"
  boundingObject Box {{ size 7.1 0.04 0.1 }}
}}
Solid {{
  translation 0 -3.5 0.05
  children [
    Shape {{
      appearance PBRAppearance {{ baseColor 0.3 0.3 0.3 roughness 1 metalness 0 }}
      geometry Box {{ size 7.1 0.04 0.1 }}
    }}
  ]
  name "wall_south"
  boundingObject Box {{ size 7.1 0.04 0.1 }}
}}
Solid {{
  translation 3.5 0 0.05
  children [
    Shape {{
      appearance PBRAppearance {{ baseColor 0.3 0.3 0.3 roughness 1 metalness 0 }}
      geometry Box {{ size 0.04 7.1 0.1 }}
    }}
  ]
  name "wall_east"
  boundingObject Box {{ size 0.04 7.1 0.1 }}
}}
Solid {{
  translation -3.5 0 0.05
  children [
    Shape {{
      appearance PBRAppearance {{ baseColor 0.3 0.3 0.3 roughness 1 metalness 0 }}
      geometry Box {{ size 0.04 7.1 0.1 }}
    }}
  ]
  name "wall_west"
  boundingObject Box {{ size 0.04 7.1 0.1 }}
}}

# Base Station (Red Cylinder)
DEF BASE_STATION Solid {{
  translation 0 0 0.001
  children [
    Shape {{
      appearance PBRAppearance {{
        baseColor 0.8 0.1 0.1
        roughness 0.5
        metalness 0
        emissiveColor 0.3 0 0
      }}
      geometry Cylinder {{
        height 0.002
        radius 0.1
      }}
    }}
  ]
  name "base_station"
  boundingObject Cylinder {{
    height 0.002
    radius 0.1
  }}
}}

"""


# ── world file generator ──────────────────────────────────────────────────────
def generate_world(sample, key, suffix, n_robots, n_tags, phero):
    n_clusters = n_tags // TAGS_PER_CLUSTER

    # Unique seed per (config, sample) — stable across runs
    seed = {"4r": 1000, "8r": 2000, "12r": 3000, "16r": 4000}[key] + sample
    rng  = random.Random(seed)

    centers = place_clusters(n_clusters, rng)

    parts = [_header(sample, n_robots, n_tags)]
    parts.append(
        f'# ---------------------------------------------------------\n'
        f'# AprilTags ({n_tags} Total) — '
        f'{n_clusters} clusters × {TAGS_PER_CLUSTER} tags (4×4 grid)\n'
        f'# Density: {n_tags/49:.2f} /m²  |  Arena: 7×7 m\n'
        f'# ---------------------------------------------------------\n'
    )

    d = TAG_SPACING
    offsets = [-1.5*d, -0.5*d, 0.5*d, 1.5*d]

    tag_num = 1
    for ci, (cx, cy) in enumerate(centers):
        parts.append(f'# Cluster {ci+1}  center=({cx:.4f}, {cy:.4f})')
        for row in range(CLUSTER_ROWS):
            for col in range(CLUSTER_COLS):
                tx = cx + offsets[col]
                ty = cy + offsets[row]
                parts.append(_tag_line(tag_num, tx, ty))
                tag_num += 1
        parts.append('')

    parts.append('')
    for i, (rx, ry) in enumerate(ROBOT_XY[:n_robots], start=1):
        parts.append(_robot_block(i, rx, ry, phero))
        parts.append('')

    parts.append(_supervisor_block(n_robots))
    parts.append('')

    return '\n'.join(parts)


# ── main ──────────────────────────────────────────────────────────────────────
def main():
    WORLDS_DIR.mkdir(exist_ok=True)
    grand_total = 0

    for key, suffix, n_robots, n_tags, phero in CONFIGS:
        n_clusters = n_tags // TAGS_PER_CLUSTER
        written = 0
        for sample in range(1, 21):
            fname = f"eval_sample{sample}_7x7{suffix}.wbt"
            content = generate_world(sample, key, suffix, n_robots, n_tags, phero)
            (WORLDS_DIR / fname).write_text(content)
            written += 1

        density = n_tags / 49
        print(f"  {key:4s}  {n_robots:2d}r  {n_tags:3d} tags  "
              f"{density:.2f}/m²  {n_clusters} clusters  "
              f"→ eval_sampleN_7x7{suffix}.wbt  ({written} files)")
        grand_total += written

    print(f"\n{grand_total} world files written to {WORLDS_DIR}/")


if __name__ == "__main__":
    main()
