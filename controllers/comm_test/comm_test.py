# controllers/comm_test/comm_test.py
# Decentralized foraging (no RL):
# - Explore via RANDOM WAYPOINTS (so robots go far from nest)
# - Detect tag with camera (pupil_apriltags)
# - CLAIM,<robot>,<x>,<y> to supervisor -> supervisor removes nearest APRILTAG_* visually
# - Logical carry 1 at a time
# - Go to nest (0,0), deposit
# - Broadcast HOT,<from>,<x>,<y>,<time> so others search near that location

from controller import Robot
import math
import time
import sys
import random
import numpy as np
from pupil_apriltags import Detector

TIME_STEP = 32
DT = TIME_STEP / 1000.0

WHEEL_RADIUS = 0.0205
AXLE_LENGTH  = 0.052
MAX_W        = 6.28

# ----------- Tunables -----------
PICKUP_AREA_THRESH = 2000.0     # decrease if never triggers, increase if too early
NEST_X, NEST_Y     = 0.0, 0.0
NEST_RADIUS        = 0.10

HOTSPOT_RADIUS     = 0.12
SEARCH_TOTAL_TIME  = 8.0
HOTSPOT_TTL        = 20.0

PICKUP_COOLDOWN    = 1.0        # seconds to avoid double-pick due to visual removal delay

# Arena bounds (your floor is ~3x3 centered at 0,0, so keep inside +/-1.35-ish)
ARENA_X_MIN, ARENA_X_MAX = -1.35, 1.35
ARENA_Y_MIN, ARENA_Y_MAX = -1.35, 1.35

# Key: force exploration AWAY from nest
MIN_WAYPOINT_DIST_FROM_NEST = 0.60   # meters
WAYPOINT_REACHED_RADIUS     = 0.15
WAYPOINT_TIMEOUT            = 12.0   # seconds then pick a new one
# --------------------------------

def wrap_to_pi(a):
    while a > math.pi:
        a -= 2.0 * math.pi
    while a < -math.pi:
        a += 2.0 * math.pi
    return a

def clamp(v, lo, hi):
    return max(lo, min(hi, v))

def get_gray(camera):
    w = camera.getWidth()
    h = camera.getHeight()
    buf = camera.getImage()
    if buf is None:
        return None
    img = np.frombuffer(buf, dtype=np.uint8).reshape((h, w, 4))  # BGRA
    return img[:, :, 0]

def poly_area(corners):
    x = corners[:, 0]
    y = corners[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))

def dist2(a, b):
    dx = a[0] - b[0]
    dy = a[1] - b[1]
    return dx*dx + dy*dy

# -------------------- init --------------------
robot = Robot()
name = robot.getName()
print(f"[START] {name}")
sys.stdout.flush()

emitter  = robot.getDevice("emitter")
receiver = robot.getDevice("receiver")
receiver.enable(TIME_STEP)

left_motor  = robot.getDevice("left wheel motor")
right_motor = robot.getDevice("right wheel motor")
left_motor.setPosition(float("inf"))
right_motor.setPosition(float("inf"))

camera = robot.getDevice("camera")
camera.enable(TIME_STEP)

# optional GPS/Compass (HIGHLY recommended)
gps = None
compass = None
try:
    gps = robot.getDevice("gps")
    gps.enable(TIME_STEP)
except Exception:
    gps = None

try:
    compass = robot.getDevice("compass")
    compass.enable(TIME_STEP)
except Exception:
    compass = None

# distance sensors optional
ds = []
for dn in ["ps0","ps1","ps2","ps3","ps4","ps5","ps6","ps7"]:
    try:
        s = robot.getDevice(dn)
        s.enable(TIME_STEP)
        ds.append(s)
    except Exception:
        ds = []
        break

detector = Detector(families="tag36h11")

# -------------------- pose init --------------------
# fallback if no GPS/Compass
if name == "robot1":
    x_est, y_est, theta = -0.5,  0.0, 0.0
elif name == "robot2":
    x_est, y_est, theta =  0.5,  0.0, 0.0
elif name == "robot3":
    x_est, y_est, theta =  0.0,  0.5, 0.0
else:
    x_est, y_est, theta =  0.0, -0.5, 0.0

# deterministic but different seeds per robot
random.seed(hash(name) & 0xffffffff)

# -------------------- state --------------------
EXPLORE   = "EXPLORE"     # go to random waypoint
GOTO_HOT  = "GOTO_HOT"
SEARCH    = "SEARCH"
DELIVER   = "DELIVER"

state = EXPLORE

carrying = False
pickups  = 0
deposits = 0

hotspots = []  # dicts: {"x":..,"y":..,"t":..,"from":..}

search_timer = 0.0
burst_timer  = 0.0
last_debug   = time.time()
last_pick_time = -1e9

# exploration waypoint
wx, wy = 0.0, 0.0
waypoint_start_time = time.time()

def pose_update_from_gps_compass():
    """If GPS/Compass exists, overwrite x_est,y_est,theta for accurate visual navigation."""
    global x_est, y_est, theta
    if gps is None or compass is None:
        return
    p = gps.getValues()         # [x,y,z] in Webots world coords
    c = compass.getValues()     # 3D vector
    x_est = p[0]
    y_est = p[1]
    # heading: Webots compass points to north; convert to yaw angle
    theta = math.atan2(c[0], c[1])  # works well for flat worlds

def pick_random_waypoint_far_from_nest():
    """Pick a random point in arena, but not too close to nest."""
    global wx, wy, waypoint_start_time
    for _ in range(100):
        x = random.uniform(ARENA_X_MIN, ARENA_X_MAX)
        y = random.uniform(ARENA_Y_MIN, ARENA_Y_MAX)
        if dist2((x, y), (NEST_X, NEST_Y)) >= MIN_WAYPOINT_DIST_FROM_NEST**2:
            wx, wy = x, y
            waypoint_start_time = time.time()
            return
    # fallback (if something odd)
    wx, wy = ARENA_X_MAX * 0.7, ARENA_Y_MAX * 0.7
    waypoint_start_time = time.time()

def go_to(xg, yg):
    dx = xg - x_est
    dy = yg - y_est
    dist = math.sqrt(dx*dx + dy*dy)
    target = math.atan2(dy, dx)
    err = wrap_to_pi(target - theta)

    forward = 4.0
    turn = 2.2 * err
    if abs(err) > 0.9:
        forward = 1.2

    vl = clamp(forward - turn, -MAX_W, MAX_W)
    vr = clamp(forward + turn, -MAX_W, MAX_W)
    return vl, vr, dist

def broadcast_hotspot(hx, hy):
    out = f"HOT,{name},{hx:.3f},{hy:.3f},{time.time():.3f}"
    emitter.send(out.encode("utf-8"))
    print(f"[{name}] BROADCAST {out}")
    sys.stdout.flush()

def send_claim(cx, cy):
    out = f"CLAIM,{name},{cx:.3f},{cy:.3f}"
    emitter.send(out.encode("utf-8"))
    print(f"[{name}] CLAIM at ({cx:.3f},{cy:.3f})")
    sys.stdout.flush()

def prune_hotspots(now):
    global hotspots
    hotspots = [h for h in hotspots if (now - h["t"]) <= HOTSPOT_TTL]

# start with a far waypoint so they immediately spread out
pick_random_waypoint_far_from_nest()

# -------------------- main loop --------------------
while robot.step(TIME_STEP) != -1:
    now = time.time()

    # 1) Use GPS/Compass if available
    pose_update_from_gps_compass()

    # 2) Receive messages
    while receiver.getQueueLength() > 0:
        msg = receiver.getString()
        receiver.nextPacket()

        if msg.startswith("HOT,"):
            parts = msg.split(",")
            if len(parts) >= 5:
                frm = parts[1]
                hx  = float(parts[2])
                hy  = float(parts[3])
                _ts = float(parts[4])

                if frm != name and (not carrying):
                    hotspots.append({"x": hx, "y": hy, "t": now, "from": frm})
                    # go help if free
                    if state == EXPLORE:
                        state = GOTO_HOT
                    print(f"[{name}] RECEIVED HOT from {frm} at ({hx:.2f},{hy:.2f})")
                    sys.stdout.flush()

    prune_hotspots(now)

    # 3) Detection (only when NOT carrying)
    det_n = 0
    if (not carrying) and (now - last_pick_time) > PICKUP_COOLDOWN:
        gray = get_gray(camera)
        if gray is not None:
            dets = detector.detect(gray)
            det_n = len(dets)

            best = None
            best_area = 0.0
            for d in dets:
                a = poly_area(d.corners)
                if a > best_area:
                    best_area = a
                    best = d

            if best is not None and best_area > PICKUP_AREA_THRESH:
                carrying = True
                pickups += 1
                last_pick_time = now

                # claim + remove visually
                send_claim(x_est, y_est)
                broadcast_hotspot(x_est, y_est)

                # go straight to nest
                state = DELIVER

                print(f"[{name}] PICKUP area={best_area:.1f} -> state=DELIVER")
                sys.stdout.flush()

    # 4) Behavior
    vl_cmd = 0.0
    vr_cmd = 0.0

    # obstacle avoid helper
    def obstacle_avoid(vl, vr):
        if not ds:
            return vl, vr
        front = max(ds[0].getValue(), ds[7].getValue(), ds[1].getValue(), ds[6].getValue())
        if front > 80.0:
            return -2.0, 2.0
        return vl, vr

    if state == DELIVER:
        vl_cmd, vr_cmd, dist = go_to(NEST_X, NEST_Y)
        vl_cmd, vr_cmd = obstacle_avoid(vl_cmd, vr_cmd)

        if dist < NEST_RADIUS:
            carrying = False
            deposits += 1
            # IMPORTANT: after deposit, pick a far waypoint so they don't loiter at nest
            pick_random_waypoint_far_from_nest()
            state = EXPLORE
            print(f"[{name}] DEPOSIT at nest -> deposits={deposits} | new waypoint=({wx:.2f},{wy:.2f})")
            sys.stdout.flush()

    elif state == GOTO_HOT and (not carrying):
        if hotspots:
            h = hotspots[-1]
            vl_cmd, vr_cmd, dist = go_to(h["x"], h["y"])
            vl_cmd, vr_cmd = obstacle_avoid(vl_cmd, vr_cmd)

            if dist < HOTSPOT_RADIUS:
                state = SEARCH
                search_timer = 0.0
                burst_timer = 0.0
                print(f"[{name}] REACHED HOTSPOT -> state=SEARCH")
                sys.stdout.flush()
        else:
            state = EXPLORE

    elif state == SEARCH and (not carrying):
        search_timer += DT
        burst_timer += DT

        # search pattern
        avoid = 0.0
        if ds:
            front = max(ds[0].getValue(), ds[7].getValue(), ds[1].getValue(), ds[6].getValue())
            if front > 80.0:
                avoid = 1.0

        if avoid > 0.0:
            vl_cmd, vr_cmd = -2.5, 2.5
        else:
            if burst_timer < 1.2:
                vl_cmd, vr_cmd = 4.0, 4.0
            elif burst_timer < 2.8:
                vl_cmd, vr_cmd = -2.0, 2.0
            else:
                burst_timer = 0.0

        if search_timer > SEARCH_TOTAL_TIME:
            # after searching, go back to exploration waypoint (far away)
            state = EXPLORE
            # refresh waypoint too (keeps them moving outward)
            pick_random_waypoint_far_from_nest()

    else:
        # EXPLORE = go to waypoint
        # If waypoint reached or timed out -> pick a new far waypoint
        if (now - waypoint_start_time) > WAYPOINT_TIMEOUT:
            pick_random_waypoint_far_from_nest()

        vl_cmd, vr_cmd, dist = go_to(wx, wy)
        vl_cmd, vr_cmd = obstacle_avoid(vl_cmd, vr_cmd)

        if dist < WAYPOINT_REACHED_RADIUS:
            pick_random_waypoint_far_from_nest()

    # 5) apply motors
    left_motor.setVelocity(vl_cmd)
    right_motor.setVelocity(vr_cmd)

    # 6) fallback odometry update ONLY if no GPS/Compass (otherwise pose is overwritten)
    if gps is None or compass is None:
        v_l = vl_cmd * WHEEL_RADIUS
        v_r = vr_cmd * WHEEL_RADIUS
        v = (v_r + v_l) / 2.0
        w = (v_r - v_l) / AXLE_LENGTH

        theta = wrap_to_pi(theta + w * DT)
        x_est += v * math.cos(theta) * DT
        y_est += v * math.sin(theta) * DT

    # 7) debug
    if now - last_debug > 2.0:
        print(f"[{name} DEBUG] state={state} x={x_est:.2f} y={y_est:.2f} carrying={carrying} "
              f"pickups={pickups} deposits={deposits} hotspots={len(hotspots)} wp=({wx:.2f},{wy:.2f})")
        print(f"[{name} DETECT] n={det_n}")
        sys.stdout.flush()
        last_debug = now