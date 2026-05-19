# Testing Guide — CPFA-RL Centralized (CoRL 2026)

Evaluate the trained PPO model and compare against the hand-coded CPFA baseline.
Both systems run in the same world with the same fixed tag layout. Any performance
difference is attributable solely to learned vs. hand-coded navigation strategy.

---

## Prerequisites

```bash
export WEBOTS_HOME=/usr/local/webots
export PYTHONPATH=$PYTHONPATH:$WEBOTS_HOME/lib/controller/python
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$WEBOTS_HOME/lib/controller
```

---

## Files Overview

| File | Role |
|------|------|
| `worlds/eval_best_5x5.wbt` | Shared evaluation world — 5×5m, 4 robots, 64 tags at fixed positions |
| `controllers/eval_best_model/eval_best_model_5x5.py` | Extern eval supervisor — loads PPO model, deterministic inference |
| `controllers/cpfa_baseline/cpfa_baseline.py` | Extern CPFA supervisor — hand-coded state machine, no PPO |
| `controllers/epuck_driver/epuck_driver.py` | Robot controller — shared by both, unchanged |

---

## Evaluating the Trained PPO Model

**Step 1: Launch Webots with the eval world (separate instance from training)**

```bash
webots worlds/eval_best_5x5.wbt &
sleep 10
```

**Step 2: Run the eval supervisor** (use port 1235 if training is running on 1234)

```bash
WEBOTS_PORT=1235 python3 controllers/eval_best_model/eval_best_model_5x5.py \
    logs/ppo_cpfa_v5/ppo_cpfa_v5_4000000_steps
```

Pass any checkpoint without `.zip` extension. Default path if none given:
`logs/ppo_cpfa_v5/ppo_cpfa_v5_200000_steps`

The model runs with `deterministic=True` — no action sampling noise during eval.

### Observation Space (must match training exactly)

```
18 values per robot × 4 robots = 72 total

[0:8]   proximity sensors (÷ 4096)
[8]     carrying
[9]     dist_to_base_norm    (÷ 3.5m)
[10]    angle_to_base_norm   (÷ π)
[11]    site_known
[12]    site_dist_norm       (÷ 3.5m)
[13]    site_angle_norm      (÷ π)
[14]    phero_known
[15]    phero_dist_norm      (÷ 3.5m)
[16]    phero_angle_norm     (÷ π)
[17]    search_duration_norm (steps_without_pickup ÷ 4000)

obs[11–17] zeroed when carrying=True
```

Tag sensing is **not** included — removed to ensure a fair comparison with CPFA baseline
which has no equivalent sensing advantage.

### Eval Log Format

Logged to console and `eval_cpfa_log.txt` every 500 steps:

```
======================================================================
Step 3500 (1.9 min) | Pickups: 3 | Deposits: 3 | Rate: 1.61 tags/min
phero_entries=3 phero_max=0.994
======================================================================
R1[BASE_ESC  ]: L=+1.00 R=+0.00 | carry=0 | base=0.07 ba=+0.17 |
               site=1 sd=0.04 sa=+0.83 | phero=0 pd=0.00 pa=+0.00 | srch=0.00 | wall=2.29
R2[SITE      ]: L=-0.95 R=-0.19 | carry=0 | base=0.08 ba=+0.78 |
               site=1 sd=0.09 sa=+0.05 | phero=0 pd=0.00 pa=+0.00 | srch=0.08 | wall=2.24
R3[GIVE_UP   ]: L=+1.00 R=+0.88 | carry=0 | base=0.08 ba=-0.05 |
               site=0 sd=0.00 sa=+0.00 | phero=0 pd=0.00 pa=+0.00 | srch=0.76 | wall=2.20
R4[BASE_ESC  ]: L=+1.00 R=+0.00 | carry=0 | base=0.07 ba=+0.09 |
               site=0 sd=0.00 sa=+0.00 | phero=1 pd=0.09 pa=+0.41 | srch=0.10 | wall=2.27
```

**Column meanings:**
- `L=, R=` — left/right motor command sent to robot (overridden action, not raw PPO output)
- `carry` — 1 if carrying food
- `base=` — normalised distance to nest (÷ 3.5m)
- `ba=` — signed angle to nest (÷ π)
- `site=` / `phero=` — 1 if target assigned
- `sd=, pd=` — normalised distance to site/phero target
- `sa=, pa=` — signed angle to site/phero target
- `srch=` — search_duration_norm (saturates at 1.0 near give-up)
- `wall=` — absolute distance to nearest wall face

**MODE values:**
- `WALL_ESC` — P1 override active (wall proximity or obstacle)
- `BASE_ESC` — BASE_ESC override (not carrying, not gave_up, dist < 0.25m)
- `RTB` — P2 override (carrying, returning to nest)
- `GIVE_UP` — gave_up=True, P2 steering robot home empty-handed
- `SITE` — PPO navigating toward site fidelity target
- `PHERO` — PPO navigating toward pheromone target
- `EXPLORE` — PPO in free exploration (no target assigned)

### Healthy Behaviour Signs

- Actions are varied small floats (e.g. `+0.12, +0.07`) — NOT `+1.00 / -1.00` oscillation
- Robots cycle: `SITE`/`PHERO` → (deposit) → `BASE_ESC` → `SITE`/`PHERO`
- `phero_entries` grows after first few deposits
- `srch=` approaching 1.0 triggers give-up return (robots don't stay lost forever)
- Rate should be stable or improving — not declining toward 0 after step 5000

### Warning Signs

- `L=+1.00 R=-1.00` or `L=-1.00 R=+1.00` repeating — entropy-diverged model (std too large)
- All robots in `EXPLORE` with `base=0.07–0.15` — robots stuck near nest, model undertrained
- Rate declining steeply after step 3000 — nearby clusters depleted, model not navigating farther

### Meaningful Eval Checkpoints

Early checkpoints (< 1M steps) show Phase 1 behaviour only — robots find nearby clusters
but don't navigate confidently to farther ones. Meaningful comparison against CPFA requires
training through Phase 3 or Phase 4:

| Checkpoint | Training phase | What to expect |
|------------|---------------|----------------|
| 600K steps | Phase 1 (~ep 37) | Basic pickup/deposit at 0.4–1.2m range |
| 1.5M steps | Phase 2 (~ep 92) | Learning navigation at 1.6m |
| 3M steps | Phase 3 (~ep 183) | Developing mid-range navigation |
| 6M+ steps | Phase 4 (~ep 366+) | Full arena, meaningful CPFA comparison |

---

## Running the CPFA Baseline

**Step 1: Reload the eval world to reset all 64 tags**

```bash
webots worlds/eval_best_5x5.wbt &
sleep 10
```

**Step 2: Run the CPFA supervisor** (no model path needed)

```bash
WEBOTS_PORT=1235 python3 controllers/cpfa_baseline/cpfa_baseline.py
```

### CPFA Parameters (matched to RL training supervisor)

| Parameter | Value |
|-----------|-------|
| `RATE_OF_LAYING_PHEROMONE` | 3.0 |
| `RATE_OF_SITE_FIDELITY` | 1.376 (ARGoS-evolved) |
| `RATE_OF_PHEROMONE_DECAY` | 0.05 /sec (τ ≈ 20s) |
| `ProbabilityOfReturningToNest` | 0.0189 per 5-second check |
| Local search | CRW (Correlated Random Walk), 30° Gaussian heading variation |
| `RATE_OF_INFORMED_SEARCH_DECAY` | 0.0002 /step |

**CPFA MODE values in log:**
- `WALL_ESC` — collision avoidance
- `RTB` — carrying, returning to nest
- `GIVE_UP` — returning empty (give-up triggered)
- `SITE` — DEPARTING toward site fidelity target
- `PHERO` — DEPARTING toward pheromone roulette target
- `SEARCH` — CRW random walk (uninformed or informed)

---

## Comparing the Two Systems

Both supervisors run in `eval_best_5x5.wbt` with the **same fixed tag positions**.
Primary metric: **tags deposited per simulated minute** (printed every 500 steps).

| Component | CPFA Baseline | PPO-CPFA v5 |
|-----------|--------------|-------------|
| Pheromone model (list, Poisson CDF, roulette-wheel) | hand-coded | **identical** |
| Site fidelity (Poisson CDF priority) | hand-coded | **identical** |
| Pheromone decay (τ ≈ 20s) | hand-coded | **identical** |
| Nest-only information timing | yes | **identical** |
| Give-up mechanism (P=0.0189 per 5s) | hand-coded | **identical** |
| BASE_ESC (steer away, dist < 0.25m) | hand-coded | **identical** |
| P1 wall escape | hard override | **identical** |
| P2 RTB when carrying | hard override | **identical** |
| DEPARTING to site/phero target | deterministic heading | **PPO (learned)** |
| Local search at cluster | CRW random walk | **PPO (learned)** |
| Tag seek | none (removed) | **none (removed)** |
| Exploration strategy | CRW with informed decay | **PPO (learned)** |

The pheromone infrastructure is held constant. Performance difference = learned vs. hand-coded
navigation strategy. This is the paper's core claim.

### Procedure for Paper Results

1. Run 5+ trials of each system on the same eval world (reload Webots between trials)
2. Each trial: let run for 30 simulated minutes (≈28,125 steps at 64ms/step)
3. Record total deposits per trial → compute tags/min
4. Report mean ± std for both systems

---

## Headless Evaluation

```bash
webots --mode=fast --minimize --no-rendering worlds/eval_best_5x5.wbt &
sleep 10
WEBOTS_PORT=1235 python3 controllers/eval_best_model/eval_best_model_5x5.py \
    logs/ppo_cpfa_v5/ppo_cpfa_v5_6000000_steps
```

---

## Troubleshooting

**`[ERROR]` on model load mentioning obs size mismatch** — Model must match `obs_per_robot=18`
(72D total). Do not load v3 or earlier checkpoints trained with 21D (84D total).

**Port conflict** — If training is on port 1234, run eval on port 1235 with `WEBOTS_PORT=1235`.

**Robots not moving** — Press Play in Webots before running the supervisor.

**Rate drops to 0 after several minutes** — All 64 tags collected. Reload Webots to reset.

**CPFA robots repeatedly exploring depleted areas** — Normal. Give-up will redirect them.
Expected behaviour after dense cluster depletion.

**`numpy.dtype size changed`** — Run `pip install "numpy<2"`.