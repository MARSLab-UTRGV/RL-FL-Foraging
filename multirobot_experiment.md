# Multirobot 7x7 Experiment

This experiment evaluates foraging performance across 4, 8, 12, and 16 robots in the 7x7 arena. It uses the worlds:

```text
worlds/eval_sample{N}_7x7_{R}r.wbt
```

where `N` is sample `1-10` and `R` is robot count `4,8,12,16`.

The runner is:

```bash
python run_multirobot_foraging_experiments.py
```

## Defaults

By default, the runner evaluates all 40 jobs:

```text
robot counts: 4, 8, 12, 16
samples:      1-10
active tags:  4r=32, 8r=64, 12r=128, 16r=208
arena:        7x7
```

`--foraging-time` is required so the run cannot accidentally use an old arena default.

## Dry Runs

Use dry-runs first to verify world paths, ports, active tag counts, and controller commands.

CPFA baseline:

```bash
python run_multirobot_foraging_experiments.py \
  --method cpfa_baseline \
  --foraging-time 25 \
  --dry-run
```

Centralized PPO:

```bash
python run_multirobot_foraging_experiments.py \
  --method centralized_ppo \
  --model ppo_cpfa_v9.zip \
  --foraging-time 25 \
  --dry-run
```

Each dry-run should print `Jobs: 40`.

## Full Runs

Run CPFA baseline:

```bash
python run_multirobot_foraging_experiments.py \
  --method cpfa_baseline \
  --foraging-time 25 \
  --max-parallel 4 \
  --results-csv results/multirobot_cpfa_baseline_7x7.csv
```

Run centralized PPO:

```bash
python run_multirobot_foraging_experiments.py \
  --method centralized_ppo \
  --model ppo_cpfa_v9.zip \
  --foraging-time 25 \
  --max-parallel 4 \
  --results-csv results/multirobot_centralized_ppo_7x7.csv
```

Run both methods:

```bash
python run_multirobot_foraging_experiments.py \
  --method cpfa_baseline \
  --foraging-time 25 \
  --max-parallel 4 \
  --results-csv results/multirobot_cpfa_baseline_7x7.csv

python run_multirobot_foraging_experiments.py \
  --method centralized_ppo \
  --model ppo_cpfa_v9.zip \
  --foraging-time 25 \
  --max-parallel 4 \
  --results-csv results/multirobot_centralized_ppo_7x7.csv
```

## One CSV Per Robot Count

Use these commands when you want one CSV per robot size and method. Each command runs all 10 samples for one robot count with 10 simulated minutes.

CPFA baseline:

```bash
python run_multirobot_foraging_experiments.py \
  --method cpfa_baseline \
  --robot-counts 4 \
  --samples 1-10 \
  --foraging-time 10 \
  --tag-counts 4:32,8:64,12:128,16:208 \
  --max-parallel 4 \
  --results-csv results/multirobot_cpfa_baseline_4r_10min.csv

python run_multirobot_foraging_experiments.py \
  --method cpfa_baseline \
  --robot-counts 8 \
  --samples 1-10 \
  --foraging-time 10 \
  --tag-counts 4:32,8:64,12:128,16:208 \
  --max-parallel 4 \
  --results-csv results/multirobot_cpfa_baseline_8r_10min.csv

python run_multirobot_foraging_experiments.py \
  --method cpfa_baseline \
  --robot-counts 12 \
  --samples 1-10 \
  --foraging-time 10 \
  --tag-counts 4:32,8:64,12:128,16:208 \
  --max-parallel 4 \
  --results-csv results/multirobot_cpfa_baseline_12r_10min.csv

python run_multirobot_foraging_experiments.py \
  --method cpfa_baseline \
  --robot-counts 16 \
  --samples 1-10 \
  --foraging-time 10 \
  --tag-counts 4:32,8:64,12:128,16:208 \
  --max-parallel 4 \
  --results-csv results/multirobot_cpfa_baseline_16r_10min.csv
```

Centralized PPO:

```bash
python run_multirobot_foraging_experiments.py \
  --method centralized_ppo \
  --model ppo_cpfa_v9.zip \
  --robot-counts 4 \
  --samples 1-10 \
  --foraging-time 10 \
  --tag-counts 4:32,8:64,12:128,16:208 \
  --max-parallel 4 \
  --results-csv results/multirobot_centralized_ppo_4r_10min.csv

python run_multirobot_foraging_experiments.py \
  --method centralized_ppo \
  --model ppo_cpfa_v9.zip \
  --robot-counts 8 \
  --samples 1-10 \
  --foraging-time 10 \
  --tag-counts 4:32,8:64,12:128,16:208 \
  --max-parallel 4 \
  --results-csv results/multirobot_centralized_ppo_8r_10min.csv

python run_multirobot_foraging_experiments.py \
  --method centralized_ppo \
  --model ppo_cpfa_v9.zip \
  --robot-counts 12 \
  --samples 1-10 \
  --foraging-time 10 \
  --tag-counts 4:32,8:64,12:128,16:208 \
  --max-parallel 4 \
  --results-csv results/multirobot_centralized_ppo_12r_10min.csv

python run_multirobot_foraging_experiments.py \
  --method centralized_ppo \
  --model ppo_cpfa_v9.zip \
  --robot-counts 16 \
  --samples 1-10 \
  --foraging-time 10 \
  --tag-counts 4:32,8:64,12:128,16:208 \
  --max-parallel 4 \
  --results-csv results/multirobot_centralized_ppo_16r_10min.csv
```

## Smoke Tests

Use short smoke tests before launching the full 40-job batches.

CPFA, one 4-robot sample:

```bash
python run_multirobot_foraging_experiments.py \
  --method cpfa_baseline \
  --robot-counts 4 \
  --samples 1 \
  --foraging-time 0.1 \
  --max-parallel 1
```

PPO, one 8-robot sample:

```bash
python run_multirobot_foraging_experiments.py \
  --method centralized_ppo \
  --model ppo_cpfa_v9.zip \
  --robot-counts 8 \
  --samples 1 \
  --foraging-time 0.1 \
  --max-parallel 1
```

## Useful Subsets

Run only 16 robots:

```bash
python run_multirobot_foraging_experiments.py \
  --method cpfa_baseline \
  --robot-counts 16 \
  --samples 1-10 \
  --foraging-time 25
```

Run selected samples:

```bash
python run_multirobot_foraging_experiments.py \
  --method centralized_ppo \
  --model ppo_cpfa_v9.zip \
  --robot-counts 4,8,12,16 \
  --samples 1,3,5 \
  --foraging-time 25
```

Override active tag counts:

```bash
python run_multirobot_foraging_experiments.py \
  --method cpfa_baseline \
  --foraging-time 25 \
  --tag-counts 4:32,8:64,12:128,16:208
```

## Output

The CSV columns are:

```text
method,arena,robots,sample,active_tags,foraging_time_min,pickups,deposits,deposit_fraction
```

Logs are written next to the CSV in a matching `_logs/logs/` directory.

## PPO Multirobot Behavior

The PPO model was trained for 4 robots, with 72 observations and 8 actions. For 8, 12, and 16 robots, the evaluator slices the full robot observation into groups of 4, runs the same PPO policy once per group, concatenates the actions, and applies the shared supervisor logic.

The pheromone list, pickup/deposit metrics, and active resource pool are shared across all robots in the world.
