# Running Foraging Experiments

This project has two experiment runners:

- `run_foraging_experiments.py`: fixed-duration evaluations that measure pickups and deposits within a set simulated time.
- `run_foraging_completion_experiments.py`: 5x5 completion-time evaluations that run until all 64 tags are deposited, or until a safety cap is reached.

Both scripts launch one Webots instance and one extern supervisor controller per sample. Use `--dry-run` first to verify worlds, ports, controller commands, and output paths.

## Fixed-Duration Experiments

Use this script when you want to match the earlier evaluation setup with a fixed foraging time per arena.

```bash
python run_foraging_experiments.py \
  --method cpfa_baseline \
  --arena 5x5
```

```bash
python run_foraging_experiments.py \
  --method centralized_ppo \
  --arena 12x12 \
  --model ppo_cpfa_v9.zip
```

### Common Flags

- `--method {cpfa_baseline,centralized_ppo}`: which controller to evaluate.
- `--arena {5x5,7x7,9x9,12x12}`: arena size and matching world/controller set.
- `--samples 1-10`: sample list. Supports ranges and comma lists, such as `1-5`, `1,3,5`, or `1-3,7`.
- `--foraging-time 10`: simulated minutes to run. If omitted, defaults are `5x5=10`, `7x7=25`, `9x9=50`, `12x12=95`.
- `--model ppo_cpfa_v9.zip`: model file for `centralized_ppo`; ignored by `cpfa_baseline`.
- `--base-port 1438`: first Webots port. Each parallel sample uses the next port.
- `--max-parallel 5`: number of samples to run concurrently. Defaults to all selected samples.
- `--results-csv results/my_run.csv`: custom output CSV path. If omitted, a timestamped CSV is written under `results/`.
- `--webots-bin webots`: Webots executable.
- `--startup-seconds 30`: time allowed for Webots to expose the extern supervisor URL.
- `--controller-start-retries 3`: controller startup retry count.
- `--dry-run`: print commands without launching Webots.

### Output

The fixed-duration CSV columns are:

```text
method,arena,sample,foraging_time_min,pickups,deposits
```

Logs are written next to the CSV in a timestamped `_logs/logs/` directory.

## Completion-Time Distribution Experiments

Use this script when you want the 5x5 completion-time experiment across resource distributions. Each sample runs until all 64 tags are deposited, or until `--max-sim-min` is reached.

```bash
python run_foraging_completion_experiments.py \
  --method cpfa_baseline \
  --distribution clustered
```

```bash
python run_foraging_completion_experiments.py \
  --method centralized_ppo \
  --distribution powerlaw \
  --model ppo_cpfa_v9.zip
```

### Distribution Worlds

- `--distribution clustered`: uses `worlds/eval_sample{N}_5x5.wbt`.
- `--distribution random`: uses `worlds/eval_random{N}_5x5.wbt`.
- `--distribution powerlaw`: uses `worlds/eval_powerlaw{N}_5x5.wbt`.

By default, the runner uses `--samples 1-5`, so each distribution runs five samples.

### Common Flags

- `--method {cpfa_baseline,centralized_ppo}`: which controller to evaluate.
- `--distribution {clustered,random,powerlaw}`: resource distribution to test.
- `--samples 1-5`: sample list. Supports ranges and comma lists.
- `--max-sim-min 120`: simulated-minute safety cap. Completed runs stop earlier.
- `--model ppo_cpfa_v9.zip`: model file for `centralized_ppo`; ignored by `cpfa_baseline`.
- `--base-port 1438`: first Webots port. Each parallel sample uses the next port.
- `--max-parallel 5`: number of samples to run concurrently. Defaults to all selected samples.
- `--results-csv results/my_completion_run.csv`: custom output CSV path. You can also pass a directory, such as `results/completion_results`, and the runner will create a timestamped CSV inside it.
- `--webots-bin webots`: Webots executable.
- `--startup-seconds 30`: time allowed for Webots to expose the extern supervisor URL.
- `--controller-start-retries 3`: controller startup retry count.
- `--dry-run`: print commands without launching Webots.

### Output

The completion-time CSV columns are:

```text
method,distribution,arena,sample,completed,status,completion_time_min,elapsed_time_min,steps,pickups,deposits
```

- `completed=true` and `status=completed`: all 64 tags were deposited.
- `completed=false` and `status=timeout`: the run reached `--max-sim-min` before completion.
- `completion_time_min` is filled only for completed runs.
- `elapsed_time_min` is always filled and equals either completion time or timeout time.

Logs are written next to the CSV in a timestamped `_logs/logs/` directory.

## Example Batch Commands

Run all three distributions for CPFA:

```bash
python run_foraging_completion_experiments.py --method cpfa_baseline --distribution clustered
python run_foraging_completion_experiments.py --method cpfa_baseline --distribution random
python run_foraging_completion_experiments.py --method cpfa_baseline --distribution powerlaw
```

Run all three distributions for centralized PPO:

```bash
python run_foraging_completion_experiments.py --method centralized_ppo --distribution clustered --model ppo_cpfa_v9.zip
python run_foraging_completion_experiments.py --method centralized_ppo --distribution random --model ppo_cpfa_v9.zip
python run_foraging_completion_experiments.py --method centralized_ppo --distribution powerlaw --model ppo_cpfa_v9.zip
```

Check commands before running:

```bash
python run_foraging_experiments.py --method cpfa_baseline --arena 5x5 --dry-run
python run_foraging_completion_experiments.py --method centralized_ppo --distribution random --dry-run
```
