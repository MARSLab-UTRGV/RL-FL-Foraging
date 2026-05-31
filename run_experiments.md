# Run Decentralized Experiments

## Start From The Repo Root

```bash
source venv/bin/activate
```

Both runners use Webots. If `webots` is not on your `PATH`, pass the executable with `--webots-bin`.

Both runners load per-robot models from the project root:

```text
robot1_<run-name>.zip
robot2_<run-name>.zip
robot3_<run-name>.zip
robot4_<run-name>.zip
```

For example, `--run-name decentralized_indep_v9` requires:

```text
robot1_decentralized_indep_v9.zip
robot2_decentralized_indep_v9.zip
robot3_decentralized_indep_v9.zip
robot4_decentralized_indep_v9.zip
```

## Available Distributions

Use one of these values for `--distribution`:

```text
clustered
powerlaw
random
```

The 5x5 world files are selected from the distribution and sample id:

```text
clustered -> worlds/eval_sample<sample>_5x5.wbt
powerlaw  -> worlds/eval_powerlaw<sample>_5x5.wbt
random    -> worlds/eval_random<sample>_5x5.wbt
```

Sample values can be written as:

```text
1
1-5
1,3,7
```

## Completion-Time Experiments

Script:

```bash
python run_decentralized_completion_experiments.py [flags]
```

This runner stops each sample when all tags are deposited or when `--max-sim-min` is reached.

### Completion Examples

Single clustered sample with `decentralized_indep_v9`:

```bash
python run_decentralized_completion_experiments.py \
  --distribution clustered \
  --samples 1 \
  --run-name decentralized_indep_v9 \
  --max-parallel 1
```

Single random sample with a 60 simulated-minute cap:

```bash
python run_decentralized_completion_experiments.py \
  --distribution random \
  --samples 3 \
  --run-name decentralized_indep_v9 \
  --max-sim-min 60 \
  --max-parallel 1
```

Powerlaw samples 1 through 5, two Webots jobs at a time:

```bash
python run_decentralized_completion_experiments.py \
  --distribution powerlaw \
  --samples 1-5 \
  --run-name decentralized_indep_v9 \
  --max-sim-min 120 \
  --max-parallel 2
```

Clustered non-contiguous samples:

```bash
python run_decentralized_completion_experiments.py \
  --distribution clustered \
  --samples 1,4,7 \
  --run-name decentralized_indep_v9 \
  --max-parallel 3
```

Write results into a specific directory:

```bash
python run_decentralized_completion_experiments.py \
  --distribution random \
  --samples 1-10 \
  --run-name decentralized_indep_v9 \
  --results-csv results/experiment2/decentralized_completion \
  --max-parallel 4
```

Write results to an exact CSV path:

```bash
python run_decentralized_completion_experiments.py \
  --distribution clustered \
  --samples 1-5 \
  --run-name decentralized_indep_v9 \
  --results-csv results/experiment2/decentralized_completion_clustered_v9.csv \
  --max-parallel 2
```

Use a specific Webots executable:

```bash
python run_decentralized_completion_experiments.py \
  --distribution powerlaw \
  --samples 1 \
  --run-name decentralized_indep_v9 \
  --webots-bin /usr/local/webots/webots \
  --max-parallel 1
```

Use a different model checkpoint set:

```bash
python run_decentralized_completion_experiments.py \
  --distribution clustered \
  --samples 1 \
  --run-name decentralized_indep_v9_4000000_steps \
  --max-parallel 1
```

Preview commands without launching Webots:

```bash
python run_decentralized_completion_experiments.py \
  --distribution clustered \
  --samples 1 \
  --run-name decentralized_indep_v9 \
  --dry-run
```

### Completion Flags

| Flag | Required | Default | Values | Description |
| --- | --- | --- | --- | --- |
| `--distribution` | Yes | None | `clustered`, `powerlaw`, `random` | Selects which world-file family to run. |
| `--samples` | No | `1-5` | `1`, `1-5`, `1,3,7` | Sample ids to run. |
| `--run-name` | No | `decentralized_indep_v9` | Any run name with matching `robotN_<run-name>.zip` files | Per-robot model run name. |
| `--max-sim-min` | No | `120` | Positive float | Simulated-minute cap per sample. |
| `--base-port` | No | `1438` | Integer | First Webots port. Parallel samples use sequential ports. |
| `--max-parallel` | No | Number of samples | Positive integer | Maximum Webots/controller jobs to run at once. |
| `--results-csv` | No | Timestamped CSV under `results/` | File path or directory path | Output CSV location. Directory paths get a timestamped file inside. Exact file paths are appended to if they already exist. |
| `--webots-bin` | No | `webots` | Executable name or path | Webots executable to launch. |
| `--startup-seconds` | No | `30` | Positive float | Seconds to wait for the Webots extern controller URL. |
| `--controller-start-retries` | No | `3` | Positive integer | Startup retry count for the extern controller process. |
| `--dry-run` | No | Off | Flag only | Prints the Webots and controller commands without running them. |

## Fixed-Time Foraging Experiments

Script:

```bash
python run_decentralized_foraging_time_experiments.py [flags]
```

This runner runs each sample for a fixed simulated duration. Completion stop is disabled, so the sample runs until the requested duration, then records whether all tags were deposited by the end.

### Foraging-Time Examples

Single clustered sample for 30 simulated minutes:

```bash
python run_decentralized_foraging_time_experiments.py \
  --distribution clustered \
  --samples 1 \
  --run-name decentralized_indep_v9 \
  --foraging-time-min 30 \
  --max-parallel 1
```

Random samples 1 through 10 for 15 simulated minutes, four jobs at a time:

```bash
python run_decentralized_foraging_time_experiments.py \
  --distribution random \
  --samples 1-10 \
  --run-name decentralized_indep_v9 \
  --foraging-time-min 15 \
  --max-parallel 4
```

Powerlaw non-contiguous samples for 45 simulated minutes:

```bash
python run_decentralized_foraging_time_experiments.py \
  --distribution powerlaw \
  --samples 2,5,8 \
  --run-name decentralized_indep_v9 \
  --foraging-time-min 45 \
  --max-parallel 3
```

Use the `--duration` alias instead of `--foraging-time-min`:

```bash
python run_decentralized_foraging_time_experiments.py \
  --distribution clustered \
  --samples 1-5 \
  --run-name decentralized_indep_v9 \
  --duration 15 \
  --max-parallel 2
```

Write fixed-time results to a specific directory:

```bash
python run_decentralized_foraging_time_experiments.py \
  --distribution random \
  --samples 1-10 \
  --run-name decentralized_indep_v9 \
  --foraging-time-min 30 \
  --results-csv results/experiment2/decentralized_foraging_time \
  --max-parallel 4
```

Write fixed-time results to an exact CSV path:

```bash
python run_decentralized_foraging_time_experiments.py \
  --distribution clustered \
  --samples 1-5 \
  --run-name decentralized_indep_v9 \
  --foraging-time-min 15 \
  --results-csv results/experiment2/decentralized_foraging_time_clustered_v9.csv \
  --max-parallel 2
```

Use a different base port:

```bash
python run_decentralized_foraging_time_experiments.py \
  --distribution powerlaw \
  --samples 1-3 \
  --run-name decentralized_indep_v9 \
  --foraging-time-min 30 \
  --base-port 1500 \
  --max-parallel 3
```

Increase startup wait time and controller retries:

```bash
python run_decentralized_foraging_time_experiments.py \
  --distribution clustered \
  --samples 1 \
  --run-name decentralized_indep_v9 \
  --foraging-time-min 30 \
  --startup-seconds 60 \
  --controller-start-retries 5 \
  --max-parallel 1
```

Preview commands without launching Webots:

```bash
python run_decentralized_foraging_time_experiments.py \
  --distribution random \
  --samples 1 \
  --run-name decentralized_indep_v9 \
  --foraging-time-min 30 \
  --dry-run
```

### Foraging-Time Flags

| Flag | Required | Default | Values | Description |
| --- | --- | --- | --- | --- |
| `--distribution` | Yes | None | `clustered`, `powerlaw`, `random` | Selects which world-file family to run. |
| `--samples` | No | `1-10` | `1`, `1-5`, `1,3,7` | Sample ids to run. |
| `--run-name` | No | `decentralized_indep_v9` | Any run name with matching `robotN_<run-name>.zip` files | Per-robot model run name. |
| `--foraging-time-min` | No | `30` | Positive float | Fixed simulated-minute horizon per sample. |
| `--duration-sim-min` | No | Alias for `--foraging-time-min` | Positive float | Same as `--foraging-time-min`. |
| `--duration` | No | Alias for `--foraging-time-min` | Positive float | Same as `--foraging-time-min`. |
| `--base-port` | No | `1438` | Integer | First Webots port. Parallel samples use sequential ports. |
| `--max-parallel` | No | Number of samples | Positive integer | Maximum Webots/controller jobs to run at once. |
| `--results-csv` | No | Timestamped CSV under `results/` | File path or directory path | Output CSV location. Directory paths get a timestamped file inside. Exact file paths are appended to if they already exist. |
| `--webots-bin` | No | `webots` | Executable name or path | Webots executable to launch. |
| `--startup-seconds` | No | `30` | Positive float | Seconds to wait for the Webots extern controller URL. |
| `--controller-start-retries` | No | `3` | Positive integer | Startup retry count for the extern controller process. |
| `--dry-run` | No | Off | Flag only | Prints the Webots and controller commands without running them. |

## Output Locations

Default completion CSV:

```text
results/decentralized_completion_<run-name>_<distribution>_5x5_<timestamp>.csv
```

Default fixed-time foraging CSV:

```text
results/decentralized_foraging_time_<run-name>_<distribution>_5x5_<minutes>min_<timestamp>.csv
```

Default logs directory:

```text
<results-csv-parent>/<results-csv-stem>_logs/logs/
```

## Common Commands

Run all completion distributions for samples 1 through 5:

```bash
python run_decentralized_completion_experiments.py --distribution clustered --samples 1-5 --run-name decentralized_indep_v9 --max-parallel 3
python run_decentralized_completion_experiments.py --distribution powerlaw  --samples 1-5 --run-name decentralized_indep_v9 --max-parallel 3
python run_decentralized_completion_experiments.py --distribution random    --samples 1-5 --run-name decentralized_indep_v9 --max-parallel 3
```

Run all fixed-time foraging distributions for 15 simulated minutes:

```bash
python run_decentralized_foraging_time_experiments.py --distribution clustered --samples 1-10 --run-name decentralized_indep_v9 --foraging-time-min 15 --max-parallel 4
python run_decentralized_foraging_time_experiments.py --distribution powerlaw  --samples 1-10 --run-name decentralized_indep_v9 --foraging-time-min 15 --max-parallel 4
python run_decentralized_foraging_time_experiments.py --distribution random    --samples 1-10 --run-name decentralized_indep_v9 --foraging-time-min 15 --max-parallel 4
```
