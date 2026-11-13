## Project specification and experiment artifacts

[project.pdf](project.pdf) contains the project specification.

The repository includes a saved training dataset and selected experiment results:

| Path | Contents |
| --- | --- |
| `training/pilot/data.jsonl` | Teacher-labelled game positions with training, validation, and test split assignments. |
| `training/pilot/metrics.json` | Cycle 72 training metrics, selected parameters, and fitted coefficients. |
| `training/pilot/evaluation/batch-00000000/results.json` | Completed independent evaluation of the cycle 65 checkpoint. |
| `training/pilot/evaluation/batch-00000001/results.json` | Completed independent evaluation of the cycle 69 checkpoint. |
| `benchmarks/tournament.json` | Recorded tournament results. |
| `benchmarks/pretable_tournament.json` | Tournament results for the earlier lookup-table experiment. |
| `benchmarks/matched_timing.json` | Timing measurements on matched positions. |
| `benchmarks/lookup_timing.json` | Lookup-kernel timing measurements. |

These files preserve the recorded experiments; the completed evaluations refer to
cycles 65 and 69, while the saved training metrics refer to cycle 72. The cycle 72
evaluation was still running when these artifacts were selected and is omitted.
Result files retain their original source paths and environment metadata; paths
may refer to the original local filename `project_skeleton.py`.

The saved dataset is an experiment snapshot. Update it when publishing a finished
training run. Runtime inference uses the
coefficients embedded in the program and does not require the training dataset.
The local `progress.json` file is used for resuming training and is not included.
