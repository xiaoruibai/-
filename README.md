# ACIC/JOBS Online Transfer Learning Experiments

This repository contains ACIC16 and JOBS semi-synthetic online stream generation code and online transfer learning runners.

## Structure

- `stream_generation_common.py`: shared stream generation utilities.
- `stability_generation_profiles.py`: low-noise stability stream profiles.
- `moderate_generation_profiles.py`: moderate-complexity stream profiles.
- `otl_runner_common.py`: shared DR-only OTL runner with the original no-share weighting design.
- `otl_runner_two_layer_common.py`: shared DR-only OTL runner with discounted two-layer weights.
- `acic16代码/`: ACIC16 data generation and runner entrypoints.
- `jobs代码/`: JOBS data generation and runner entrypoints.

## Main Entry Points

Original no-share OTL:

```powershell
python "acic16代码\双线OTL.py"
python "jobs代码\双线OTL.py"
```

Discounted two-layer OTL:

```powershell
python "acic16代码\双层权重双线OTL.py"
python "jobs代码\双层权重双线OTL.py"
```

Moderate and stability scenarios also have matching entrypoints in each dataset folder.

## Data and Outputs

Generated stream directories and experiment output directories are intentionally ignored by Git. Regenerate them locally with the corresponding data generation scripts when needed.
