# BEV and Residual Evaluation Core

Core code for BEV reference planner modeling, training, open-loop validation, and residual analysis. Data, images, checkpoints, sample manifests, and output directories are supplied through runtime arguments, with no default data paths or environment-variable fallbacks.

## Installation and Entry Points

Requires Python 3.10+. Install `requirements.txt` and ensure that PyTorch and torchvision versions are compatible. The variables below represent file locations you supply at runtime; the code does not automatically read these environment variables.

```bash
python -m pip install -r requirements.txt

# Build the index, check calibration, and split the data without training.
python -m bev.train --data-json "$DATA_JSON" --images-root "$IMAGES_ROOT" --output-dir "$INDEX_OUTPUT" --dry-run-index

# Train using an empty output directory. Existing results are never deleted automatically.
python -m bev.train --data-json "$DATA_JSON" --images-root "$IMAGES_ROOT" --output-dir "$TRAIN_OUTPUT"

# Validate using a sample manifest exported during training.
python -m bev.evaluate --data-json "$DATA_JSON" --images-root "$IMAGES_ROOT" --checkpoint "$CHECKPOINT" --manifest "$VALIDATION_MANIFEST" --output-dir "$EVALUATION_OUTPUT"

# Analyze aligned trajectories or residuals.jsonl exported during validation.
python -m residual.analyze --input "$TRAJECTORIES_JSONL" --output-dir "$RESIDUAL_OUTPUT"
```

Run modules from the project root. To run from another directory, add this project to the Python module search path. Use `--help` for the complete arguments. Training and validation both support `--device`, `--batch-size`, and `--num-workers`.

## BEV Model and Data

`bev/` provides shared implementations of the model, calibration, data loading, and temporal splitting. It retains the ResNet50 backbone, BEVFormer-style encoder, trajectory history encoder, and fusion decoder, along with existing checkpoint parameter names and camera embedding compatibility logic. Default inputs use 6 seconds of history and a 5-second future horizon at 10 Hz, with three exterior camera slots. Images are resized to 450×800 and padded to a height of 480.

A single data source is a JSON array. Each record contains `framegroup_id`, a millisecond `timestamp`, `vehicle_name`, `driver_mode`, `frames`, and `trajectory`. Camera records contain `camera_type`, a relative `image_name`, and `camera_calibration` (3×3 `intrinsic`, 4×4 `extrinsic`, and `distortion`). Trajectory points contain a millisecond `timestamp` and meter-based `x_m` and `y_m`, covering 6 seconds of history through 5 seconds into the future. Coordinates use the current ego position as the origin, with x forward and y left.

For multiple data sources, use `--sources` instead of `--data-json` and `--images-root`. The source manifest is a JSON array. Each entry requires `key`, `data_json`, and `images_root`, with optional `vehicle_id`, `vehicle_name`, and `platform_label`. Relative file locations are resolved against the directory containing the source manifest. There are no predefined vehicle IDs or directory mappings. Sample manifests retain `source_key` to distinguish frame IDs from different sources.

By default, training creates a shared HV model and an AV model for each vehicle. Select a subset with `--run-mode hv|av|all`. The original trainer's HV checkpoint initialization remains enabled by default; `--no-hv-warmstart` disables it, while `--av-init-checkpoint` explicitly selects an initialization checkpoint. These two options cannot be combined. Unknown and non-AV modes remain in the original HV group. Training defaults are 30 epochs, batch size 2, AdamW, and a learning rate of 3e-4. Settings are centralized in `bev/config.py`.

## Validation and Residuals

Validation requires a checkpoint and at least one sample manifest. Repeat `--manifest` to merge multiple splits; each sample is evaluated once. Metrics include ADE/FDE at 1, 3, and 5 seconds, 5-second miss rate (endpoint error greater than 2 meters), heading error, and speed error. Validation also exports predicted and executed trajectories, pointwise residuals, and statistics by split. Outputs are `metrics.json`, `residuals.jsonl`, and `residual_fits.csv`; no plots are generated.

`residual/` depends only on NumPy. Its standalone input is JSONL. Each record contains an arbitrary nonempty `platform` label, an optional unique `sample_id`, and either:

- Aligned trajectories: `past_xy` has shape 60×2; `reference_xy` and `executed_xy` each have shape 50×2. History timestamps range from -6.0 to -0.1 seconds, and future timestamps range from +0.1 to +5.0 seconds. Coordinates are in meters.
- Precomputed residuals: 50 entries in `signed_l2_residual`, with an optional boolean `signed_l2_valid_mask`. Invalid values use JSON null. BEV validation outputs can be read directly.

Scenario fields are `driving_state_group`, `lighting_group`, and `area_group`. The categories Cruise, Left turn, Right turn, Day, Night, Urban, and Highway use independent filters; night variants are merged. Missing scenario fields exclude the sample from the corresponding categories. No fixed platform list or cohort size is required. In BEV data, scenario values come from `driving_state`, `lighting`, and `area_type` within `scenario_annotation`. Platform labels use explicit labels when provided, otherwise HV or the vehicle name.

The signed residual is the magnitude of the reference-minus-executed position difference. Its sign comes from the cross product of the executed trajectory tangent and that difference: positive on the left, negative on the right, and positive when the cross product is zero. Degenerate tangent lookup and invalid-point masking retain their original behavior. Gaussian fitting retains the original fitting windows, deterministic sampling, and support criteria. Cells with at least 20 samples and 1,000 valid points are marked as supported; low-support fits remain explicitly labeled. `sigma` is the standard deviation, and `variance` is its square.

## Submit to GitHub

Use `scripts/git_submit_current.sh` to commit all non-ignored project changes and push to `main` in `KITE-Lab/PAVE-Residual-Benchmark`. GitHub SSH access is required. The script initializes Git and configures `origin` on its first run; an existing repository must use the expected remote and the `main` branch.

```bash
# Preview without initializing Git, committing, or pushing.
./scripts/git_submit_current.sh --dry-run

# Submit with a custom commit message.
./scripts/git_submit_current.sh "update residual evaluation"

# Submit with an automatically generated, timestamped commit message.
./scripts/git_submit_current.sh

# Show usage.
./scripts/git_submit_current.sh --help
```

When changes are present, the script stages them with `git add -A`, commits, rebases against `origin/main` if that branch exists, and pushes with upstream tracking. Resolve any reported conflicts before submitting again. If there are no local changes, the script exits without pushing.
