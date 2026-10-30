"""Validate a checkpoint on explicit sample manifests and export trajectory residuals."""

from __future__ import annotations
import argparse
import json
from contextlib import nullcontext
from pathlib import Path
import numpy as np
import torch
from . import config
from .data import build_dataset_index, create_dataloader, move_batch_to_device
from .model import BEVPlanner, load_model_checkpoint_compat
from .sources import add_source_arguments, sources_from_args
from .metrics import sample_metrics, summarize
from .utils import choose_device, configure_loader, prepare_output, save_json, log
from residual.core import compute_signed_l2_residuals
from residual.statistics import (
    ResidualDataset,
    fit_dataset,
    write_fits,
    normalize_driving_state,
    normalize_lighting_group,
    normalize_area_group,
)


def sample_identity(sample):
    return (
        sample.source_key,
        sample.framegroup_id,
        sample.timestamp_ms,
        sample.vehicle_name,
        sample.driver_mode,
    )


def match_manifests(samples, manifest_paths):
    lookup = {}
    for sample in samples:
        key = (
            sample.framegroup_id,
            sample.timestamp_ms,
            sample.vehicle_name,
            sample.driver_mode,
        )
        lookup.setdefault(key, []).append(sample)
    selected, memberships = {}, {}
    for manifest in manifest_paths:
        with Path(manifest).open(encoding="utf-8") as handle:
            rows = json.load(handle)
        if not isinstance(rows, list):
            raise ValueError("A sample manifest must contain a JSON array")
        for row in rows:
            if not isinstance(row, dict):
                raise ValueError("Each manifest row must be an object")
            key = (
                int(row["framegroup_id"]),
                int(row["timestamp_ms"]),
                str(row["vehicle_name"]),
                str(row["driver_mode"]),
            )
            matches = lookup.get(key, [])
            if "source_key" in row:
                matches = [
                    sample
                    for sample in matches
                    if sample.source_key == row["source_key"]
                ]
            if len(matches) != 1:
                raise ValueError(
                    "Manifest row has no unique data match; retain source_key for multiple sources"
                )
            sample = matches[0]
            identity = sample_identity(sample)
            selected[identity] = sample
            split = str(row.get("split") or "selected")
            if split not in memberships.setdefault(identity, []):
                memberships[identity].append(split)
    if not selected:
        raise ValueError("No samples selected by the supplied manifests")
    return list(selected.values()), memberships


def evaluate_samples(model, samples, device, output_dir, memberships=None):
    """Run model validation; output files contain numerical results and no plots."""
    if not samples:
        raise ValueError("No evaluation samples")
    memberships = memberships or {}
    rows, metadata, residual_arrays = [], [], []
    model.eval()
    loader = create_dataloader(samples, shuffle=False)
    with torch.no_grad():
        for batch in loader:
            batch = move_batch_to_device(batch, device)
            use_amp = config.MIXED_PRECISION and device.type == "cuda"
            with torch.amp.autocast(device_type="cuda") if use_amp else nullcontext():
                prediction = model(
                    **{
                        name: batch[name]
                        for name in (
                            "images",
                            "lidar2img",
                            "img_shape",
                            "pad_shape",
                            "camera_mask",
                            "past_xy",
                        )
                    }
                )
            reference = prediction.detach().float().cpu().numpy()
            executed = batch["future_xy"].detach().cpu().numpy()
            past = batch["past_xy"].detach().cpu().numpy()
            if not np.isfinite(reference).all():
                raise ValueError("Model produced non-finite predictions")
            signed, valid = compute_signed_l2_residuals(reference, executed, past)
            sample_indices = batch["record_index"].detach().cpu().tolist()
            for index, sample_index in enumerate(sample_indices):
                sample = samples[sample_index]
                scene = {
                    "platform": sample.platform_label or sample.vehicle_name,
                    "driver_mode": sample.driver_mode,
                    "driving_state_group": normalize_driving_state(
                        sample.scene.get("driving_state")
                    ),
                    "lighting_group": normalize_lighting_group(
                        sample.scene.get("lighting")
                    ),
                    "area_group": normalize_area_group(sample.scene.get("area_type")),
                }
                row = {
                    "source_key": sample.source_key,
                    "framegroup_id": sample.framegroup_id,
                    "timestamp_ms": sample.timestamp_ms,
                    "vehicle_name": sample.vehicle_name,
                    **scene,
                    "source_splits": memberships.get(
                        sample_identity(sample), ["selected"]
                    ),
                    "past_xy": past[index].tolist(),
                    "gt_future_xy": executed[index].tolist(),
                    "pred_future_xy": reference[index].tolist(),
                    "residual_xy": (reference[index] - executed[index]).tolist(),
                    "pointwise_error": np.linalg.norm(
                        reference[index] - executed[index], axis=-1
                    ).tolist(),
                    "signed_l2_residual": [
                        float(value) if ok else None
                        for value, ok in zip(signed[index], valid[index])
                    ],
                    "signed_l2_valid_mask": valid[index].tolist(),
                    **sample_metrics(reference[index], executed[index]),
                }
                rows.append(row)
                metadata.append(scene)
                residual_arrays.append(signed[index][valid[index]])
    summary = summarize(rows)
    split_names = sorted({split for row in rows for split in row["source_splits"]})
    summary["by_split"] = {
        name: summarize([row for row in rows if name in row["source_splits"]])
        for name in split_names
    }
    summary["signed_residual_valid_points"] = sum(
        array.size for array in residual_arrays
    )
    summary["signed_residual_invalid_points"] = (
        len(rows) * config.FUTURE_STEPS - summary["signed_residual_valid_points"]
    )
    fits = fit_dataset(ResidualDataset(metadata, residual_arrays))
    output_dir = prepare_output(output_dir)
    with (output_dir / "residuals.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, allow_nan=False) + "\n")
    save_json(output_dir / "metrics.json", summary)
    write_fits(output_dir / "residual_fits.csv", fits)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_source_arguments(parser)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--manifest",
        type=Path,
        action="append",
        required=True,
        help="Saved sample manifest; repeat to merge splits without duplicate samples",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default=config.DEVICE)
    parser.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    parser.add_argument("--num-workers", type=int, default=config.NUM_WORKERS)
    args = parser.parse_args()
    sources = sources_from_args(args)
    configure_loader(args)
    if not args.checkpoint.is_file():
        raise FileNotFoundError("Checkpoint does not exist")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError("Evaluation output directory must be empty")
    index = build_dataset_index(sources)
    samples, memberships = match_manifests(index.samples, args.manifest)
    device = choose_device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=True)
    if not isinstance(checkpoint, dict):
        raise ValueError("Expected checkpoint mapping")
    model = BEVPlanner().to(device)
    load_model_checkpoint_compat(model, checkpoint)
    metrics = evaluate_samples(model, samples, device, args.output_dir, memberships)
    log(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
