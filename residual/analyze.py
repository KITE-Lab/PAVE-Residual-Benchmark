"""Compute signed trajectory residuals and fit scenario-conditioned distributions."""

from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
from .core import compute_signed_l2_residuals_for_sample
from .statistics import (
    ResidualDataset,
    fit_dataset,
    write_fits,
    normalize_driving_state,
    normalize_lighting_group,
    normalize_area_group,
)


def analyze(input_path, output_dir):
    records, arrays, exports, seen = [], [], [], set()
    with Path(input_path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if (
                not isinstance(row, dict)
                or not isinstance(row.get("platform"), str)
                or not row["platform"].strip()
            ):
                raise ValueError(
                    f"Line {line_number}: provide a nonempty platform label"
                )
            default_id = f"{row.get('source_key', '')}:{row.get('framegroup_id', line_number)}:{row.get('timestamp_ms', '')}"
            sample_id = str(row.get("sample_id", default_id))
            identity = (row["platform"], sample_id)
            if identity in seen:
                raise ValueError(f"Line {line_number}: duplicate sample identity")
            seen.add(identity)
            metadata = {
                "platform": row["platform"],
                "driving_state_group": normalize_driving_state(
                    row.get("driving_state_group")
                ),
                "lighting_group": normalize_lighting_group(row.get("lighting_group")),
                "area_group": normalize_area_group(row.get("area_group")),
            }
            # Missing metadata leaves that dimension unclassified. Bad supplied labels are rejected.
            for name in ("driving_state_group", "lighting_group", "area_group"):
                if row.get(name) is not None and metadata[name] is None:
                    raise ValueError(f"Line {line_number}: unsupported {name}")
            if "signed_l2_residual" in row:
                values = np.asarray(row["signed_l2_residual"], dtype=np.float64)
                if values.shape != (50,) or np.isinf(values).any():
                    raise ValueError(
                        f"Line {line_number}: expected 50 finite or null residuals"
                    )
                valid = np.isfinite(values)
                if "signed_l2_valid_mask" in row:
                    mask = np.asarray(row["signed_l2_valid_mask"])
                    if mask.shape != (50,) or mask.dtype != np.bool_:
                        raise ValueError(
                            f"Line {line_number}: expected a 50-element boolean mask"
                        )
                    if np.any(mask & ~valid):
                        raise ValueError(
                            f"Line {line_number}: valid mask includes null residuals"
                        )
                    valid &= mask
            else:
                reference = np.asarray(row["reference_xy"], dtype=np.float64)
                executed = np.asarray(row["executed_xy"], dtype=np.float64)
                past = np.asarray(row["past_xy"], dtype=np.float64)
                if (
                    reference.shape != (50, 2)
                    or executed.shape != (50, 2)
                    or past.shape != (60, 2)
                ):
                    raise ValueError(
                        f"Line {line_number}: expected future [50,2] and past [60,2]"
                    )
                if not all(
                    np.isfinite(array).all() for array in (reference, executed, past)
                ):
                    raise ValueError(
                        f"Line {line_number}: trajectory coordinates must be finite"
                    )
                values, valid = compute_signed_l2_residuals_for_sample(
                    reference, executed, past
                )
            records.append(metadata)
            arrays.append(values[valid])
            exports.append(
                {
                    "sample_id": sample_id,
                    **metadata,
                    "signed_l2_residual": [
                        float(value) if ok else None for value, ok in zip(values, valid)
                    ],
                    "signed_l2_valid_mask": valid.tolist(),
                }
            )
    fits = fit_dataset(ResidualDataset(records, arrays))
    output_dir = Path(output_dir).expanduser().resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError("Residual output directory must be empty")
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "residuals.jsonl").open("w", encoding="utf-8") as handle:
        for row in exports:
            handle.write(json.dumps(row, allow_nan=False) + "\n")
    write_fits(output_dir / "residual_fits.csv", fits)
    print(
        f"Analyzed {len(records)} samples across {len({row['platform'] for row in records})} platforms"
    )
    return fits


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    analyze(args.input, args.output_dir)


if __name__ == "__main__":
    main()
