"""Open-loop trajectory and motion consistency metrics at 10 Hz."""

from __future__ import annotations
import numpy as np


def wrapped_heading_error(gt_xy: np.ndarray, pred_xy: np.ndarray) -> np.ndarray:
    origin = np.zeros((1, 2), dtype=np.float64)
    gt_delta = np.diff(np.vstack((origin, gt_xy)), axis=0)
    pred_delta = np.diff(np.vstack((origin, pred_xy)), axis=0)
    gt_heading = np.arctan2(gt_delta[:, 1], gt_delta[:, 0])
    pred_heading = np.arctan2(pred_delta[:, 1], pred_delta[:, 0])
    delta = pred_heading - gt_heading
    return np.abs(np.arctan2(np.sin(delta), np.cos(delta)))


def speed_error(gt_xy: np.ndarray, pred_xy: np.ndarray, hz: float = 10.0) -> np.ndarray:
    origin = np.zeros((1, 2), dtype=np.float64)
    gt_speed = np.linalg.norm(np.diff(np.vstack((origin, gt_xy)), axis=0), axis=1) * hz
    pred_speed = (
        np.linalg.norm(np.diff(np.vstack((origin, pred_xy)), axis=0), axis=1) * hz
    )
    return np.abs(pred_speed - gt_speed)


def sample_metrics(reference, executed, hz=10):
    reference = np.asarray(reference, dtype=np.float64)
    executed = np.asarray(executed, dtype=np.float64)
    if reference.shape != (50, 2) or executed.shape != (50, 2):
        raise ValueError("Expected aligned [50, 2] trajectories")
    if not np.isfinite(reference).all() or not np.isfinite(executed).all():
        raise ValueError("Trajectories must be finite")
    distance = np.linalg.norm(reference - executed, axis=-1)
    result = {"ade": float(distance.mean()), "fde": float(distance[-1])}
    for seconds in (1, 3, 5):
        steps = seconds * hz
        result[f"ade_{seconds}s"] = float(distance[:steps].mean())
        result[f"fde_{seconds}s"] = float(distance[steps - 1])
    result["miss_rate_5s"] = float(distance[-1] > 2.0)
    result["yaw_error_rad"] = float(wrapped_heading_error(executed, reference).mean())
    result["speed_error_mps"] = float(speed_error(executed, reference, hz).mean())
    return result


def summarize(rows):
    if not rows:
        raise ValueError("No evaluation rows")
    fields = (
        "ade",
        "fde",
        "ade_1s",
        "ade_3s",
        "ade_5s",
        "fde_1s",
        "fde_3s",
        "fde_5s",
        "miss_rate_5s",
        "yaw_error_rad",
        "speed_error_mps",
    )
    return {
        "samples": len(rows),
        **{name: float(np.mean([row[name] for row in rows])) for name in fields},
    }
