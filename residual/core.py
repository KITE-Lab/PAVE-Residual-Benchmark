"""Numerical signed-residual implementation extracted from the paper evaluator."""

from __future__ import annotations
from typing import Optional, Tuple
import numpy as np

POLYLINE_EPS = 1e-6


def compute_pointwise_errors(
    pred_future_xy: np.ndarray, gt_future_xy: np.ndarray
) -> np.ndarray:
    residual_xy = pred_future_xy - gt_future_xy
    return np.linalg.norm(residual_xy, axis=-1)


def is_valid_tangent(tangent: np.ndarray) -> bool:
    tangent_np = np.asarray(tangent, dtype=np.float64)
    return bool(
        tangent_np.shape == (2,)
        and np.isfinite(tangent_np).all()
        and float(np.linalg.norm(tangent_np)) > float(POLYLINE_EPS)
    )


def nearest_valid_path_segment_tangent(
    reference_points: np.ndarray, step_index: int
) -> Optional[np.ndarray]:
    reference_points_np = np.asarray(reference_points, dtype=np.float64)
    if reference_points_np.ndim != 2 or reference_points_np.shape[-1] != 2:
        raise ValueError("reference_points must have shape [M, 2]")
    if reference_points_np.shape[0] < 2:
        return None

    segment_vectors = reference_points_np[1:] - reference_points_np[:-1]
    valid_segment_mask = np.array(
        [is_valid_tangent(segment) for segment in segment_vectors], dtype=bool
    )
    if not np.any(valid_segment_mask):
        return None

    candidate_segment_indices = np.flatnonzero(valid_segment_mask)
    anchor_segment_index = int(np.clip(step_index, 0, segment_vectors.shape[0] - 1))
    nearest_index = int(
        candidate_segment_indices[
            np.argmin(
                np.abs(
                    candidate_segment_indices.astype(np.int64) - anchor_segment_index
                )
            )
        ]
    )
    return segment_vectors[nearest_index]


def compute_gt_tangents(
    gt_future_xy: np.ndarray, past_xy: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    gt_future_xy_np = np.asarray(gt_future_xy, dtype=np.float64)
    past_xy_np = np.asarray(past_xy, dtype=np.float64)

    if gt_future_xy_np.ndim != 2 or gt_future_xy_np.shape[-1] != 2:
        raise ValueError("gt_future_xy must have shape [T, 2]")
    if past_xy_np.ndim != 2 or past_xy_np.shape[-1] != 2:
        raise ValueError("past_xy must have shape [P, 2]")

    future_steps = gt_future_xy_np.shape[0]
    tangents = np.full((future_steps, 2), np.nan, dtype=np.float64)
    valid_mask = np.zeros(future_steps, dtype=bool)
    if future_steps == 0:
        return tangents, valid_mask

    origin_xy = np.zeros(2, dtype=np.float64)
    first_segment = gt_future_xy_np[0] - origin_xy
    reference_points = np.concatenate(
        (origin_xy.reshape(1, 2), gt_future_xy_np), axis=0
    )

    for step_index in range(future_steps):
        if step_index == 0:
            if future_steps > 1:
                tangent = gt_future_xy_np[1] - origin_xy
            else:
                tangent = first_segment
        elif step_index == future_steps - 1:
            tangent = gt_future_xy_np[step_index] - gt_future_xy_np[step_index - 1]
        else:
            tangent = gt_future_xy_np[step_index + 1] - gt_future_xy_np[step_index - 1]

        if not is_valid_tangent(tangent):
            nearest_segment_tangent = nearest_valid_path_segment_tangent(
                reference_points=reference_points,
                step_index=step_index,
            )
            if nearest_segment_tangent is not None:
                tangent = nearest_segment_tangent

        if step_index == 0 and not is_valid_tangent(tangent):
            tangent = first_segment
            if (
                not is_valid_tangent(tangent)
                and past_xy_np.shape[0] > 0
                and np.isfinite(past_xy_np[-1]).all()
            ):
                tangent = origin_xy - past_xy_np[-1]

        if not is_valid_tangent(tangent):
            continue

        tangents[step_index] = tangent
        valid_mask[step_index] = True

    return tangents, valid_mask


def compute_signed_l2_residuals_for_sample(
    pred_future_xy: np.ndarray,
    gt_future_xy: np.ndarray,
    past_xy: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    pred_future_xy_np = np.asarray(pred_future_xy, dtype=np.float64)
    gt_future_xy_np = np.asarray(gt_future_xy, dtype=np.float64)
    if pred_future_xy_np.ndim != 2 or pred_future_xy_np.shape[-1] != 2:
        raise ValueError("pred_future_xy must have shape [T, 2]")
    if gt_future_xy_np.shape != pred_future_xy_np.shape:
        raise ValueError("gt_future_xy must have the same shape as pred_future_xy")

    signed_residual = np.full(pred_future_xy_np.shape[0], np.nan, dtype=np.float64)
    valid_mask = np.zeros(pred_future_xy_np.shape[0], dtype=bool)
    if pred_future_xy_np.shape[0] == 0:
        return signed_residual, valid_mask

    pointwise_error = compute_pointwise_errors(pred_future_xy_np, gt_future_xy_np)
    finite_point_mask = (
        np.isfinite(pred_future_xy_np).all(axis=-1)
        & np.isfinite(gt_future_xy_np).all(axis=-1)
        & np.isfinite(pointwise_error)
    )
    if not np.any(finite_point_mask):
        return signed_residual, valid_mask

    tangents, tangent_valid_mask = compute_gt_tangents(
        gt_future_xy=gt_future_xy_np,
        past_xy=past_xy,
    )
    offsets = pred_future_xy_np - gt_future_xy_np
    finite_tangent_point_mask = finite_point_mask & tangent_valid_mask
    zero_distance_mask = finite_tangent_point_mask & (
        pointwise_error <= float(POLYLINE_EPS)
    )
    signed_residual[zero_distance_mask] = 0.0
    valid_mask[zero_distance_mask] = True

    signed_mask = finite_tangent_point_mask & ~zero_distance_mask
    if np.any(signed_mask):
        cross_values = (
            tangents[signed_mask, 0] * offsets[signed_mask, 1]
            - tangents[signed_mask, 1] * offsets[signed_mask, 0]
        )
        finite_cross_mask = np.isfinite(cross_values)
        signed_indices = np.flatnonzero(signed_mask)
        signed_indices = signed_indices[finite_cross_mask]
        signs = np.where(cross_values[finite_cross_mask] < 0.0, -1.0, 1.0)
        signed_residual[signed_indices] = signs * pointwise_error[signed_indices]
        valid_mask[signed_indices] = True

    return signed_residual, valid_mask


def compute_signed_l2_residuals(
    pred_future_xy: np.ndarray,
    gt_future_xy: np.ndarray,
    past_xy: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    pred_future_xy_np = np.asarray(pred_future_xy, dtype=np.float64)
    gt_future_xy_np = np.asarray(gt_future_xy, dtype=np.float64)
    past_xy_np = np.asarray(past_xy, dtype=np.float64)

    if pred_future_xy_np.ndim != 3 or pred_future_xy_np.shape[-1] != 2:
        raise ValueError("pred_future_xy must have shape [N, T, 2]")
    if gt_future_xy_np.shape != pred_future_xy_np.shape:
        raise ValueError("gt_future_xy must have the same shape as pred_future_xy")
    if (
        past_xy_np.ndim != 3
        or past_xy_np.shape[0] != pred_future_xy_np.shape[0]
        or past_xy_np.shape[-1] != 2
    ):
        raise ValueError("past_xy must have shape [N, P, 2]")

    signed_residual = np.full(pred_future_xy_np.shape[:2], np.nan, dtype=np.float64)
    valid_mask = np.zeros(pred_future_xy_np.shape[:2], dtype=bool)
    for sample_index in range(pred_future_xy_np.shape[0]):
        sample_signed_residual, sample_valid_mask = (
            compute_signed_l2_residuals_for_sample(
                pred_future_xy=pred_future_xy_np[sample_index],
                gt_future_xy=gt_future_xy_np[sample_index],
                past_xy=past_xy_np[sample_index],
            )
        )
        signed_residual[sample_index] = sample_signed_residual
        valid_mask[sample_index] = sample_valid_mask
    return signed_residual, valid_mask
