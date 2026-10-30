"""Published-data indexing, calibration, temporal splitting and image loading."""

from __future__ import annotations
import ast
import json
import sys
import math
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple
import numpy as np
import torch
from PIL import Image
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from . import config
from .sources import PublishedSource, iter_published_records
from .utils import slugify


@dataclass
class CameraRecord:
    camera_type: str
    image_name: str
    image_path: Path
    raw_calibration: Any
    calibration_issues: List[str] = field(default_factory=list)


@dataclass
class SampleRecord:
    framegroup_id: int
    vehicle_name: str
    timestamp_ms: int
    driver_mode: str
    camera_signature: str
    sequence_signature: str
    cameras: Dict[str, CameraRecord]
    past_xy: np.ndarray
    future_xy: np.ndarray
    record_index: int
    source_key: str = ""
    platform_label: str = ""
    scene: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DatasetIndex:
    samples: List[SampleRecord]
    dropped_reasons: Dict[str, int]
    audit_entries: List[Dict[str, Any]]
    mode_counts: Dict[str, int]
    samples_by_vehicle_mode: Dict[str, Dict[str, int]]


@dataclass
class SplitBundle:
    train_samples: List[SampleRecord]
    val_samples: List[SampleRecord]
    shift_val_samples: List[SampleRecord] = field(default_factory=list)
    split_strategy: str = config.SPLIT_STRATEGY
    grouped_chunk_count: int = 0
    train_only_group_count: int = 0
    shift_group_count: int = 0


def canonical_camera_type(raw_value: Any) -> Optional[str]:
    raw = str(raw_value or "").strip().lower()
    if not raw:
        return None
    normalized = (
        raw.replace("-", "_").replace("/", "_").replace(" ", "_").replace("__", "_")
    )
    if "inside" in normalized or "cabin" in normalized:
        return "inside"
    if normalized in {"rear_tele", "rear_30", "cam_rear_30"}:
        return "rear_tele"
    if "rear" in normalized and ("tele" in normalized or "30" in normalized):
        return "rear_tele"
    if "front_tele" in normalized or "front_30" in normalized:
        return "front_tele"
    if "tele" in normalized:
        return "front_tele"
    if "front" in normalized:
        return "front_wide"
    if "left" in normalized:
        return "left_wide"
    if "right" in normalized:
        return "right_wide"
    return normalized if normalized in config.CAMERA_ORDER else None


def normalize_driver_mode(raw_value: Any) -> str:
    raw = str(raw_value or "").strip().lower()
    if raw in {"av", "auto", "autonomous", "assisted", "adas"}:
        return "AV"
    if raw in {"hv", "human", "manual", "unknown", ""}:
        return "HV"
    if "auto" in raw or "av" == raw:
        return "AV"
    return "HV"


def parse_literal_value(value: Any) -> Tuple[Optional[Any], Optional[str]]:
    if value is None:
        return (None, "missing")
    if not isinstance(value, str):
        return (value, None)
    text = value.strip()
    if not text:
        return (None, "missing")
    try:
        return (json.loads(text), None)
    except Exception:
        try:
            return (ast.literal_eval(text), None)
        except Exception:
            return (None, "parse_error")


def parse_mapping_like(value: Any) -> Tuple[Optional[Mapping[str, Any]], Optional[str]]:
    parsed, issue = parse_literal_value(value)
    if issue is not None:
        return (None, issue)
    if not isinstance(parsed, Mapping):
        return (None, "parse_error")
    return (parsed, None)


def validate_calibration_array(
    field_name: str, value: Any, expected_shape: Optional[Tuple[int, ...]] = None
) -> Tuple[Optional[np.ndarray], Optional[str]]:
    parsed, issue = parse_literal_value(value)
    if issue is not None:
        return (None, f"{field_name}_{issue}")
    try:
        arr = np.asarray(parsed, dtype=np.float32)
    except Exception:
        return (None, f"{field_name}_parse_error")
    if arr.size == 0:
        return (None, f"{field_name}_shape_invalid")
    if expected_shape is not None and arr.shape != expected_shape:
        return (None, f"{field_name}_shape_invalid")
    if not np.isfinite(arr).all():
        return (None, f"{field_name}_nonfinite")
    return (arr.astype(np.float32), None)


def detect_calibration_issues(raw_calibration: Any) -> List[str]:
    raw_mapping, mapping_issue = parse_mapping_like(raw_calibration)
    if raw_mapping is None:
        issue_suffix = mapping_issue or "parse_error"
        return [
            f"{field_name}_{issue_suffix}"
            for field_name, _ in config.CALIBRATION_FIELD_SPECS
        ]
    issues: List[str] = []
    for field_name, expected_shape in config.CALIBRATION_FIELD_SPECS:
        _array, issue = validate_calibration_array(
            field_name=field_name,
            value=raw_mapping.get(field_name),
            expected_shape=expected_shape,
        )
        if issue is not None:
            issues.append(issue)
    return issues


def has_blocking_calibration_issues(issues: Sequence[str]) -> bool:
    return any(
        (
            str(issue).startswith("intrinsic_") or str(issue).startswith("extrinsic_")
            for issue in issues
        )
    )


def can_impute_blocking_calibration_issues(
    vehicle_name: str, camera_type: str, issues: Sequence[str]
) -> bool:
    if (
        vehicle_name not in config.IMPUTED_CALIBRATION_VEHICLES
        or camera_type not in config.CAMERA_ORDER
    ):
        return False
    blocking_issues = [
        str(issue)
        for issue in issues
        if str(issue).startswith("intrinsic_") or str(issue).startswith("extrinsic_")
    ]
    return bool(blocking_issues) and all(
        (
            issue in {"intrinsic_missing", "extrinsic_missing"}
            for issue in blocking_issues
        )
    )


def has_unresolved_calibration_blockers(
    vehicle_name: str, camera_type: str, issues: Sequence[str]
) -> bool:
    if not has_blocking_calibration_issues(issues):
        return False
    return not can_impute_blocking_calibration_issues(vehicle_name, camera_type, issues)


def camera_has_usable_image(image_name: str, image_path: Path) -> bool:
    return bool(image_name and image_path.exists())


def role_default_hfov_deg(camera_type: str) -> float:
    if camera_type in {"front_tele", "rear_tele"}:
        return 30.0
    return 120.0


def role_default_intrinsic(camera_type: str, width: int, height: int) -> np.ndarray:
    hfov = math.radians(role_default_hfov_deg(camera_type))
    fx = 0.5 * float(width) / math.tan(hfov / 2.0)
    fy = fx
    cx = (float(width) - 1.0) / 2.0
    cy = (float(height) - 1.0) / 2.0
    return np.asarray([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float32)


def role_default_extrinsic(camera_type: str) -> np.ndarray:
    if camera_type in {"front_wide", "front_tele"}:
        rotation = np.asarray(
            [[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]], dtype=np.float32
        )
    elif camera_type == "left_wide":
        rotation = np.asarray(
            [[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]], dtype=np.float32
        )
    elif camera_type == "right_wide":
        rotation = np.asarray(
            [[-1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, -1.0, 0.0]], dtype=np.float32
        )
    elif camera_type == "rear_tele":
        rotation = np.asarray(
            [[0.0, 0.0, -1.0], [1.0, 0.0, 0.0], [0.0, -1.0, 0.0]], dtype=np.float32
        )
    else:
        rotation = np.eye(3, dtype=np.float32)
    extrinsic = np.eye(4, dtype=np.float32)
    extrinsic[:3, :3] = rotation
    return extrinsic


def scale_intrinsic_matrix(
    intrinsic: np.ndarray,
    old_width: int,
    old_height: int,
    new_width: int,
    new_height: int,
) -> np.ndarray:
    scaled = intrinsic.astype(np.float32).copy()
    sx = float(new_width) / float(old_width)
    sy = float(new_height) / float(old_height)
    scaled[0, 0] *= sx
    scaled[0, 2] *= sx
    scaled[1, 1] *= sy
    scaled[1, 2] *= sy
    return scaled


def make_intrinsic_4x4(intrinsic: np.ndarray) -> np.ndarray:
    intrinsic_4x4 = np.eye(4, dtype=np.float32)
    intrinsic_4x4[:3, :3] = intrinsic.astype(np.float32)
    return intrinsic_4x4


def build_lidar2img_matrix(intrinsic: np.ndarray, extrinsic: np.ndarray) -> np.ndarray:
    try:
        vehicle_to_camera = np.linalg.inv(extrinsic.astype(np.float32))
    except np.linalg.LinAlgError:
        vehicle_to_camera = np.eye(4, dtype=np.float32)
    return (make_intrinsic_4x4(intrinsic) @ vehicle_to_camera).astype(np.float32)


def resolve_camera_calibration(
    vehicle_name: str,
    camera_type: str,
    raw_calibration: Any,
    old_width: int,
    old_height: int,
    new_width: int,
    new_height: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
    imputed_fields: List[str] = []
    calibration_mapping, _mapping_issue = parse_mapping_like(raw_calibration)
    allow_default_calibration = (
        vehicle_name in config.IMPUTED_CALIBRATION_VEHICLES
        and camera_type in config.CAMERA_ORDER
    )
    if calibration_mapping is None:
        if allow_default_calibration and _mapping_issue == "missing":
            calibration_mapping = {}
        else:
            raise ValueError(
                f"{camera_type} calibration must be a mapping with valid intrinsic/extrinsic fields."
            )
    intrinsic, _intrinsic_issue = validate_calibration_array(
        field_name="intrinsic",
        value=calibration_mapping.get("intrinsic"),
        expected_shape=(3, 3),
    )
    if intrinsic is None:
        if allow_default_calibration and _intrinsic_issue == "intrinsic_missing":
            intrinsic = role_default_intrinsic(camera_type, old_width, old_height)
            imputed_fields.append("intrinsic")
        else:
            raise ValueError(f"{camera_type} calibration has invalid intrinsic.")
    intrinsic = scale_intrinsic_matrix(
        intrinsic, old_width, old_height, new_width, new_height
    )
    distortion, _distortion_issue = validate_calibration_array(
        field_name="distortion",
        value=calibration_mapping.get("distortion"),
        expected_shape=None,
    )
    if distortion is None:
        distortion = np.zeros((5,), dtype=np.float32)
        imputed_fields.append("distortion")
    distortion = distortion.astype(np.float32).reshape(-1)
    extrinsic, _extrinsic_issue = validate_calibration_array(
        field_name="extrinsic",
        value=calibration_mapping.get("extrinsic"),
        expected_shape=(4, 4),
    )
    if extrinsic is None:
        if allow_default_calibration and _extrinsic_issue == "extrinsic_missing":
            extrinsic = role_default_extrinsic(camera_type)
            imputed_fields.append("extrinsic")
        else:
            raise ValueError(f"{camera_type} calibration has invalid extrinsic.")
    return (
        intrinsic.astype(np.float32),
        distortion,
        extrinsic.astype(np.float32),
        imputed_fields,
    )


def dedupe_and_sort_trajectory(
    raw_traj: Sequence[Mapping[str, Any]],
) -> Optional[np.ndarray]:
    points: List[Tuple[int, float, float]] = []
    for point in raw_traj:
        if not isinstance(point, Mapping):
            continue
        timestamp = point.get("timestamp")
        x_m = point.get("x_m")
        y_m = point.get("y_m")
        if not isinstance(timestamp, int):
            continue
        try:
            x_val = float(x_m)
            y_val = float(y_m)
        except Exception:
            continue
        if not np.isfinite([x_val, y_val]).all():
            continue
        points.append((timestamp, x_val, y_val))
    if len(points) < 2:
        return None
    points.sort(key=lambda p: p[0])
    deduped: List[Tuple[int, float, float]] = []
    last_ts: Optional[int] = None
    for point in points:
        if last_ts is not None and point[0] == last_ts:
            deduped[-1] = point
        else:
            deduped.append(point)
            last_ts = point[0]
    if len(deduped) < 2:
        return None
    return np.asarray(deduped, dtype=np.float64)


def extract_past_future_xy(
    raw_traj: Sequence[Mapping[str, Any]], anchor_timestamp_ms: int
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[str]]:
    traj = dedupe_and_sort_trajectory(raw_traj)
    if traj is None:
        return (None, None, "trajectory_too_short")
    anchor_time = float(anchor_timestamp_ms)
    ts = traj[:, 0].astype(np.float64) - anchor_time
    xs = traj[:, 1].astype(np.float64)
    ys = traj[:, 2].astype(np.float64)
    past_times = np.asarray(
        [-step * config.RESAMPLE_DT_MS for step in range(config.PAST_STEPS, 0, -1)],
        dtype=np.float64,
    )
    future_times = np.asarray(
        [step * config.RESAMPLE_DT_MS for step in range(1, config.FUTURE_STEPS + 1)],
        dtype=np.float64,
    )
    required_min = float(past_times[0])
    required_max = float(future_times[-1])
    boundary_tolerance_ms = float(config.RESAMPLE_DT_MS)
    if (
        ts[0] > required_min + boundary_tolerance_ms
        or ts[-1] < required_max - boundary_tolerance_ms
    ):
        return (None, None, "trajectory_window_incomplete")
    past_x = np.interp(past_times, ts, xs)
    past_y = np.interp(past_times, ts, ys)
    future_x = np.interp(future_times, ts, xs)
    future_y = np.interp(future_times, ts, ys)
    past_xy = np.stack([past_x, past_y], axis=-1).astype(np.float32)
    future_xy = np.stack([future_x, future_y], axis=-1).astype(np.float32)
    return (past_xy, future_xy, None)


def ordered_camera_types(cameras: Mapping[str, CameraRecord]) -> List[str]:
    return [
        camera_type for camera_type in config.CAMERA_ORDER if camera_type in cameras
    ]


def camera_signature_from_cameras(cameras: Mapping[str, CameraRecord]) -> str:
    ordered = ordered_camera_types(cameras)
    return "|".join(ordered) if ordered else "none"


def sort_samples(samples: Sequence[SampleRecord]) -> List[SampleRecord]:
    return sorted(samples, key=lambda s: (s.timestamp_ms, s.framegroup_id))


def group_sort_key(group: Sequence[SampleRecord]) -> Tuple[str, int, int, str, str]:
    first = group[0]
    return (
        first.vehicle_name,
        first.timestamp_ms,
        first.framegroup_id,
        first.camera_signature,
        first.sequence_signature,
    )


def flatten_sample_groups(
    groups: Sequence[Sequence[SampleRecord]],
) -> List[SampleRecord]:
    flattened: List[SampleRecord] = []
    for group in groups:
        flattened.extend(group)
    return sort_samples(flattened)


def assign_sequence_signatures(samples: Sequence[SampleRecord]) -> None:
    by_vehicle: Dict[str, List[SampleRecord]] = defaultdict(list)
    for sample in samples:
        by_vehicle[sample.vehicle_name].append(sample)
    for vehicle_name, vehicle_samples in by_vehicle.items():
        sequence_index = 0
        last_timestamp_ms: Optional[int] = None
        for sample in sort_samples(vehicle_samples):
            if (
                last_timestamp_ms is None
                or sample.timestamp_ms - last_timestamp_ms > config.SEQUENCE_GAP_MS
            ):
                sequence_index += 1
            sample.sequence_signature = (
                f"{slugify(vehicle_name)}_seq_{sequence_index:04d}"
            )
            last_timestamp_ms = sample.timestamp_ms


def count_samples_by_camera_signature(
    samples: Sequence[SampleRecord],
) -> Dict[str, int]:
    return dict(Counter((sample.camera_signature for sample in samples)))


def build_grouped_sample_chunks(
    samples: Sequence[SampleRecord],
) -> List[List[SampleRecord]]:
    grouped: Dict[Tuple[str, str, str], List[SampleRecord]] = defaultdict(list)
    for sample in samples:
        grouped[
            sample.vehicle_name, sample.camera_signature, sample.sequence_signature
        ].append(sample)
    groups = [sort_samples(group) for group in grouped.values() if group]
    return sorted(groups, key=group_sort_key)


def group_distance_ms(
    left_group: Sequence[SampleRecord], right_group: Sequence[SampleRecord]
) -> int:
    return abs(int(right_group[0].timestamp_ms) - int(left_group[-1].timestamp_ms))


def merge_small_groups(
    groups: Sequence[Sequence[SampleRecord]],
) -> List[List[SampleRecord]]:
    grouped_by_signature: Dict[Tuple[str, str], List[List[SampleRecord]]] = defaultdict(
        list
    )
    for group in groups:
        if not group:
            continue
        first = group[0]
        grouped_by_signature[first.vehicle_name, first.camera_signature].append(
            sort_samples(group)
        )
    merged_groups: List[List[SampleRecord]] = []
    for signature_groups in grouped_by_signature.values():
        current_groups = sorted(signature_groups, key=group_sort_key)
        index = 0
        while index < len(current_groups):
            group = current_groups[index]
            if len(group) >= 2 or len(current_groups) == 1:
                index += 1
                continue
            prev_gap = (
                group_distance_ms(current_groups[index - 1], group)
                if index > 0
                else sys.maxsize
            )
            next_gap = (
                group_distance_ms(group, current_groups[index + 1])
                if index + 1 < len(current_groups)
                else sys.maxsize
            )
            if prev_gap <= next_gap:
                current_groups[index - 1] = sort_samples(
                    current_groups[index - 1] + group
                )
                del current_groups[index]
                index = max(index - 1, 0)
            else:
                current_groups[index + 1] = sort_samples(
                    group + current_groups[index + 1]
                )
                del current_groups[index]
        merged_groups.extend(current_groups)
    return sorted(merged_groups, key=group_sort_key)


def select_shift_val_groups(
    groups: Sequence[Sequence[SampleRecord]],
) -> Tuple[List[List[SampleRecord]], List[List[SampleRecord]]]:
    ordered_groups = sorted(
        [sort_samples(group) for group in groups if group], key=group_sort_key
    )
    if len(ordered_groups) < config.MIN_GROUPS_FOR_GROUPED_SPLIT + 1:
        return ([], ordered_groups)
    distinct_signatures = {group[0].camera_signature for group in ordered_groups}
    if len(distinct_signatures) < 2:
        return ([], ordered_groups)
    latest_signature = ordered_groups[-1][0].camera_signature
    latest_signature_indices = [
        index
        for index, group in enumerate(ordered_groups)
        if group[0].camera_signature == latest_signature
    ]
    if len(latest_signature_indices) < 2:
        return ([], ordered_groups)
    shift_index = latest_signature_indices[-1]
    shift_groups = [ordered_groups[shift_index]]
    remaining_groups = [
        group for index, group in enumerate(ordered_groups) if index != shift_index
    ]
    if len(remaining_groups) < config.MIN_GROUPS_FOR_GROUPED_SPLIT:
        return ([], ordered_groups)
    return (shift_groups, remaining_groups)


def split_groups_by_time(
    groups: Sequence[Sequence[SampleRecord]], val_ratio: float
) -> Tuple[List[SampleRecord], List[SampleRecord], int]:
    train_samples: List[SampleRecord] = []
    val_samples: List[SampleRecord] = []
    train_only_group_count = 0
    for group in groups:
        group_train, group_val = split_samples_by_time(group, val_ratio)
        train_samples.extend(group_train)
        val_samples.extend(group_val)
        if group_train and (not group_val):
            train_only_group_count += 1
    return (
        sort_samples(train_samples),
        sort_samples(val_samples),
        train_only_group_count,
    )


def collapse_split_strategies(strategies: Sequence[str]) -> str:
    unique = sorted({str(strategy) for strategy in strategies if strategy})
    if not unique:
        return config.SPLIT_STRATEGY
    if len(unique) == 1:
        return unique[0]
    return "mixed:" + ",".join(unique)


def split_samples_grouped_by_time(
    samples: Sequence[SampleRecord], val_ratio: float
) -> SplitBundle:
    ordered_samples = sort_samples(samples)
    if not ordered_samples:
        return SplitBundle(
            train_samples=[], val_samples=[], split_strategy=config.SPLIT_STRATEGY
        )
    if config.SPLIT_STRATEGY != "grouped_time":
        train_samples, val_samples = split_samples_by_time(ordered_samples, val_ratio)
        return SplitBundle(
            train_samples=train_samples, val_samples=val_samples, split_strategy="time"
        )
    grouped_chunks = build_grouped_sample_chunks(ordered_samples)
    merged_chunks = merge_small_groups(grouped_chunks)
    chunk_count = len(merged_chunks)
    if chunk_count < config.MIN_GROUPS_FOR_GROUPED_SPLIT:
        train_samples, val_samples = split_samples_by_time(ordered_samples, val_ratio)
        return SplitBundle(
            train_samples=train_samples,
            val_samples=val_samples,
            split_strategy="time_fallback_small_group_count",
            grouped_chunk_count=chunk_count,
        )
    shift_groups, main_groups = select_shift_val_groups(merged_chunks)
    train_samples, val_samples, train_only_group_count = split_groups_by_time(
        main_groups, val_ratio
    )
    if not val_samples and len(ordered_samples) >= 2:
        train_samples, val_samples = split_samples_by_time(ordered_samples, val_ratio)
        return SplitBundle(
            train_samples=train_samples,
            val_samples=val_samples,
            shift_val_samples=[],
            split_strategy="time_fallback_empty_grouped_val",
            grouped_chunk_count=chunk_count,
            train_only_group_count=train_only_group_count,
            shift_group_count=0,
        )
    return SplitBundle(
        train_samples=train_samples,
        val_samples=val_samples,
        shift_val_samples=flatten_sample_groups(shift_groups),
        split_strategy="grouped_time",
        grouped_chunk_count=chunk_count,
        train_only_group_count=train_only_group_count,
        shift_group_count=len(shift_groups),
    )


def build_split_map(
    samples: Sequence[SampleRecord], split_by_vehicle: bool
) -> Dict[str, List[SampleRecord]]:
    if split_by_vehicle:
        grouped_samples: Dict[str, List[SampleRecord]] = defaultdict(list)
        for sample in samples:
            grouped_samples[sample.vehicle_name].append(sample)
        train_samples: List[SampleRecord] = []
        val_samples: List[SampleRecord] = []
        shift_val_samples: List[SampleRecord] = []
        for vehicle_samples in grouped_samples.values():
            split_bundle = split_samples_grouped_by_time(
                vehicle_samples, config.VAL_RATIO
            )
            train_samples.extend(split_bundle.train_samples)
            val_samples.extend(split_bundle.val_samples)
            shift_val_samples.extend(split_bundle.shift_val_samples)
        return {
            "train": sort_samples(train_samples),
            "val": sort_samples(val_samples),
            "shift_val": sort_samples(shift_val_samples),
            "all": sort_samples(samples),
        }
    split_bundle = split_samples_grouped_by_time(samples, config.VAL_RATIO)
    return {
        "train": split_bundle.train_samples,
        "val": split_bundle.val_samples,
        "shift_val": split_bundle.shift_val_samples,
        "all": sort_samples(samples),
    }


def to_manifest_rows(
    samples: Sequence[SampleRecord], split_name: str, split_strategy: str
) -> List[Dict[str, Any]]:
    return [
        {
            "source_key": sample.source_key,
            "framegroup_id": sample.framegroup_id,
            "vehicle_name": sample.vehicle_name,
            "driver_mode": sample.driver_mode,
            "timestamp_ms": sample.timestamp_ms,
            "available_cameras": ordered_camera_types(sample.cameras),
            "camera_signature": sample.camera_signature,
            "sequence_signature": sample.sequence_signature,
            "split": split_name,
            "split_strategy": split_strategy,
        }
        for sample in samples
    ]


def split_samples_by_time(
    samples: Sequence[SampleRecord], val_ratio: float
) -> Tuple[List[SampleRecord], List[SampleRecord]]:
    if not samples:
        return ([], [])
    ordered = sorted(samples, key=lambda s: (s.timestamp_ms, s.framegroup_id))
    if len(ordered) == 1:
        return (ordered[:], [])
    val_count = max(1, int(math.ceil(len(ordered) * val_ratio)))
    val_count = min(val_count, len(ordered) - 1)
    return (ordered[:-val_count], ordered[-val_count:])


def coerce_published_sources(data_json_path_or_sources, images_root=None):
    if isinstance(data_json_path_or_sources, (str, Path)):
        if images_root is None:
            raise ValueError("images_root must be supplied with a data JSON")
        return (
            PublishedSource(
                "single", Path(data_json_path_or_sources), Path(images_root)
            ),
        )
    sources = tuple(data_json_path_or_sources)
    if not sources:
        raise ValueError("At least one published source is required")
    return sources


def build_dataset_index(
    data_json_path_or_sources: Path | Sequence[PublishedSource],
    images_root: Optional[Path] = None,
) -> DatasetIndex:
    sources = coerce_published_sources(data_json_path_or_sources, images_root)
    samples: List[SampleRecord] = []
    dropped_reasons: Counter[str] = Counter()
    audit_entries: List[Dict[str, Any]] = []
    mode_counts: Counter[str] = Counter()
    samples_by_vehicle_mode: Dict[str, Counter[str]] = defaultdict(Counter)
    for index, published_record in enumerate(iter_published_records(sources)):
        item = published_record.payload
        record_images_root = published_record.source.images_root
        framegroup_id = item.get("framegroup_id")
        timestamp_ms = item.get("timestamp")
        if not isinstance(framegroup_id, int):
            dropped_reasons["missing_framegroup_id"] += 1
            continue
        if not isinstance(timestamp_ms, int):
            dropped_reasons["missing_timestamp"] += 1
            continue
        vehicle_name = str(item.get("vehicle_name") or "UNKNOWN_VEHICLE")
        driver_mode = normalize_driver_mode(
            item.get("driver_mode", item.get("drive_mode"))
        )
        mode_counts[driver_mode] += 1
        samples_by_vehicle_mode[vehicle_name][driver_mode] += 1
        frames = item.get("frames")
        if not isinstance(frames, list):
            dropped_reasons["missing_frames"] += 1
            continue
        camera_records: Dict[str, CameraRecord] = {}
        for frame in frames:
            if not isinstance(frame, Mapping):
                continue
            camera_type = canonical_camera_type(frame.get("camera_type"))
            if camera_type is None or camera_type in config.IGNORE_CAMERA_TYPES:
                continue
            image_name = str(frame.get("image_name") or "")
            raw_calibration = frame.get("camera_calibration")
            calibration_issues = detect_calibration_issues(raw_calibration)
            if calibration_issues:
                audit_entries.append(
                    {
                        "framegroup_id": framegroup_id,
                        "vehicle_name": vehicle_name,
                        "driver_mode": driver_mode,
                        "camera_type": camera_type,
                        "image_name": image_name,
                        "calibration_issues": calibration_issues,
                    }
                )
            candidate = CameraRecord(
                camera_type=camera_type,
                image_name=image_name,
                image_path=record_images_root / image_name if image_name else Path(),
                raw_calibration=raw_calibration,
                calibration_issues=calibration_issues,
            )
            existing = camera_records.get(camera_type)
            candidate_has_image = camera_has_usable_image(
                candidate.image_name, candidate.image_path
            )
            candidate_is_valid = candidate_has_image and (
                not has_unresolved_calibration_blockers(
                    vehicle_name, candidate.camera_type, candidate.calibration_issues
                )
            )
            existing_has_image = bool(
                existing
                and camera_has_usable_image(existing.image_name, existing.image_path)
            )
            existing_is_valid = bool(
                existing_has_image
                and existing
                and (
                    not has_unresolved_calibration_blockers(
                        vehicle_name, existing.camera_type, existing.calibration_issues
                    )
                )
            )
            candidate_score = (
                candidate_is_valid,
                candidate_has_image,
                bool(candidate.image_name),
            )
            existing_score = (
                existing_is_valid,
                existing_has_image,
                bool(existing.image_name) if existing is not None else False,
            )
            if existing is None or candidate_score > existing_score:
                camera_records[camera_type] = candidate
        if not camera_records:
            dropped_reasons["no_exterior_cameras"] += 1
            continue
        has_usable_exterior_camera = any(
            (
                camera_has_usable_image(camera.image_name, camera.image_path)
                for camera in camera_records.values()
            )
        )
        if not has_usable_exterior_camera:
            dropped_reasons["no_usable_exterior_cameras"] += 1
            continue
        valid_camera_records = {
            camera_type: camera
            for camera_type, camera in camera_records.items()
            if camera_type in config.CAMERA_ORDER
            and camera_has_usable_image(camera.image_name, camera.image_path)
            and (
                not has_unresolved_calibration_blockers(
                    vehicle_name, camera_type, camera.calibration_issues
                )
            )
        }
        if not valid_camera_records:
            dropped_reasons["no_valid_calibrated_exterior_cameras"] += 1
            continue
        past_xy, future_xy, reason = extract_past_future_xy(
            item.get("trajectory") or [], timestamp_ms
        )
        if past_xy is None or future_xy is None:
            dropped_reasons[reason or "trajectory_invalid"] += 1
            continue
        samples.append(
            SampleRecord(
                framegroup_id=framegroup_id,
                vehicle_name=vehicle_name,
                timestamp_ms=timestamp_ms,
                driver_mode=driver_mode,
                camera_signature=camera_signature_from_cameras(valid_camera_records),
                sequence_signature="",
                cameras=valid_camera_records,
                past_xy=past_xy,
                future_xy=future_xy,
                record_index=index,
                source_key=published_record.source.key,
                platform_label=str(
                    item.get("platform_label")
                    or ("HV" if driver_mode != "AV" else vehicle_name)
                ),
                scene=dict(item["scenario_annotation"])
                if isinstance(item.get("scenario_annotation"), Mapping)
                else {},
            )
        )
    assign_sequence_signatures(samples)
    return DatasetIndex(
        samples=samples,
        dropped_reasons=dict(dropped_reasons),
        audit_entries=audit_entries,
        mode_counts=dict(mode_counts),
        samples_by_vehicle_mode={
            k: dict(v) for k, v in samples_by_vehicle_mode.items()
        },
    )


def calibration_audit_summary(entries: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    by_issue: Counter[str] = Counter()
    by_camera: Counter[str] = Counter()
    by_vehicle: Counter[str] = Counter()
    by_mode: Counter[str] = Counter()
    for entry in entries:
        vehicle_name = str(entry.get("vehicle_name") or "UNKNOWN")
        camera_type = str(entry.get("camera_type") or "UNKNOWN")
        driver_mode = str(entry.get("driver_mode") or "UNKNOWN")
        by_camera[camera_type] += 1
        by_vehicle[vehicle_name] += 1
        by_mode[driver_mode] += 1
        for issue_code in entry.get("calibration_issues", []):
            by_issue[str(issue_code)] += 1
    return {
        "total_entries": len(entries),
        "counts_by_issue": dict(by_issue),
        "counts_by_camera_type": dict(by_camera),
        "counts_by_vehicle_name": dict(by_vehicle),
        "counts_by_driver_mode": dict(by_mode),
        "entries": list(entries),
    }


def dataset_summary_payload(dataset_index: DatasetIndex) -> Dict[str, Any]:
    camera_signature_counts = count_samples_by_camera_signature(dataset_index.samples)
    return {
        "total_kept_samples": len(dataset_index.samples),
        "dropped_reasons": dataset_index.dropped_reasons,
        "mode_counts": dataset_index.mode_counts,
        "samples_by_vehicle_mode": dataset_index.samples_by_vehicle_mode,
        "camera_signature_counts": camera_signature_counts,
        "split_strategy": config.SPLIT_STRATEGY,
    }


class PublishedPlanningDataset(Dataset):
    def __init__(self, samples: Sequence[SampleRecord]) -> None:
        self.samples = list(samples)
        self.target_size = (config.IMAGE_RESIZE_WIDTH, config.IMAGE_RESIZE_HEIGHT)
        self.image_mean = torch.tensor(config.IMAGE_MEAN, dtype=torch.float32).view(
            3, 1, 1
        )
        self.image_std = torch.tensor(config.IMAGE_STD, dtype=torch.float32).view(
            3, 1, 1
        )
        if hasattr(Image, "Resampling"):
            self._resample = Image.Resampling.BILINEAR
        else:
            self._resample = Image.BILINEAR

    def __len__(self) -> int:
        return len(self.samples)

    def _load_image_tensor(
        self, image_path: Path
    ) -> Tuple[torch.Tensor, Tuple[int, int], bool]:
        if not image_path or not image_path.exists():
            blank = torch.zeros(
                (3, config.IMAGE_HEIGHT, config.IMAGE_WIDTH), dtype=torch.float32
            )
            return (blank, (config.IMAGE_WIDTH, config.IMAGE_HEIGHT), False)
        try:
            with Image.open(image_path) as pil_image:
                pil_image = pil_image.convert("RGB")
                old_width, old_height = pil_image.size
                resized = pil_image.resize(self.target_size, self._resample)
                image_np = np.asarray(resized, dtype=np.float32)
        except Exception:
            blank = torch.zeros(
                (3, config.IMAGE_HEIGHT, config.IMAGE_WIDTH), dtype=torch.float32
            )
            return (blank, (config.IMAGE_WIDTH, config.IMAGE_HEIGHT), False)
        tensor = torch.from_numpy(image_np).permute(2, 0, 1)
        tensor = (tensor - self.image_mean) / self.image_std
        pad_h = config.IMAGE_HEIGHT - config.IMAGE_RESIZE_HEIGHT
        pad_w = config.IMAGE_WIDTH - config.IMAGE_RESIZE_WIDTH
        if pad_h > 0 or pad_w > 0:
            tensor = F.pad(tensor, (0, pad_w, 0, pad_h), value=0.0)
        return (tensor, (old_width, old_height), True)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        sample = self.samples[index]
        images: List[torch.Tensor] = []
        intrinsics: List[torch.Tensor] = []
        extrinsics: List[torch.Tensor] = []
        lidar2img_mats: List[torch.Tensor] = []
        img_shapes: List[torch.Tensor] = []
        pad_shapes: List[torch.Tensor] = []
        camera_mask: List[float] = []
        camera_missing_flags: List[float] = []
        for camera_type in config.CAMERA_ORDER:
            camera = sample.cameras.get(camera_type)
            if camera is None or not camera.image_name:
                images.append(
                    torch.zeros(
                        (3, config.IMAGE_HEIGHT, config.IMAGE_WIDTH),
                        dtype=torch.float32,
                    )
                )
                intrinsics.append(torch.eye(3, dtype=torch.float32))
                extrinsics.append(torch.eye(4, dtype=torch.float32))
                lidar2img_mats.append(torch.eye(4, dtype=torch.float32))
                img_shapes.append(
                    torch.tensor(
                        [config.IMAGE_RESIZE_HEIGHT, config.IMAGE_RESIZE_WIDTH],
                        dtype=torch.long,
                    )
                )
                pad_shapes.append(
                    torch.tensor(
                        [config.IMAGE_HEIGHT, config.IMAGE_WIDTH], dtype=torch.long
                    )
                )
                camera_mask.append(0.0)
                camera_missing_flags.append(1.0)
                continue
            image_tensor, (old_width, old_height), loaded_ok = self._load_image_tensor(
                camera.image_path
            )
            images.append(image_tensor)
            if loaded_ok:
                intrinsic, _distortion, extrinsic, imputed_fields = (
                    resolve_camera_calibration(
                        vehicle_name=sample.vehicle_name,
                        camera_type=camera_type,
                        raw_calibration=camera.raw_calibration,
                        old_width=old_width,
                        old_height=old_height,
                        new_width=config.IMAGE_RESIZE_WIDTH,
                        new_height=config.IMAGE_RESIZE_HEIGHT,
                    )
                )
                intrinsics.append(torch.from_numpy(intrinsic))
                extrinsics.append(torch.from_numpy(extrinsic))
                lidar2img_mats.append(
                    torch.from_numpy(build_lidar2img_matrix(intrinsic, extrinsic))
                )
                img_shapes.append(
                    torch.tensor(
                        [config.IMAGE_RESIZE_HEIGHT, config.IMAGE_RESIZE_WIDTH],
                        dtype=torch.long,
                    )
                )
                pad_shapes.append(
                    torch.tensor(
                        [config.IMAGE_HEIGHT, config.IMAGE_WIDTH], dtype=torch.long
                    )
                )
                camera_mask.append(1.0)
                camera_missing_flags.append(1.0 if imputed_fields else 0.0)
            else:
                intrinsics.append(torch.eye(3, dtype=torch.float32))
                extrinsics.append(torch.eye(4, dtype=torch.float32))
                lidar2img_mats.append(torch.eye(4, dtype=torch.float32))
                img_shapes.append(
                    torch.tensor(
                        [config.IMAGE_RESIZE_HEIGHT, config.IMAGE_RESIZE_WIDTH],
                        dtype=torch.long,
                    )
                )
                pad_shapes.append(
                    torch.tensor(
                        [config.IMAGE_HEIGHT, config.IMAGE_WIDTH], dtype=torch.long
                    )
                )
                camera_mask.append(0.0)
                camera_missing_flags.append(1.0)
        return {
            "images": torch.stack(images, dim=0),
            "intrinsics": torch.stack(intrinsics, dim=0).float(),
            "extrinsics": torch.stack(extrinsics, dim=0).float(),
            "lidar2img": torch.stack(lidar2img_mats, dim=0).float(),
            "img_shape": torch.stack(img_shapes, dim=0).long(),
            "pad_shape": torch.stack(pad_shapes, dim=0).long(),
            "camera_mask": torch.tensor(camera_mask, dtype=torch.float32),
            "camera_missing_flags": torch.tensor(
                camera_missing_flags, dtype=torch.float32
            ),
            "past_xy": torch.from_numpy(sample.past_xy).float(),
            "future_xy": torch.from_numpy(sample.future_xy).float(),
            "framegroup_id": torch.tensor(sample.framegroup_id, dtype=torch.long),
            "timestamp_ms": torch.tensor(sample.timestamp_ms, dtype=torch.long),
            "vehicle_name": sample.vehicle_name,
            "driver_mode": sample.driver_mode,
            "record_index": index,
            "source_key": sample.source_key,
        }


def create_dataloader(samples: Sequence[SampleRecord], shuffle: bool) -> DataLoader:
    dataset = PublishedPlanningDataset(samples)
    workers = config.NUM_WORKERS if len(dataset) > 0 else 0
    return DataLoader(
        dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=config.PIN_MEMORY,
        persistent_workers=config.PERSISTENT_WORKERS if workers > 0 else False,
        drop_last=False,
    )


def move_batch_to_device(
    batch: Mapping[str, Any], device: torch.device
) -> Dict[str, Any]:
    moved: Dict[str, Any] = {}
    for key, value in batch.items():
        if torch.is_tensor(value):
            moved[key] = value.to(device, non_blocking=True)
        else:
            moved[key] = value
    return moved
