"""Scenario-conditioned residual fitting; platforms are supplied by input records."""

from __future__ import annotations
from typing import Any, Dict, List, Mapping, NamedTuple, Sequence, Tuple
import csv
from pathlib import Path
import numpy as np

RANDOM_SEED = 20260520


MAX_DENSITY_SAMPLES = 800_000


LOW_SUPPORT_SAMPLES = 20


LOW_SUPPORT_POINTS = 1_000


TAIL_THRESHOLD_METERS = 2.0


SUPPLEMENT_SOURCE = "raw_scene_supplement"


DISPLAY_PERCENTILE_RANGE = (5.0, 95.0)


DISPLAY_LIMIT_PADDING_RATIO = 0.08


MARGINAL_DISPLAY_PERCENTILE = 90.0


MARGINAL_DISPLAY_LIMIT_CAP = 8.0


MIN_HIST_BINS = 30


MAX_HIST_BINS = 80


MIN_SIGNED_RESIDUAL_PLOT_LIMIT = 0.20


MAX_SIGNED_RESIDUAL_PLOT_LIMIT = 10.0


class ResidualDataset(NamedTuple):
    records: List[Dict[str, Any]]
    residual_rows: List[np.ndarray]


class ResidualVarianceSpec(NamedTuple):
    dimension: str
    label: str
    field: str
    value: str


DRIVING_STATE_LABELS: Tuple[str, ...] = (
    "cruise",
    "follow",
    "stop",
    "turn_left",
    "turn_right",
    "lane_change_left",
    "lane_change_right",
)


MARGINAL_RESIDUAL_VARIANCE_SPECS: Tuple[ResidualVarianceSpec, ...] = (
    ResidualVarianceSpec("Driving state", "Cruise", "driving_state_group", "cruise"),
    ResidualVarianceSpec(
        "Driving state", "Left turn", "driving_state_group", "turn_left"
    ),
    ResidualVarianceSpec(
        "Driving state", "Right turn", "driving_state_group", "turn_right"
    ),
    ResidualVarianceSpec("Lighting", "Day", "lighting_group", "day"),
    ResidualVarianceSpec("Lighting", "Night", "lighting_group", "night"),
    ResidualVarianceSpec("Area type", "Urban", "area_group", "urban"),
    ResidualVarianceSpec("Area type", "Highway", "area_group", "highway"),
)


MarginalPlatformKey = Tuple[str, str]


def finite_values(values: np.ndarray, display_limit: float | None = None) -> np.ndarray:
    flat = np.asarray(values, dtype=np.float64).reshape(-1)
    finite = flat[np.isfinite(flat)]
    if display_limit is None:
        return finite
    return finite[np.abs(finite) <= float(display_limit) + 1e-12]


def normalize_driving_state(value: Any) -> str | None:
    token = str(value or "").strip().lower()
    valid_states = set(DRIVING_STATE_LABELS)
    return token if token in valid_states else None


def normalize_lighting_group(value: Any) -> str | None:
    token = str(value or "").strip().lower()
    if token == "day":
        return "day"
    if token in {"night", "night_lit", "night_unlit"}:
        return "night"
    return None


def normalize_area_group(value: Any) -> str | None:
    token = str(value or "").strip().lower()
    if token in {"urban", "highway"}:
        return token
    return None


def residual_values_for_indices(
    dataset: ResidualDataset, indices: Sequence[int]
) -> np.ndarray:
    if not indices:
        return np.asarray([], dtype=np.float32)
    return np.concatenate(
        [dataset.residual_rows[index].reshape(-1) for index in indices], axis=0
    )


def count_records_by_source(
    dataset: ResidualDataset, indices: Sequence[int]
) -> Tuple[int, int]:
    existing = 0
    supplemental = 0
    for index in indices:
        if dataset.records[index].get("record_source") == SUPPLEMENT_SOURCE:
            supplemental += 1
        else:
            existing += 1
    return existing, supplemental


def scene_support_flag(samples: int, points: int) -> str:
    if samples == 0 or points == 0:
        return "no_data"
    if samples < LOW_SUPPORT_SAMPLES or points < LOW_SUPPORT_POINTS:
        return "low_support"
    return "ok"


def sample_for_fit(
    values: np.ndarray, max_samples: int, seed_offset: int
) -> np.ndarray:
    if values.size <= max_samples:
        return values
    rng = np.random.default_rng(RANDOM_SEED + seed_offset)
    indices = rng.choice(values.size, size=max_samples, replace=False)
    return values[indices]


def signed_residual_display_limit(
    values: np.ndarray, fallback: float = MIN_SIGNED_RESIDUAL_PLOT_LIMIT
) -> float:
    flat = np.asarray(values, dtype=np.float64).reshape(-1)
    finite = flat[np.isfinite(flat)]
    if finite.size == 0:
        return fallback

    lower, upper = np.percentile(finite, DISPLAY_PERCENTILE_RANGE)
    limit = max(abs(float(lower)), abs(float(upper)), fallback)
    if np.isfinite(limit) and limit > 0.0:
        limit = float(limit) * (1.0 + float(DISPLAY_LIMIT_PADDING_RATIO))
    else:
        max_abs = float(np.max(np.abs(finite)))
        limit = (
            max(max_abs, fallback) * (1.0 + float(DISPLAY_LIMIT_PADDING_RATIO))
            if np.isfinite(max_abs)
            else fallback
        )

    if np.isfinite(limit) and limit > 0.0:
        return float(
            np.clip(
                limit, MIN_SIGNED_RESIDUAL_PLOT_LIMIT, MAX_SIGNED_RESIDUAL_PLOT_LIMIT
            )
        )
    return fallback


def signed_residual_fit_window(
    values: np.ndarray, fallback: float = MIN_SIGNED_RESIDUAL_PLOT_LIMIT
) -> Tuple[float, float]:
    limit = signed_residual_display_limit(values, fallback=fallback)
    return -float(limit), float(limit)


def select_signed_residual_values_in_window(
    values: np.ndarray,
    window_min: float,
    window_max: float,
) -> np.ndarray:
    finite = finite_values(values)
    if finite.size == 0:
        return finite
    lower = float(window_min)
    upper = float(window_max)
    if not np.isfinite(lower) or not np.isfinite(upper) or lower >= upper:
        return finite
    return finite[(finite >= lower - 1e-12) & (finite <= upper + 1e-12)]


def choose_histogram_bin_edges(values: np.ndarray, limit: float) -> np.ndarray:
    flat = np.asarray(values, dtype=np.float64).reshape(-1)
    finite = flat[np.isfinite(flat)]
    limit = float(limit)
    if not np.isfinite(limit) or limit <= 0.0:
        limit = 1.0
    if finite.size < 2:
        return np.linspace(-limit, limit, MIN_HIST_BINS + 1, dtype=np.float64)

    q25, q75 = np.percentile(finite, [25.0, 75.0])
    iqr = float(q75 - q25)
    if np.isfinite(iqr) and iqr > 0.0:
        bin_width = 2.0 * iqr / np.cbrt(float(finite.size))
    else:
        bin_width = 0.0

    if not np.isfinite(bin_width) or bin_width <= 0.0:
        bin_count = int(np.sqrt(float(finite.size)))
    else:
        bin_count = int(np.ceil((2.0 * limit) / bin_width))

    bin_count = int(np.clip(max(bin_count, 1), MIN_HIST_BINS, MAX_HIST_BINS))
    return np.linspace(-limit, limit, bin_count + 1, dtype=np.float64)


def select_visible_signed_residual_values(
    values: np.ndarray, limit: float
) -> np.ndarray:
    finite = finite_values(values)
    if finite.size == 0:
        return finite
    visible = finite[np.abs(finite) <= float(limit) + 1e-12]
    if visible.size:
        return visible
    return finite


def fit_seed_offset(scene: str, platform: str) -> int:
    token = f"{scene}:{platform}"
    return 7000 + sum(
        (index + 1) * ord(character) for index, character in enumerate(token)
    )


def bin_width_from_edges(bin_edges: np.ndarray) -> float:
    edges = np.asarray(bin_edges, dtype=np.float64)
    if edges.size < 2:
        return float("nan")
    width = float(np.diff(edges)[0])
    return width if np.isfinite(width) and width > 0.0 else float("nan")


def fit_status_for_row(
    samples: int,
    points: int,
    fit_points: int,
    support_flag: str,
    raw_available_samples: int,
    supplemental_samples: int,
    override: str | None,
) -> str:
    if override in {"supplement_setup_failed", "inference_failed"}:
        return override
    if samples == 0 or points == 0:
        return "no_raw_data" if raw_available_samples == 0 else "no_residual_values"
    if fit_points < 2:
        return "insufficient_fit_points"
    if support_flag == "ok":
        return "fit_ok_supplemented" if supplemental_samples > 0 else "fit_ok"
    if override == "supplement_requested" and supplemental_samples > 0:
        return "low_after_supplement"
    if raw_available_samples <= samples:
        return "low_raw_exhausted"
    return "low_support"


def compute_gaussian_row(
    scene: str,
    platform: str,
    samples: int,
    existing_samples: int,
    supplemental_samples: int,
    raw_available_samples: int,
    values: np.ndarray,
    display_limit: float,
    bin_edges: np.ndarray,
    fit_status_override: str | None,
    allow_low_support_fit: bool = False,
) -> Dict[str, Any]:
    finite = finite_values(values)
    points = int(finite.size)
    if points:
        fit_window_min, fit_window_max = signed_residual_fit_window(finite)
    else:
        fit_window_min, fit_window_max = -float(display_limit), float(display_limit)
    visible = select_visible_signed_residual_values(finite, display_limit)
    fit_visible = select_signed_residual_values_in_window(
        finite, fit_window_min, fit_window_max
    )
    plotted_points = int(visible.size)
    fit_points = int(fit_visible.size)
    raw_available_total = max(int(raw_available_samples), int(samples))
    support_flag = scene_support_flag(int(samples), points)
    row: Dict[str, Any] = {
        "scene": scene,
        "platform": platform,
        "samples": int(samples),
        "existing_samples": int(existing_samples),
        "supplemental_samples": int(supplemental_samples),
        "raw_available_samples": raw_available_total,
        "uncovered_raw_samples": max(raw_available_total - int(samples), 0),
        "points": points,
        "plotted_points": plotted_points,
        "display_limit": float(display_limit),
        "fit_window_min": float(fit_window_min),
        "fit_window_max": float(fit_window_max),
        "hist_bin_count": max(int(np.asarray(bin_edges).size) - 1, 0),
        "hist_bin_width": bin_width_from_edges(bin_edges),
        "mu": float("nan"),
        "sigma": float("nan"),
        "mean_abs_r": float("nan"),
        "p95_abs": float("nan"),
        "tail_rate": float("nan"),
        "support_flag": support_flag,
        "fit_status": fit_status_for_row(
            samples=int(samples),
            points=points,
            fit_points=fit_points,
            support_flag=support_flag,
            raw_available_samples=raw_available_total,
            supplemental_samples=int(supplemental_samples),
            override=fit_status_override,
        ),
    }
    if points == 0:
        return row

    abs_values = np.abs(finite)
    row.update(
        {
            "mean_abs_r": float(np.mean(abs_values)),
            "p95_abs": float(np.percentile(abs_values, 95.0)),
            "tail_rate": float(np.mean(abs_values > TAIL_THRESHOLD_METERS)),
        }
    )
    if (support_flag == "ok" or allow_low_support_fit) and fit_points >= 2:
        fit_values = sample_for_fit(
            fit_visible,
            MAX_DENSITY_SAMPLES,
            seed_offset=fit_seed_offset(scene, platform),
        )
        row["mu"] = float(np.mean(fit_values))
        row["sigma"] = float(np.std(fit_values))
    return row


def csv_float(value: Any) -> str:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return ""
    if not np.isfinite(numeric):
        return ""
    return f"{numeric:.6f}"


def marginal_display_limit(
    values: np.ndarray, fallback: float = MIN_SIGNED_RESIDUAL_PLOT_LIMIT
) -> float:
    """Return a robust visual limit from the pooled absolute residuals."""
    finite = finite_values(values)
    if finite.size == 0:
        return float(fallback)
    percentile = float(np.percentile(np.abs(finite), MARGINAL_DISPLAY_PERCENTILE))
    limit = max(float(fallback), percentile)
    limit *= 1.0 + float(DISPLAY_LIMIT_PADDING_RATIO)
    return float(
        np.clip(limit, MIN_SIGNED_RESIDUAL_PLOT_LIMIT, MARGINAL_DISPLAY_LIMIT_CAP)
    )


def compute_marginal_display_limits(
    dataset: ResidualDataset,
    index_map: Mapping[MarginalPlatformKey, Sequence[int]],
) -> Dict[str, float]:
    """Compute one common x-axis limit from pooled values in every category."""
    category_limits: Dict[str, float] = {}
    for spec in MARGINAL_RESIDUAL_VARIANCE_SPECS:
        pooled: List[np.ndarray] = []
        for platform in platform_labels(dataset):
            values = residual_values_for_indices(
                dataset, index_map.get((spec.label, platform), [])
            )
            finite = finite_values(values)
            if finite.size:
                pooled.append(finite)
        category_limits[spec.label] = marginal_display_limit(
            np.concatenate(pooled) if pooled else np.asarray([], dtype=np.float64)
        )
    common_limit = max(category_limits.values(), default=MIN_SIGNED_RESIDUAL_PLOT_LIMIT)
    common_limit = float(
        np.clip(
            common_limit, MIN_SIGNED_RESIDUAL_PLOT_LIMIT, MARGINAL_DISPLAY_LIMIT_CAP
        )
    )
    return {spec.label: common_limit for spec in MARGINAL_RESIDUAL_VARIANCE_SPECS}


def collect_visible_marginal_values(
    dataset: ResidualDataset,
    index_map: Mapping[MarginalPlatformKey, Sequence[int]],
    display_limit: float,
    category: str,
) -> np.ndarray:
    pooled: List[np.ndarray] = []
    for platform in platform_labels(dataset):
        values = residual_values_for_indices(
            dataset, index_map.get((category, platform), [])
        )
        visible = select_visible_signed_residual_values(values, display_limit)
        if visible.size:
            pooled.append(visible)
    return np.concatenate(pooled) if pooled else np.asarray([], dtype=np.float64)


def compute_marginal_bin_edges(
    dataset: ResidualDataset,
    index_map: Mapping[MarginalPlatformKey, Sequence[int]],
    display_limits: Mapping[str, float],
) -> Dict[str, np.ndarray]:
    edges: Dict[str, np.ndarray] = {}
    for spec in MARGINAL_RESIDUAL_VARIANCE_SPECS:
        limit = float(display_limits.get(spec.label, MIN_SIGNED_RESIDUAL_PLOT_LIMIT))
        visible = collect_visible_marginal_values(dataset, index_map, limit, spec.label)
        edges[spec.label] = choose_histogram_bin_edges(visible, limit)
    return edges


def marginal_gaussian_rows(
    dataset: ResidualDataset,
    index_map: Mapping[MarginalPlatformKey, Sequence[int]],
    display_limits: Mapping[str, float],
    bin_edges_by_category: Mapping[str, np.ndarray],
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for spec in MARGINAL_RESIDUAL_VARIANCE_SPECS:
        category = spec.label
        limit = float(display_limits.get(category, MIN_SIGNED_RESIDUAL_PLOT_LIMIT))
        bins = bin_edges_by_category.get(
            category,
            np.linspace(-limit, limit, MIN_HIST_BINS + 1, dtype=np.float64),
        )
        for platform in platform_labels(dataset):
            key = (category, platform)
            indices = index_map.get(key, [])
            existing, supplemental = count_records_by_source(dataset, indices)
            row = compute_gaussian_row(
                scene=category,
                platform=platform,
                samples=len(indices),
                existing_samples=existing,
                supplemental_samples=supplemental,
                raw_available_samples=len(indices),
                values=residual_values_for_indices(dataset, indices),
                display_limit=limit,
                bin_edges=bins,
                fit_status_override=None,
                allow_low_support_fit=True,
            )
            row["category"] = category
            row["dimension"] = spec.dimension
            row["filter_field"] = spec.field
            row["filter_value"] = spec.value
            rows.append(row)
    return rows


def platform_labels(dataset):
    return sorted({str(row["platform"]) for row in dataset.records})


def marginal_platform_indices(dataset):
    indices = {}
    for index, record in enumerate(dataset.records):
        label = str(record["platform"])
        for spec in MARGINAL_RESIDUAL_VARIANCE_SPECS:
            if record.get(spec.field) == spec.value:
                indices.setdefault((spec.label, label), []).append(index)
    return indices


def fit_dataset(dataset):
    if not dataset.records or len(dataset.records) != len(dataset.residual_rows):
        raise ValueError("Expected nonempty aligned records and residual arrays")
    indices = marginal_platform_indices(dataset)
    limits = compute_marginal_display_limits(dataset, indices)
    bins = compute_marginal_bin_edges(dataset, indices, limits)
    rows = marginal_gaussian_rows(dataset, indices, limits, bins)
    for row in rows:
        row["variance"] = row["sigma"] ** 2
    return rows


def write_fits(path, rows):
    fields = (
        "category",
        "dimension",
        "filter_field",
        "filter_value",
        "platform",
        "samples",
        "points",
        "mu",
        "sigma",
        "variance",
        "mean_abs_r",
        "p95_abs",
        "tail_rate",
        "support_flag",
        "fit_status",
        "display_limit",
        "fit_window_min",
        "fit_window_max",
        "hist_bin_count",
        "hist_bin_width",
    )
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    name: (
                        csv_float(row[name])
                        if isinstance(row[name], (float, np.floating))
                        else row[name]
                    )
                    for name in fields
                }
            )
