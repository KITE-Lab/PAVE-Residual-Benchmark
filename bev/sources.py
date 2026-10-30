"""Stream explicit published-data sources without a built-in source registry."""

from __future__ import annotations
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence
import ijson


@dataclass(frozen=True)
class PublishedSource:
    key: str
    data_json: Path
    images_root: Path
    vehicle_id: int | None = None
    vehicle_name: str | None = None
    platform_label: str | None = None


@dataclass(frozen=True)
class PublishedRecord:
    source: PublishedSource
    record_index: int
    payload: dict[str, Any]


def validate_sources(sources: Sequence[PublishedSource]) -> None:
    if not sources:
        raise ValueError("At least one published source is required.")
    seen_keys: set[str] = set()
    for source in sources:
        if source.key in seen_keys:
            raise ValueError(f"Duplicate published source key: {source.key}")
        seen_keys.add(source.key)
        if not source.data_json.is_file():
            raise FileNotFoundError(
                f"Published data.json not found: {source.data_json}"
            )
        if not source.images_root.is_dir():
            raise FileNotFoundError(
                f"Published images directory not found: {source.images_root}"
            )


def normalize_record(
    source: PublishedSource, payload: Mapping[str, Any]
) -> dict[str, Any]:
    record = dict(payload)
    if source.vehicle_id is not None:
        record_vehicle_id = record.get("vehicle_id")
        if record_vehicle_id != source.vehicle_id:
            raise ValueError(
                f"Source {source.key} expected vehicle_id={source.vehicle_id}, "
                f"observed {record_vehicle_id!r} in framegroup_id={record.get('framegroup_id')!r}."
            )
    if source.vehicle_name:
        existing_name = str(record.get("vehicle_name") or "").strip()
        if existing_name and existing_name != source.vehicle_name:
            raise ValueError(
                f"Source {source.key} expected vehicle_name={source.vehicle_name!r}, "
                f"observed {existing_name!r}."
            )
        record["vehicle_name"] = source.vehicle_name
    if source.platform_label:
        record["platform_label"] = source.platform_label
    return record


def iter_source_records(source: PublishedSource) -> Iterator[PublishedRecord]:
    if ijson is None:
        raise RuntimeError(
            "Streaming published records requires the optional 'ijson' dependency."
        )
    with source.data_json.open("rb") as handle:
        for record_index, payload in enumerate(ijson.items(handle, "item")):
            if not isinstance(payload, Mapping):
                continue
            yield PublishedRecord(
                source=source,
                record_index=record_index,
                payload=normalize_record(source, payload),
            )


def iter_published_records(
    sources: Sequence[PublishedSource],
) -> Iterator[PublishedRecord]:
    validate_sources(sources)
    for source in sources:
        yield from iter_source_records(source)


def add_source_arguments(parser):
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--data-json", type=Path, help="Single published-format JSON array"
    )
    group.add_argument("--sources", type=Path, help="JSON array of source descriptions")
    parser.add_argument("--images-root", type=Path, help="Required for --data-json")


def sources_from_args(args):
    if args.data_json is not None:
        if args.images_root is None:
            raise ValueError("--data-json requires --images-root")
        sources = (
            PublishedSource(
                "single",
                args.data_json.expanduser().resolve(),
                args.images_root.expanduser().resolve(),
            ),
        )
    else:
        if args.images_root is not None:
            raise ValueError("--images-root is only used with --data-json")
        manifest = args.sources.expanduser().resolve()
        with manifest.open(encoding="utf-8") as handle:
            rows = json.load(handle)
        if not isinstance(rows, list) or not rows:
            raise ValueError("--sources must contain a nonempty JSON array")
        sources = []
        for row in rows:
            if not isinstance(row, dict) or not all(
                row.get(k) for k in ("key", "data_json", "images_root")
            ):
                raise ValueError("Each source requires key, data_json and images_root")

            def supplied_path(value):
                path = Path(value).expanduser()
                return (
                    (manifest.parent / path).resolve()
                    if not path.is_absolute()
                    else path.resolve()
                )

            sources.append(
                PublishedSource(
                    key=str(row["key"]),
                    data_json=supplied_path(row["data_json"]),
                    images_root=supplied_path(row["images_root"]),
                    vehicle_id=row.get("vehicle_id"),
                    vehicle_name=row.get("vehicle_name"),
                    platform_label=row.get("platform_label"),
                )
            )
        sources = tuple(sources)
    validate_sources(sources)
    return sources
