"""Train HV and per-vehicle AV reference planners from explicit input sources."""

from __future__ import annotations
import argparse
import sys
from collections import defaultdict
from contextlib import nullcontext
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple
import torch
from torch.utils.data import DataLoader
from torch import nn
from torch.nn import functional as F
from . import config
from .data import (
    SampleRecord,
    DatasetIndex,
    build_dataset_index,
    create_dataloader,
    move_batch_to_device,
    to_manifest_rows,
    count_samples_by_camera_signature,
    sort_samples,
    assign_sequence_signatures,
    split_samples_grouped_by_time,
    collapse_split_strategies,
    calibration_audit_summary,
)
from .model import BEVPlanner, load_model_checkpoint_compat
from .sources import add_source_arguments, sources_from_args
from .utils import (
    log,
    slugify,
    ensure_dir,
    save_json,
    seed_everything,
    choose_device,
    prepare_output,
    configure_loader,
    batch_field_to_list,
)

try:
    from tqdm.auto import tqdm
except ImportError:
    tqdm = None

RUN_MODE = "all"
TARGET_AV_VEHICLE = None
TARGET_AV_VEHICLES = ()
AV_INIT_CHECKPOINT = None
OUTPUT_DIR: Path


@dataclass
class TrainingJob:
    name: str
    model_kind: str
    vehicle_name: Optional[str]
    train_samples: List[SampleRecord]
    val_samples: List[SampleRecord]
    output_dir: Path
    shift_val_samples: List[SampleRecord] = field(default_factory=list)
    split_strategy: str = config.SPLIT_STRATEGY
    grouped_chunk_count: int = 0
    train_only_group_count: int = 0
    shift_group_count: int = 0
    init_checkpoint: Optional[Path] = None


def planning_loss(
    pred_future_xy: torch.Tensor, target_future_xy: torch.Tensor
) -> torch.Tensor:
    point_loss = F.smooth_l1_loss(pred_future_xy, target_future_xy)
    final_loss = F.smooth_l1_loss(pred_future_xy[:, -1], target_future_xy[:, -1])
    return point_loss + config.FINAL_POINT_LOSS_WEIGHT * final_loss


def compute_ade_fde(
    pred_future_xy: torch.Tensor, target_future_xy: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    distances = torch.linalg.norm(pred_future_xy - target_future_xy, dim=-1)
    ade = distances.mean()
    fde = distances[:, -1].mean()
    return (ade, fde)


def supports_live_progress() -> bool:
    if tqdm is None:
        return False
    for stream in (getattr(sys, "stderr", None), getattr(sys, "stdout", None)):
        if stream is not None and hasattr(stream, "isatty") and stream.isatty():
            return True
    return False


def batch_debug_summary(batch: Mapping[str, Any]) -> str:
    framegroup_ids = [int(v) for v in batch_field_to_list(batch.get("framegroup_id"))]
    timestamps = [int(v) for v in batch_field_to_list(batch.get("timestamp_ms"))]
    vehicle_names = [str(v) for v in batch_field_to_list(batch.get("vehicle_name"))]
    return f"framegroup_id={framegroup_ids} timestamp_ms={timestamps} vehicle_name={vehicle_names}"


def ensure_finite_tensor(
    name: str,
    tensor: torch.Tensor,
    batch: Mapping[str, Any],
    job_name: str = "",
    stage_name: str = "",
    epoch: Optional[int] = None,
) -> None:
    finite_mask = torch.isfinite(tensor)
    if bool(finite_mask.all().item()):
        return
    bad_count = int((~finite_mask).sum().item())
    context_parts = [f"non_finite_tensor={name}", f"count={bad_count}"]
    if job_name:
        context_parts.append(f"job={job_name}")
    if stage_name:
        context_parts.append(f"stage={stage_name}")
    if epoch is not None:
        context_parts.append(f"epoch={epoch}")
    context_parts.append(batch_debug_summary(batch))
    raise RuntimeError(" | ".join(context_parts))


def run_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    optimizer: Optional[torch.optim.Optimizer],
    scaler: Optional[torch.amp.GradScaler],
    use_amp: bool,
    job_name: str = "",
    stage_name: str = "",
    epoch: Optional[int] = None,
    total_epochs: Optional[int] = None,
) -> Dict[str, float]:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_ade = 0.0
    total_fde = 0.0
    total_batches = 0
    autocast_enabled = use_amp and device.type == "cuda"
    iterator: Iterable[Mapping[str, Any]] = dataloader
    progress_bar = None
    if supports_live_progress():
        stage_label = stage_name or ("train" if training else "val")
        epoch_label = ""
        if epoch is not None:
            epoch_label = f" e{epoch}"
            if total_epochs is not None:
                epoch_label += f"/{total_epochs}"
        desc = f"{job_name or 'job'} {stage_label}{epoch_label}"
        progress_bar = tqdm(
            dataloader,
            total=len(dataloader),
            desc=desc,
            dynamic_ncols=True,
            leave=False,
        )
        iterator = progress_bar
    for batch in iterator:
        batch = move_batch_to_device(batch, device)
        if training:
            optimizer.zero_grad(set_to_none=True)
        ensure_finite_tensor(
            "images", batch["images"], batch, job_name, stage_name, epoch
        )
        ensure_finite_tensor(
            "past_xy", batch["past_xy"], batch, job_name, stage_name, epoch
        )
        ensure_finite_tensor(
            "future_xy", batch["future_xy"], batch, job_name, stage_name, epoch
        )
        ensure_finite_tensor(
            "intrinsics", batch["intrinsics"], batch, job_name, stage_name, epoch
        )
        ensure_finite_tensor(
            "extrinsics", batch["extrinsics"], batch, job_name, stage_name, epoch
        )
        ensure_finite_tensor(
            "lidar2img", batch["lidar2img"], batch, job_name, stage_name, epoch
        )
        amp_context = (
            torch.amp.autocast(device_type="cuda", enabled=True)
            if autocast_enabled
            else nullcontext()
        )
        with amp_context:
            pred_future_xy = model(
                images=batch["images"],
                lidar2img=batch["lidar2img"],
                img_shape=batch["img_shape"],
                pad_shape=batch["pad_shape"],
                camera_mask=batch["camera_mask"],
                past_xy=batch["past_xy"],
            )
            loss = planning_loss(pred_future_xy, batch["future_xy"])
            ensure_finite_tensor(
                "pred_future_xy", pred_future_xy, batch, job_name, stage_name, epoch
            )
            ensure_finite_tensor("loss", loss, batch, job_name, stage_name, epoch)
        if training:
            assert optimizer is not None
            if autocast_enabled and scaler is not None:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), config.GRAD_CLIP_NORM
                )
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), config.GRAD_CLIP_NORM
                )
                optimizer.step()
        ade, fde = compute_ade_fde(pred_future_xy.detach(), batch["future_xy"])
        ensure_finite_tensor("ade", ade, batch, job_name, stage_name, epoch)
        ensure_finite_tensor("fde", fde, batch, job_name, stage_name, epoch)
        total_loss += float(loss.detach().item())
        total_ade += float(ade.detach().item())
        total_fde += float(fde.detach().item())
        total_batches += 1
        if progress_bar is not None:
            postfix = {
                "loss": f"{total_loss / total_batches:.4f}",
                "ade": f"{total_ade / total_batches:.4f}",
                "fde": f"{total_fde / total_batches:.4f}",
            }
            if training and optimizer is not None and optimizer.param_groups:
                postfix["lr"] = f"{float(optimizer.param_groups[0]['lr']):.2e}"
            progress_bar.set_postfix(postfix)
    if progress_bar is not None:
        progress_bar.close()
    if total_batches == 0:
        return {"loss": float("nan"), "ade": float("nan"), "fde": float("nan")}
    return {
        "loss": total_loss / total_batches,
        "ade": total_ade / total_batches,
        "fde": total_fde / total_batches,
    }


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    epoch: int,
    metrics: Mapping[str, float],
    job: TrainingJob,
) -> None:
    payload = {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "epoch": epoch,
        "metrics": dict(metrics),
        "model_arch": config.MODEL_ARCH_NAME,
        "model_kind": job.model_kind,
        "vehicle_name": job.vehicle_name,
        "camera_order": list(config.CAMERA_ORDER),
        "input_size": [config.IMAGE_HEIGHT, config.IMAGE_WIDTH],
        "image_resize": [config.IMAGE_RESIZE_HEIGHT, config.IMAGE_RESIZE_WIDTH],
        "resample_hz": config.RESAMPLE_HZ,
        "past_steps": config.PAST_STEPS,
        "future_steps": config.FUTURE_STEPS,
        "pc_range": list(config.PC_RANGE),
        "bev_h": config.BEV_HEIGHT,
        "bev_w": config.BEV_WIDTH,
        "split_strategy": job.split_strategy,
        "init_checkpoint": str(job.init_checkpoint)
        if job.init_checkpoint is not None
        else None,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def train_job(job: TrainingJob, device: torch.device) -> None:
    if len(job.train_samples) < 1:
        log(f"Skipping {job.name}: train split is empty.")
        return
    if len(job.val_samples) < 1:
        log(f"Skipping {job.name}: val split is empty after selected split strategy.")
        return
    log(
        f"Training job={job.name} | train={len(job.train_samples)} | val={len(job.val_samples)} | vehicle={job.vehicle_name or 'ALL'}"
    )
    log(
        f"{job.name} split_strategy={job.split_strategy} | grouped_chunks={job.grouped_chunk_count} | train_only_groups={job.train_only_group_count} | shift_groups={job.shift_group_count}"
    )
    log(
        f"{job.name} train camera_signature counts: {count_samples_by_camera_signature(job.train_samples)}"
    )
    log(
        f"{job.name} val camera_signature counts: {count_samples_by_camera_signature(job.val_samples)}"
    )
    if job.shift_val_samples:
        log(
            f"{job.name} shift_val camera_signature counts: {count_samples_by_camera_signature(job.shift_val_samples)}"
        )
    ensure_dir(job.output_dir)
    save_json(
        job.output_dir / "train_manifest.json",
        to_manifest_rows(job.train_samples, "train", job.split_strategy),
    )
    save_json(
        job.output_dir / "val_manifest.json",
        to_manifest_rows(job.val_samples, "val", job.split_strategy),
    )
    save_json(
        job.output_dir / "shift_val_manifest.json",
        to_manifest_rows(job.shift_val_samples, "shift_val", job.split_strategy),
    )
    train_loader = create_dataloader(job.train_samples, shuffle=True)
    val_loader = create_dataloader(job.val_samples, shuffle=False)
    shift_val_loader = (
        create_dataloader(job.shift_val_samples, shuffle=False)
        if job.shift_val_samples
        else None
    )
    model = BEVPlanner().to(device)
    loaded_init_checkpoint: Optional[Path] = None
    if job.init_checkpoint is not None:
        checkpoint_path = Path(job.init_checkpoint)
        if checkpoint_path.exists():
            checkpoint_obj = torch.load(checkpoint_path, map_location=device)
            if not isinstance(checkpoint_obj, Mapping):
                raise ValueError(
                    f"Unsupported init checkpoint format: {type(checkpoint_obj)!r}"
                )
            load_model_checkpoint_compat(model, checkpoint_obj)
            loaded_init_checkpoint = checkpoint_path
            log(f"{job.name} initialized from checkpoint: {checkpoint_path}")
        else:
            raise FileNotFoundError(
                f"{job.name} init checkpoint not found: {checkpoint_path}"
            )
    job.init_checkpoint = loaded_init_checkpoint
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=config.LR_PLATEAU_FACTOR,
        patience=config.LR_PLATEAU_PATIENCE,
        min_lr=config.MIN_LEARNING_RATE,
    )
    use_amp = config.MIXED_PRECISION and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    history: List[Dict[str, Any]] = []
    best_val_fde = float("inf")
    best_epoch = -1
    patience_left = config.EARLY_STOP_PATIENCE
    for epoch in range(1, config.NUM_EPOCHS + 1):
        current_lr = float(optimizer.param_groups[0]["lr"])
        train_metrics = run_epoch(
            model=model,
            dataloader=train_loader,
            device=device,
            optimizer=optimizer,
            scaler=scaler,
            use_amp=use_amp,
            job_name=job.name,
            stage_name="train",
            epoch=epoch,
            total_epochs=config.NUM_EPOCHS,
        )
        with torch.no_grad():
            val_metrics = run_epoch(
                model=model,
                dataloader=val_loader,
                device=device,
                optimizer=None,
                scaler=None,
                use_amp=use_amp,
                job_name=job.name,
                stage_name="val",
                epoch=epoch,
                total_epochs=config.NUM_EPOCHS,
            )
        epoch_row = {
            "epoch": epoch,
            "lr": current_lr,
            "train_loss": train_metrics["loss"],
            "train_ade": train_metrics["ade"],
            "train_fde": train_metrics["fde"],
            "val_loss": val_metrics["loss"],
            "val_ade": val_metrics["ade"],
            "val_fde": val_metrics["fde"],
        }
        if shift_val_loader is not None:
            with torch.no_grad():
                shift_val_metrics = run_epoch(
                    model=model,
                    dataloader=shift_val_loader,
                    device=device,
                    optimizer=None,
                    scaler=None,
                    use_amp=use_amp,
                    job_name=job.name,
                    stage_name="shift_val",
                    epoch=epoch,
                    total_epochs=config.NUM_EPOCHS,
                )
            epoch_row["shift_val_loss"] = shift_val_metrics["loss"]
            epoch_row["shift_val_ade"] = shift_val_metrics["ade"]
            epoch_row["shift_val_fde"] = shift_val_metrics["fde"]
        current_val_fde = float(val_metrics["fde"])
        scheduler.step(current_val_fde)
        next_lr = float(optimizer.param_groups[0]["lr"])
        epoch_row["lr_next"] = next_lr
        history.append(epoch_row)
        save_json(job.output_dir / "metrics_history.json", history)
        save_checkpoint(
            job.output_dir / "last.pt", model, optimizer, epoch, val_metrics, job
        )
        log_message = f"{job.name} | epoch {epoch:03d} | lr={current_lr:.2e} -> {next_lr:.2e} | train loss={train_metrics['loss']:.4f} ade={train_metrics['ade']:.4f} fde={train_metrics['fde']:.4f} | val loss={val_metrics['loss']:.4f} ade={val_metrics['ade']:.4f} fde={val_metrics['fde']:.4f}"
        if shift_val_loader is not None:
            log_message += f" | shift_val loss={epoch_row['shift_val_loss']:.4f} ade={epoch_row['shift_val_ade']:.4f} fde={epoch_row['shift_val_fde']:.4f}"
        log(log_message)
        if current_val_fde < best_val_fde - config.EARLY_STOP_MIN_DELTA_FDE:
            best_val_fde = current_val_fde
            best_epoch = epoch
            patience_left = config.EARLY_STOP_PATIENCE
            save_checkpoint(
                job.output_dir / "best.pt", model, optimizer, epoch, val_metrics, job
            )
        else:
            patience_left -= 1
            if patience_left <= 0:
                log(
                    f"{job.name} early stopped at epoch {epoch}, best epoch was {best_epoch}."
                )
                break
    summary = {
        "job_name": job.name,
        "model_kind": job.model_kind,
        "vehicle_name": job.vehicle_name,
        "train_samples": len(job.train_samples),
        "val_samples": len(job.val_samples),
        "shift_val_samples": len(job.shift_val_samples),
        "best_epoch": best_epoch,
        "best_val_fde": best_val_fde,
        "split_strategy": job.split_strategy,
        "grouped_chunk_count": job.grouped_chunk_count,
        "train_only_group_count": job.train_only_group_count,
        "shift_group_count": job.shift_group_count,
        "init_checkpoint": str(loaded_init_checkpoint)
        if loaded_init_checkpoint is not None
        else None,
        "train_camera_signature_counts": count_samples_by_camera_signature(
            job.train_samples
        ),
        "val_camera_signature_counts": count_samples_by_camera_signature(
            job.val_samples
        ),
        "shift_val_camera_signature_counts": count_samples_by_camera_signature(
            job.shift_val_samples
        ),
    }
    save_json(job.output_dir / "job_summary.json", summary)


def build_training_jobs(dataset_index: DatasetIndex) -> List[TrainingJob]:
    jobs: List[TrainingJob] = []
    hv_by_vehicle: Dict[str, List[SampleRecord]] = defaultdict(list)
    av_by_vehicle: Dict[str, List[SampleRecord]] = defaultdict(list)
    for sample in dataset_index.samples:
        if sample.driver_mode == "AV":
            av_by_vehicle[sample.vehicle_name].append(sample)
        else:
            hv_by_vehicle[sample.vehicle_name].append(sample)
    if RUN_MODE in {"all", "hv"}:
        hv_train: List[SampleRecord] = []
        hv_val: List[SampleRecord] = []
        hv_shift_val: List[SampleRecord] = []
        hv_split_strategies: List[str] = []
        hv_grouped_chunk_count = 0
        hv_train_only_group_count = 0
        hv_shift_group_count = 0
        for vehicle_name, vehicle_samples in sorted(hv_by_vehicle.items()):
            split_bundle = split_samples_grouped_by_time(
                vehicle_samples, config.VAL_RATIO
            )
            hv_train.extend(split_bundle.train_samples)
            hv_val.extend(split_bundle.val_samples)
            hv_shift_val.extend(split_bundle.shift_val_samples)
            hv_split_strategies.append(split_bundle.split_strategy)
            hv_grouped_chunk_count += split_bundle.grouped_chunk_count
            hv_train_only_group_count += split_bundle.train_only_group_count
            hv_shift_group_count += split_bundle.shift_group_count
        if (
            len(hv_train) >= 1
            and len(hv_val) >= 1
            and (len(hv_train) + len(hv_val) >= config.MIN_SAMPLES_FOR_TRAINING)
        ):
            jobs.append(
                TrainingJob(
                    name="hv_shared",
                    model_kind="hv",
                    vehicle_name=None,
                    train_samples=sort_samples(hv_train),
                    val_samples=sort_samples(hv_val),
                    output_dir=OUTPUT_DIR / "hv_shared",
                    shift_val_samples=sort_samples(hv_shift_val),
                    split_strategy=collapse_split_strategies(hv_split_strategies),
                    grouped_chunk_count=hv_grouped_chunk_count,
                    train_only_group_count=hv_train_only_group_count,
                    shift_group_count=hv_shift_group_count,
                )
            )
        else:
            log("HV job was not created because the split is too small.")
    if RUN_MODE in {"all", "av"}:
        requested_av_vehicles = {
            str(name)
            for name in (
                *TARGET_AV_VEHICLES,
                *((TARGET_AV_VEHICLE,) if TARGET_AV_VEHICLE else ()),
            )
            if str(name).strip()
        }
        for vehicle_name, vehicle_samples in sorted(av_by_vehicle.items()):
            if (
                RUN_MODE == "av"
                and requested_av_vehicles
                and (vehicle_name not in requested_av_vehicles)
            ):
                continue
            assign_sequence_signatures(vehicle_samples)
            if len(vehicle_samples) < config.MIN_AV_SAMPLES_FOR_TRAINING:
                log(
                    f"Skipping AV vehicle={vehicle_name}: samples={len(vehicle_samples)} < MIN_AV_SAMPLES_FOR_TRAINING={config.MIN_AV_SAMPLES_FOR_TRAINING}."
                )
                continue
            split_bundle = split_samples_grouped_by_time(
                vehicle_samples, config.VAL_RATIO
            )
            if len(split_bundle.train_samples) < 1 or len(split_bundle.val_samples) < 1:
                log(
                    f"Skipping AV vehicle={vehicle_name}: not enough samples after grouped split (train={len(split_bundle.train_samples)}, val={len(split_bundle.val_samples)}, samples={len(vehicle_samples)})."
                )
                continue
            jobs.append(
                TrainingJob(
                    name=f"av_{slugify(vehicle_name)}",
                    model_kind="av",
                    vehicle_name=vehicle_name,
                    train_samples=split_bundle.train_samples,
                    val_samples=split_bundle.val_samples,
                    output_dir=OUTPUT_DIR / f"av_{slugify(vehicle_name)}",
                    shift_val_samples=split_bundle.shift_val_samples,
                    split_strategy=split_bundle.split_strategy,
                    grouped_chunk_count=split_bundle.grouped_chunk_count,
                    train_only_group_count=split_bundle.train_only_group_count,
                    shift_group_count=split_bundle.shift_group_count,
                    init_checkpoint=Path(AV_INIT_CHECKPOINT)
                    if config.ENABLE_HV_WARMSTART_FOR_AV
                    and AV_INIT_CHECKPOINT is not None
                    else OUTPUT_DIR / "hv_shared" / "best.pt"
                    if config.ENABLE_HV_WARMSTART_FOR_AV
                    else None,
                )
            )
    return jobs


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    add_source_arguments(parser)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-mode", choices=("all", "hv", "av"), default="all")
    parser.add_argument("--target-av-vehicle", action="append", default=[])
    parser.add_argument("--av-init-checkpoint", type=Path)
    parser.add_argument("--no-hv-warmstart", action="store_true")
    parser.add_argument("--device", default=config.DEVICE)
    parser.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    parser.add_argument("--num-workers", type=int, default=config.NUM_WORKERS)
    parser.add_argument("--epochs", type=int, default=config.NUM_EPOCHS)
    parser.add_argument("--dry-run-index", action="store_true")
    parser.add_argument("--incremental", action="store_true")
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error("epochs must be positive")
    if args.target_av_vehicle and args.run_mode != "av":
        parser.error("--target-av-vehicle requires --run-mode av")
    if args.no_hv_warmstart and args.av_init_checkpoint is not None:
        parser.error("--av-init-checkpoint conflicts with --no-hv-warmstart")
    return args


def main():
    global OUTPUT_DIR, RUN_MODE, TARGET_AV_VEHICLES, AV_INIT_CHECKPOINT
    args = parse_args()
    sources = sources_from_args(args)
    configure_loader(args)
    config.NUM_EPOCHS = args.epochs
    config.ENABLE_HV_WARMSTART_FOR_AV = not args.no_hv_warmstart
    RUN_MODE = args.run_mode
    TARGET_AV_VEHICLES = tuple(args.target_av_vehicle)
    AV_INIT_CHECKPOINT = (
        args.av_init_checkpoint.expanduser().resolve()
        if args.av_init_checkpoint
        else None
    )
    if AV_INIT_CHECKPOINT is not None and not AV_INIT_CHECKPOINT.is_file():
        raise FileNotFoundError("Initialization checkpoint does not exist")
    OUTPUT_DIR = args.output_dir.expanduser().resolve()
    seed_everything(config.RANDOM_SEED)
    device = choose_device(args.device)
    index = build_dataset_index(sources)
    jobs = build_training_jobs(index)
    if not jobs:
        raise ValueError(
            "No training jobs: check data, driving modes, vehicle names and split support"
        )
    names = [job.name for job in jobs]
    if len(names) != len(set(names)):
        raise ValueError("Vehicle names produce colliding model directory names")
    # Inspect every selected job before creating output or writing manifests.
    if OUTPUT_DIR.exists() and any(OUTPUT_DIR.iterdir()) and not args.incremental:
        raise FileExistsError(
            "Training output directory must be empty; --incremental permits only new jobs"
        )
    for job in jobs:
        if job.output_dir.exists() and any(job.output_dir.iterdir()):
            raise FileExistsError("Selected model job already has output")
        if not args.dry_run_index and job.init_checkpoint is not None:
            implicit_hv = OUTPUT_DIR / "hv_shared" / "best.pt"
            will_train_hv = any(candidate.model_kind == "hv" for candidate in jobs)
            if not job.init_checkpoint.is_file() and not (
                will_train_hv and job.init_checkpoint == implicit_hv
            ):
                raise FileNotFoundError(
                    "HV initialization checkpoint is missing; supply one or use --no-hv-warmstart"
                )
    prepare_output(OUTPUT_DIR, incremental=args.incremental)
    save_json(
        OUTPUT_DIR / "index_summary.json",
        {
            "kept_samples": len(index.samples),
            "dropped_reasons": index.dropped_reasons,
            "driver_mode_counts": index.mode_counts,
            "calibration_audit": calibration_audit_summary(index.audit_entries),
        },
    )
    for job in jobs:
        ensure_dir(job.output_dir)
        for split, samples in (
            ("train", job.train_samples),
            ("val", job.val_samples),
            ("shift_val", job.shift_val_samples),
        ):
            save_json(
                job.output_dir / f"{split}_manifest.json",
                to_manifest_rows(samples, split, job.split_strategy),
            )
    if args.dry_run_index:
        save_json(
            OUTPUT_DIR / "dry_run.json",
            {
                "kept_samples": len(index.samples),
                "jobs": [
                    {
                        "name": job.name,
                        "train_samples": len(job.train_samples),
                        "val_samples": len(job.val_samples),
                        "shift_val_samples": len(job.shift_val_samples),
                        "split_strategy": job.split_strategy,
                        "uses_initialization": job.init_checkpoint is not None,
                    }
                    for job in jobs
                ],
            },
        )
        log(
            f"Index validated: {len(index.samples)} samples, {len(jobs)} jobs; no training performed"
        )
        return
    for job in jobs:
        train_job(job, device)


if __name__ == "__main__":
    main()
