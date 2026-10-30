"""Runtime helpers; importing the package does not read or create files."""

from __future__ import annotations
import json
import random
from pathlib import Path
import numpy as np
import torch
from . import config


def log(message):
    print(message, flush=True)


def slugify(text):
    value = "".join(ch.lower() if ch.isalnum() else "_" for ch in str(text))
    return "_".join(part for part in value.split("_") if part) or "unknown"


def ensure_dir(path):
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def choose_device(name):
    if str(name).startswith("cuda") and not torch.cuda.is_available():
        log("CUDA unavailable; using CPU")
        return torch.device("cpu")
    return torch.device(name)


def configure_loader(args):
    if args.batch_size < 1 or args.num_workers < 0:
        raise ValueError("batch-size must be positive and num-workers nonnegative")
    config.BATCH_SIZE = args.batch_size
    config.NUM_WORKERS = args.num_workers
    config.PERSISTENT_WORKERS = args.num_workers > 0


def prepare_output(path, incremental=False):
    path = Path(path).expanduser().resolve()
    if path.exists() and any(path.iterdir()) and not incremental:
        raise FileExistsError(
            "Output directory is not empty; supply a new directory or use --incremental for new model jobs"
        )
    path.mkdir(parents=True, exist_ok=True)
    return path


def batch_field_to_list(value):
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]
