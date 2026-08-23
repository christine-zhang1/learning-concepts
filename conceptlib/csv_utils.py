"""Shared CSV helpers for the eval scripts.

Layout: one long-format CSV per eval type under RESULTS_ROOT, with all
dimensions (training_dataset, dataset_type, model, ablation,
max_content_words, randomized_synonyms, use_weighted_ce, max_train_samples,
use_data_augmentation, num_train_epochs, adapter_weight, ce_weight) as
explicit columns. Per-eval-type extras (e.g., evaluation_dataset) and metric
columns are appended by the caller.

Each call upserts: rows whose key columns match the suite key are deleted
first, then the new rows are written. Re-running an eval suite therefore
produces no duplicate rows.
"""
from __future__ import annotations

import csv
import os
from typing import Dict, Iterable, List, Sequence

from conceptlib.paths import RESULTS_ROOT

# Universal dimension columns written by every eval script.
COMMON_DIM_COLS: List[str] = [
    "training_dataset",
    "dataset_type",
    "model",
    "ablation",
    "max_content_words",
    "randomized_synonyms",
    "use_weighted_ce",
    "max_train_samples",
    "use_data_augmentation",
    "num_train_epochs",
    "adapter_weight",
    "ce_weight",
]


def model_suffix(base_model: str) -> str:
    return base_model.split("/")[-1].lower()


def derive_ablation(args) -> str:
    """Canonical ablation tag derived from variant flags.

    If args has an explicit non-empty `ablation` attribute, use it directly.
    """
    explicit = getattr(args, "ablation", None)
    if explicit:
        return explicit
    if getattr(args, "randomized_synonyms", False):
        return "randomized"
    if getattr(args, "use_weighted_ce", False):
        return "wce"
    if getattr(args, "use_data_augmentation", False):
        n = getattr(args, "num_train_epochs", None)
        return "data_aug_1" if n == 1 else "data_aug_2"
    mts = getattr(args, "max_train_samples", None)
    if mts is not None:
        if mts == 4000:
            return "half_train"
        if mts == 2000:
            return "quarter_train"
        return f"mts_{mts}"
    mcw = getattr(args, "max_content_words", None)
    if mcw == "half":
        return "half_words"
    if mcw == "quarter":
        return "quarter_words"
    if mcw == "one":
        return "one_words"
    return "normal"


def _bool_str(v) -> str:
    return "True" if bool(v) else "False"


def _opt_str(v) -> str:
    return "" if v is None else str(v)


def base_dim_row(args, training_dataset: str) -> Dict[str, str]:
    """Universal dim cols (no adapter_weight / ce_weight; caller fills those)."""
    return {
        "training_dataset": training_dataset,
        "dataset_type": args.dataset_type,
        "model": model_suffix(args.base_model),
        "ablation": derive_ablation(args),
        "max_content_words": getattr(args, "max_content_words", None) or "all",
        "randomized_synonyms": _bool_str(getattr(args, "randomized_synonyms", False)),
        "use_weighted_ce": _bool_str(getattr(args, "use_weighted_ce", False)),
        "max_train_samples": _opt_str(getattr(args, "max_train_samples", None)),
        "use_data_augmentation": _bool_str(getattr(args, "use_data_augmentation", False)),
        "num_train_epochs": _opt_str(getattr(args, "num_train_epochs", None)),
        # adapter_weight / ce_weight set by caller per row
        "adapter_weight": "",
        "ce_weight": "",
    }


def add_ablation_arg(parser):
    """Optional --ablation override; otherwise derive_ablation() is used."""
    parser.add_argument(
        "--ablation", default=None,
        help="Canonical ablation tag (e.g. normal, randomized, wce, data_aug_1, "
             "half_words). If unset, derived from variant flags.")


def results_csv(name: str) -> str:
    """Default csv path for a given eval-type basename."""
    return os.path.join(RESULTS_ROOT, f"{name}.csv")


def upsert_csv(
    csv_path: str,
    fieldnames: Sequence[str],
    key_cols: Sequence[str],
    key_values: Dict[str, object],
    new_rows: Iterable[Dict[str, object]],
) -> int:
    """Read csv_path, drop rows whose key_cols all match key_values, then write
    the kept rows + new_rows back. Creates parent dir as needed.

    Returns the number of new rows written.
    """
    key_str = {k: str(key_values[k]) for k in key_cols}
    kept: List[Dict[str, str]] = []
    if os.path.exists(csv_path) and os.path.getsize(csv_path) > 0:
        with open(csv_path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if all(row.get(k, "") == key_str[k] for k in key_cols):
                    continue
                kept.append(row)

    parent = os.path.dirname(csv_path)
    if parent:
        os.makedirs(parent, exist_ok=True)

    new_rows_list = [{k: ("" if v is None else str(v)) for k, v in r.items()}
                     for r in new_rows]

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in kept:
            writer.writerow({k: row.get(k, "") for k in fieldnames})
        for row in new_rows_list:
            writer.writerow({k: row.get(k, "") for k in fieldnames})

    return len(new_rows_list)
