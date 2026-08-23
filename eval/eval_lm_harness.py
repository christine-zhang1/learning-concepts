"""Sweep lm-evaluation-harness across concept-trained LoRA adapters.

For each adapter weight in a training configuration, invokes the `lm_eval` CLI
with `--model hf --model_args pretrained=BASE,peft=ADAPTER,load_in_4bit=True,...`
and writes the harness JSON to a per-adapter subdirectory under
`CONCEPT_RESULTS_ROOT/lm_harness_runs/`. After all adapters finish, the main
metric of every task is flattened and upserted into a long-format CSV (one row
per adapter/task/metric), matching the other eval scripts.

Adapters are located with `conceptlib.paths.adapter_dir` — the same builder
`train.py` uses to write checkpoints — so the lookup never drifts from training.

The default task suite is the six likelihood-scored benchmarks reported in the
paper: ARC-Challenge, ARC-Easy, OpenBookQA, HellaSwag, PIQA, and WinoGrande.
All are scored by likelihood, so they suit non-instruction-tuned base LMs.
Override with `--tasks-suite custom --tasks ...`.

Example:
  python eval/eval_lm_harness.py \
      --base-model Qwen/Qwen3-1.7B-Base \
      --model-training-dataset c4 \
      --dataset-type embedding \
      --limit 1000
"""
import sys
import pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import argparse
import json
import os
import subprocess
import time
from pathlib import Path

from conceptlib import csv_utils as ecu
from conceptlib import paths
from conceptlib.eval_utils import format_elapsed


ADAPTER_WEIGHTS = ["base", "0.0", "0.25", "0.5", "0.75", "1.0"]

TASK_SUITES = {
    "core": [
        "arc_challenge",
        "arc_easy",
        "openbookqa",
        "hellaswag",
        "piqa",
        "winogrande",
    ],
}


def build_adapter_dir(args, adapter_weight, ce_concept_token_weight=None):
    """Absolute adapter directory matching the training layout
    (conceptlib.paths.adapter_dir). Returns None for the bare base model."""
    if adapter_weight == "base":
        return None
    return paths.adapter_dir(
        args.model_training_dataset, args.base_model, args.dataset_type,
        max_content_words=args.max_content_words,
        randomized_synonyms=args.randomized_synonyms,
        use_weighted_ce=args.use_weighted_ce,
        ce_concept_token_weight=ce_concept_token_weight,
        max_train_samples=args.max_train_samples,
        use_data_augmentation=args.use_data_augmentation,
        num_train_epochs=args.num_train_epochs,
        concept_loss_weight=adapter_weight,
    )


def build_model_args(base_model, adapter_path):
    parts = [
        f"pretrained={base_model}",
        "dtype=float16",
        "load_in_4bit=True",
        "bnb_4bit_quant_type=nf4",
        "bnb_4bit_compute_dtype=float16",
        "bnb_4bit_use_double_quant=False",
    ]
    if adapter_path is not None:
        parts.insert(1, f"peft={adapter_path}")
    return ",".join(parts)


def run_lm_eval(args, adapter_path, out_subdir, log_path):
    out_subdir.mkdir(parents=True, exist_ok=True)
    model_args = build_model_args(args.base_model, adapter_path)
    cmd = [
        "lm_eval",
        "--model", "hf",
        "--model_args", model_args,
        "--tasks", ",".join(args.tasks),
        "--batch_size", str(args.batch_size),
        "--device", args.device,
        "--output_path", str(out_subdir),
    ]
    if args.num_fewshot is not None:
        cmd += ["--num_fewshot", str(args.num_fewshot)]
    if args.limit is not None:
        cmd += ["--limit", str(args.limit)]
    if args.trust_remote_code:
        cmd += ["--trust_remote_code"]
    if args.extra_args:
        cmd += args.extra_args.split()

    print("$ " + " ".join(cmd), flush=True)
    with open(log_path, "w") as logf:
        logf.write("$ " + " ".join(cmd) + "\n")
        logf.flush()
        subprocess.run(cmd, check=True, stdout=logf, stderr=subprocess.STDOUT)


def find_results_json(out_subdir):
    """The harness writes results_<timestamp>.json (possibly nested). Pick the newest."""
    candidates = sorted(out_subdir.rglob("results_*.json"), key=lambda p: p.stat().st_mtime)
    return candidates[-1] if candidates else None


def flatten_results(json_path):
    with open(json_path) as f:
        data = json.load(f)
    rows = []
    for task, metrics in data.get("results", {}).items():
        for k, v in metrics.items():
            # harness keys look like "acc,none" / "acc_stderr,none" — keep main metrics
            if k == "alias" or k.endswith("_stderr") or ("," in k and k.split(",")[0].endswith("_stderr")):
                continue
            if isinstance(v, (int, float)):
                rows.append((task, k, v))
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--base-model", required=True)
    p.add_argument("--model-training-dataset", required=True,
                   help="Training dataset name used in the adapter path (c4, openwebtext, ...)")
    p.add_argument("--tasks-suite", default="core",
                   choices=list(TASK_SUITES.keys()) + ["custom"])
    p.add_argument("--tasks", default=None,
                   help="Comma-separated task list. Required when --tasks-suite=custom; "
                        "otherwise overrides the suite.")
    p.add_argument("--randomized-synonyms", action="store_true")
    p.add_argument("--max-content-words", default=None,
                   help="Choose between all, half, quarter, or one (default: all)")
    p.add_argument("--use-weighted-ce", action="store_true",
                   help="Sweep CE weights instead of concept weights.")
    p.add_argument("--dataset-type", required=True, choices=["embedding", "prompting"],
                   help="Dataset type used during training; part of the adapter path.")
    p.add_argument("--max-train-samples", type=int, default=None,
                   help="If set, look under max-train-samples-{N}/ subdir.")
    p.add_argument("--use-data-augmentation", action="store_true",
                   help="If set, look under aug_{N}epochs/ subdir.")
    p.add_argument("--num-train-epochs", type=int, default=None,
                   help="Epoch count used to name the aug subdir (required with --use-data-augmentation).")
    p.add_argument("--adapters", default=None,
                   help="Comma-separated subset of adapter weights to run (default: all)")
    p.add_argument("--num-fewshot", type=int, default=0)
    p.add_argument("--limit", type=int, default=None,
                   help="Limit examples per task (for fast iteration).")
    p.add_argument("--batch-size", default="auto",
                   help="Pass through to lm_eval; 'auto' lets harness pick.")
    p.add_argument("--device", default="cuda")
    p.add_argument("--trust-remote-code", action="store_true")
    p.add_argument("--extra-args", default=None,
                   help="Extra raw args appended to the lm_eval invocation.")
    p.add_argument("--out-dir", default=None,
                   help="Override output directory (default: <results>/lm_harness_runs/<sweep>/)")
    p.add_argument("--rerun", action="store_true",
                   help="Re-run even if a results JSON already exists for an adapter.")
    p.add_argument("--csv-output", default=ecu.results_csv("lm_harness"),
                   help=f"Long-format CSV to upsert results into "
                        f"(default: {ecu.results_csv('lm_harness')}).")
    ecu.add_ablation_arg(p)
    args = p.parse_args()

    if args.use_data_augmentation and args.num_train_epochs is None:
        p.error("--use-data-augmentation requires --num-train-epochs")

    if args.tasks:
        args.tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
    elif args.tasks_suite == "custom":
        p.error("--tasks is required when --tasks-suite=custom")
    else:
        args.tasks = TASK_SUITES[args.tasks_suite]

    adapter_weights = list(ADAPTER_WEIGHTS)
    if (args.randomized_synonyms
            or (args.max_content_words and args.max_content_words != "all")):
        adapter_weights = ["0.25", "0.5", "0.75", "1.0"]
    elif args.use_data_augmentation:
        adapter_weights = ["base", "0.0"]
    if args.adapters:
        adapter_weights = [a.strip() for a in args.adapters.split(",") if a.strip()]

    if args.use_weighted_ce:
        ce_weights = [2.0, 5.0, 10.0, 20.0, 50.0, 100.0]
        configs = [("0.0", w) for w in ce_weights]
    else:
        configs = [(w, None) for w in adapter_weights]

    model_suffix = args.base_model.split("/")[-1].lower()
    suite_tag = args.tasks_suite
    content_tag = f"{args.max_content_words or 'all'}_content_words"
    parts = [args.model_training_dataset, model_suffix, args.dataset_type, suite_tag, content_tag]
    if args.randomized_synonyms:
        parts.append("randomized_synonyms")
    elif args.use_weighted_ce:
        parts.append("wce")
    elif args.max_train_samples is not None:
        parts.append(f"max-train-samples-{args.max_train_samples}")
    elif args.use_data_augmentation:
        parts.append(f"aug_{args.num_train_epochs}epochs")
    out_dir = Path(args.out_dir or os.path.join(paths.RESULTS_ROOT, "lm_harness_runs", *parts))
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {out_dir}")
    print(f"Tasks: {args.tasks}")
    print(f"Configs: {configs}")

    csv_rows = []
    for adapter_weight, ce_weight in configs:
        tag = adapter_weight + (f"-wce{ce_weight}" if ce_weight is not None else "")
        sub = out_dir / tag
        log_path = out_dir / f"{tag}.log"

        adir = build_adapter_dir(args, adapter_weight, ce_concept_token_weight=ce_weight)
        if adir is not None and not Path(adir).exists():
            raise FileNotFoundError(f"adapter dir does not exist: {adir}")

        existing = find_results_json(sub)
        if existing and not args.rerun:
            print(f"[cache] {existing} (use --rerun to redo)", flush=True)
        else:
            _t_start = time.time()
            run_lm_eval(args, adir, sub, log_path)
            elapsed = time.time() - _t_start
            print(f"  Time for adapter {tag}: {format_elapsed(elapsed)}", flush=True)
            existing = find_results_json(sub)
            if existing is None:
                raise RuntimeError(f"no results JSON in {sub} after lm_eval run")

        base_dim = ecu.base_dim_row(args, training_dataset=args.model_training_dataset)
        base_dim["tasks_suite"] = args.tasks_suite
        for task, metric, value in flatten_results(existing):
            row = dict(base_dim)
            row["adapter_weight"] = adapter_weight
            row["ce_weight"] = "" if ce_weight is None else ce_weight
            row["task"] = task
            row["metric"] = metric
            row["value"] = value
            csv_rows.append(row)

    if csv_rows:
        fieldnames = ecu.COMMON_DIM_COLS + ["tasks_suite", "task", "metric", "value"]
        key_cols = ["training_dataset", "dataset_type", "model", "ablation",
                    "max_content_words", "tasks_suite"]
        # All emitted rows share the same suite key; reuse the first.
        n = ecu.upsert_csv(args.csv_output, fieldnames, key_cols, csv_rows[0], csv_rows)
        print(f"\nUpserted {n} rows into {args.csv_output}")
    else:
        print("\nNo results to write.")


if __name__ == "__main__":
    main()
