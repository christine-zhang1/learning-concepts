"""Evaluate next-token prediction only at content-word positions.

Reads from the synonyms_test.jsonl validation file with fields:
  - input_sequence: str
  - content_word_responses: list[dict] with keys:
      - word: str       (content word string, used for verification)
      - position: int   (token-level position in the sequence)
      - synonyms: list[str]  (unused here)
      - scores: list[float]  (unused here)

Metrics are computed per content-word position (each valid content-word position
is one example), then averaged. With --non-target, the complement set (every
non-padding position that is NOT a content word) is scored instead.
"""
import sys
import pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import argparse
import os
import time
import json
import logging
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer
from tqdm import tqdm

from conceptlib import csv_utils as ecu
from conceptlib import paths
from conceptlib.eval_utils import (
    load_model_and_tokenizer,
    paired_bootstrap_cis,
    format_elapsed,
    free_model_memory,
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


ADAPTER_WEIGHTS = ["base", "0.0", "0.25", "0.5", "0.75", "1.0"]


def build_adapter_dir(args, adapter_weight, ce_concept_token_weight=None):
    """Construct the absolute adapter directory matching the training layout
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


# -------------------------
# Dataset
# -------------------------

class TargetWordValidationDataset(Dataset):
    """Pre-tokenized validation dataset with content_words and content_words_indices."""

    def __init__(
        self,
        data_path: str,
        tokenizer: AutoTokenizer,
        max_length: int = 256,
        max_samples: Optional[int] = None,
    ):
        self.encodings = []

        with open(data_path, "r", encoding="utf-8") as f:
            for line in tqdm(f, desc=f"Loading {data_path}"):
                line = line.strip()
                if not line:
                    continue
                item = json.loads(line)
                responses = item.get("content_word_responses", [])
                # Filter positions that fall within the (pre-truncation) valid range.
                valid = [
                    (r["position"], r["word"])
                    for r in responses
                    if 0 < r["position"] < max_length
                ]
                if not valid:
                    continue

                encoding = tokenizer(
                    item["input_sequence"],
                    truncation=True,
                    max_length=max_length,
                    return_tensors="pt",
                )
                actual_len = encoding["input_ids"].shape[1]
                valid = [(pos, word) for pos, word in valid if pos < actual_len]
                if not valid:
                    continue
                valid_indices, valid_words = zip(*valid)

                self.encodings.append({
                    "input_ids":            encoding["input_ids"].squeeze(0),
                    "attention_mask":       encoding["attention_mask"].squeeze(0),
                    "content_words_indices": list(valid_indices),
                    "content_words":         list(valid_words),
                })

                if max_samples and len(self.encodings) >= max_samples:
                    break

        logger.info(f"Loaded {len(self.encodings)} samples from {data_path}")

    def __len__(self):
        return len(self.encodings)

    def __getitem__(self, idx):
        return self.encodings[idx]


def collate_fn(batch):
    """Dynamically pad to the longest sequence in the batch; keep variable-length lists as lists-of-lists."""
    max_len = max(b["input_ids"].shape[0] for b in batch)
    padded_input_ids = []
    padded_attention_mask = []
    for b in batch:
        seq_len = b["input_ids"].shape[0]
        pad_len = max_len - seq_len
        padded_input_ids.append(torch.cat([b["input_ids"], b["input_ids"].new_zeros(pad_len)]))
        padded_attention_mask.append(torch.cat([b["attention_mask"], b["attention_mask"].new_zeros(pad_len)]))
    return {
        "input_ids": torch.stack(padded_input_ids),
        "attention_mask": torch.stack(padded_attention_mask),
        "content_words_indices": [b["content_words_indices"] for b in batch],
        "content_words": [b["content_words"] for b in batch],
    }


# -------------------------
# Evaluation
# -------------------------

@torch.no_grad()
def evaluate(model, tokenizer, dataset, batch_size=8, device="cuda", return_per_example=False, verify=True, non_target=False):
    """
    Evaluate NTP metrics over token positions.

    When non_target=False (default): evaluate only at content word positions.
    When non_target=True: evaluate at every non-padding position EXCEPT content
      word positions (i.e. the complement set).

    For each selected position `pos`, uses logits[b, pos-1] to predict
    input_ids[b, pos] and records confidence, correctness, and NLL.

    If verify=True (only meaningful in content-word mode), decodes the token at
    each target position and logs a warning when it does not match the expected
    content word string.
    """
    model.eval()
    model.to(device)

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    all_confidences = []
    all_correctness = []
    all_nll = []
    n_verified = 0
    n_mismatch = 0
    proportions = []  # fraction of (predictable) positions that are content words, per sequence

    desc = "Evaluating (non-target positions)" if non_target else "Evaluating (target positions)"
    for batch in tqdm(dataloader, desc=desc):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        content_words_indices_batch = batch["content_words_indices"]   # list[list[int]]
        content_words_batch = batch["content_words"]                   # list[list[str]]

        seq_lens_b = attention_mask.sum(dim=1).tolist()
        for b_i, (target_positions, seq_len) in enumerate(
            zip(content_words_indices_batch, seq_lens_b)
        ):
            predictable = int(seq_len) - 1  # positions 1..seq_len-1
            if predictable > 0:
                proportions.append(len(target_positions) / predictable)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits  # [B, seq_len, vocab]

        if non_target:
            # For each sequence, collect every valid non-padding position (pos > 0)
            # that is NOT a content word position.
            seq_lens = attention_mask.sum(dim=1).tolist()  # actual length per sequence
            valid_b = []
            valid_pos = []
            for b_i, (target_positions, seq_len) in enumerate(
                zip(content_words_indices_batch, seq_lens)
            ):
                target_set = set(target_positions)
                for pos in range(1, int(seq_len)):
                    if pos not in target_set:
                        valid_b.append(b_i)
                        valid_pos.append(pos)
            valid_words_flat = []  # not used in non-target mode

        else:
            # Default: content word positions only.
            valid_b = [b for b, positions in enumerate(content_words_indices_batch) for _ in positions]
            valid_pos = [pos for positions in content_words_indices_batch for pos in positions]
            valid_words_flat = [word for words in content_words_batch for word in words]

        if not valid_b:
            continue

        b_idx = torch.tensor(valid_b, device=device)
        pos_idx = torch.tensor(valid_pos, device=device)

        if torch.any(pos_idx == 0):
            logger.error("Target index 0 detected! Cannot predict the first token with no context.")
            mask = pos_idx > 0
            b_idx, pos_idx = b_idx[mask], pos_idx[mask]

        # logits[b, pos-1] predicts the token at pos — gather all at once.
        context_logits = logits[b_idx, pos_idx - 1]               # [N, vocab]
        log_probs = F.log_softmax(context_logits.float(), dim=-1)  # [N, vocab]
        probs = log_probs.exp()                                    # [N, vocab]

        true_tokens = input_ids[b_idx, pos_idx]              # [N]
        pred_tokens = probs.argmax(dim=-1)                   # [N]
        n_idx = torch.arange(b_idx.shape[0], device=device)

        confidences = probs.max(dim=-1).values               # [N]
        correctness = (pred_tokens == true_tokens)           # [N]
        nll_vals = -log_probs[n_idx, true_tokens]            # [N]

        if verify and not non_target:
            for i, expected_word in enumerate(valid_words_flat):
                decoded = tokenizer.decode([true_tokens[i].item()]).strip()
                n_verified += 1
                if (expected_word.strip().lower() not in decoded.lower()
                        and decoded.lower() not in expected_word.strip().lower()):
                    logger.warning(
                        f"Token mismatch at pos {valid_pos[i]}: "
                        f"expected '{expected_word}', got '{decoded}'"
                    )
                    n_mismatch += 1

        all_confidences.extend(confidences.tolist())
        all_correctness.extend(correctness.tolist())
        all_nll.extend(nll_vals.tolist())

    if proportions:
        logger.info(
            f"Content word proportion per sequence — "
            f"mean: {np.mean(proportions):.3f}, "
            f"min: {np.min(proportions):.3f}, "
            f"max: {np.max(proportions):.3f}"
        )

    if verify and not non_target:
        logger.info(
            f"Verification: {n_verified} positions checked, "
            f"{n_mismatch} mismatches ({100*n_mismatch/max(n_verified,1):.1f}%)"
        )

    total_examples = len(all_confidences)
    avg_conf = float(np.mean(all_confidences))
    accuracy = float(np.mean(all_correctness))
    avg_nll = float(np.mean(all_nll))
    perplexity = float(np.exp(avg_nll))

    out = {
        "accuracy": accuracy,
        "avg_confidence": avg_conf,
        "nll": avg_nll,
        "perplexity": perplexity,
        "num_examples": total_examples,
    }

    if return_per_example:
        out["_confidences"] = np.asarray(all_confidences)
        out["_correctness"] = np.asarray(all_correctness)
        out["_nll"] = np.asarray(all_nll)
    return out


def run_eval_for_adapter(adapter_path, dataset, args, return_per_example=False, ce_concept_token_weight=None):
    full_path = build_adapter_dir(args, adapter_path, ce_concept_token_weight)
    model, tokenizer = load_model_and_tokenizer(
        adapter_full_path=full_path,
        base_model_name=args.base_model,
    )
    metrics = evaluate(
        model,
        tokenizer,
        dataset,
        batch_size=args.batch_size,
        device=args.device,
        return_per_example=return_per_example,
        verify=args.verify,
        non_target=args.non_target,
    )
    return metrics


# -------------------------
# CLI
# -------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", type=str, required=True)
    parser.add_argument("--model-training-dataset", type=str, required=True)
    parser.add_argument("--evaluation-dataset", type=str, required=True,
                        help="Name of the evaluation dataset")
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--max-samples", type=int, default=1000)
    parser.add_argument("--csv-output", type=str, default=ecu.results_csv("ntp_content_words"),
                        help=f"Long-format CSV to upsert results into (default: {ecu.results_csv('ntp_content_words')}).")
    parser.add_argument("--randomized-synonyms", action="store_true")
    parser.add_argument("--max-content-words", type=str, default=None,
                        help="Choose between all, half, quarter, or one (default: all)")
    parser.add_argument("--no-verify", dest="verify", action="store_false", default=True,
                        help="Skip token/word verification logging")
    parser.add_argument("--non-target", action="store_true", default=False,
                        help="Evaluate on non-content-word positions instead of content-word positions")
    parser.add_argument("--use-weighted-ce", action="store_true")
    parser.add_argument("--dataset-type", type=str, required=True, choices=["embedding", "prompting"],
                        help="Dataset type used during training; part of the adapter path.")
    parser.add_argument("--max-train-samples", type=int, default=None,
                        help="If set, look under max-train-samples-{N}/ subdir.")
    parser.add_argument("--use-data-augmentation", action="store_true",
                        help="If set, look under aug_{N}epochs/ subdir.")
    parser.add_argument("--num-train-epochs", type=int, default=None,
                        help="Epoch count used to name the aug subdir (required with --use-data-augmentation).")
    ecu.add_ablation_arg(parser)

    args = parser.parse_args()

    if args.use_data_augmentation and args.num_train_epochs is None:
        parser.error("--use-data-augmentation requires --num-train-epochs")

    adapter_weights = list(ADAPTER_WEIGHTS)
    if (args.randomized_synonyms
            or (args.max_content_words and args.max_content_words != "all")):
        # Prepend "base" so paired-bootstrap diff CIs are computed vs. base
        # instead of vs. the first NONZERO adapter.
        adapter_weights = ["base", "0.25", "0.5", "0.75", "1.0"]
    elif args.use_data_augmentation:
        adapter_weights = ["base", "0.0"]

    logger.info(f"Loading tokenizer for {args.base_model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    data_path = os.path.join(
        paths.dataset_dir(args.evaluation_dataset, args.base_model, args.dataset_type),
        "synonyms_test.jsonl",
    )
    logger.info(f"Loading validation dataset from {data_path}...")
    dataset = TargetWordValidationDataset(
        data_path=data_path,
        tokenizer=tokenizer,
        max_length=args.max_length,
        max_samples=args.max_samples,
    )
    logger.info(f"Dataset: {len(dataset)} samples")

    per_model_metrics = []
    metrics_per_adapter = []

    if args.use_weighted_ce:
        # ce_weight=1.0 isn't trained; prepend base model so wce runs get
        # paired diff CIs against base rather than against an untrained ce=1.0.
        ce_weights = [2.0, 5.0, 10.0, 20.0, 50.0, 100.0]
        iter_configs = [("base", None)] + [("0.0", w) for w in ce_weights]
    else:
        iter_configs = [(ap, None) for ap in adapter_weights]

    for adapter_path, ce_weight in iter_configs:
        logger.info(f"\n--- Adapter: {adapter_path}{f', ce_weight={ce_weight}' if ce_weight is not None else ''} ---")
        _t_start = time.time()
        try:
            logger.info(f"Running evaluation for adapter: {adapter_path}")
            metrics = run_eval_for_adapter(
                adapter_path, dataset, args, return_per_example=True, ce_concept_token_weight=ce_weight
            )
            per_model_metrics.append({
                "_confidences": metrics["_confidences"],
                "_correctness": metrics["_correctness"],
                "_nll": metrics["_nll"],
            })
            metrics_per_adapter.append(metrics)
            logger.info(
                f"  accuracy={metrics['accuracy']:.4f}  "
                f"avg_conf={metrics['avg_confidence']:.4f}  "
                f"nll={metrics['nll']:.4f}  "
                f"ppl={metrics['perplexity']:.4f}  "
                f"n={metrics['num_examples']}"
            )
        finally:
            elapsed = time.time() - _t_start
            logger.info(f"  Time for adapter {adapter_path}{f', ce_weight={ce_weight}' if ce_weight is not None else ''}: {format_elapsed(elapsed)}")
            free_model_memory()

    if args.use_weighted_ce:
        run_labels = ["base" if ce_w is None else str(ce_w) for _, ce_w in iter_configs]
    else:
        run_labels = [ap for ap, _ in iter_configs]
    per_model_cis, diff_cis = paired_bootstrap_cis(per_model_metrics)
    print("\n--- Paired-bootstrap CIs ---")
    for m, label in enumerate(run_labels):
        c = per_model_cis[m]
        print(f"  {label}: acc {c['accuracy_ci']}  ppl {c['perplexity_ci']}")
    print("--- CI of difference vs first run ---")
    for j in range(1, len(run_labels)):
        d = diff_cis[(0, j)]
        print(f"  {run_labels[j]} - {run_labels[0]}: acc {d['accuracy_ci']}  ppl {d['perplexity_ci']}")

    if args.csv_output:
        base_dim = ecu.base_dim_row(args, training_dataset=args.model_training_dataset)
        base_dim["evaluation_dataset"] = args.evaluation_dataset
        metric_cols = [
            "accuracy", "accuracy_ci_lo", "accuracy_ci_hi",
            "accuracy_diff_ci_lo", "accuracy_diff_ci_hi",
            "avg_confidence",
            "nll",
            "perplexity", "perplexity_ci_lo", "perplexity_ci_hi",
            "perplexity_diff_ci_lo", "perplexity_diff_ci_hi",
            "num_examples",
        ]
        fieldnames = ecu.COMMON_DIM_COLS + ["evaluation_dataset"] + metric_cols
        key_cols = ["training_dataset", "dataset_type", "model", "ablation",
                    "max_content_words", "evaluation_dataset"]
        new_rows = []
        for m, ((adapter_path, ce_w), metrics) in enumerate(zip(iter_configs, metrics_per_adapter)):
            c = per_model_cis[m]
            d = diff_cis.get((0, m))
            row = dict(base_dim)
            row["adapter_weight"] = adapter_path
            row["ce_weight"] = "" if ce_w is None else ce_w
            row["accuracy"] = f"{metrics['accuracy']:.6f}"
            row["accuracy_ci_lo"] = f"{c['accuracy_ci'][0]:.6f}"
            row["accuracy_ci_hi"] = f"{c['accuracy_ci'][1]:.6f}"
            row["accuracy_diff_ci_lo"] = "" if d is None else f"{d['accuracy_ci'][0]:.6f}"
            row["accuracy_diff_ci_hi"] = "" if d is None else f"{d['accuracy_ci'][1]:.6f}"
            row["avg_confidence"] = f"{metrics['avg_confidence']:.6f}"
            row["nll"] = f"{metrics['nll']:.6f}"
            row["perplexity"] = f"{metrics['perplexity']:.6f}"
            row["perplexity_ci_lo"] = f"{c['perplexity_ci'][0]:.6f}"
            row["perplexity_ci_hi"] = f"{c['perplexity_ci'][1]:.6f}"
            row["perplexity_diff_ci_lo"] = "" if d is None else f"{d['perplexity_ci'][0]:.6f}"
            row["perplexity_diff_ci_hi"] = "" if d is None else f"{d['perplexity_ci'][1]:.6f}"
            row["num_examples"] = metrics["num_examples"]
            new_rows.append(row)
        n = ecu.upsert_csv(args.csv_output, fieldnames, key_cols, base_dim, new_rows)
        logger.info(f"Upserted {n} rows into {args.csv_output}")


if __name__ == "__main__":
    main()
