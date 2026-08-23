"""Evaluate base + LoRA adapters on MTEB tasks.

- Uses MTEB's official task suite (correct splits, multiple tasks).
- Defaults to 4-bit NF4 to match QLoRA training; pass --no-quantize for bf16.
- Mean pooling (over real tokens) by default; pass --pooling last for last-token.

Adapter paths and the results CSV are resolved through conceptlib (shared with
the training and other eval scripts).

Requires: pip install mteb (>=1.12)

Examples:
  # Quick: STS suite only
  python eval/eval_mteb.py --base-model Qwen/Qwen3-4B-Base \\
      --dataset c4 --dataset-type embedding --tasks sts

  # Short-text subset of MTEB(eng, v2)
  python eval/eval_mteb.py ... --tasks short

  # Specific tasks
  python eval/eval_mteb.py ... --tasks STSBenchmark,SICK-R,Banking77Classification

"""
import sys
import pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import argparse
import gc
import os
import time
import traceback
from typing import List, Optional

import numpy as np
import torch
import torch.nn.functional as F

torch.set_grad_enabled(False)
torch.backends.cuda.enable_cudnn_sdp(False)
from peft import PeftModel
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from mteb.models.abs_encoder import AbsEncoder

from conceptlib import csv_utils as ecu
from conceptlib import paths
from conceptlib.eval_utils import format_elapsed

ADAPTER_WEIGHTS = ["base", "0.0", "0.25", "0.5", "0.75", "1.0"]


# -------------------------------------------------
# CLI
# -------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--base-model", type=str, required=True)
parser.add_argument("--dataset", type=str, required=True,
                    help="Training dataset name (used for adapter path lookup, e.g. c4).")
parser.add_argument("--batch_size", type=int, default=16)
parser.add_argument("--max_length", type=int, default=256,
                    help="Max sequence length for the encoder. 256 matches the "
                         "short-text preset bound (every task <=256 tokens under "
                         "the Qwen3 tokenizer).")
parser.add_argument("--seed", type=int, default=42,
                    help="Seed for torch / numpy / MTEB. Clustering tasks (KMeans) "
                         "are sensitive to init; setting a seed makes ablation deltas "
                         "reproducible.")
parser.add_argument("--layer", type=int, default=-1,
                    help="Hidden-states index to pool from (-1 = final layer).")
parser.add_argument("--randomized-synonyms", action="store_true")
parser.add_argument(
    "--csv-output",
    type=str,
    default=ecu.results_csv("mteb"),
    help="Long-format CSV (one row per adapter_weight x task).",
)
parser.add_argument("--mteb-output-root", type=str, default="mteb_runs",
                    help="MTEB writes per-task JSONs under {root}/{model_tag}/{adapter_tag}/")
parser.add_argument("--max-content-words", type=str, default=None,
                    choices=[None, "all", "half", "quarter", "one"])
parser.add_argument("--use-weighted-ce", action="store_true")
parser.add_argument("--dataset-type", type=str, required=True,
                    choices=["embedding", "prompting"])
parser.add_argument("--max-train-samples", type=int, default=None)
parser.add_argument("--use-data-augmentation", action="store_true")
parser.add_argument("--num-train-epochs", type=int, default=None)

# MTEB-specific
parser.add_argument("--tasks", type=str, default="sts",
                    help='Comma-separated task names, OR a preset: '
                         '"sts" (9 STS tasks, taken directly from MTEB(eng,v2), English-only), '
                         '"short" (short-text subset of MTEB(eng,v2), every text <=256 tokens).')
parser.add_argument("--no-quantize", dest="quantize_4bit", action="store_false",
                    help="Load base model in bf16 instead of 4-bit NF4. "
                         "Default is 4-bit to match QLoRA training conditions.")
parser.set_defaults(quantize_4bit=True)
parser.add_argument("--pooling", choices=["mean", "last"], default="mean",
                    help="Pooling strategy. mean (default) is preferred for "
                         "non-contrastively-trained decoder LMs; last matches "
                         "the SOTA contrastive embedders (E5-Mistral, GritLM).")
parser.add_argument("--eval-splits", type=str, default=None,
                    help="Comma-separated splits override (default: use MTEB's default per task).")
parser.add_argument("--only-adapter-weights", type=str, default=None,
                    help="Comma-list filter applied to the default adapter-weight "
                         "sweep (e.g. 'base,0.0,0.25'). Must be a subset of the "
                         "ablation's sweep; an unknown weight errors out.")
parser.add_argument("--only-ce-weights", type=str, default=None,
                    help="Like --only-adapter-weights but for the --use-weighted-ce "
                         "ce-weight sweep (e.g. '2.0,5.0,10.0').")
ecu.add_ablation_arg(parser)
args = parser.parse_args()

if args.use_data_augmentation and args.num_train_epochs is None:
    parser.error("--use-data-augmentation requires --num-train-epochs")

if (args.randomized_synonyms
        or (args.max_content_words and args.max_content_words != "all")):
    ADAPTER_WEIGHTS = ["0.25", "0.5", "0.75", "1.0"]
elif args.use_data_augmentation:
    ADAPTER_WEIGHTS = ["base", "0.0"]

if args.only_adapter_weights:
    _keep = [s.strip() for s in args.only_adapter_weights.split(",") if s.strip()]
    _missing = sorted(set(_keep) - set(ADAPTER_WEIGHTS))
    assert not _missing, (
        f"--only-adapter-weights has values not in this ablation's sweep: "
        f"{_missing}; available={ADAPTER_WEIGHTS}")
    ADAPTER_WEIGHTS = [w for w in ADAPTER_WEIGHTS if w in set(_keep)]


# -------------------------------------------------
# Adapter path (shared builder in conceptlib.paths)
# -------------------------------------------------
def build_adapter_dir(adapter_weight: str, ce_concept_token_weight: Optional[float] = None):
    if adapter_weight == "base":
        return None
    return paths.adapter_dir(
        args.dataset, args.base_model, args.dataset_type,
        max_content_words=args.max_content_words,
        randomized_synonyms=args.randomized_synonyms,
        use_weighted_ce=args.use_weighted_ce,
        ce_concept_token_weight=ce_concept_token_weight,
        max_train_samples=args.max_train_samples,
        use_data_augmentation=args.use_data_augmentation,
        num_train_epochs=args.num_train_epochs,
        concept_loss_weight=adapter_weight,
    )


# -------------------------------------------------
# Model loading
# -------------------------------------------------
def load_model_and_tokenizer(adapter_full_path: Optional[str]):
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    # Right-pad: mean pooling uses the attention_mask explicitly to ignore pad
    # positions, so padding side doesn't affect the result. Right-pad is the
    # HF default and avoids odd interactions with truncation.
    tokenizer.padding_side = "right"

    load_kwargs = {"device_map": {"": "cuda:0"}}
    if args.quantize_4bit:
        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=False,
        )
    else:
        load_kwargs["torch_dtype"] = torch.bfloat16

    base = AutoModelForCausalLM.from_pretrained(args.base_model, **load_kwargs)
    if adapter_full_path is None:
        return base, tokenizer
    model = PeftModel.from_pretrained(base, adapter_full_path)
    return model, tokenizer


# -------------------------------------------------
# Encoder wrapper that MTEB calls
# -------------------------------------------------
class PooledEncoder(AbsEncoder):
    """MTEB AbsEncoder over a (PEFT-wrapped) causal LM with configurable pooling.

    MTEB 2.x passes a `DataLoader[BatchedInput]` to `encode`. For text tasks
    each batch is `{"text": [<sentences...>]}`. We tokenize on the fly so we
    don't need to pre-load the whole dataset.

    Pooling modes:
      - "mean": mask-aware mean over real (non-pad) token hidden states.
                Preferred for non-contrastively-trained decoder LMs.
      - "last": hidden state at the final real token (per row, via mask).
                Matches E5-Mistral / GritLM / SGPT convention.
    """

    mteb_model_meta = None  # Required attribute on the EncoderProtocol.

    def __init__(self, model, tokenizer, *, max_length: int,
                 layer: int, pooling: str):
        assert pooling in ("mean", "last")
        self.model = model
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.layer = layer
        self.pooling = pooling
        self.device = next(model.parameters()).device

    def _pool(self, hidden, attention_mask):
        mask = attention_mask.to(hidden.dtype).unsqueeze(-1)  # (B, T, 1)
        if self.pooling == "mean":
            summed = (hidden * mask).sum(dim=1)
            counts = mask.sum(dim=1).clamp(min=1.0)
            return summed / counts
        # "last": index of the last real token per row (works with right-pad).
        assert self.tokenizer.padding_side == "right"
        lengths = attention_mask.sum(dim=1) - 1
        rows = torch.arange(hidden.size(0), device=hidden.device)
        return hidden[rows, lengths]

    @torch.no_grad()
    def encode(self, inputs, *, task_metadata=None, hf_split=None,
               hf_subset=None, prompt_type=None, **kwargs):
        task_name = getattr(task_metadata, "name", None) or "task"
        out: List[np.ndarray] = []
        disable_tqdm = os.environ.get("SLURM_JOB_ID") is not None
        for batch in tqdm(
            inputs,
            desc=f"encode({task_name})",
            leave=False,
            disable=disable_tqdm,
        ):
            sentences = batch["text"]
            if isinstance(sentences, str):
                sentences = [sentences]
            texts = list(sentences)
            enc = self.tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            ).to(self.device)
            outputs = self.model(
                input_ids=enc["input_ids"],
                attention_mask=enc["attention_mask"],
                output_hidden_states=True,
            )
            hidden = outputs.hidden_states[self.layer]
            pooled = self._pool(hidden, enc["attention_mask"])
            pooled = F.normalize(pooled, p=2, dim=1)
            out.append(pooled.cpu().float().numpy())
        return np.concatenate(out, axis=0)


# -------------------------------------------------
# Task selection
# -------------------------------------------------
# STS tasks from MTEB(eng, v2); resolve_tasks pulls the benchmark's own task
# objects so the English subsets and eval splits match it exactly.
STS_PRESET = [
    "STS12", "STS13", "STS14", "STS15", "STS17", "STS22.v2",
    "STSBenchmark", "SICK-R", "BIOSSES",
]

# Short-text subset of MTEB(eng, v2): tasks whose texts fit within 256 tokens.
# STS tasks are short too but are covered by the "sts" preset, so they're omitted.
MTEB_SHORT_PRESET = [
    "AmazonCounterfactualClassification",
    "ArXivHierarchicalClusteringS2S",
    "AskUbuntuDupQuestions",
    "Banking77Classification",
    "MassiveIntentClassification",
    "MassiveScenarioClassification",
    "MedrxivClusteringS2S.v2",
    "MTOPDomainClassification",
    "SprintDuplicateQuestions",
    "StackExchangeClustering.v2",
    "ToxicConversationsClassification",
    "TweetSentimentExtractionClassification",
    "TwentyNewsgroupsClustering.v2",
    "TwitterSemEval2015",
    "TwitterURLCorpus",
]


def resolve_tasks(spec: str):
    import mteb
    spec = spec.strip()
    # English-only filter: keep only subsets whose languages are all English,
    # dropping the cross-lingual pairs that multilingual tasks otherwise include.
    eng_only = dict(languages=["eng"], exclusive_language_filter=True)
    if spec == "sts":
        bench = {t.metadata.name: t for t in mteb.get_benchmark("MTEB(eng, v2)").tasks}
        missing = [n for n in STS_PRESET if n not in bench]
        assert not missing, f"STS_PRESET tasks absent from MTEB(eng, v2): {missing}"
        return [bench[n] for n in STS_PRESET]
    if spec == "short":
        bench = {t.metadata.name: t for t in mteb.get_benchmark("MTEB(eng, v2)").tasks}
        missing = [n for n in MTEB_SHORT_PRESET if n not in bench]
        assert not missing, f"MTEB_SHORT_PRESET tasks absent from MTEB(eng, v2): {missing}"
        return [bench[n] for n in MTEB_SHORT_PRESET]
    names = [t.strip() for t in spec.split(",") if t.strip()]
    return mteb.get_tasks(tasks=names, **eng_only)


# -------------------------------------------------
# Score extraction
# -------------------------------------------------
def extract_main_scores(task_result) -> dict:
    """Return {split: main_score} from an MTEB TaskResult, robust to API drift."""
    scores = getattr(task_result, "scores", None) or {}
    out = {}
    for split, payload in scores.items():
        # payload is usually a list of dicts (one per hf subset/language) or a dict.
        items = payload if isinstance(payload, list) else [payload]
        vals = []
        for item in items:
            if not isinstance(item, dict):
                continue
            if "main_score" in item:
                vals.append(item["main_score"])
        if vals:
            out[split] = float(np.mean(vals))
    return out


# -------------------------------------------------
# Per-adapter run
# -------------------------------------------------
def run_mteb_for_adapter(adapter_weight: str, tasks, ce_weight: Optional[float] = None):
    import mteb
    full_path = build_adapter_dir(adapter_weight, ce_weight)
    model_obj, tok = load_model_and_tokenizer(full_path)
    model_obj.eval()

    encoder = PooledEncoder(
        model_obj, tok,
        max_length=args.max_length,
        layer=args.layer,
        pooling=args.pooling,
    )

    model_tag = ecu.model_suffix(args.base_model)
    adapter_tag = adapter_weight if ce_weight is None else f"wce-{ce_weight}_aw-{adapter_weight}"
    ablation_tag = ecu.derive_ablation(args)
    out_dir = os.path.join(
        args.mteb_output_root,
        args.dataset,
        args.dataset_type,
        model_tag,
        ablation_tag,
        adapter_tag,
    )
    os.makedirs(out_dir, exist_ok=True)

    run_kwargs = dict(
        output_folder=out_dir,
        encode_kwargs={"batch_size": args.batch_size},
    )
    if args.eval_splits:
        run_kwargs["eval_splits"] = [s.strip() for s in args.eval_splits.split(",") if s.strip()]

    evaluation = mteb.MTEB(tasks=tasks)
    try:
        # raise_error=False so one failing task doesn't abort the whole adapter;
        # finished tasks are cached under out_dir and skipped on re-run.
        results = evaluation.run(encoder, raise_error=False, **run_kwargs)

        # Flatten: {task_name: {split: score}}
        flat = {}
        for tr in results:
            if hasattr(tr, "task") and hasattr(tr.task, "metadata"):
                task_name = tr.task.metadata.name
            else:
                task_name = getattr(tr, "task_name", "unknown_task")
            flat[task_name] = extract_main_scores(tr)
        return flat
    finally:
        # Always free the GPU, even if run()/flatten raised, so the next adapter
        # in the suite starts from a clean slate.
        del model_obj, encoder
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# -------------------------------------------------
# Main
# -------------------------------------------------
def main():
    import random
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    tasks = resolve_tasks(args.tasks)
    # MTEB AbsTask carries self.seed (default 42); override for non-default seeds
    # so KMeans / downsampling RNGs match across ablation runs.
    if args.seed != 42:
        from mteb.abstasks.abstask import _set_seed
        for t in tasks:
            t.seed = args.seed
            t.rng_state, t.np_rng = _set_seed(args.seed)
    task_names = [getattr(t, "metadata", t).name if hasattr(t, "metadata") else str(t) for t in tasks]
    print(f"Running {len(tasks)} task(s) with seed={args.seed}, max_length={args.max_length}")
    print(f"Tasks: {task_names}")

    # adapter_weight -> {task_name: {split: score}}
    all_results: dict = {}

    # CSV schema fixed up-front so we can flush after *every* adapter: a crash on
    # a later adapter then never discards the adapters already finished.
    base_dim = ecu.base_dim_row(args, training_dataset=args.dataset)
    fieldnames = ecu.COMMON_DIM_COLS + ["eval_task", "eval_split", "main_score"]
    key_cols = ["training_dataset", "dataset_type", "model", "ablation",
                "max_content_words"]

    def flush_csv() -> int:
        """Rewrite this suite's rows from everything completed so far.

        upsert_csv deletes the suite's existing rows (key_cols are identical
        across adapter weights) and writes the accumulated set, so calling it
        after each adapter is idempotent and grows the file monotonically.
        """
        rows = []
        for (aw_, cw_), per_task in all_results.items():
            for task_name, split_scores in per_task.items():
                for split, score in split_scores.items():
                    row = dict(base_dim)
                    row["adapter_weight"] = aw_
                    row["ce_weight"] = "" if cw_ is None else cw_
                    row["eval_task"] = task_name
                    row["eval_split"] = split
                    row["main_score"] = f"{score:.6f}"
                    rows.append(row)
        return ecu.upsert_csv(args.csv_output, fieldnames, key_cols, base_dim, rows)

    if args.use_weighted_ce:
        ce_weights = (2.0, 5.0, 10.0, 20.0, 50.0, 100.0)
        if args.only_ce_weights:
            keep = [float(s.strip()) for s in args.only_ce_weights.split(",")
                    if s.strip()]
            missing = sorted(set(keep) - set(ce_weights))
            assert not missing, (
                f"--only-ce-weights has values not in the wce sweep: {missing}; "
                f"available={list(ce_weights)}")
            ce_weights = tuple(w for w in ce_weights if w in set(keep))
        run_plan = [("0.0", cw) for cw in ce_weights]
    else:
        if args.only_ce_weights:
            parser.error("--only-ce-weights requires --use-weighted-ce")
        run_plan = [(aw, None) for aw in ADAPTER_WEIGHTS]

    failures = []
    for aw, cw in run_plan:
        label = f"ce_weight={cw}" if cw is not None else f"adapter={aw}"
        print(f"\n=== {label} ===")
        t0 = time.time()
        try:
            all_results[(aw, cw)] = run_mteb_for_adapter(aw, tasks, cw)
            n = flush_csv()
            print(f"  Wrote {n} row(s) so far -> {args.csv_output}")
        except Exception as e:  # keep going so the other runs still persist
            failures.append((label, repr(e)))
            print(f"  ERROR ({label}): {e!r} -- continuing with remaining runs")
            traceback.print_exc()
        finally:
            print(f"  Elapsed: {format_elapsed(time.time() - t0)}")

    n = flush_csv()
    print(f"\nUpserted {n} rows into {args.csv_output}")
    if failures:
        print(f"\n{len(failures)} run(s) failed; suite incomplete -- exiting "
              f"non-zero so the .done marker is NOT written and the suite is "
              f"retried (completed tasks are cached and skipped on re-run):")
        for label, err in failures:
            print(f"  - {label}: {err}")
        sys.exit(1)


if __name__ == "__main__":
    main()
