"""Helpers shared by the next-token-prediction evaluation scripts."""
import gc

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel


def format_elapsed(seconds: float) -> str:
    seconds = int(round(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h}h {m}m {s}s"


def free_model_memory():
    """Release GPU memory after unloading a model."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def load_model_and_tokenizer(adapter_full_path, base_model_name: str,
                             device_map=None):
    """Load the 4-bit base model and (optionally) wrap it with a LoRA adapter.

    `adapter_full_path=None` returns the bare base model. The 4-bit NF4 config
    matches the QLoRA training conditions.
    """
    if device_map is None:
        device_map = {"": "cuda:0"}
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=False,
    )
    base = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        device_map=device_map,
        quantization_config=bnb_config,
    )
    if adapter_full_path is None:
        return base, tokenizer
    model = PeftModel.from_pretrained(base, adapter_full_path)
    return model, tokenizer


def paired_bootstrap_cis(per_model_metrics, n_bootstrap=1000, ci=0.95, seed=42):
    """Paired bootstrap: the same resample of test examples is used for every
    model on each iteration.

    Returns per-model CIs and CIs for the pairwise difference (run 0 - run j).
    `per_model_metrics` is a list of dicts each with keys `_correctness` and
    `_nll` (equal-length arrays). Only accuracy and perplexity CIs are computed.
    """
    rng = np.random.default_rng(seed)
    n_models = len(per_model_metrics)
    n = len(per_model_metrics[0]["_correctness"])
    for m in range(n_models):
        assert len(per_model_metrics[m]["_correctness"]) == n, \
            "All models must have the same number of examples."

    alpha = 1 - ci
    lo_pct = 100 * alpha / 2
    hi_pct = 100 * (1 - alpha / 2)

    def ci_tuple(arr):
        return (float(np.percentile(arr, lo_pct)), float(np.percentile(arr, hi_pct)))

    boot_acc = np.zeros((n_bootstrap, n_models))
    boot_ppl = np.zeros((n_bootstrap, n_models))

    for b in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)  # same resample for all models
        for m in range(n_models):
            r = per_model_metrics[m]["_correctness"][idx]
            nll = per_model_metrics[m]["_nll"][idx]
            boot_acc[b, m] = np.mean(r)
            boot_ppl[b, m] = np.exp(np.mean(nll))

    per_model_cis = []
    for m in range(n_models):
        per_model_cis.append({
            "accuracy_ci": ci_tuple(boot_acc[:, m]),
            "perplexity_ci": ci_tuple(boot_ppl[:, m]),
        })

    diff_cis = {}
    for j in range(1, n_models):
        diff_cis[(0, j)] = {
            "accuracy_ci": ci_tuple(boot_acc[:, j] - boot_acc[:, 0]),
            "perplexity_ci": ci_tuple(boot_ppl[:, j] - boot_ppl[:, 0]),
        }

    return per_model_cis, diff_cis
