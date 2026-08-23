"""Step 1 of data collection: extract single-token content words.

Streams a raw text corpus (C4 / OpenWebText), tokenizes each sample
with the model tokenizer, and selects content words (NOUN / VERB / ADJ) that are
a single whole word-piece token. Writes one `combined.jsonl` per (dataset, model)
under CONCEPT_DATA_ROOT with the tokenization, character offsets, and the chosen
content words + their token indices.

Output: {CONCEPT_DATA_ROOT}/{dataset}/{model}/combined.jsonl
"""
import sys
import pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import argparse
import json
import os
from typing import Tuple

import numpy as np
import spacy
from datasets import load_dataset
from transformers import AutoTokenizer
from tqdm import tqdm

from conceptlib.paths import dataset_dir

parser = argparse.ArgumentParser()
parser.add_argument("--model", default="meta-llama/Llama-3.2-1B", help="HuggingFace model name for the tokenizer")
parser.add_argument("--dataset", default="c4", choices=["c4", "openwebtext"], help="Short dataset name")
parser.add_argument("--max_length", type=int, default=256, help="Max number of tokens per sample for tokenizer truncation")
args = parser.parse_args()

MODEL = args.model
tokenizer = AutoTokenizer.from_pretrained(MODEL)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

nlp = spacy.load("en_core_web_sm")

content_word_lengths = []

DATASET = args.dataset
SPLIT = "train"
MAX_SAMPLES = 10000
MAX_LENGTH = args.max_length

print(f"Getting content words directly for {DATASET} {SPLIT} with max {MAX_SAMPLES} samples using {MODEL}")
print(f"Max length: {MAX_LENGTH}")

# Resolve HF dataset spec from a short name (e.g. "c4")
if DATASET == "c4":
    ds_spec: Tuple[str, str] = ("allenai/c4", "en")
elif DATASET == "openwebtext":
    ds_spec = ("Skylion007/openwebtext", None)
else:
    raise ValueError(f"Unsupported dataset: {DATASET}")

# Load raw texts from the HF dataset (streaming)
if ds_spec[1] is None:
    ds = load_dataset(ds_spec[0], streaming=True, split=SPLIT)
else:
    ds = load_dataset(ds_spec[0], ds_spec[1], streaming=True, split=SPLIT)
ds = ds.shuffle(seed=42, buffer_size=10_000)


def _iter_texts():
    count = 0
    for item in ds:
        text = item.get("text", "")
        if not isinstance(text, str):
            continue
        text = text.strip()
        yield text
        count += 1
        if count >= MAX_SAMPLES:
            break


out_path = os.path.join(dataset_dir(DATASET, MODEL), "combined.jsonl")
os.makedirs(os.path.dirname(out_path), exist_ok=True)

with open(out_path, "w") as f:
    for sample_idx, text in enumerate(tqdm(_iter_texts(), desc="Processing texts", total=MAX_SAMPLES)):
        content_words = []
        content_word_indices = []
        # tokenization with character spans for alignment
        enc = tokenizer(
            text,
            return_offsets_mapping=True,
            add_special_tokens=True,
            truncation=True,
            max_length=MAX_LENGTH,
        )
        input_ids = enc["input_ids"]
        offsets = [list(o) for o in enc["offset_mapping"]]

        # One-token words only: the token starts with a space (word boundary) and
        # the next token starts with a space or punctuation (no continuation piece).
        next_token_starts = (" ", ".", ",", "!", "?", ":", ";", "(", ")", "[", "]", "{", "}", "'", "\"")
        token_strs = tokenizer.batch_decode([[tid] for tid in input_ids])
        token_strs_with_idx = [(i, s) for i, s in enumerate(token_strs)]
        words = {  # dict keyed by token string also dedupes
            s: i for i, s in token_strs_with_idx
            if s.strip() and s.startswith(" ")
            and (i + 1 == len(token_strs) or token_strs[i + 1].startswith(next_token_starts))
        }
        docs = list(nlp.pipe(list(words.keys())))
        for i, (word, token_idx) in enumerate(words.items()):
            d = docs[i]
            if word.startswith(" "):
                # Skip when the model and spaCy tokenize the word differently.
                if len(d) != 2 or word != " " + d[1].text:
                    continue
                spacy_tok = d[1]
            else:
                if len(d) != 1 or word != d[0].text:
                    continue
                spacy_tok = d[0]
            if spacy_tok.pos_ in ["NOUN", "VERB", "ADJ"]:
                content_words.append(spacy_tok.text)
                content_word_indices.append(token_idx)

        content_word_lengths.append(len(content_words))

        # Truncate input_sequence to the char range covered by the (truncated)
        # tokens so spaCy doesn't parse text the model never saw. Special tokens
        # have span (0, 0); skip those when finding the last real end offset.
        last_end = max((e for _, e in offsets if e > 0), default=len(text))
        f.write(json.dumps({
            "input_sequence": text[:last_end],
            "input_ids": input_ids,
            "offsets": offsets,
            "content_words": content_words,
            "content_words_indices": content_word_indices,
        }) + "\n")

print("Mean number of content words: ", np.mean(content_word_lengths))
print("Std number of content words: ", np.std(content_word_lengths))
print("Min number of content words: ", np.min(content_word_lengths))
print("Max number of content words: ", np.max(content_word_lengths))
