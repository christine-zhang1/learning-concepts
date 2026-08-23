"""Embedding-based synonym pipeline (one of two synonym sources).

For each (input_sequence, content_word) we:
  1. take the top-K next-token candidates from logits at position idx-1
  2. filter to alphanumeric word-pieces
  3. POS-filter via spaCy by substituting each candidate into the sentence
  4. rank surviving candidates by cosine similarity between the candidate's
     input embedding and the model's *context-aware* hidden state at the
     target position (last layer)

Output schema matches llm_synonyms.py so downstream code is unchanged: each
entry has `content_word_responses: [{word, position, synonyms}]`.

Reads:  {CONCEPT_DATA_ROOT}/{dataset}/{model}/combined.jsonl
Writes: {CONCEPT_DATA_ROOT}/{dataset}/{model}/embedding/synonyms_{s}_{e}.jsonl
        {CONCEPT_DATA_ROOT}/{dataset}/{model}/prompting/topk_{s}_{e}.jsonl
"""
import sys
import pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import re
import os
import json
import argparse
import logging

import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)
from transformers.cache_utils import DynamicCache
import spacy

from conceptlib.config import TrainingConfig
from conceptlib.paths import dataset_dir

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

nlp = spacy.load("en_core_web_sm")
# Keep attribute_ruler enabled: it maps token.tag_ -> token.pos_, which the
# POS filter depends on.
SPACY_DISABLE = ["parser", "ner", "lemmatizer"]
SPACY_BATCH = 128

TOPK_LOGITS = 100      # initial pool from next-token distribution
POS_CHECK_TOP = 10     # how many top-cosine candidates to actually POS-check
FINAL_TOPN = 10        # cap on synonyms returned per target
MIN_COSINE = 0.75      # floor on cosine similarity (set >0 to be stricter)
CAND_CTX_BATCH = 64    # candidates per forward call when computing ctx reps
TOKEN_RE = re.compile(r"^ ?[A-Za-z0-9]+$")
# Whole-word check: a candidate is a full word if its decoded form starts with
# a space (BPE word-start marker) and the following token in the sequence is
# end-of-sequence or itself starts with a space/punctuation.
NEXT_TOKEN_STARTS = (
    " ", ".", ",", "!", "?", ":", ";", "(", ")", "[", "]", "{", "}", "'", "\"",
)


class EmbeddingSynonymExtractor:
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.model, self.tokenizer = self._setup_model_and_tokenizer()

    def _setup_model_and_tokenizer(self):
        logger.info(f"Loading tokenizer from {self.config.model_name}...")
        tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name, trust_remote_code=True
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id
        tokenizer.padding_side = "right"

        quantization_config = None
        if self.config.use_4bit:
            logger.info("Using 4-bit quantization")
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type=self.config.bnb_4bit_quant_type,
                bnb_4bit_use_double_quant=self.config.use_nested_quant,
            )

        logger.info(f"Loading model from {self.config.model_name}...")
        model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            quantization_config=quantization_config,
            device_map="auto",
            trust_remote_code=True,
        )
        model.eval()
        return model, tokenizer

    @staticmethod
    def _token_at_char(doc, char_start):
        for t in doc:
            if t.idx <= char_start < t.idx + len(t.text):
                return t
        return None

    @staticmethod
    def _kv_layers(past_kv):
        """Return (keys, values) lists from either DynamicCache or legacy tuple."""
        if hasattr(past_kv, "key_cache"):
            return past_kv.key_cache, past_kv.value_cache
        keys = [layer[0] for layer in past_kv]
        values = [layer[1] for layer in past_kv]
        return keys, values

    def _candidate_contextual_reps(self, past_kv, idx, cand_ids):
        """Contextual hidden state at position `idx` for each candidate token.

        Reuses the base forward's prefix K/V cache (positions [0, idx)) and runs
        a batched 1-token forward where each batch element substitutes one
        candidate at position idx. Causal attention means the suffix doesn't
        affect this hidden state, so we get the true contextual rep without
        re-running the full sequence per candidate.

        Returns: (N, hidden_dim) float tensor on the model's device.
        """
        keys, values = self._kv_layers(past_kv)
        device = cand_ids.device
        n = cand_ids.shape[0]

        outs = []
        for s in range(0, n, CAND_CTX_BATCH):
            chunk = cand_ids[s : s + CAND_CTX_BATCH]
            b = chunk.shape[0]
            # No .contiguous(): SDPA / flash backends accept the broadcasted view,
            # avoiding a prefix-K/V copy per chunk per layer.
            prefix_legacy = tuple(
                (
                    k[:, :, :idx, :].expand(b, -1, -1, -1),
                    v[:, :, :idx, :].expand(b, -1, -1, -1),
                )
                for k, v in zip(keys, values)
            )
            cache = DynamicCache.from_legacy_cache(prefix_legacy)
            attn_mask = torch.ones(b, idx + 1, device=device, dtype=torch.long)
            position_ids = torch.full((b, 1), idx, device=device, dtype=torch.long)
            cache_position = torch.tensor([idx], device=device, dtype=torch.long)
            with torch.inference_mode():
                out = self.model(
                    input_ids=chunk.unsqueeze(-1),
                    attention_mask=attn_mask,
                    position_ids=position_ids,
                    past_key_values=cache,
                    cache_position=cache_position,
                    output_hidden_states=True,
                    use_cache=False,
                )
            outs.append(out.hidden_states[-1][:, 0, :].float())
        return torch.cat(outs, dim=0)

    def _rank_target(self, input_ids, logits, hidden_last, past_kv, idx, word):
        """Rank candidates for one target by cosine sim to the contextual rep.

        Returns `(pairs, topk_words)`:
          - `pairs`: top POS_CHECK_TOP `(cand_word_clean, score)` sorted desc.
          - `topk_words`: the full filtered candidate pool (≤TOPK_LOGITS, post
            alphanumeric + whole-word + ≠target filter), stripped.

        POS filtering is done later, batched at the sequence level.
        """
        decoded_target = self.tokenizer.decode([input_ids[0, idx].item()])
        if decoded_target.strip() != str(word).strip():
            logger.warning(
                f"Target '{word}' at idx {idx} does not match decoded token "
                f"'{decoded_target}'; skipping."
            )
            return [], []

        prev_logits = logits[0, idx - 1, :]
        topk = torch.topk(prev_logits, k=TOPK_LOGITS)
        cand_ids = topk.indices

        decoded = self.tokenizer.batch_decode(cand_ids.unsqueeze(-1).tolist())
        target_norm = str(word).strip().lower()
        # Whole-word feasibility check on the *fixed* token following idx —
        # substituting a candidate at idx doesn't change input_ids[idx+1], so
        # this is a per-target constant. If false, no candidate can form a
        # complete word here.
        seq_len = input_ids.shape[1]
        if idx + 1 >= seq_len:
            next_ok = True
        else:
            next_tok = self.tokenizer.decode([input_ids[0, idx + 1].item()])
            next_ok = next_tok.startswith(NEXT_TOKEN_STARTS)
        keep = [
            bool(TOKEN_RE.match(t))
            and t.strip().lower() != target_norm
            and t.startswith(" ")
            and next_ok
            for t in decoded
        ]
        if not any(keep):
            return [], []
        keep_t = torch.tensor(keep, device=cand_ids.device)
        cand_ids = cand_ids[keep_t]
        decoded = [t for t, k in zip(decoded, keep) if k]
        topk_words = [t.strip() for t in decoded]

        ctx_vec = hidden_last[idx].float()
        cand_vecs = self._candidate_contextual_reps(past_kv, idx, cand_ids)
        sims = F.cosine_similarity(cand_vecs, ctx_vec.unsqueeze(0), dim=-1)
        sims_list = sims.tolist()

        pairs = [
            (decoded[i].strip(), sims_list[i])
            for i in range(len(decoded))
            if sims_list[i] >= MIN_COSINE
        ]
        pairs.sort(key=lambda x: -x[1])
        return pairs[:POS_CHECK_TOP], topk_words

    def analyze_sequence(self, row, save_path, topk_save_path):
        input_text = row["input_sequence"]
        content_words = row["content_words"]
        content_words_indices = row["content_words_indices"]

        # Use precomputed input_ids/offsets from the data file. Re-tokenizing
        # here would drift on the BPE tokenizer because input_text was saved
        # before truncation and the saved indices/offsets are tied to the exact
        # tokenization that produced them.
        input_ids = torch.tensor(
            [row["input_ids"]], device=self.model.device, dtype=torch.long
        )
        offsets = row["offsets"]
        attention_mask = torch.ones_like(input_ids)
        model_inputs = {"input_ids": input_ids, "attention_mask": attention_mask}
        assert input_ids.shape[0] == 1

        with torch.inference_mode():
            outputs = self.model(
                **model_inputs, output_hidden_states=True, use_cache=True
            )
            logits = outputs.logits
            hidden_last = outputs.hidden_states[-1][0]  # (T, H)
            past_kv = outputs.past_key_values  # used for candidate ctx reps

        # Phase 1: per-target cosine ranking (cheap, vectorised).
        targets = []
        for word, idx in zip(content_words, content_words_indices):
            if idx < 1:
                continue
            cs, ce = offsets[idx]
            # Strip whitespace edges off the tokenizer span so we replace just
            # the bare word and keep the surrounding spaces from input_text.
            span = input_text[cs:ce]
            ws_lead = len(span) - len(span.lstrip())
            ws_trail = len(span) - len(span.rstrip())
            word_start = cs + ws_lead
            word_end = ce - ws_trail
            if word_end <= word_start:
                # Special token (e.g. BOS has span [0,0]) or empty — skip POS work.
                targets.append({
                    "word": word, "idx": idx,
                    "word_start": None, "word_end": None,
                    "ranked": [], "topk_words": [],
                })
                continue
            ranked, topk_words = self._rank_target(
                input_ids, logits, hidden_last, past_kv, idx, word
            )
            targets.append({
                "word": word, "idx": idx,
                "word_start": word_start, "word_end": word_end,
                "ranked": ranked, "topk_words": topk_words,
            })

        # Phase 2: batch every spaCy parse for this sequence into one pipe call.
        texts = [input_text]
        plan = []  # (target_i, cand_i, doc_i, char_offset)
        for ti, t in enumerate(targets):
            if not t["ranked"] or t["word_start"] is None:
                continue
            ws, we = t["word_start"], t["word_end"]
            for ci, (cand_word, _score) in enumerate(t["ranked"]):
                # cand_word is already stripped (no BPE leading space).
                sub = input_text[:ws] + cand_word + input_text[we:]
                texts.append(sub)
                plan.append((ti, ci, len(texts) - 1, ws))

        docs = list(nlp.pipe(texts, disable=SPACY_DISABLE, batch_size=SPACY_BATCH))
        orig_doc = docs[0]

        target_pos = []
        for t in targets:
            if t["word_start"] is None:
                target_pos.append(None)
                continue
            tok = self._token_at_char(orig_doc, t["word_start"])
            target_pos.append(tok.pos_ if tok is not None else None)

        pos_keep = [[False] * len(t["ranked"]) for t in targets]
        for ti, ci, di, off in plan:
            tp = target_pos[ti]
            if not tp:
                continue
            tok = self._token_at_char(docs[di], off)
            if tok is not None and tok.pos_ == tp:
                pos_keep[ti][ci] = True

        responses = []
        for ti, t in enumerate(targets):
            survivors = [
                t["ranked"][ci]
                for ci in range(len(t["ranked"]))
                if pos_keep[ti][ci]
            ][:FINAL_TOPN]
            responses.append({
                "word": t["word"],
                "position": t["idx"],
                "synonyms": [w for w, _ in survivors],
                "scores": [round(s, 4) for _, s in survivors],
            })

        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, "a") as f:
            f.write(json.dumps({
                "input_sequence": input_text,
                "content_word_responses": responses,
            }) + "\n")

        topk_records = [
            {"word": t["word"], "position": t["idx"], "topk_tokens": t["topk_words"]}
            for t in targets
        ]
        os.makedirs(os.path.dirname(topk_save_path), exist_ok=True)
        with open(topk_save_path, "a") as f:
            f.write(json.dumps({
                "input_sequence": input_text,
                "content_word_topk": topk_records,
            }) + "\n")
        return 1


def run(dataset: str, start_line=None, end_line=None, model: str | None = None):
    config = TrainingConfig()
    if model is not None:
        config.model_name = model
    extractor = EmbeddingSynonymExtractor(config)

    base_dir = dataset_dir(dataset, config.model_name)
    in_path = os.path.join(base_dir, "combined.jsonl")
    out_dir = os.path.join(base_dir, "embedding")
    topk_dir = os.path.join(base_dir, "prompting")
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(topk_dir, exist_ok=True)

    with open(in_path) as f:
        lines = [ln.strip() for ln in f if ln.strip()]

    s = start_line if start_line is not None else 0
    e = end_line if end_line is not None else len(lines)
    out_path = f"{out_dir}/synonyms_{s}_{e}.jsonl"
    topk_path = f"{topk_dir}/topk_{s}_{e}.jsonl"
    if os.path.exists(out_path):
        os.remove(out_path)
    if os.path.exists(topk_path):
        os.remove(topk_path)

    logger.info(f"Reading {in_path} [{s}:{e}] -> {out_path}, {topk_path}")
    for line in tqdm(lines[s:e], desc=dataset):
        row = json.loads(line)
        extractor.analyze_sequence(row, save_path=out_path, topk_save_path=topk_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Single-model embedding+POS synonym extraction."
    )
    parser.add_argument("dataset", type=str, default="c4", nargs="?")
    parser.add_argument("--start", type=int, default=None, help="Start line index (0-based, inclusive)")
    parser.add_argument("--end", type=int, default=None, help="End line index (0-based, exclusive)")
    parser.add_argument("--model", type=str, default=None, help="HuggingFace model name; overrides TrainingConfig.model_name")
    args = parser.parse_args()
    run(args.dataset, args.start, args.end, model=args.model)
