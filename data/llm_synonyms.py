"""LLM-prompted synonym pipeline (the other synonym source).

Uses an instruction-tuned LLM to pick contextual synonyms for each content word
from the top-K candidate tokens produced by embedding_synonyms.py, then
post-processes the response to extract the synonym list.

vLLM-backed batch version: all (entry, content_word) prompts within a chunk are
generated in one continuous-batch call. Prefix caching makes the shared
input_sequence prefix across an entry's content words effectively free.

Reads:  {CONCEPT_DATA_ROOT}/{dataset}/{model}/prompting/topk_{s}_{e}.jsonl
Writes: {CONCEPT_DATA_ROOT}/{dataset}/{model}/prompting/synonyms_{s}_{e}.jsonl
"""
import sys
import pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import json
import os
import argparse

from tqdm import tqdm
from vllm import LLM, SamplingParams

from conceptlib.paths import dataset_dir

base_to_instruct = {
    "meta-llama/Llama-3.2-3B": "meta-llama/Llama-3.2-3B-Instruct",
    "meta-llama/Llama-3.1-8B": "meta-llama/Llama-3.1-8B-Instruct",
    "meta-llama/Llama-3.2-1B": "meta-llama/Llama-3.2-1B-Instruct",
    "Qwen/Qwen3-1.7B-Base": "Qwen/Qwen3-1.7B",
    "Qwen/Qwen3-4B-Base": "Qwen/Qwen3-4B",
    "Qwen/Qwen3-8B-Base": "Qwen/Qwen3-8B",
    "tiiuae/Falcon3-7B-Base": "tiiuae/Falcon3-7B-Instruct",
    "tiiuae/Falcon3-3B-Base": "tiiuae/Falcon3-3B-Instruct",
    "tiiuae/Falcon3-1B-Base": "tiiuae/Falcon3-1B-Instruct",
}

# Chunk size in *entries* (jsonl lines). Each entry produces len(content_word_topk)
# prompts; vLLM will continuously batch them. Larger = more for the scheduler to
# pack across, but we still flush periodically so output is checkpointed.
CHUNK_SIZE = 1024

sampling_params = SamplingParams(
    temperature=0.0,
    max_tokens=20,
    repetition_penalty=1.1,
)


def build_prompt(input_seq: str, content_word: str, decoded_tokens: list) -> str:
    target = content_word.strip() if isinstance(content_word, str) else str(content_word).strip()
    # input_seq comes first so all prompts within an entry share it as a prefix
    # and hit prefix cache. target / decoded_tokens (which differ per target) go after.
    return f"""Text: "{input_seq}"

Find contextual synonyms for the word "{target}" in the text above.

Available tokens: {decoded_tokens}

Instructions:
- Find ALL possible synonyms from the available tokens that could replace "{target}" in this context
- Return ONLY a comma-separated list of synonyms from the available tokens, surrounded by square brackets
- Include every relevant synonym - don't miss any
- NO duplicates allowed
- NO explanations or extra text
- If no synonyms found, return: []

Example format: [word1, word2, word3]

Synonyms for "{target}":"""


def parse_bracket_synonyms(response: str):
    """Extract comma/space-separated words from the last [...] in the LLM response."""
    current_idx = 0
    open_idx = -1
    for _ in range(4):
        open_idx = response.find("[", current_idx)
        if open_idx == -1:
            break
        current_idx = open_idx + 1
    if open_idx == -1:
        return []
    close_idx = response.find("]", open_idx + 1)
    content = response[open_idx + 1 : close_idx] if close_idx != -1 else response[open_idx + 1 :]
    if "," in content:
        candidates = [w.strip() for w in content.split(",")]
    else:
        candidates = content.split()
    filtered = [w for w in candidates if w]
    return list(dict.fromkeys(filtered))


def process_chunk(entries, f_out):
    """Generate synonyms for a chunk of entries with one batched vLLM call."""
    prompts = []
    # owners[i] = (entry_idx_within_chunk, target_idx_within_entry, allowed_set)
    owners = []
    per_entry_responses = [[] for _ in entries]

    for ei, entry in enumerate(entries):
        input_seq = entry["input_sequence"]
        for ti, item in enumerate(entry["content_word_topk"]):
            word = item["word"]
            position = item.get("position")
            topk_tokens = item.get("topk_tokens", [])
            if isinstance(topk_tokens, str):
                topk_tokens = topk_tokens.split()
            allowed = set(w.strip() for w in topk_tokens if w)
            prompts.append(build_prompt(input_seq, word, topk_tokens))
            owners.append((ei, ti, word, position, allowed))
            per_entry_responses[ei].append(None)  # placeholder, filled below

    if not prompts:
        return

    outputs = llm.generate(prompts, sampling_params, use_tqdm=True)

    for (ei, ti, word, position, allowed), out in zip(owners, outputs):
        gen_text = out.outputs[0].text
        # parse_bracket_synonyms looks for the 4th "[" in the *full* response
        # (the prompt itself contains 3 brackets), so prepend the prompt to the
        # generated text before parsing.
        full = out.prompt + gen_text
        parsed = parse_bracket_synonyms(full)
        synonyms = [w for w in parsed if w in allowed]
        per_entry_responses[ei][ti] = {
            "word": word,
            "position": position,
            "synonyms": synonyms,
        }

    for entry, responses in zip(entries, per_entry_responses):
        entry.pop("content_word_topk")
        entry.pop("content_words", None)
        entry.pop("content_words_indices", None)
        entry["content_word_responses"] = responses
        f_out.write(json.dumps(entry) + "\n")


def get_llm_synonyms(dataset, start_line=None, end_line=None, model_name=None):
    print(f"Getting LLM synonyms for {dataset}...")
    prompting_dir = os.path.join(dataset_dir(dataset, model_name), "prompting")
    input_path = os.path.join(prompting_dir, f"topk_{start_line}_{end_line}.jsonl")

    with open(input_path, 'r') as f:
        output_path = os.path.join(prompting_dir, f"synonyms_{start_line}_{end_line}.jsonl")
        if os.path.exists(output_path):
            os.remove(output_path)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        print(f"Processing from {input_path}")
        print(f"Output path: {output_path}")

        with open(output_path, 'a') as f_out:
            counter = 0
            chunk = []

            with tqdm(total=end_line-start_line, desc="Processing dataset") as pbar:
                for line in f:
                    chunk.append(json.loads(line))
                    counter += 1
                    if len(chunk) >= CHUNK_SIZE:
                        process_chunk(chunk, f_out)
                        pbar.update(len(chunk))
                        chunk = []

                if chunk:
                    process_chunk(chunk, f_out)
                    pbar.update(len(chunk))

    print(f"Total: {counter}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Get LLM synonyms for a dataset.")
    parser.add_argument(
        "dataset",
        type=str,
        default="c4",
        nargs="?",
        help="Dataset name (e.g. c4)",
    )
    parser.add_argument(
        "--start",
        type=int,
        default=None,
        help="Start line index (0-based, inclusive)",
    )
    parser.add_argument(
        "--end",
        type=int,
        default=None,
        help="End line index (0-based, exclusive)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model name",
    )
    args = parser.parse_args()
    llm = LLM(
        model=base_to_instruct[args.model],
        dtype="bfloat16",
        gpu_memory_utilization=0.9,
        enable_prefix_caching=True,
        max_model_len=1024,
    )
    get_llm_synonyms(args.dataset, args.start, args.end, args.model)
