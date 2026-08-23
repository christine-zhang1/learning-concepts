"""Baseline data processing: augment synonyms_train.jsonl by replacing content
words with synonyms (a non-concept-loss baseline that bakes synonym variety
into the training text instead).

For each input sequence in
  {base}/{dataset_name}/{model_suffix}/{prompting,embedding}/synonyms_train.jsonl
emits the original sequence followed by `--num-augmentations` augmented
copies. In each copy, every content word is independently replaced. The
replacement is drawn per content word as follows:
  - if len(synonyms) >= num_augmentations: sample without replacement from
    the synonyms across the augmentations (the original is excluded here,
    since the original sequence is emitted separately).
  - otherwise: sample with replacement from [original] + synonyms.

Character spans for each target are looked up via the sibling
`combined.jsonl` (which contains per-token `offsets`).

Run:
  python data/augment_synonyms.py --num-augmentations 4
  # --base-dir defaults to CONCEPT_DATA_ROOT
"""
import sys
import pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import argparse
import json
import random
from pathlib import Path

from conceptlib.paths import DATA_ROOT


TARGET_KEY = "content_word_responses"
ALT_KEY = "synonyms"
INPUT_FILENAME = "synonyms_train.jsonl"
LEAF_NAMES = ("prompting", "embedding")


def load_offsets_index(combined_path):
    """Map input_sequence -> offsets list, or None if combined lacks offsets."""
    index = {}
    has_offsets = None
    with open(combined_path) as f:
        for line in f:
            row = json.loads(line)
            if has_offsets is None:
                has_offsets = "offsets" in row
                if not has_offsets:
                    return None
            index[row["input_sequence"]] = row["offsets"]
    return index


def pick_choices_per_target(targets, num_augmentations, rng):
    """Return a list (one entry per target) of `num_augmentations` replacements."""
    per_target = []
    for t in targets:
        original = t["word"]
        synonyms = list(t.get(ALT_KEY, []) or [])
        if len(synonyms) >= num_augmentations:
            chosen = rng.sample(synonyms, num_augmentations)
        else:
            pool = [original] + synonyms
            chosen = [rng.choice(pool) for _ in range(num_augmentations)]
        per_target.append(chosen)
    return per_target


def build_with_offsets(input_sequence, targets, offsets, choices):
    edits = []
    for t, chosen in zip(targets, choices):
        pos = t["position"]
        start, end = offsets[pos]
        span = input_sequence[start:end]
        original = t["word"]
        idx = span.rfind(original)
        if idx < 0:
            replacement = span
        else:
            replacement = span[:idx] + chosen + span[idx + len(original):]
        edits.append((start, end, replacement))

    edits.sort(key=lambda e: e[0])
    return _apply_edits(input_sequence, edits)


def build_by_search(input_sequence, targets, choices):
    """Fallback when offsets are unavailable: pair targets with chosen
    replacements, sort by token position, then walk the input left-to-right
    replacing the next occurrence of each content word. Token position is
    monotonic with character position, so this recovers the intended order
    even when same-spelling words repeat."""
    pairs = sorted(zip(targets, choices), key=lambda tc: tc[0]["position"])
    edits = []
    cursor = 0
    for t, chosen in pairs:
        original = t["word"]
        idx = input_sequence.find(original, cursor)
        if idx < 0:
            continue
        edits.append((idx, idx + len(original), chosen))
        cursor = idx + len(original)
    return _apply_edits(input_sequence, edits)


def _apply_edits(input_sequence, edits):
    out = []
    cursor = 0
    for start, end, replacement in edits:
        if start < cursor:
            continue
        out.append(input_sequence[cursor:start])
        out.append(replacement)
        cursor = end
    out.append(input_sequence[cursor:])
    return "".join(out)


def augment_file(in_path, out_path, offsets_index, num_augmentations, rng):
    n_in = 0
    n_out = 0
    n_missing = 0
    with open(in_path) as fin, open(out_path, "w") as fout:
        for line in fin:
            row = json.loads(line)
            n_in += 1
            seq = row["input_sequence"]
            targets = row.get(TARGET_KEY, []) or []
            offsets = offsets_index.get(seq) if offsets_index is not None else None
            if offsets_index is not None and offsets is None:
                n_missing += 1

            fout.write(json.dumps({
                "input_sequence": seq,
                "original_input_sequence": seq,
                TARGET_KEY: targets,
                "augmentation_index": 0,
            }) + "\n")
            n_out += 1

            per_target_choices = pick_choices_per_target(targets, num_augmentations, rng)
            for k in range(num_augmentations):
                choices_k = [per_target_choices[i][k] for i in range(len(targets))]
                if offsets is not None:
                    new_seq = build_with_offsets(seq, targets, offsets, choices_k)
                else:
                    new_seq = build_by_search(seq, targets, choices_k)
                fout.write(json.dumps({
                    "input_sequence": new_seq,
                    "original_input_sequence": seq,
                    TARGET_KEY: targets,
                    "augmentation_index": k + 1,
                }) + "\n")
                n_out += 1
    return n_in, n_out, n_missing


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base-dir", default=DATA_ROOT)
    p.add_argument("--num-augmentations", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output-name", default="synonyms_train_aug5x.jsonl",
                   help="Filename written next to synonyms_train.jsonl.")
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()

    rng = random.Random(args.seed)
    base = Path(args.base_dir)

    in_paths = []
    for leaf in LEAF_NAMES:
        in_paths.extend(base.glob(f"*/*/{leaf}/{INPUT_FILENAME}"))
    in_paths = sorted(in_paths)
    if not in_paths:
        leaves = ",".join(LEAF_NAMES)
        print(f"No files found matching */*/{{{leaves}}}/{INPUT_FILENAME} under {base}")
        return

    for in_path in in_paths:
        model_dir = in_path.parent.parent
        combined_path = model_dir / "combined.jsonl"
        out_path = in_path.parent / args.output_name
        if out_path.exists() and not args.overwrite:
            print(f"[skip] exists: {out_path}")
            continue
        if combined_path.exists():
            print(f"[load] {combined_path}")
            offsets_index = load_offsets_index(combined_path)
            if offsets_index is None:
                print(f"           (combined.jsonl has no offsets; "
                      f"falling back to string search)")
        else:
            print(f"[warn] no combined.jsonl in {model_dir}; "
                  f"falling back to string search")
            offsets_index = None
        print(f"[augment] {in_path} -> {out_path}")
        n_in, n_out, n_missing = augment_file(
            in_path, out_path, offsets_index, args.num_augmentations, rng,
        )
        print(f"           in={n_in} out={n_out} missing_offsets={n_missing}")


if __name__ == "__main__":
    main()
