"""Baseline data processing: build a randomized-synonym control set.

Walks {CONCEPT_DATA_ROOT}/{c4,openwebtext}/<model>/{embedding,prompting} and,
for every directory containing synonyms_{split}.jsonl, writes the randomized
version synonyms_{split}_randomized.jsonl. The split(s) to randomize are chosen
with --split (default: test).

Randomization logic: pool all synonyms across the file, then for each content
word emit n+1 random samples (with replacement) from the pool, where n is the
number of real synonyms for that target. Preserves input_sequence, word, and
position. This isolates "does training toward *these specific* synonyms matter"
from "does training toward *any* set of same-size token groups matter".
"""
import sys
import pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import argparse
import json
import random
from pathlib import Path

from conceptlib.paths import DATA_ROOT

ROOT = Path(DATA_ROOT)
DATASETS = ("c4", "openwebtext")
SUBDIRS = ("embedding", "prompting")
SPLITS = ("train", "val", "test")
SEED = 42


def randomize_file(in_path: Path, out_path: Path) -> None:
    rows = []
    pool = []
    with open(in_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            rows.append(obj)
            for tw in obj.get("content_word_responses", []):
                pool.extend(tw.get("synonyms", []))

    if not pool:
        raise ValueError(f"No synonym words found in {in_path}")

    rng = random.Random(SEED)

    with open(out_path, "w", encoding="utf-8") as f:
        for obj in rows:
            new_responses = []
            for tw in obj.get("content_word_responses", []):
                n = len(tw.get("synonyms", []))
                new_responses.append({
                    "word": tw["word"],
                    "position": tw["position"],
                    "synonyms": rng.choices(pool, k=n + 1),
                })
            out_obj = {
                "input_sequence": obj["input_sequence"],
                "content_word_responses": new_responses,
            }
            f.write(json.dumps(out_obj, ensure_ascii=False) + "\n")

    print(f"  wrote {out_path} ({len(rows)} rows, pool={len(pool)})")


def main():
    parser = argparse.ArgumentParser(
        description="Build randomized-synonym control files for the chosen split(s)."
    )
    parser.add_argument(
        "--split",
        nargs="+",
        choices=SPLITS,
        default=["test"],
        help="Which split file(s) to randomize: synonyms_{split}.jsonl -> "
             "synonyms_{split}_randomized.jsonl (default: test).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-randomize even if the _randomized.jsonl output already exists.",
    )
    args = parser.parse_args()

    targets = []
    for ds in DATASETS:
        ds_dir = ROOT / ds
        if not ds_dir.is_dir():
            continue
        for model_dir in sorted(p for p in ds_dir.iterdir() if p.is_dir()):
            for sub in SUBDIRS:
                d = model_dir / sub
                if not d.is_dir():
                    continue
                for split in args.split:
                    in_path = d / f"synonyms_{split}.jsonl"
                    out_path = d / f"synonyms_{split}_randomized.jsonl"
                    if in_path.is_file() and (args.overwrite or not out_path.is_file()):
                        targets.append((in_path, out_path))

    if not targets:
        print("Nothing to do.")
        return

    print(f"Found {len(targets)} file(s) to process:")
    for in_path, _ in targets:
        print(f"  {in_path}")

    for in_path, out_path in targets:
        print(f"Processing {in_path}")
        randomize_file(in_path, out_path)

    print(f"Done. Processed {len(targets)} files.")


if __name__ == "__main__":
    main()
