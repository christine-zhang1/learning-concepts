"""Merge per-shard synonym files into train/val/test splits.

Traverses {CONCEPT_DATA_ROOT}/{dataset}/{model}/{prompting,embedding} and, for
each such directory, finds all shard files of the form
    synonyms_{start}_{end}.jsonl
(written by embedding_synonyms.py / llm_synonyms.py), verifies they form a
contiguous range with no gaps or overlaps, and splits them into
    synonyms_train.jsonl  (first  TRAIN_SIZE lines)
    synonyms_val.jsonl    (next   VAL_SIZE  lines)
    synonyms_test.jsonl   (next   TEST_SIZE lines)
The shard files are deleted only after the split files are verified.

Before splitting, audits the full cartesian product of
{dataset} x {model} x {prompting,embedding} and flags any directory that is
missing, empty, or has incomplete shards (i.e. not yet ready and not already
split).
"""
import sys
import pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import os
import re
import argparse

from conceptlib.paths import DATA_ROOT

SPLITS = ("train", "val", "test")
LEAF_NAMES = ("prompting", "embedding")
PART_RE = re.compile(r"^synonyms_(\d+)_(\d+)\.jsonl$")
TRAIN_FILENAME = "synonyms_train.jsonl"


def collect_parts(out_dir: str):
    parts = []
    for name in os.listdir(out_dir):
        m = PART_RE.match(name)
        if m:
            parts.append((int(m.group(1)), int(m.group(2)), os.path.join(out_dir, name)))
    parts.sort(key=lambda x: x[0])
    return parts


def parts_are_complete(parts, total_expected: int):
    """Return (ok, reason) for a sorted list of part tuples."""
    if not parts:
        return False, "no parts"
    if parts[0][0] != 0:
        return False, f"first part starts at {parts[0][0]}, expected 0"
    for i, (s, e, p) in enumerate(parts):
        if s >= e:
            return False, f"bad range in {os.path.basename(p)}: [{s}, {e})"
        if i > 0 and s != parts[i - 1][1]:
            return False, (
                f"gap/overlap between {os.path.basename(parts[i-1][2])} "
                f"(ends {parts[i-1][1]}) and {os.path.basename(p)} (starts {s})"
            )
    covered = parts[-1][1] - parts[0][0]
    if covered != total_expected:
        return False, f"parts cover {covered} rows, expected {total_expected}"
    return True, "complete"


def split_parts(
    out_dir: str,
    parts,
    train_size: int = 8000,
    val_size: int = 1000,
    test_size: int = 1000,
    dry_run: bool = False,
):
    total_expected = train_size + val_size + test_size
    split_sizes = {"train": train_size, "val": val_size, "test": test_size}
    split_paths = {s: os.path.join(out_dir, f"synonyms_{s}.jsonl") for s in SPLITS}
    tmp_paths = {s: split_paths[s] + ".tmp" for s in SPLITS}

    print(
        f"[{out_dir}] Splitting {len(parts)} parts covering "
        f"[{parts[0][0]}, {parts[-1][1]}) into "
        f"train={train_size}, val={val_size}, test={test_size}"
    )
    for s, e, p in parts:
        print(f"  {os.path.basename(p)}  [{s}, {e})")

    if dry_run:
        print("Dry run; not writing.")
        return

    boundaries = {
        "train": (0, train_size),
        "val": (train_size, train_size + val_size),
        "test": (train_size + val_size, total_expected),
    }

    written = {s: 0 for s in SPLITS}
    files = {s: open(tmp_paths[s], "w") for s in SPLITS}
    try:
        idx = 0
        for _, _, p in parts:
            with open(p) as f:
                for line in f:
                    for s in SPLITS:
                        lo, hi = boundaries[s]
                        if lo <= idx < hi:
                            files[s].write(line)
                            written[s] += 1
                            break
                    idx += 1
    finally:
        for f in files.values():
            f.close()

    for s in SPLITS:
        if written[s] != split_sizes[s]:
            for tmp in tmp_paths.values():
                if os.path.exists(tmp):
                    os.remove(tmp)
            raise SystemExit(
                f"Wrote {written[s]} lines for {s}, expected {split_sizes[s]}"
            )

    for s in SPLITS:
        with open(tmp_paths[s]) as f:
            n = sum(1 for _ in f)
        if n != split_sizes[s]:
            for tmp in tmp_paths.values():
                if os.path.exists(tmp):
                    os.remove(tmp)
            raise SystemExit(
                f"Verification failed for {s}: file has {n} lines, expected {split_sizes[s]}"
            )

    for s in SPLITS:
        os.replace(tmp_paths[s], split_paths[s])
        print(f"Wrote {split_paths[s]} ({split_sizes[s]} lines)")

    for _, _, p in parts:
        os.remove(p)
        print(f"Deleted {os.path.basename(p)}")


def discover_expected_dirs(base_dir: str):
    """Return (datasets, model_tags, expected_paths). model_tags is the union
    across datasets so a model present for only one dataset still gets flagged
    as missing for the other."""
    if not os.path.isdir(base_dir):
        raise SystemExit(f"Base directory does not exist: {base_dir}")
    datasets = sorted(
        d for d in os.listdir(base_dir)
        if os.path.isdir(os.path.join(base_dir, d))
    )
    model_tags = set()
    for dataset in datasets:
        for m in os.listdir(os.path.join(base_dir, dataset)):
            if os.path.isdir(os.path.join(base_dir, dataset, m)):
                model_tags.add(m)
    model_tags = sorted(model_tags)
    expected = []
    for dataset in datasets:
        for model_tag in model_tags:
            for leaf in LEAF_NAMES:
                expected.append(os.path.join(base_dir, dataset, model_tag, leaf))
    return datasets, model_tags, expected


def audit_dir(path: str, total_expected: int):
    """Classify a leaf dir. Returns (status, detail, parts_or_none).

    status in {"missing", "ready", "parts_ok", "parts_incomplete", "empty"}.
    """
    if not os.path.isdir(path):
        return "missing", "directory does not exist", None
    has_train = os.path.isfile(os.path.join(path, TRAIN_FILENAME))
    parts = collect_parts(path)
    if has_train:
        return "ready", f"{TRAIN_FILENAME} present", parts
    if not parts:
        return "empty", f"no parts and no {TRAIN_FILENAME}", None
    ok, reason = parts_are_complete(parts, total_expected)
    if ok:
        return "parts_ok", reason, parts
    return "parts_incomplete", reason, parts


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Audit and split embedding/prompting synonym part files into "
                    "train/val/test jsonl files for every dataset/model under the "
                    "base directory."
    )
    parser.add_argument("--base-dir", type=str, default=DATA_ROOT)
    parser.add_argument("--train-size", type=int, default=8000)
    parser.add_argument("--val-size", type=int, default=1000)
    parser.add_argument("--test-size", type=int, default=1000)
    parser.add_argument("--expected-count", type=int, default=36,
                        help="Expected number of leaf directories "
                             "(default 9 models x 2 datasets x 2 leaves = 36).")
    parser.add_argument("--dry-run", action="store_true",
                        help="Audit and list parts without writing or deleting.")
    parser.add_argument("--force", action="store_true",
                        help="Proceed with splitting even if some directories "
                             "are flagged as missing/empty/incomplete.")
    args = parser.parse_args()

    total_expected = args.train_size + args.val_size + args.test_size
    datasets, model_tags, expected = discover_expected_dirs(args.base_dir)

    print(f"Base: {args.base_dir}")
    print(f"Datasets ({len(datasets)}): {datasets}")
    print(f"Model tags ({len(model_tags)}): {model_tags}")
    print(f"Leaves: {list(LEAF_NAMES)}")
    print(f"Expected leaf directories: {len(expected)} "
          f"(target {args.expected_count})")
    print()

    audits = []
    for path in expected:
        status, detail, parts = audit_dir(path, total_expected)
        audits.append((path, status, detail, parts))

    by_status = {k: [] for k in ("ready", "parts_ok", "parts_incomplete", "empty", "missing")}
    for path, status, detail, _ in audits:
        by_status[status].append((path, detail))

    print("Audit summary:")
    for status in ("ready", "parts_ok", "parts_incomplete", "empty", "missing"):
        print(f"  {status:18s} {len(by_status[status])}")
    print()

    flagged = by_status["missing"] + by_status["empty"] + by_status["parts_incomplete"]
    if flagged:
        print("FLAGGED directories (missing or not ready to split):")
        for path, detail in flagged:
            rel = os.path.relpath(path, args.base_dir)
            print(f"  [{rel}] {detail}")
        print()

    if len(expected) != args.expected_count:
        print(f"WARNING: found {len(expected)} expected leaf paths, "
              f"but expected {args.expected_count}.")
        print()

    if flagged and not args.force:
        raise SystemExit(
            "Refusing to split because some directories are flagged. "
            "Re-run with --force to split the ones that are ready."
        )

    to_split = [(path, parts) for path, status, _, parts in audits if status == "parts_ok"]
    if not to_split:
        print("Nothing to split.")
    else:
        print(f"Splitting {len(to_split)} directories with complete parts...")
        for path, parts in to_split:
            split_parts(
                path,
                parts,
                train_size=args.train_size,
                val_size=args.val_size,
                test_size=args.test_size,
                dry_run=args.dry_run,
            )
