"""Dependency-free manifest checks and complete song-balanced epoch ordering."""
from collections import defaultdict
import json
from pathlib import Path
import random


def load_manifest(path, split):
    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line]
    if not rows or any(row["split"] != split for row in rows):
        raise ValueError(f"Expected nonempty {split} manifest")
    if len({row["id"] for row in rows}) != len(rows):
        raise ValueError("Duplicate clip IDs")
    return rows


def epoch_order(rows, seed, epoch):
    """Round-robin songs within each shuffled epoch; expose every clip once."""
    groups = defaultdict(list)
    for row in rows:
        groups[row["song"]].append(row)
    rng = random.Random(seed + epoch)
    keys = sorted(groups)
    for clips in groups.values():
        rng.shuffle(clips)
    result = []
    while any(groups.values()):
        rng.shuffle(keys)
        for key in keys:
            if groups[key]:
                result.append(groups[key].pop())
    return result
