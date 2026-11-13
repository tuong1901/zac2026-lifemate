from __future__ import annotations

from collections import defaultdict

from .parser import step_index


def group_trajectories(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[row["source_id"]].append(row)
    for source, items in groups.items():
        items.sort(key=lambda r: step_index(r["id"]))
        indices = [step_index(r["id"]) for r in items]
        if len(indices) != len(set(indices)):
            raise ValueError(f"{source}: duplicate step indices")
        if indices and indices != list(range(indices[0], indices[-1] + 1)):
            raise ValueError(f"{source}: discontinuous step sequence {indices}")
    return dict(groups)
