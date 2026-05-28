from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from semdedup import SemDeDupResult


@dataclass
class PairEvalMetrics:
    precision: float
    recall: float
    f1: float
    n_true_pairs: int
    n_predicted_pairs: int
    n_true_positive_pairs: int
    n_input: int
    n_kept: int
    n_removed: int
    reduction_ratio: float
    n_singletons_lost: int   # удалённые документы не из true_pairs

    def to_dict(self) -> dict:
        return {
            "precision":             round(self.precision, 4),
            "recall":                round(self.recall, 4),
            "f1":                    round(self.f1, 4),
            "n_true_pairs":          self.n_true_pairs,
            "n_predicted_pairs":     self.n_predicted_pairs,
            "n_true_positive_pairs": self.n_true_positive_pairs,
            "n_input":               self.n_input,
            "n_kept":                self.n_kept,
            "n_removed":             self.n_removed,
            "reduction_ratio":       round(self.reduction_ratio, 4),
            "n_singletons_lost":     self.n_singletons_lost,
        }


def evaluate_pairs(
    result: SemDeDupResult,
    true_pairs: set[tuple[int, int]],
) -> PairEvalMetrics:

    N = result.n_kept + result.n_removed

    # Predicted pairs — из duplicate_pairs алгоритма
    pred_pairs: set[tuple[int, int]] = set()
    for a, b, _ in result.duplicate_pairs:
        x, y = int(a), int(b)
        pred_pairs.add((min(x, y), max(x, y)))

    tp = true_pairs & pred_pairs
    precision = len(tp) / max(1, len(pred_pairs))
    recall    = len(tp) / max(1, len(true_pairs))
    f1 = (2 * precision * recall / (precision + recall)
          if (precision + recall) > 0 else 0.0)

    # Документы, входящие хотя бы в одну true pair
    docs_in_true_pairs: set[int] = set()
    for i, j in true_pairs:
        docs_in_true_pairs.add(i)
        docs_in_true_pairs.add(j)

    # Синглтоны = документы НЕ из true_pairs
    remove_set = set(result.remove_indices.tolist())
    n_singletons_lost = sum(
        1 for idx in remove_set
        if idx not in docs_in_true_pairs
    )

    return PairEvalMetrics(
        precision=precision,
        recall=recall,
        f1=f1,
        n_true_pairs=len(true_pairs),
        n_predicted_pairs=len(pred_pairs),
        n_true_positive_pairs=len(tp),
        n_input=N,
        n_kept=result.n_kept,
        n_removed=result.n_removed,
        reduction_ratio=result.reduction_ratio,
        n_singletons_lost=n_singletons_lost,
    )


def groups_to_pairs(group_ids: list[int]) -> set[tuple[int, int]]:

    from collections import defaultdict
    groups: dict[int, list[int]] = defaultdict(list)
    for idx, gid in enumerate(group_ids):
        if gid >= 0:
            groups[gid].append(idx)

    pairs: set[tuple[int, int]] = set()
    for members in groups.values():
        if len(members) < 2:
            continue
        for i in range(len(members)):
            for j in range(i + 1, len(members)):
                a, b = sorted((members[i], members[j]))
                pairs.add((a, b))
    return pairs
