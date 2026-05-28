from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class QQPDataset:
    texts: list[str]
    true_pairs: set[tuple[int, int]]   # индексы в texts, i < j

    def __len__(self) -> int:
        return len(self.texts)

    @property
    def n_true_pairs(self) -> int:
        return len(self.true_pairs)

    @property
    def n_duplicate_docs(self) -> int:
        involved = set()
        for i, j in self.true_pairs:
            involved.add(i)
            involved.add(j)
        return len(involved)


def _load_raw_qqp() -> tuple[dict, list, list]:

    global _QQP_CACHE
    if _QQP_CACHE is not None:
        return _QQP_CACHE

    from datasets import load_dataset


    SOURCES = [
        ("AlekseyKorshuk/quora-question-pairs", "train", "flat"),
        ("michaelrglass/quora-question-pairs", "train", "flat"),
    ]

    ds = None
    fmt = None
    for dataset_name, split, structure in SOURCES:
        try:
            print(f"Loading {dataset_name} ({split})...")
            ds = load_dataset(dataset_name, split=split)
            fmt = structure
            print(f"  Loaded {len(ds):,} rows.")
            break
        except Exception as e:
            print(f"  Failed ({e}), trying next source...")

    if ds is None:
        raise RuntimeError(
            "Не удалось загрузить QQP ни из одного источника.\n"
        )

    texts_by_id: dict[int, str] = {}
    dup_pairs: list[tuple[int, int]] = []

    if fmt == "flat":
        # Структура: qid1, qid2, question1, question2, is_duplicate
        for row in ds:
            qid1 = int(row["qid1"])
            qid2 = int(row["qid2"])
            q1 = str(row.get("question1", "") or "").strip()
            q2 = str(row.get("question2", "") or "").strip()
            is_dup = bool(row["is_duplicate"])

            if q1 and qid1 not in texts_by_id:
                texts_by_id[qid1] = q1
            if q2 and qid2 not in texts_by_id:
                texts_by_id[qid2] = q2

            if is_dup:
                dup_pairs.append((qid1, qid2))

    # Синглтоны = qids не в дубликатных парах
    dup_ids: set[int] = set()
    for a, b in dup_pairs:
        dup_ids.add(a)
        dup_ids.add(b)
    singleton_ids = [qid for qid in texts_by_id if qid not in dup_ids]

    print(f"  Unique questions: {len(texts_by_id):,}")
    print(f"  Duplicate pairs:  {len(dup_pairs):,}")
    print(f"  Singletons:       {len(singleton_ids):,}")

    _QQP_CACHE = (texts_by_id, dup_pairs, singleton_ids)
    return _QQP_CACHE


_QQP_CACHE = None


def load_qqp(
    n_docs: int,
    dup_fraction: float = 0.30,
    seed: int = 42,
) -> QQPDataset:
    
    texts_by_id, dup_pairs, singleton_ids = _load_raw_qqp()
    rng = np.random.RandomState(seed)

    # Сколько документов из дубликатных пар
    n_dup_docs = max(4, int(round(n_docs * dup_fraction)))
    # Сколько пар нам нужно (каждая пара = 2 уникальных документа минимум)
    n_pairs_needed = max(2, n_dup_docs // 2)
    n_pairs_needed = min(n_pairs_needed, len(dup_pairs))

    # Сэмплируем пары
    pair_indices = rng.choice(len(dup_pairs), size=n_pairs_needed, replace=False)
    chosen_pairs = [dup_pairs[i] for i in pair_indices]

    # Собираем уникальные qid из выбранных пар
    dup_qids_ordered: list[int] = []
    seen_dup = set()
    for a, b in chosen_pairs:
        for qid in (a, b):
            if qid not in seen_dup and qid in texts_by_id:
                dup_qids_ordered.append(qid)
                seen_dup.add(qid)

    # Синглтоны до n_docs
    n_singletons = max(0, n_docs - len(dup_qids_ordered))
    n_singletons = min(n_singletons, len(singleton_ids))

    sing_indices = rng.choice(len(singleton_ids), size=n_singletons, replace=False)
    chosen_singletons = [singleton_ids[i] for i in sing_indices]

    # Собираем корпус
    all_qids = dup_qids_ordered + chosen_singletons
    # Перемешиваем
    perm = rng.permutation(len(all_qids)).tolist()
    all_qids = [all_qids[i] for i in perm]

    texts = [texts_by_id[qid] for qid in all_qids]
    qid_to_idx = {qid: i for i, qid in enumerate(all_qids)}

    # True pairs: только те пары из chosen_pairs, где ОБА qid попали в выборку
    true_pairs: set[tuple[int, int]] = set()
    for a, b in chosen_pairs:
        if a in qid_to_idx and b in qid_to_idx:
            i, j = qid_to_idx[a], qid_to_idx[b]
            true_pairs.add((min(i, j), max(i, j)))

    return QQPDataset(texts=texts, true_pairs=true_pairs)


def print_stats(ds: QQPDataset, label: str = ""):
    tag = f"[{label}] " if label else ""
    print(f"{tag}N={len(ds)}, true_pairs={ds.n_true_pairs}, "
          f"dup_docs={ds.n_duplicate_docs} "
          f"({ds.n_duplicate_docs/len(ds)*100:.0f}%)")


if __name__ == "__main__":
    for n in [50, 100, 500, 1000]:
        ds = load_qqp(n_docs=n, seed=42)
        print_stats(ds, label=f"N={n}")