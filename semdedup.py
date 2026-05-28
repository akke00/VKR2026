from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
from sklearn.cluster import KMeans


@dataclass
class SemDeDupResult:
    """Результат дедупликации."""
    keep_indices: np.ndarray              # индексы документов, которые оставляем
    remove_indices: np.ndarray            # индексы, которые удаляем
    duplicate_pairs: list[tuple[int, int, float]]  # (kept_idx, removed_idx, similarity)
    cluster_assignments: np.ndarray
    n_clusters: int
    threshold_info: dict = field(default_factory=dict)
    timing: dict = field(default_factory=dict)

    @property
    def n_kept(self) -> int:
        return len(self.keep_indices)

    @property
    def n_removed(self) -> int:
        return len(self.remove_indices)

    @property
    def reduction_ratio(self) -> float:
        total = self.n_kept + self.n_removed
        return self.n_removed / total if total > 0 else 0.0


def _kmeans_cluster(emb: np.ndarray, n_clusters: int, random_state: int = 42):
 
    n_clusters = min(n_clusters, max(2, len(emb) // 2))
    km = KMeans(
        n_clusters=n_clusters,
        random_state=random_state,
        n_init=5,
    )
    labels = km.fit_predict(emb)
    return labels, km.cluster_centers_


def semdedup(
    embeddings: np.ndarray,
    threshold: float = 0.85,
    n_clusters: int | None = None,
    random_state: int = 42,
    keep_strategy: str = "centroid",   # "centroid" | "first" | "random"
) -> SemDeDupResult:

    N = len(embeddings)
    timing = {}

    if n_clusters is None:
        n_clusters = max(2, int(np.sqrt(N)))

    # 1. Кластеризация
    t0 = time.perf_counter()
    labels, centroids = _kmeans_cluster(embeddings, n_clusters, random_state)
    timing["clustering"] = time.perf_counter() - t0
    actual_n_clusters = len(np.unique(labels))

    # 2. Внутрикластерное сравнение
    t0 = time.perf_counter()
    rng = np.random.RandomState(random_state)
    to_remove: set[int] = set()
    duplicate_pairs: list[tuple[int, int, float]] = []

    for c in range(actual_n_clusters):
        idx_in_cluster = np.where(labels == c)[0]
        if len(idx_in_cluster) < 2:
            continue
        cluster_emb = embeddings[idx_in_cluster]
        # Pairwise cosine = dot product (нормализованы)
        sim = cluster_emb @ cluster_emb.T

        if keep_strategy == "centroid":
            # similarity каждого документа с центроидом
            to_centroid = cluster_emb @ centroids[c]
            # сортировка по убыванию близости к центроиду
            order = np.argsort(-to_centroid)
        elif keep_strategy == "first":
            order = np.arange(len(idx_in_cluster))
        elif keep_strategy == "random":
            order = rng.permutation(len(idx_in_cluster))
        else:
            raise ValueError(f"Unknown keep_strategy: {keep_strategy}")

        # Жадный проход: первый в order — "глава" группы дубликатов.
        # Любой неубранный документ, сходство которого с уже принятым > threshold, удаляется.
        already_removed_local = np.zeros(len(idx_in_cluster), dtype=bool)
        for pos_i, local_i in enumerate(order):
            if already_removed_local[local_i]:
                continue
            global_i = int(idx_in_cluster[local_i])
            # сравниваем с теми, кто позже в order и ещё не удалён
            for local_j in order[pos_i + 1:]:
                if already_removed_local[local_j]:
                    continue
                s = float(sim[local_i, local_j])
                if s >= threshold:
                    already_removed_local[local_j] = True
                    global_j = int(idx_in_cluster[local_j])
                    to_remove.add(global_j)
                    duplicate_pairs.append((global_i, global_j, s))

    timing["pairwise"] = time.perf_counter() - t0

    remove_arr = np.array(sorted(to_remove), dtype=np.int64)
    keep_arr = np.array(sorted(set(range(N)) - to_remove), dtype=np.int64)

    return SemDeDupResult(
        keep_indices=keep_arr,
        remove_indices=remove_arr,
        duplicate_pairs=duplicate_pairs,
        cluster_assignments=labels,
        n_clusters=actual_n_clusters,
        threshold_info={"type": "global", "threshold": threshold},
        timing=timing,
    )


if __name__ == "__main__":
    from embeddings import embed_texts

    docs = [
        "The cat sat on the mat",
        "A cat was sitting on a mat",                          # dup of 0
        "Dogs love running in the park",
        "The dog enjoyed running through the park",            # dup of 2
        "Stocks fell after the Federal Reserve announcement",
        "The Fed's announcement caused stocks to drop sharply",  # dup of 4
        "Pizza is a popular Italian dish",
        "Pizza, an Italian classic, is loved worldwide",       # dup of 6
    ] * 5  # 40 документов

    # Введём также абсолютные дубликаты
    docs = docs + ["The cat sat on the mat"] * 3

    emb = embed_texts(docs, n_components=32)
    print(f"Embeddings: {emb.shape}")

    result = semdedup(emb, threshold=0.85, n_clusters=4)
    print(f"Kept: {result.n_kept}, Removed: {result.n_removed}")
    print(f"Reduction: {result.reduction_ratio:.1%}")
    print(f"Timing: clustering={result.timing['clustering']*1000:.1f}ms, "
          f"pairwise={result.timing['pairwise']*1000:.1f}ms")
    print(f"\nFirst 5 duplicate pairs:")
    for kept, removed, sim in result.duplicate_pairs[:5]:
        print(f"  kept={kept} removed={removed} sim={sim:.3f}")
