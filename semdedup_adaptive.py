from __future__ import annotations

import time

import numpy as np
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors

from semdedup import SemDeDupResult, _kmeans_cluster



# Вариант 1: Density-Adaptive Threshold

def _cluster_density(cluster_emb: np.ndarray, centroid: np.ndarray) -> float:

    if len(cluster_emb) == 0:
        return 0.0
    sims = cluster_emb @ centroid
    return float(np.mean(sims))


def semdedup_density_adaptive(
    embeddings: np.ndarray,
    base_threshold: float = 0.80,
    alpha: float = 0.15,
    n_clusters: int | None = None,
    random_state: int = 42,
    keep_strategy: str = "centroid",
) -> SemDeDupResult:
 
    N = len(embeddings)
    timing = {}
    if n_clusters is None:
        n_clusters = max(2, int(np.sqrt(N)))

    t0 = time.perf_counter()
    labels, centroids = _kmeans_cluster(embeddings, n_clusters, random_state)
    timing["clustering"] = time.perf_counter() - t0
    actual_n_clusters = len(np.unique(labels))

    # Сначала плотности всех кластеров (для нормализации)
    densities = {}
    for c in range(actual_n_clusters):
        idx = np.where(labels == c)[0]
        if len(idx) >= 2:
            densities[c] = _cluster_density(embeddings[idx], centroids[c])
    if densities:
        mean_density = float(np.mean(list(densities.values())))
    else:
        mean_density = 0.0

    # Поиск дубликатов с per-cluster порогом
    t0 = time.perf_counter()
    rng = np.random.RandomState(random_state)
    to_remove: set[int] = set()
    duplicate_pairs: list[tuple[int, int, float]] = []
    cluster_thresholds: dict[int, float] = {}

    for c in range(actual_n_clusters):
        idx_in_cluster = np.where(labels == c)[0]
        if len(idx_in_cluster) < 2:
            continue
        cluster_emb = embeddings[idx_in_cluster]
        sim = cluster_emb @ cluster_emb.T

        local_thr = base_threshold + alpha * (densities[c] - mean_density)
        local_thr = float(np.clip(local_thr, 0.5, 0.999))
        cluster_thresholds[c] = local_thr

        if keep_strategy == "centroid":
            to_centroid = cluster_emb @ centroids[c]
            order = np.argsort(-to_centroid)
        elif keep_strategy == "first":
            order = np.arange(len(idx_in_cluster))
        else:  # random
            order = rng.permutation(len(idx_in_cluster))

        removed = np.zeros(len(idx_in_cluster), dtype=bool)
        for pos_i, li in enumerate(order):
            if removed[li]:
                continue
            gi = int(idx_in_cluster[li])
            for lj in order[pos_i + 1:]:
                if removed[lj]:
                    continue
                s = float(sim[li, lj])
                if s >= local_thr:
                    removed[lj] = True
                    gj = int(idx_in_cluster[lj])
                    to_remove.add(gj)
                    duplicate_pairs.append((gi, gj, s))

    timing["pairwise"] = time.perf_counter() - t0

    remove_arr = np.array(sorted(to_remove), dtype=np.int64)
    keep_arr = np.array(sorted(set(range(N)) - to_remove), dtype=np.int64)

    return SemDeDupResult(
        keep_indices=keep_arr,
        remove_indices=remove_arr,
        duplicate_pairs=duplicate_pairs,
        cluster_assignments=labels,
        n_clusters=actual_n_clusters,
        threshold_info={
            "type": "density_adaptive",
            "base_threshold": base_threshold,
            "alpha": alpha,
            "mean_density": mean_density,
            "cluster_thresholds": cluster_thresholds,
        },
        timing=timing,
    )


# Вариант 2: k-NN Local Threshold

def semdedup_knn_local(
    embeddings: np.ndarray,
    k: int = 10,
    quantile: float = 0.95,
    n_clusters: int | None = None,
    random_state: int = 42,
    keep_strategy: str = "centroid",
    min_threshold: float = 0.7,
) -> SemDeDupResult:

    N = len(embeddings)
    timing = {}
    if n_clusters is None:
        n_clusters = max(2, int(np.sqrt(N)))

    # 1) Локальные пороги через kNN
    t0 = time.perf_counter()
    k_eff = min(k, N - 1)
    nn = NearestNeighbors(n_neighbors=k_eff + 1, metric="cosine")
    nn.fit(embeddings)
    distances, _ = nn.kneighbors(embeddings)
    # отбрасываем сам документ (distance == 0 в первой колонке)
    distances = distances[:, 1:]
    sims_to_nn = 1.0 - distances        # cos sim = 1 - cos dist
    # quantile_q сходств с соседями. q=0.95 => высокая планка
    local_tau = np.quantile(sims_to_nn, quantile, axis=1)
    local_tau = np.maximum(local_tau, min_threshold)
    timing["knn_threshold"] = time.perf_counter() - t0

    # 2) Кластеризация
    t0 = time.perf_counter()
    labels, centroids = _kmeans_cluster(embeddings, n_clusters, random_state)
    timing["clustering"] = time.perf_counter() - t0
    actual_n_clusters = len(np.unique(labels))

    # 3) Внутрикластерное сравнение с парными порогами
    t0 = time.perf_counter()
    rng = np.random.RandomState(random_state)
    to_remove: set[int] = set()
    duplicate_pairs: list[tuple[int, int, float]] = []

    for c in range(actual_n_clusters):
        idx_in_cluster = np.where(labels == c)[0]
        if len(idx_in_cluster) < 2:
            continue
        cluster_emb = embeddings[idx_in_cluster]
        sim = cluster_emb @ cluster_emb.T
        cluster_tau = local_tau[idx_in_cluster]

        if keep_strategy == "centroid":
            order = np.argsort(-(cluster_emb @ centroids[c]))
        elif keep_strategy == "first":
            order = np.arange(len(idx_in_cluster))
        else:
            order = rng.permutation(len(idx_in_cluster))

        removed = np.zeros(len(idx_in_cluster), dtype=bool)
        for pos_i, li in enumerate(order):
            if removed[li]:
                continue
            gi = int(idx_in_cluster[li])
            for lj in order[pos_i + 1:]:
                if removed[lj]:
                    continue
                s = float(sim[li, lj])
                pair_thr = max(cluster_tau[li], cluster_tau[lj])
                if s >= pair_thr:
                    removed[lj] = True
                    gj = int(idx_in_cluster[lj])
                    to_remove.add(gj)
                    duplicate_pairs.append((gi, gj, s))

    timing["pairwise"] = time.perf_counter() - t0

    remove_arr = np.array(sorted(to_remove), dtype=np.int64)
    keep_arr = np.array(sorted(set(range(N)) - to_remove), dtype=np.int64)

    return SemDeDupResult(
        keep_indices=keep_arr,
        remove_indices=remove_arr,
        duplicate_pairs=duplicate_pairs,
        cluster_assignments=labels,
        n_clusters=actual_n_clusters,
        threshold_info={
            "type": "knn_local",
            "k": k_eff,
            "quantile": quantile,
            "min_threshold": min_threshold,
            "tau_mean": float(local_tau.mean()),
            "tau_std": float(local_tau.std()),
            "tau_min": float(local_tau.min()),
            "tau_max": float(local_tau.max()),
        },
        timing=timing,
    )



# Вариант 3: Percentile-based Cluster Threshold

def semdedup_percentile_cluster(
    embeddings: np.ndarray,
    percentile: float = 0.98,
    n_clusters: int | None = None,
    random_state: int = 42,
    keep_strategy: str = "centroid",
    global_floor: float = 0.75,
) -> SemDeDupResult:

    N = len(embeddings)
    timing = {}
    if n_clusters is None:
        n_clusters = max(2, int(np.sqrt(N)))

    t0 = time.perf_counter()
    labels, centroids = _kmeans_cluster(embeddings, n_clusters, random_state)
    timing["clustering"] = time.perf_counter() - t0
    actual_n_clusters = len(np.unique(labels))

    t0 = time.perf_counter()
    rng = np.random.RandomState(random_state)
    to_remove: set[int] = set()
    duplicate_pairs: list[tuple[int, int, float]] = []
    cluster_thresholds: dict[int, float] = {}

    for c in range(actual_n_clusters):
        idx_in_cluster = np.where(labels == c)[0]
        if len(idx_in_cluster) < 2:
            continue
        cluster_emb = embeddings[idx_in_cluster]
        sim = cluster_emb @ cluster_emb.T


        n = len(idx_in_cluster)
        triu_mask = np.triu(np.ones((n, n), dtype=bool), k=1)
        off_diag_sims = sim[triu_mask]
        if len(off_diag_sims) == 0:
            continue

        local_thr = float(np.quantile(off_diag_sims, percentile))
        local_thr = max(local_thr, global_floor)
        cluster_thresholds[c] = local_thr

        if keep_strategy == "centroid":
            order = np.argsort(-(cluster_emb @ centroids[c]))
        elif keep_strategy == "first":
            order = np.arange(len(idx_in_cluster))
        else:
            order = rng.permutation(len(idx_in_cluster))

        removed = np.zeros(len(idx_in_cluster), dtype=bool)
        for pos_i, li in enumerate(order):
            if removed[li]:
                continue
            gi = int(idx_in_cluster[li])
            for lj in order[pos_i + 1:]:
                if removed[lj]:
                    continue
                s = float(sim[li, lj])
                if s >= local_thr:
                    removed[lj] = True
                    gj = int(idx_in_cluster[lj])
                    to_remove.add(gj)
                    duplicate_pairs.append((gi, gj, s))

    timing["pairwise"] = time.perf_counter() - t0

    remove_arr = np.array(sorted(to_remove), dtype=np.int64)
    keep_arr = np.array(sorted(set(range(N)) - to_remove), dtype=np.int64)

    return SemDeDupResult(
        keep_indices=keep_arr,
        remove_indices=remove_arr,
        duplicate_pairs=duplicate_pairs,
        cluster_assignments=labels,
        n_clusters=actual_n_clusters,
        threshold_info={
            "type": "percentile_cluster",
            "percentile": percentile,
            "global_floor": global_floor,
            "cluster_thresholds": cluster_thresholds,
            "mean_threshold": float(np.mean(list(cluster_thresholds.values())))
                              if cluster_thresholds else 0.0,
        },
        timing=timing,
    )


if __name__ == "__main__":
    from embeddings import embed_texts
    from semdedup import semdedup

    docs = [
        "The cat sat on the mat",
        "A cat was sitting on a mat",
        "Dogs love running in the park",
        "The dog enjoyed running through the park",
        "Stocks fell after the Federal Reserve announcement",
        "The Fed's announcement caused stocks to drop sharply",
        "Pizza is a popular Italian dish",
        "Pizza, an Italian classic, is loved worldwide",
    ] * 5
    docs += ["The cat sat on the mat"] * 3

    emb = embed_texts(docs, n_components=32)
    print(f"Embeddings shape: {emb.shape}\n")

    print("Base SemDeDup (thr=0.85):")
    r = semdedup(emb, threshold=0.85, n_clusters=4)
    print(f"  Removed: {r.n_removed}/{len(docs)}  ({r.reduction_ratio:.1%})")

    print("\nDensity-Adaptive (base=0.80, alpha=0.15):")
    r = semdedup_density_adaptive(emb, base_threshold=0.80, alpha=0.15, n_clusters=4)
    print(f"  Removed: {r.n_removed}/{len(docs)}  ({r.reduction_ratio:.1%})")
    print(f"  Per-cluster thresholds: {r.threshold_info['cluster_thresholds']}")

    print("\nkNN-Local (k=5, q=0.95):")
    r = semdedup_knn_local(emb, k=5, quantile=0.95, n_clusters=4)
    print(f"  Removed: {r.n_removed}/{len(docs)}  ({r.reduction_ratio:.1%})")
    print(f"  tau: mean={r.threshold_info['tau_mean']:.3f} "
          f"min={r.threshold_info['tau_min']:.3f} max={r.threshold_info['tau_max']:.3f}")

    print("\nPercentile-Cluster (p=0.95):")
    r = semdedup_percentile_cluster(emb, percentile=0.95, n_clusters=4)
    print(f"  Removed: {r.n_removed}/{len(docs)}  ({r.reduction_ratio:.1%})")
    print(f"  Mean threshold: {r.threshold_info['mean_threshold']:.3f}")
