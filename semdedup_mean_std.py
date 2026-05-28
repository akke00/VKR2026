from __future__ import annotations

import time

import numpy as np

from semdedup import SemDeDupResult, _kmeans_cluster


def semdedup_cluster_mean_std(
    embeddings: np.ndarray,
    alpha: float = 1.5,
    n_clusters: int | None = None,
    random_state: int = 42,
    keep_strategy: str = "centroid",
    global_floor: float = 0.70,
    global_ceiling: float = 0.99,
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
    cluster_stats: dict[int, dict] = {}

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

        mean_c = float(np.mean(off_diag_sims))
        std_c = float(np.std(off_diag_sims))
        raw_thr = mean_c + alpha * std_c
        local_thr = float(np.clip(raw_thr, global_floor, global_ceiling))
        cluster_thresholds[c] = local_thr
        cluster_stats[c] = {
            "mean": mean_c,
            "std": std_c,
            "raw_threshold": raw_thr,
            "applied_threshold": local_thr,
            "size": int(n),
        }

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
            "type": "cluster_mean_std",
            "alpha": alpha,
            "global_floor": global_floor,
            "global_ceiling": global_ceiling,
            "cluster_thresholds": cluster_thresholds,
            "cluster_stats": cluster_stats,
            "mean_threshold": (
                float(np.mean(list(cluster_thresholds.values())))
                if cluster_thresholds else 0.0
            ),
        },
        timing=timing,
    )


if __name__ == "__main__":
    from embeddings_sbert import embed_texts

    docs = [
        "The cat sat on the mat",
        "A cat was sitting on a mat",
        "Dogs love running in the park",
        "The dog enjoyed running through the park",
        "Stocks fell after the Fed announcement",
        "The Fed's announcement caused stocks to drop sharply",
    ] * 5

    emb = embed_texts(docs)
    print(f"Embeddings: {emb.shape}\n")

    for alpha in [0.5, 1.0, 1.5, 2.0]:
        r = semdedup_cluster_mean_std(emb, alpha=alpha, n_clusters=4)
        print(f"alpha={alpha}: removed={r.n_removed}/{len(docs)}  "
              f"mean_thr={r.threshold_info['mean_threshold']:.3f}")
