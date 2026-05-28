from __future__ import annotations

import time

import numpy as np

from semdedup import SemDeDupResult, _kmeans_cluster


# Otsu's method для 1D массива значений

def _otsu_threshold(values: np.ndarray, n_bins: int = 100) -> float:

    if len(values) < 4:
        return float(np.quantile(values, 0.95))

    v_min, v_max = float(values.min()), float(values.max())
    if v_max - v_min < 1e-6:
        return v_max  # все значения одинаковы

    counts, bin_edges = np.histogram(values, bins=n_bins,
                                     range=(v_min, v_max))
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    total = counts.sum()
    if total == 0:
        return float(np.quantile(values, 0.95))

    probs = counts / total
    # Кумулятивные суммы (слева)
    cum_w = np.cumsum(probs)           # w₀(t)
    cum_mean = np.cumsum(probs * bin_centers)  # w₀(t)·μ₀(t)

    global_mean = cum_mean[-1]

    # Взвешенная межклассовая дисперсия для каждого порога
    w0 = cum_w[:-1]
    w1 = 1.0 - w0
    mu0 = np.where(w0 > 0, cum_mean[:-1] / w0, 0.0)
    mu1 = np.where(w1 > 0, (global_mean - cum_mean[:-1]) / w1, 0.0)

    sigma_between = w0 * w1 * (mu0 - mu1) ** 2

    # Находим порог
    best_idx = int(np.argmax(sigma_between))
    threshold = float(bin_centers[best_idx])

    # Sanity-check: порог должен быть правее большинства данных
    # (мы ищем порог для «верхнего хвоста» — дубликатов)
    if threshold < float(np.quantile(values, 0.70)):
        # Одномодальное распределение — Otsu нашёл шум, используем fallback
        threshold = float(np.quantile(values, 0.97))

    return threshold


#OAT: Otsu Adaptive Threshold

def semdedup_oat(
    embeddings: np.ndarray,
    n_clusters: int | None = None,
    n_bins: int = 100,
    global_floor: float = 0.80,
    global_ceiling: float = 0.999,
    random_state: int = 42,
    keep_strategy: str = "centroid",
    min_pairs_for_otsu: int = 10,
) -> SemDeDupResult:
 
    N = len(embeddings)
    timing = {}

    if n_clusters is None:
        n_clusters = max(2, int(np.sqrt(N)))

    t0 = time.perf_counter()
    labels, centroids = _kmeans_cluster(embeddings, n_clusters, random_state)
    timing["clustering"] = time.perf_counter() - t0
    actual_k = len(np.unique(labels))

    t0 = time.perf_counter()
    rng = np.random.RandomState(random_state)
    to_remove: set[int] = set()
    duplicate_pairs: list[tuple[int, int, float]] = []
    cluster_thresholds: dict[int, float] = {}
    otsu_stats: dict[int, dict] = {}

    for c in range(actual_k):
        idx = np.where(labels == c)[0]
        if len(idx) < 2:
            continue
        emb_c = embeddings[idx]
        sim = emb_c @ emb_c.T

        n = len(idx)

        triu = np.triu(np.ones((n, n), dtype=bool), k=1)
        off_sims = sim[triu]

        if len(off_sims) < min_pairs_for_otsu:
            # Мало пар — используем global_floor
            thr = global_floor
        else:
            thr = _otsu_threshold(off_sims, n_bins=n_bins)
            thr = float(np.clip(thr, global_floor, global_ceiling))

        cluster_thresholds[c] = thr
        otsu_stats[c] = {
            "n_pairs": len(off_sims),
            "sim_mean": float(off_sims.mean()),
            "sim_std": float(off_sims.std()),
            "otsu_threshold": thr,
        }

        if keep_strategy == "centroid":
            order = np.argsort(-(emb_c @ centroids[c]))
        elif keep_strategy == "first":
            order = np.arange(n)
        else:
            order = rng.permutation(n)

        removed = np.zeros(n, dtype=bool)
        for pos_i, li in enumerate(order):
            if removed[li]:
                continue
            gi = int(idx[li])
            for lj in order[pos_i + 1:]:
                if removed[lj]:
                    continue
                s = float(sim[li, lj])
                if s >= thr:
                    removed[lj] = True
                    gj = int(idx[lj])
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
        n_clusters=actual_k,
        threshold_info={
            "type": "oat",
            "n_bins": n_bins,
            "global_floor": global_floor,
            "cluster_thresholds": cluster_thresholds,
            "otsu_stats": otsu_stats,
            "mean_threshold": float(np.mean(list(cluster_thresholds.values())))
                              if cluster_thresholds else 0.0,
        },
        timing=timing,
    )


#Two-Pass OAT

def semdedup_two_pass(
    embeddings: np.ndarray,
    n_clusters: int | None = None,
    # Первый проход — поиск кандидатов (широкий)
    pass1_floor: float = 0.78,
    pass1_n_bins: int = 100,
    # Второй проход — верификация (строгий)
    pass2_floor: float = 0.85,
    pass2_n_bins: int = 50,
    random_state: int = 42,
    keep_strategy: str = "centroid",
) -> SemDeDupResult:
  
    N = len(embeddings)
    timing = {}

    if n_clusters is None:
        n_clusters = max(2, int(np.sqrt(N)))

    #Проход 1: широкий OAT
    t0 = time.perf_counter()
    labels, centroids = _kmeans_cluster(embeddings, n_clusters, random_state)
    timing["clustering"] = time.perf_counter() - t0
    actual_k = len(np.unique(labels))

    t0 = time.perf_counter()
    rng = np.random.RandomState(random_state)

    # Union-Find для группировки кандидатов
    parent = np.arange(N)
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            if ra < rb:
                parent[rb] = ra
            else:
                parent[ra] = rb
            return True
        return False

    pass1_thresholds: dict[int, float] = {}

    for c in range(actual_k):
        idx = np.where(labels == c)[0]
        if len(idx) < 2:
            continue
        emb_c = embeddings[idx]
        sim = emb_c @ emb_c.T

        n = len(idx)
        triu = np.triu(np.ones((n, n), dtype=bool), k=1)
        off_sims = sim[triu]

        thr1 = (_otsu_threshold(off_sims, n_bins=pass1_n_bins)
                if len(off_sims) >= 6 else pass1_floor)
        thr1 = float(np.clip(thr1, pass1_floor, 0.999))
        pass1_thresholds[c] = thr1

        # Объединяем кандидатные пары через Union-Find
        for li in range(n):
            for lj in range(li + 1, n):
                if sim[li, lj] >= thr1:
                    union(int(idx[li]), int(idx[lj]))

    timing["pass1"] = time.perf_counter() - t0

    #Проход 2: верификация внутри кандидатных групп
    t0 = time.perf_counter()
    from collections import defaultdict
    groups: dict[int, list[int]] = defaultdict(list)
    for i in range(N):
        groups[find(i)].append(i)

    to_remove: set[int] = set()
    duplicate_pairs: list[tuple[int, int, float]] = []
    pass2_thresholds: dict[int, float] = {}

    for root, members in groups.items():
        if len(members) < 2:
            continue
        emb_g = embeddings[members]
        sim_g = emb_g @ emb_g.T

        n = len(members)
        triu = np.triu(np.ones((n, n), dtype=bool), k=1)
        off_sims_g = sim_g[triu]

        # Второй Otsu — более строгий порог
        thr2 = (_otsu_threshold(off_sims_g, n_bins=pass2_n_bins)
                if len(off_sims_g) >= 4 else pass2_floor)
        thr2 = float(np.clip(thr2, pass2_floor, 0.999))
        pass2_thresholds[root] = thr2

        # Сортировка по близости к centroid своего кластера
        if keep_strategy == "centroid":
            cluster_of_root = int(labels[root])
            centroid = centroids[cluster_of_root]
            order = np.argsort(-(emb_g @ centroid))
        elif keep_strategy == "first":
            order = np.arange(n)
        else:
            order = rng.permutation(n)

        removed = np.zeros(n, dtype=bool)
        for pos_i, li in enumerate(order):
            if removed[li]:
                continue
            gi = int(members[li])
            for lj in order[pos_i + 1:]:
                if removed[lj]:
                    continue
                s = float(sim_g[li, lj])
                if s >= thr2:
                    removed[lj] = True
                    gj = int(members[lj])
                    to_remove.add(gj)
                    duplicate_pairs.append((gi, gj, s))

    timing["pass2"] = time.perf_counter() - t0

    remove_arr = np.array(sorted(to_remove), dtype=np.int64)
    keep_arr = np.array(sorted(set(range(N)) - to_remove), dtype=np.int64)

    return SemDeDupResult(
        keep_indices=keep_arr,
        remove_indices=remove_arr,
        duplicate_pairs=duplicate_pairs,
        cluster_assignments=labels,
        n_clusters=actual_k,
        threshold_info={
            "type": "two_pass_oat",
            "pass1_floor": pass1_floor,
            "pass2_floor": pass2_floor,
            "n_candidate_groups": len([m for m in groups.values() if len(m) >= 2]),
            "mean_pass1_threshold": float(np.mean(list(pass1_thresholds.values())))
                                    if pass1_thresholds else 0.0,
            "mean_pass2_threshold": float(np.mean(list(pass2_thresholds.values())))
                                    if pass2_thresholds else 0.0,
        },
        timing=timing,
    )


#Grid search для новых методов

def grid_oat(emb: np.ndarray, group_ids: list[int]) -> list[dict]:

    import itertools
    from evaluation import evaluate
    rows = []
    for floor, bins in itertools.product(
        [0.75, 0.78, 0.80, 0.82, 0.85],
        [50, 100, 200],
    ):
        import time as _t
        t0 = _t.perf_counter()
        r = semdedup_oat(emb, global_floor=floor, n_bins=bins)
        elapsed = _t.perf_counter() - t0
        m = evaluate(r, group_ids)
        rows.append({
            "method": "OAT",
            "global_floor": floor,
            "n_bins": bins,
            "mean_threshold": r.threshold_info["mean_threshold"],
            "time_s": round(elapsed, 3),
            **m.to_dict(),
        })
    return rows


def grid_two_pass(emb: np.ndarray, group_ids: list[int]) -> list[dict]:

    import itertools
    from evaluation import evaluate
    rows = []
    for p1, p2 in itertools.product(
        [0.75, 0.78, 0.80],
        [0.82, 0.85, 0.88],
    ):
        if p2 <= p1:
            continue
        import time as _t
        t0 = _t.perf_counter()
        r = semdedup_two_pass(emb, pass1_floor=p1, pass2_floor=p2)
        elapsed = _t.perf_counter() - t0
        m = evaluate(r, group_ids)
        rows.append({
            "method": "Two-Pass OAT",
            "pass1_floor": p1,
            "pass2_floor": p2,
            "time_s": round(elapsed, 3),
            **m.to_dict(),
        })
    return rows


#CLI demo и сравнение

if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent))

    try:
        from data_real import load_bbc_news
        from embeddings_sbert import TextEmbedder, SBERTConfig
        from evaluation import evaluate
        from semdedup import semdedup

        print("Loading BBC News...")
        ds = load_bbc_news(
            split="train", dataset_name="SetFit/bbc-news",
            n_artificial_dups=200, dup_group_size=3, seed=42,
        )
        print(f"N={len(ds)}, true_pairs={len(ds.true_duplicate_pairs())}\n")

        embedder = TextEmbedder(SBERTConfig(show_progress_bar=True))
        emb = embedder.fit_transform(ds.texts)
        group_ids = ds.group_ids

    except ImportError as e:
        print(f"Using synthetic data ({e})")
        import numpy as np
        N = 600
        emb = np.random.randn(N, 128).astype(np.float32)
        emb /= np.linalg.norm(emb, axis=1, keepdims=True)
        group_ids = list(range(N))

    from evaluation import evaluate
    from semdedup import semdedup

    algos = {
        "SemDeDup (thr=0.85)":
            lambda e: semdedup(e, threshold=0.85),
        "SemDeDup (thr=0.90)":
            lambda e: semdedup(e, threshold=0.90),
        "OAT (floor=0.80)":
            lambda e: semdedup_oat(e, global_floor=0.80),
        "OAT (floor=0.82)":
            lambda e: semdedup_oat(e, global_floor=0.82),
        "Two-Pass OAT (0.78→0.85)":
            lambda e: semdedup_two_pass(e, pass1_floor=0.78, pass2_floor=0.85),
        "Two-Pass OAT (0.80→0.88)":
            lambda e: semdedup_two_pass(e, pass1_floor=0.80, pass2_floor=0.88),
    }

    print(f"{'Algorithm':<32s} {'P':>6s} {'R':>6s} {'F1':>6s} "
          f"{'Removed':>8s} {'Singletons':>11s} {'Time':>7s}")
    print("─" * 85)
    for name, fn in algos.items():
        import time
        t0 = time.perf_counter()
        r = fn(emb)
        elapsed = time.perf_counter() - t0
        m = evaluate(r, group_ids)
        thr_info = r.threshold_info
        mean_thr = thr_info.get("mean_threshold", thr_info.get("threshold", "—"))
        if isinstance(mean_thr, float):
            mean_thr = f"{mean_thr:.3f}"
        print(f"{name:<32s} {m.precision:>6.3f} {m.recall:>6.3f} {m.f1:>6.3f} "
              f"{m.n_removed:>8d} {m.n_singletons_lost:>11d} "
              f"{elapsed*1000:>6.0f}ms  τ_mean={mean_thr}")

    # Grid search
    print("\n\n=== Grid search OAT ===")
    import pandas as pd
    rows_oat = grid_oat(emb, group_ids)
    df_oat = pd.DataFrame(rows_oat)
    best = df_oat.loc[df_oat["f1"].idxmax()]
    print(f"Best OAT: F1={best['f1']:.4f}  "
          f"P={best['precision']:.3f}  R={best['recall']:.3f}  "
          f"floor={best['global_floor']}  bins={best['n_bins']}")

    print("\n=== Grid search Two-Pass OAT ===")
    rows_tp = grid_two_pass(emb, group_ids)
    df_tp = pd.DataFrame(rows_tp)
    best_tp = df_tp.loc[df_tp["f1"].idxmax()]
    print(f"Best Two-Pass: F1={best_tp['f1']:.4f}  "
          f"P={best_tp['precision']:.3f}  R={best_tp['recall']:.3f}  "
          f"pass1={best_tp['pass1_floor']}  pass2={best_tp['pass2_floor']}")
