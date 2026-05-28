from __future__ import annotations

import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from data_qqp import load_qqp, print_stats
from embeddings_sbert import TextEmbedder, SBERTConfig
from evaluation_pairs import evaluate_pairs
from semdedup import semdedup
from semdedup_adaptive import (
    semdedup_density_adaptive,
    semdedup_knn_local,
    semdedup_percentile_cluster,
)
from semdedup_mean_std import semdedup_cluster_mean_std
from semdedup_otsu import semdedup_oat, semdedup_two_pass

RESULTS_DIR = Path("results_full")
RESULTS_DIR.mkdir(exist_ok=True)

GRID_N    = 5000   # размер корпуса для grid search
GRID_SEED = 0      # seed=0 — отдельная выборка от финального бенчмарка (seeds 1,2,3)


def _run(fn, emb, true_pairs) -> dict:
    t0 = time.perf_counter()
    r = fn(emb)
    elapsed = time.perf_counter() - t0
    m = evaluate_pairs(r, true_pairs)
    return {"time_s": round(elapsed, 3), **m.to_dict()}


def grid_semdedup(emb, true_pairs):
    rows = []
    for thr in [0.75, 0.78, 0.80, 0.82, 0.85, 0.87, 0.90]:
        d = _run(lambda e, t=thr: semdedup(e, threshold=t), emb, true_pairs)
        rows.append({"method": "SemDeDup", "threshold": thr, **d})
    return rows


def grid_dat(emb, true_pairs):
    rows = []
    for base, alpha in itertools.product(
        [0.75, 0.80, 0.85, 0.87],
        [0.05, 0.10, 0.15, 0.20],
    ):
        d = _run(lambda e, b=base, a=alpha: semdedup_density_adaptive(
            e, base_threshold=b, alpha=a), emb, true_pairs)
        rows.append({"method": "DAT", "base_threshold": base, "alpha": alpha, **d})
    return rows


def grid_knn_lt(emb, true_pairs):
    rows = []
    for k, q in itertools.product(
        [5, 10, 15, 20, 30],
        [0.80, 0.85, 0.90, 0.95],
    ):
        d = _run(lambda e, k_=k, q_=q: semdedup_knn_local(
            e, k=k_, quantile=q_, min_threshold=0.70), emb, true_pairs)
        rows.append({"method": "kNN-LT", "k": k, "quantile": q, **d})
    return rows


def grid_percentile(emb, true_pairs):
    rows = []
    for p, floor in itertools.product(
        [0.90, 0.93, 0.95, 0.97],
        [0.70, 0.75, 0.80],
    ):
        d = _run(lambda e, p_=p, f_=floor: semdedup_percentile_cluster(
            e, percentile=p_, global_floor=f_), emb, true_pairs)
        rows.append({"method": "Percentile", "percentile": p, "global_floor": floor, **d})
    return rows


def grid_mean_std(emb, true_pairs):
    rows = []
    for alpha, floor in itertools.product(
        [0.5, 1.0, 1.5, 2.0, 2.5, 3.0],
        [0.65, 0.75],
    ):
        d = _run(lambda e, a=alpha, f=floor: semdedup_cluster_mean_std(
            e, alpha=a, global_floor=f), emb, true_pairs)
        rows.append({"method": "Mean+Std", "alpha": alpha, "global_floor": floor, **d})
    return rows


def grid_oat(emb, true_pairs):
    rows = []
    for floor, bins in itertools.product(
        [0.70, 0.75, 0.78, 0.80, 0.82],
        [50, 100, 200],
    ):
        d = _run(lambda e, f=floor, b=bins: semdedup_oat(
            e, global_floor=f, n_bins=b), emb, true_pairs)
        rows.append({"method": "OAT", "global_floor": floor, "n_bins": bins, **d})
    return rows


def grid_two_pass(emb, true_pairs):
    rows = []
    for p1, p2 in itertools.product(
        [0.70, 0.73, 0.75, 0.78],
        [0.78, 0.80, 0.83, 0.85],
    ):
        if p2 <= p1:
            continue
        d = _run(lambda e, a=p1, b=p2: semdedup_two_pass(
            e, pass1_floor=a, pass2_floor=b), emb, true_pairs)
        rows.append({"method": "Two-Pass OAT", "pass1_floor": p1, "pass2_floor": p2, **d})
    return rows


GRID_RUNNERS = {
    "SemDeDup":     (grid_semdedup,   7),
    "DAT":          (grid_dat,       16),
    "kNN-LT":       (grid_knn_lt,    20),
    "Percentile":   (grid_percentile,12),
    "Mean+Std":     (grid_mean_std,  12),
    "OAT":          (grid_oat,       15),
    "Two-Pass OAT": (grid_two_pass,  "~9"),
}


def main():
    # 1) Данные
    print(f"Loading QQP (N={GRID_N}, seed={GRID_SEED})...")
    ds = load_qqp(n_docs=GRID_N, seed=GRID_SEED)
    print_stats(ds, label="grid_search")

    # 2) Эмбеддинги
    print("\nComputing SBERT embeddings...")
    embedder = TextEmbedder(SBERTConfig(
        model_name="all-MiniLM-L6-v2",
        show_progress_bar=True,
    ))
    t0 = time.perf_counter()
    emb = embedder.fit_transform(ds.texts)
    print(f"Embeddings: {emb.shape}  [{time.perf_counter()-t0:.1f}s]\n")

    # 3) Grid search
    all_rows = []
    for method_name, (runner, n_configs) in GRID_RUNNERS.items():
        print(f"{'='*60}")
        print(f"{method_name} ({n_configs} configs)")
        print(f"{'='*60}")
        rows = runner(emb, ds.true_pairs)
        for r in rows:
            param_cols = [c for c in ["threshold", "base_threshold", "alpha",
                                       "k", "quantile", "percentile",
                                       "global_floor", "n_bins",
                                       "pass1_floor", "pass2_floor"]
                          if c in r and r[c] is not None]
            params = "  ".join(f"{c}={r[c]}" for c in param_cols)
            print(f"  F1={r['f1']:.3f}  P={r['precision']:.3f}  "
                  f"R={r['recall']:.3f}  [{params}]")
        all_rows.extend(rows)

    # 4) Сохранение
    df = pd.DataFrame(all_rows)
    df.to_csv(RESULTS_DIR / "grid_search_qqp.csv", index=False)
    print(f"\nSaved {len(df)} configs → {RESULTS_DIR}/grid_search_qqp.csv")

    # 5) Лучшие параметры
    best_params: dict[str, dict] = {}
    print(f"\n{'='*60}")
    print("BEST CONFIG PER METHOD (by F1)")
    print(f"{'='*60}")

    for method in df["method"].unique():
        sub = df[df["method"] == method].sort_values("f1", ascending=False)
        best = sub.iloc[0].to_dict()
        param_cols = [c for c in ["threshold", "base_threshold", "alpha",
                                   "k", "quantile", "percentile",
                                   "global_floor", "n_bins",
                                   "pass1_floor", "pass2_floor"]
                      if c in best and pd.notna(best.get(c))]
        params = {c: best[c] for c in param_cols}
        best_params[method] = params
        params_str = "  ".join(f"{k}={v}" for k, v in params.items())
        print(f"  {method:<16s}  F1={best['f1']:.4f}  "
              f"P={best['precision']:.3f}  R={best['recall']:.3f}  "
              f"[{params_str}]")

    # Сохраняем лучшие параметры в JSON
    with open(RESULTS_DIR / "best_params_qqp.json", "w") as f:
        json.dump(best_params, f, indent=2)
    print(f"\nSaved best params → {RESULTS_DIR}/best_params_qqp.json")

    return best_params


if __name__ == "__main__":
    main()