from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from data_qqp import load_qqp, print_stats, _load_raw_qqp
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
FIGURES_DIR = Path("figures_full")
RESULTS_DIR.mkdir(exist_ok=True)
FIGURES_DIR.mkdir(exist_ok=True)

SIZES = [50, 100, 500, 1000, 2000, 5000, 10000]
SEEDS = [1, 2, 3]        
ILLUSTRATIVE_N = {50, 100}


#Загрузка лучших параметров

def load_best_params() -> dict:
    path = RESULTS_DIR / "best_params_qqp.json"
    if path.exists():
        with open(path) as f:
            params = json.load(f)
        print(f"Loaded best params from {path}")
        return params
    else:
        print("best_params_qqp.json not found — running grid search first...")
        from grid_search_qqp import main as run_grid
        return run_grid()


def build_algos(params: dict, n_docs: int | None = None) -> dict:

    def p(method, key, default):
        return params.get(method, {}).get(key, default)

    
    k_best = int(p("kNN-LT", "k", 20))
    k_adaptive = k_best if n_docs is None else max(2, min(k_best, n_docs // 10))

    return {
        "SemDeDup": lambda emb: semdedup(
            emb,
            threshold=p("SemDeDup", "threshold", 0.82),
        ),
        "DAT": lambda emb: semdedup_density_adaptive(
            emb,
            base_threshold=p("DAT", "base_threshold", 0.82),
            alpha=p("DAT", "alpha", 0.10),
        ),
        "kNN-LT": lambda emb, k=k_adaptive: semdedup_knn_local(
            emb,
            k=k,
            quantile=p("kNN-LT", "quantile", 0.90),
            min_threshold=0.70,
        ),
        "Percentile": lambda emb: semdedup_percentile_cluster(
            emb,
            percentile=p("Percentile", "percentile", 0.95),
            global_floor=p("Percentile", "global_floor", 0.75),
        ),
        "Mean+Std": lambda emb: semdedup_cluster_mean_std(
            emb,
            alpha=p("Mean+Std", "alpha", 1.5),
            global_floor=p("Mean+Std", "global_floor", 0.75),
        ),
        "OAT": lambda emb: semdedup_oat(
            emb,
            global_floor=p("OAT", "global_floor", 0.75),
            n_bins=int(p("OAT", "n_bins", 100)),
        ),
        "Two-Pass OAT": lambda emb: semdedup_two_pass(
            emb,
            pass1_floor=p("Two-Pass OAT", "pass1_floor", 0.73),
            pass2_floor=p("Two-Pass OAT", "pass2_floor", 0.80),
        ),
    }


#Основной бенч

def run_single(
    n: int,
    best_params: dict,
    embedder: TextEmbedder,
    seed: int,
) -> list[dict]:

    ds = load_qqp(n_docs=n, seed=seed)

    if ds.n_true_pairs == 0:
        return []

    t0 = time.perf_counter()
    emb = embedder.fit_transform(ds.texts)
    embed_time = time.perf_counter() - t0

    algos = build_algos(best_params, n_docs=n)
    rows = []
    for name, fn in algos.items():
        try:
            t0 = time.perf_counter()
            result = fn(emb)
            elapsed = time.perf_counter() - t0
            m = evaluate_pairs(result, ds.true_pairs)
            rows.append({
                "n_docs": n,
                "seed": seed,
                "algorithm": name,
                "total_time_s": round(elapsed, 3),
                "embed_time_s": round(embed_time, 3),
                "n_true_pairs": ds.n_true_pairs,
                **m.to_dict(),
            })
        except Exception as e:
            print(f"    {name}: FAILED ({e})")
            rows.append({"n_docs": n, "seed": seed, "algorithm": name, "error": str(e)})
    return rows


def run_size(
    n: int,
    best_params: dict,
    embedder: TextEmbedder,
    seeds: list[int] = SEEDS,
) -> list[dict]:

    illustrative = " [иллюстративно — мало пар]" if n in ILLUSTRATIVE_N else ""
    print(f"\n{'='*65}")
    print(f"N = {n:,}{illustrative}")
    print(f"{'='*65}")

    all_rows = []
    for seed in seeds:
        print(f"  seed={seed}...")
        ds = load_qqp(n_docs=n, seed=seed)
        print_stats(ds, label=f"  N={n} seed={seed}")

        if ds.n_true_pairs == 0:
            print("    WARNING: no true pairs — skipping seed.")
            continue

        t0 = time.perf_counter()
        emb = embedder.fit_transform(ds.texts)
        embed_time = time.perf_counter() - t0

        algos = build_algos(best_params, n_docs=n)
        for name, fn in algos.items():
            try:
                t0 = time.perf_counter()
                result = fn(emb)
                elapsed = time.perf_counter() - t0
                m = evaluate_pairs(result, ds.true_pairs)
                all_rows.append({
                    "n_docs": n,
                    "seed": seed,
                    "algorithm": name,
                    "total_time_s": round(elapsed, 3),
                    "embed_time_s": round(embed_time, 3),
                    "n_true_pairs": ds.n_true_pairs,
                    **m.to_dict(),
                })
                print(f"    {name:<20s} F1={m.f1:.3f}  "
                      f"P={m.precision:.3f}  R={m.recall:.3f}  "
                      f"singletons_lost={m.n_singletons_lost}")
            except Exception as e:
                print(f"    {name}: FAILED ({e})")
                all_rows.append({
                    "n_docs": n, "seed": seed,
                    "algorithm": name, "error": str(e),
                })

    return all_rows


#графики

def plot_results(df: pd.DataFrame):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        valid = df.dropna(subset=["f1"]).copy()
        algos = valid["algorithm"].unique()
        colors = {a: c for a, c in zip(algos, plt.cm.tab10(np.linspace(0, 1, len(algos))))}

        #метрики
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        for ax, metric, title in zip(
            axes, ["precision", "recall", "f1"],
            ["Precision", "Recall", "F1"]
        ):
            for algo in algos:
                sub = valid[valid["algorithm"] == algo].sort_values("n_docs")
                if sub.empty:
                    continue
                ax.plot(sub["n_docs"], sub[metric],
                        marker="o", label=algo,
                        color=colors[algo], linewidth=2, markersize=5)
            ax.set_xscale("log")
            ax.set_xlabel("N documents (log scale)")
            ax.set_ylabel(title)
            ax.set_title(f"{title} vs N")
            ax.grid(True, alpha=0.3)
        axes[2].legend(fontsize=9, loc="lower right")
        fig.suptitle("Quora QQP: scaling of metrics with corpus size", fontsize=13)
        fig.tight_layout()
        fig.savefig(FIGURES_DIR / "qqp_scaling_prf1.png", dpi=120, bbox_inches="tight")
        plt.close(fig)

        #лишние удаления
        fig, ax = plt.subplots(figsize=(10, 6))
        for algo in algos:
            sub = valid[valid["algorithm"] == algo].sort_values("n_docs")
            ax.plot(sub["n_docs"], sub["n_singletons_lost"],
                    marker="o", label=algo,
                    color=colors[algo], linewidth=2, markersize=5)
        ax.set_xscale("log")
        ax.set_xlabel("N documents (log scale)")
        ax.set_ylabel("Singletons incorrectly removed")
        ax.set_title("Unique documents incorrectly removed vs N")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(FIGURES_DIR / "qqp_singletons_lost.png", dpi=120, bbox_inches="tight")
        plt.close(fig)

        #по времени
        fig, ax = plt.subplots(figsize=(10, 6))
        for algo in algos:
            sub = valid[valid["algorithm"] == algo].sort_values("n_docs")
            ax.plot(sub["n_docs"], sub["total_time_s"],
                    marker="o", label=algo,
                    color=colors[algo], linewidth=2, markersize=5)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("N documents (log scale)")
        ax.set_ylabel("Total time (seconds, log scale)")
        ax.set_title("Runtime scaling")
        ax.legend(fontsize=9)
        ax.grid(True, which="both", alpha=0.3)
        fig.tight_layout()
        fig.savefig(FIGURES_DIR / "qqp_runtime.png", dpi=120, bbox_inches="tight")
        plt.close(fig)


        max_n = valid["n_docs"].max()
        final = valid[valid["n_docs"] == max_n].sort_values("f1")
        fig, ax = plt.subplots(figsize=(10, 5))
        bar_colors = [colors[a] for a in final["algorithm"]]
        bars = ax.barh(final["algorithm"], final["f1"],
                       color=bar_colors, edgecolor="black", linewidth=0.5)
        for bar, v in zip(bars, final["f1"]):
            ax.text(bar.get_width() + 0.005,
                    bar.get_y() + bar.get_height() / 2,
                    f"{v:.3f}", va="center", fontsize=10)
        ax.set_xlabel("F1")
        ax.set_xlim(0, min(1.0, final["f1"].max() * 1.15 + 0.05))
        ax.set_title(f"Final F1 comparison at N={max_n:,} (Quora QQP)")
        ax.grid(True, alpha=0.3, axis="x")
        fig.tight_layout()
        fig.savefig(FIGURES_DIR / "qqp_final_f1.png", dpi=120, bbox_inches="tight")
        plt.close(fig)

        print(f"\nFigures saved to {FIGURES_DIR}/")
        print("  qqp_scaling_prf1.png")
        print("  qqp_singletons_lost.png")
        print("  qqp_runtime.png")
        print("  qqp_final_f1.png")

    except Exception as e:
        print(f"Plotting failed: {e}")




def aggregate_seeds(df_raw: pd.DataFrame) -> pd.DataFrame:
 
    valid = df_raw.dropna(subset=["f1"])
    agg = valid.groupby(["n_docs", "algorithm"]).agg(
        f1_mean=("f1", "mean"),
        f1_std=("f1", "std"),
        precision_mean=("precision", "mean"),
        precision_std=("precision", "std"),
        recall_mean=("recall", "mean"),
        recall_std=("recall", "std"),
        singletons_mean=("n_singletons_lost", "mean"),
        singletons_std=("n_singletons_lost", "std"),
        n_removed_mean=("n_removed", "mean"),
        n_seeds=("seed", "count"),
    ).reset_index()
    return agg.round(4)


def print_summary(df_agg: pd.DataFrame):
    """Выводит сводную таблицу mean ± std."""
    print(f"\n{'='*70}")
    print("SUMMARY: F1 mean ± std by N (averaged over 3 seeds)")
    print(f"{'='*70}")
    print(f"  {'Algorithm':<20s}", end="")
    reliable_ns = [n for n in SIZES if n not in ILLUSTRATIVE_N]
    for n in reliable_ns:
        print(f"  {'N='+str(n):>12s}", end="")
    print()
    print("  " + "─" * (20 + 14 * len(reliable_ns)))

    algos = df_agg["algorithm"].unique()
    for algo in algos:
        sub = df_agg[df_agg["algorithm"] == algo].set_index("n_docs")
        print(f"  {algo:<20s}", end="")
        for n in reliable_ns:
            if n in sub.index:
                mean = sub.loc[n, "f1_mean"]
                std = sub.loc[n, "f1_std"]
                std_str = f"±{std:.3f}" if not np.isnan(std) else ""
                print(f"  {mean:.3f}{std_str:>6s}", end="")
            else:
                print(f"  {'—':>12s}", end="")
        print()

    if any(n in ILLUSTRATIVE_N for n in SIZES):
        print(f"\n  * N={sorted(ILLUSTRATIVE_N)} — иллюстративно "
              f"(мало пар, высокий шум, не использовать для выводов)")

    print(f"\n{'='*70}")
    print("SUMMARY: Singletons lost mean ± std")
    print(f"{'='*70}")
    for algo in algos:
        sub = df_agg[df_agg["algorithm"] == algo].set_index("n_docs")
        print(f"  {algo:<20s}", end="")
        for n in reliable_ns:
            if n in sub.index:
                mean = sub.loc[n, "singletons_mean"]
                std = sub.loc[n, "singletons_std"]
                std_str = f"±{std:.1f}" if not np.isnan(std) else ""
                print(f"  {mean:.1f}{std_str:>7s}", end="")
            else:
                print(f"  {'—':>12s}", end="")
        print()


def main():
    # 1) Лучшие параметры
    best_params = load_best_params()

    print("\nAlgorithms with best params:")
    for name in best_params:
        print(f"  {name:<20s}  {best_params[name]}")

    # 2) Embedder и прогрев кэша
    print("\nInitializing SBERT embedder...")
    embedder = TextEmbedder(SBERTConfig(
        model_name="all-MiniLM-L6-v2",
        show_progress_bar=False,
    ))
    _load_raw_qqp()

    # 3) Бенчмарк: каждый размер × 3 seeds
    all_rows = []
    for n in SIZES:
        rows = run_size(n, best_params, embedder, seeds=SEEDS)
        all_rows.extend(rows)

    # 4) Сохранение сырых данных (все seeds)
    df_raw = pd.DataFrame(all_rows)
    df_raw.to_csv(RESULTS_DIR / "qqp_benchmark_raw.csv", index=False)
    print(f"\nRaw results (all seeds) → {RESULTS_DIR}/qqp_benchmark_raw.csv")

    # 5) Агрегация
    df_agg = aggregate_seeds(df_raw)
    df_agg.to_csv(RESULTS_DIR / "qqp_benchmark_final.csv", index=False)
    print(f"Aggregated results → {RESULTS_DIR}/qqp_benchmark_final.csv")

    # 6) Сводка и графики
    print_summary(df_agg)
    plot_results(df_agg.rename(columns={
        "f1_mean": "f1",
        "precision_mean": "precision",
        "recall_mean": "recall",
        "singletons_mean": "n_singletons_lost",
        "n_removed_mean": "n_removed",
    }))


if __name__ == "__main__":
    main()