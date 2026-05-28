
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

FIG_DIR = Path("figures")
FIG_DIR.mkdir(exist_ok=True)

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "legend.fontsize": 9,
    "figure.dpi": 150,
})


ALGO_ORDER = [
    "SemDeDup", "DAT", "kNN-LT", "Percentile",
    "Mean+Std", "OAT", "Two-Pass OAT",
]
COLORS = {
    "SemDeDup":     "#3266ad",
    "DAT":          "#1d9e75",
    "kNN-LT":       "#8e44ad",
    "Percentile":   "#e07b27",
    "Mean+Std":     "#c0392b",
    "OAT":          "#a6761d",
    "Two-Pass OAT": "#2c9c3f",
}

RELIABLE_N = [500, 1000, 2000, 5000, 10000]


def load_data():
    df_final = pd.read_csv("qqp_benchmark_final.csv")
    df_raw = pd.read_csv("qqp_benchmark_raw.csv")
    df_grid = pd.read_csv("grid_search_qqp.csv")
    return df_final, df_raw, df_grid




def fig_f1_scaling(df_final):
    fig, ax = plt.subplots(figsize=(9, 5.5))
    d = df_final[df_final["n_docs"].isin(RELIABLE_N)]

    for algo in ALGO_ORDER:
        sub = d[d["algorithm"] == algo].sort_values("n_docs")
        if sub.empty:
            continue
        ax.errorbar(
            sub["n_docs"], sub["f1_mean"], yerr=sub["f1_std"],
            marker="o", markersize=6, linewidth=2, capsize=4,
            color=COLORS[algo], label=algo,
        )

    ax.set_xscale("log")
    ax.set_xticks(RELIABLE_N)
    ax.set_xticklabels([str(n) for n in RELIABLE_N])
    ax.set_xlabel("Размер корпуса N (логарифмическая шкала)")
    ax.set_ylabel("F1-мера")
    ax.set_title("Зависимость F1-меры от размера корпуса (среднее по 3 запускам)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", ncol=2)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_3_1_f1_scaling.png", bbox_inches="tight")
    plt.close(fig)
    print("fig_3_1_f1_scaling.png")




def fig_pr_scatter(df_grid):
    fig, ax = plt.subplots(figsize=(8, 7))

    for algo in ALGO_ORDER:
        sub = df_grid[df_grid["method"] == algo]
        if sub.empty:
            continue
        ax.scatter(
            sub["recall"], sub["precision"],
            s=55, alpha=0.7, color=COLORS[algo], label=algo,
            edgecolors="black", linewidths=0.5,
        )


    rr, pp = np.meshgrid(np.linspace(0.3, 1.0, 200),
                         np.linspace(0.5, 1.0, 200))
    f1 = 2 * pp * rr / (pp + rr + 1e-9)
    cs = ax.contour(rr, pp, f1, levels=[0.5, 0.6, 0.7, 0.8, 0.9],
                    colors="gray", alpha=0.5, linewidths=0.8)
    ax.clabel(cs, inline=True, fontsize=8, fmt="F1=%.1f")

    ax.set_xlabel("Полнота (Recall)")
    ax.set_ylabel("Точность (Precision)")
    ax.set_title("Пространство «точность–полнота» для всех конфигураций\n"
                 "подбора гиперпараметров")
    ax.legend(loc="lower left", ncol=2)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_3_2_pr_scatter.png", bbox_inches="tight")
    plt.close(fig)
    print("fig_3_2_pr_scatter.png")




def fig_final_f1(df_final):
    d = df_final[df_final["n_docs"] == 10000].copy()
    d = d.set_index("algorithm").reindex(ALGO_ORDER).reset_index()
    d = d.sort_values("f1_mean")

    fig, ax = plt.subplots(figsize=(9, 5))
    colors = [COLORS[a] for a in d["algorithm"]]
    bars = ax.barh(d["algorithm"], d["f1_mean"],
                   xerr=d["f1_std"], capsize=4,
                   color=colors, edgecolor="black", linewidth=0.6)
    for bar, v, s in zip(bars, d["f1_mean"], d["f1_std"]):
        ax.text(v + s + 0.008, bar.get_y() + bar.get_height() / 2,
                f"{v:.3f}", va="center", fontsize=10)

    ax.set_xlabel("F1-мера")
    ax.set_xlim(0, max(d["f1_mean"]) * 1.2)
    ax.set_title("Итоговое сравнение методов при N = 10 000")
    ax.grid(True, alpha=0.3, axis="x")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_3_3_final_f1.png", bbox_inches="tight")
    plt.close(fig)
    print("fig_3_3_final_f1.png")




def fig_singletons(df_final):
    fig, ax = plt.subplots(figsize=(9, 5.5))
    d = df_final[df_final["n_docs"].isin(RELIABLE_N)]

    for algo in ALGO_ORDER:
        sub = d[d["algorithm"] == algo].sort_values("n_docs")
        if sub.empty:
            continue
        ax.errorbar(
            sub["n_docs"], sub["singletons_mean"], yerr=sub["singletons_std"],
            marker="s", markersize=6, linewidth=2, capsize=4,
            color=COLORS[algo], label=algo,
        )

    ax.set_xscale("log")
    ax.set_xticks(RELIABLE_N)
    ax.set_xticklabels([str(n) for n in RELIABLE_N])
    ax.set_xlabel("Размер корпуса N (логарифмическая шкала)")
    ax.set_ylabel("Число ложно удалённых уникальных документов")
    ax.set_title("Потери уникальных документов в зависимости от размера корпуса")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left", ncol=2)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_3_4_singletons.png", bbox_inches="tight")
    plt.close(fig)
    print("fig_3_4_singletons.png")




def fig_runtime(df_raw):
    fig, ax = plt.subplots(figsize=(9, 5.5))
    timing = df_raw.groupby(["algorithm", "n_docs"])["total_time_s"].mean().reset_index()

    for algo in ALGO_ORDER:
        sub = timing[timing["algorithm"] == algo].sort_values("n_docs")
        if sub.empty:
            continue
        ax.plot(sub["n_docs"], sub["total_time_s"],
                marker="o", markersize=5, linewidth=2,
                color=COLORS[algo], label=algo)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Размер корпуса N (логарифмическая шкала)")
    ax.set_ylabel("Время работы, с (логарифмическая шкала)")
    ax.set_title("Время работы алгоритмов в зависимости от размера корпуса")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="upper left", ncol=2)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_3_5_runtime.png", bbox_inches="tight")
    plt.close(fig)
    print("fig_3_5_runtime.png")


def main():
    df_final, df_raw, df_grid = load_data()
    print("Building figures...")
    fig_f1_scaling(df_final)
    fig_pr_scatter(df_grid)
    fig_final_f1(df_final)
    fig_singletons(df_final)
    fig_runtime(df_raw)
    print(f"\nAll figures saved to {FIG_DIR}/")


if __name__ == "__main__":
    main()
