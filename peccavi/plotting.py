"""
peccavi/plotting.py
Chart helpers for the PECCAVI evaluation dashboard.
"""

from __future__ import annotations
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def bar_chart(all_models: dict):
    models = list(all_models.keys())
    groups = {
        "Robustness (%)":     [m["robustness"]       for m in all_models.values()],
        "Resilience (%)":     [m["resilience"]        for m in all_models.values()],
        "AUC-ROC (×100)":    [m["auc"] * 100         for m in all_models.values()],
        "Readability (×20)":  [m["readability"] * 20  for m in all_models.values()],
    }
    thresholds = [85.0, None, 90.0, 90.0]
    colors = ["#1976D2", "#388E3C", "#F57C00", "#7B1FA2"]

    x, w = np.arange(len(models)), 0.19
    fig, ax = plt.subplots(figsize=(14, 6))
    fig.patch.set_facecolor("#F9F9F9")
    ax.set_facecolor("#F9F9F9")

    for i, (label, vals, color, thresh) in enumerate(
        zip(groups.keys(), groups.values(), colors, thresholds)
    ):
        offsets = x + (i - 1.5) * w
        bars = ax.bar(offsets, vals, w, label=label, color=color, alpha=0.85, zorder=3)
        bars[0].set_edgecolor("#111111")
        bars[0].set_linewidth(2.0)
        if thresh is not None:
            ax.axhline(thresh, color=color, ls="--", lw=0.9, alpha=0.45, zorder=2)

    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=16, ha="right", fontsize=9)
    ax.set_ylabel("Score", fontsize=10)
    ax.set_ylim(0, 118)
    ax.set_title(
        "PECCAVI vs Baseline Models — Evaluation Metrics\n"
        "(PECCAVI bars outlined in black  |  dashed lines = success targets)",
        fontsize=12, fontweight="bold", pad=10,
    )
    ax.legend(fontsize=8, loc="upper right", framealpha=0.85)
    ax.grid(axis="y", alpha=0.25, zorder=0)
    ax.annotate("◀ PECCAVI\n   (live)", xy=(0, 109), fontsize=7.5,
                color="#333", style="italic", ha="center")
    ax.text(len(models) - 0.55, 85.8, "85% target", fontsize=7, color="#1976D2", alpha=0.75)
    ax.text(len(models) - 0.55, 90.8, "90% target", fontsize=7, color="#F57C00", alpha=0.75)

    plt.tight_layout()
    return fig


def radar_chart(all_models: dict):
    categories = ["Robustness", "Resilience", "AUC-ROC", "Readability", "Low FPR"]
    N = len(categories)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
    fig.patch.set_facecolor("#F9F9F9")
    ax.set_facecolor("#F9F9F9")

    palette = ["#E53935", "#43A047", "#1E88E5", "#FB8C00", "#8E24AA"]

    for (name, m), color in zip(all_models.items(), palette):
        vals = [
            m["robustness"] / 100.0,
            m["resilience"] / 100.0,
            m["auc"],
            m["readability"] / 5.0,
            max(0.0, 1.0 - m["fpr"] / 15.0),
        ]
        vals += vals[:1]
        lw = 2.8 if name == "PECCAVI (Ours)" else 1.4
        ls = "-" if name == "PECCAVI (Ours)" else "--"
        ax.plot(angles, vals, "o-", lw=lw, ls=ls, color=color,
                label=name, markersize=4, zorder=3)
        ax.fill(angles, vals, alpha=0.07, color=color)

    crit = [0.85, 0.85, 0.90, 0.90, 1.0 - 5.0 / 15.0]
    crit += crit[:1]
    ax.plot(angles, crit, "k--", lw=1.1, alpha=0.35, label="Success Criteria", zorder=2)
    ax.fill(angles, crit, alpha=0.04, color="black")

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=10, fontweight="bold")
    ax.set_ylim(0, 1)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["25%", "50%", "75%", "100%"], fontsize=7, color="gray")
    ax.set_title(
        "Overall Performance Radar\n(normalised — higher is better on all axes)",
        fontsize=10, fontweight="bold", pad=22,
    )
    ax.legend(loc="upper right", bbox_to_anchor=(1.42, 1.18), fontsize=8, framealpha=0.85)
    ax.grid(alpha=0.2)

    plt.tight_layout()
    return fig
