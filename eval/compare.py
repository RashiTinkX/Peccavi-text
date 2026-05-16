"""
eval/compare.py
Loads multiple benchmark result JSON files and prints a comparison table.
Usage:
    python eval/compare.py results/peccavi.json results/kgw.json results/ablation_fixed_theta.json
"""

from __future__ import annotations
import json
import sys
import os
from typing import Dict, List


METRICS = [
    ("auc_roc",               "AUC-ROC",         "{:.4f}", "≥0.90"),
    ("false_positive_rate",   "FPR",              "{:.4f}", "≤0.05"),
    ("effective_score_final", "S_eff",            "{:.4f}", ">0.50"),
    ("theta_final",           "θ_final",          "{:.4f}", "—"),
    ("improvement_pct",       "Improvement %",    "{:.1f}", "—"),
    ("avg_readability",       "Readability",      "{:.2f}", "≥3.0"),
]


def load_results(path: str) -> Dict:
    with open(path) as f:
        return json.load(f)


def flatten(results: Dict) -> Dict[str, Dict]:
    """Handle both single-model and multi-model result files."""
    flat = {}
    for key, val in results.items():
        if isinstance(val, dict) and "auc_roc" in val:
            flat[key] = val
        elif isinstance(val, dict) and not val.get("error"):
            flat[key] = val
    return flat


def print_table(all_results: Dict[str, Dict]):
    models = list(all_results.keys())
    col_w = max(20, max(len(m) for m in models) + 2)

    header = f"{'Metric':<22}" + "".join(f"{m:^{col_w}}" for m in models)
    print("\n" + "═" * len(header))
    print("  PECCAVI COMPARISON TABLE")
    print("═" * len(header))
    print(header)
    print("─" * len(header))

    for key, label, fmt, target in METRICS:
        row = f"{label + ' (' + target + ')':<22}"
        for model in models:
            val = all_results[model].get(key)
            if val is None:
                row += f"{'N/A':^{col_w}}"
            else:
                row += f"{fmt.format(val):^{col_w}}"
        print(row)

    print("═" * len(header))


def latex_table(all_results: Dict[str, Dict]) -> str:
    models = list(all_results.keys())
    col_spec = "l" + "r" * len(models)
    lines = [
        "\\begin{table}[h]",
        "\\centering",
        f"\\begin{{tabular}}{{{col_spec}}}",
        "\\toprule",
        "Metric & " + " & ".join(models) + " \\\\",
        "\\midrule",
    ]
    for key, label, fmt, target in METRICS:
        vals = []
        for model in models:
            val = all_results[model].get(key)
            vals.append(fmt.format(val) if val is not None else "—")
        lines.append(f"{label} & " + " & ".join(vals) + " \\\\")
    lines += [
        "\\bottomrule",
        "\\end{tabular}",
        "\\caption{PECCAVI vs baselines and ablations}",
        "\\label{tab:results}",
        "\\end{table}",
    ]
    return "\n".join(lines)


def main(paths: List[str]):
    all_results: Dict[str, Dict] = {}

    for path in paths:
        if not os.path.exists(path):
            print(f"  WARNING: {path} not found — skipping")
            continue
        raw = load_results(path)
        flat = flatten(raw)
        all_results.update(flat)

    if not all_results:
        print("No results found.")
        return

    print_table(all_results)

    print("\n  LaTeX table:\n")
    print(latex_table(all_results))

    out = "./results/comparison_table.json"
    os.makedirs("results", exist_ok=True)
    with open(out, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n  Saved → {out}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python eval/compare.py <result1.json> [result2.json ...]")
        sys.exit(1)
    main(sys.argv[1:])
