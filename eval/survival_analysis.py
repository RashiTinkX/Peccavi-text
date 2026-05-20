"""
eval/survival_analysis.py
Post-hoc survival-vs-threshold analysis from saved result JSONs.

Reads attack_z_scores saved by eval/watermark.py and computes survival at
multiple detection thresholds — converting "0% at z≥4.0" into a curve that
shows where each method actually breaks down.

Usage:
    python eval/survival_analysis.py results/peccavi_s42.json results/kgw_s42.json
    python eval/survival_analysis.py results/*.json          # all at once
"""

from __future__ import annotations
import json
import sys
from pathlib import Path
from typing import Dict, List

THRESHOLDS = [1.5, 2.0, 2.5, 3.0, 3.5, 4.0]


def _inner(data: dict) -> dict:
    """Unwrap nested JSON like {"peccavi": {...}} → inner dict."""
    keys = list(data.keys())
    if len(keys) == 1 and isinstance(data[keys[0]], dict):
        inner = data[keys[0]]
        if "watermark_mode" in inner or "attack_z_scores" in inner:
            return inner
    return data


def survival_curve(z_list: List[float]) -> Dict[str, float]:
    n = max(len(z_list), 1)
    return {f"z≥{t:.1f}": round(sum(1 for z in z_list if z >= t) / n, 4) for t in THRESHOLDS}


def analyse(path: str) -> None:
    p = Path(path)
    with open(p) as f:
        data = json.load(f)
    inner = _inner(data)

    mode = inner.get("watermark_mode", p.stem)
    seed = inner.get("seed", "?")
    attack_z_scores: Dict[str, List[float]] = inner.get("attack_z_scores", {})

    if not attack_z_scores:
        print(f"  {p.name}: no attack_z_scores — re-run with updated eval/watermark.py")
        return

    print(f"\n{'='*68}")
    print(f"  {p.name}  |  mode={mode}  seed={seed}")
    print(f"{'='*68}")
    header = f"  {'Attack':<22}" + "".join(f"  z≥{t:.1f}" for t in THRESHOLDS)
    print(header)
    print("  " + "-" * (len(header) - 2))

    for attack, z_list in attack_z_scores.items():
        curve = survival_curve(z_list)
        row = f"  {attack:<22}" + "".join(f"  {curve[f'z≥{t:.1f}']:.3f}" for t in THRESHOLDS)
        print(row)

    # Summary: mean survival across attacks at each threshold
    all_z = [z for zs in attack_z_scores.values() for z in zs]
    mean_curve = survival_curve(all_z)
    print("  " + "-" * (len(header) - 2))
    row = f"  {'MEAN (all attacks)':<22}" + "".join(f"  {mean_curve[f'z≥{t:.1f}']:.3f}" for t in THRESHOLDS)
    print(row)


def main():
    paths = sys.argv[1:]
    if not paths:
        print("Usage: python eval/survival_analysis.py <results.json> [...]")
        sys.exit(1)
    for path in paths:
        try:
            analyse(path)
        except Exception as e:
            print(f"  ERROR {path}: {e}")


if __name__ == "__main__":
    main()
