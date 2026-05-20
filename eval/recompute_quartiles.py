"""
eval/recompute_quartiles.py
Recomputes theta_by_quartile from saved theta_by_prompt data in result JSONs.
Run after fixing the >= bug without needing to re-run experiments.

Usage:
    python eval/recompute_quartiles.py results/peccavi_seed42.json [...]
    python eval/recompute_quartiles.py results/*.json
"""

from __future__ import annotations
import json
import sys
import numpy as np
from pathlib import Path


def recompute_quartile(theta_by_prompt: list) -> dict:
    if not theta_by_prompt:
        return {}
    _ents = np.array([d["entropy"] for d in theta_by_prompt])
    _thts = np.array([d["theta_context"] for d in theta_by_prompt])
    q25, q50, q75 = np.percentile(_ents, [25, 50, 75])
    _q1 = _thts[_ents <= q25]
    _q2 = _thts[(_ents > q25) & (_ents < q50)]
    _q3 = _thts[(_ents >= q50) & (_ents < q75)]
    _q4 = _thts[_ents >= q75]          # fixed: >= not >

    def _safe_mean(arr):
        return round(float(np.mean(arr)), 4) if len(arr) > 0 else None

    spread = (
        round(float(np.mean(_q4) - np.mean(_q1)), 4)
        if (len(_q1) > 0 and len(_q4) > 0) else None
    )
    return {
        "Q1_factual":  _safe_mean(_q1),
        "Q2":          _safe_mean(_q2),
        "Q3":          _safe_mean(_q3),
        "Q4_creative": _safe_mean(_q4),
        "spread":      spread,
    }


def process(path: str):
    p = Path(path)
    with open(p) as f:
        data = json.load(f)

    # Results are stored nested under an experiment-name key (e.g. {"peccavi": {...}})
    keys = list(data.keys())
    if len(keys) == 1 and isinstance(data[keys[0]], dict) and "watermark_mode" in data[keys[0]]:
        exp_key = keys[0]
        inner = data[exp_key]
    else:
        exp_key = None
        inner = data

    theta_by_prompt = inner.get("theta_by_prompt", [])
    if not theta_by_prompt:
        print(f"  {p.name}: no theta_by_prompt data — skipping")
        return

    old = inner.get("theta_by_quartile", {})
    new = recompute_quartile(theta_by_prompt)
    inner["theta_by_quartile"] = new

    with open(p, "w") as f:
        json.dump(data, f, indent=2)

    print(f"  {p.name}: Q1={new.get('Q1_factual')} Q4={new.get('Q4_creative')} spread={new.get('spread')}  (was Q4={old.get('Q4_creative')})")


def main():
    files = sys.argv[1:]
    if not files:
        print("Usage: python eval/recompute_quartiles.py results/*.json")
        sys.exit(1)
    for f in files:
        process(f)


if __name__ == "__main__":
    main()
