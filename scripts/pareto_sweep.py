"""
scripts/pareto_sweep.py
Generates Pareto curve data by sweeping delta values for KGW and SIR baselines,
then loading PECCAVI's trained result for comparison.

This supports the unified framework framing: KGW and SIR are special cases of PECCAVI
with frozen theta. The sweep shows their operating curves; PECCAVI's learned theta
finds a better point automatically.

Output: results/pareto_data.json
Run from repo root:
    python scripts/pareto_sweep.py
"""

from __future__ import annotations
import sys
import os
import json
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# Delta values to sweep — covers the full tradeoff range
DELTA_SWEEP = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]

# Smaller eval set for speed — this is for a figure, not the main results table
N_EVAL_SAMPLES = 100
N_PARAPHRASES = 5
GENERATIONS = 1   # KGW/SIR don't learn, so 1 generation is enough
SEED = 42


def _safe(val):
    """Return None if val is NaN, else val."""
    import math
    if val is None:
        return None
    try:
        return None if math.isnan(float(val)) else val
    except (TypeError, ValueError):
        return None


def sweep_method(backbone, watermark_mode: str, delta: float) -> dict:
    from eval.watermark import run_peccavi
    kwargs = dict(
        generations=GENERATIONS,
        n_paraphrases=N_PARAPHRASES,
        n_eval_samples=N_EVAL_SAMPLES,
        verbose=False,
        theta_init=2.0,
        watermark_mode=watermark_mode,
        seed=SEED,
    )
    if watermark_mode == "kgw":
        kwargs["kgw_delta"] = delta
        kwargs["kgw_gamma"] = 0.5
    elif watermark_mode == "sir":
        kwargs["sir_delta"] = delta
        kwargs["sir_gamma"] = 0.5
        kwargs["sir_entropy_threshold"] = 1.0

    out = run_peccavi(backbone, **kwargs)
    return {
        "delta": delta,
        "auc_roc": _safe(out.get("auc_roc")),
        "ppl_ratio": _safe(out.get("ppl_ratio")),
        "tpr_at_1fpr": _safe(out.get("tpr_at_1fpr")),
        "gpt4_survival": _safe(out.get("gpt4_survival")),
    }


def load_peccavi_point(path: str = "results/peccavi.json") -> list:
    if not os.path.exists(path):
        logger.warning(f"{path} not found — run main PECCAVI experiment first")
        return []
    with open(path) as f:
        raw = json.load(f)
    # Handle both {"peccavi": {...}} and flat {"auc_roc": ...} formats
    data = raw.get("peccavi", raw)
    return [{
        "theta": _safe(data.get("theta_final")),
        "auc_roc": _safe(data.get("auc_roc")),
        "ppl_ratio": _safe(data.get("ppl_ratio")),
        "tpr_at_1fpr": _safe(data.get("tpr_at_1fpr")),
        "gpt4_survival": _safe(data.get("gpt4_survival")),
    }]


def main():
    os.makedirs("results", exist_ok=True)

    from main import init_backbone
    logger.info("Initialising backbone for Pareto sweep...")
    backbone = init_backbone("configs/peccavi.yaml")

    pareto_data = {"kgw": [], "sir": [], "peccavi": []}

    for method in ("kgw", "sir"):
        for delta in DELTA_SWEEP:
            logger.info(f"Sweeping {method.upper()} delta={delta:.1f} ...")
            try:
                pt = sweep_method(backbone, method, delta)
                pareto_data[method].append(pt)
                logger.info(
                    f"  {method.upper()} δ={delta:.1f} → "
                    f"AUC={pt['auc_roc']}, PPL={pt['ppl_ratio']}, "
                    f"GPT-4 surv={pt['gpt4_survival']}"
                )
            except Exception as e:
                logger.warning(f"  {method.upper()} δ={delta:.1f} failed: {e}")

    pareto_data["peccavi"] = load_peccavi_point()

    out_path = "results/pareto_data.json"
    with open(out_path, "w") as f:
        json.dump(pareto_data, f, indent=2)
    logger.info(f"Pareto data saved → {out_path}")

    print("\n  Summary")
    print("  " + "─" * 56)
    for method, pts in pareto_data.items():
        for pt in pts:
            key_val = pt.get("delta") or pt.get("theta")
            label = f"δ={key_val:.2f}" if key_val is not None else "δ=?"
            auc = pt.get("auc_roc")
            ppl = pt.get("ppl_ratio")
            g4  = pt.get("gpt4_survival")
            auc_s = f"{auc:.4f}" if auc is not None else "N/A"
            ppl_s = f"{ppl:.4f}" if ppl is not None else "N/A"
            g4_s  = f"{g4:.4f}"  if g4  is not None else "N/A"
            print(
                f"  {method.upper():<8} {label:<10} "
                f"AUC={auc_s:<8} PPL={ppl_s:<8} GPT-4={g4_s}"
            )

    print(f"\n  Run  python eval/plot_pareto.py  to generate the figure.")


if __name__ == "__main__":
    main()
