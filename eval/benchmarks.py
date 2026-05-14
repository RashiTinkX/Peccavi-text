"""
eval/benchmarks.py
Runs the PECCAVI benchmark suite and produces a unified report.
Prints a summary table and saves results to JSON.
"""

from __future__ import annotations
from backbone.model import LLaMABackbone
from eval.watermark import run_peccavi
from typing import Dict
import json
import os
import logging

logger = logging.getLogger(__name__)


def _peccavi_summary(pec_out: Dict) -> Dict:
    robustness = pec_out["effective_score_final"] * 100
    reliability = pec_out["effective_score_final"] * 100
    return {
        "theta_final": pec_out["theta_final"],
        "effective_score_final": pec_out["effective_score_final"],
        "improvement_pct": pec_out["effective_score_improvement_pct"],
        "auc_roc": pec_out["auc_roc"],
        "false_positive_rate": pec_out["false_positive_rate"],
        "avg_readability": pec_out["avg_readability"],
        "pass_retention": pec_out["meets_85pct_retention"],
        "pass_auc": pec_out["meets_90pct_auc"],
        "pass_readability": pec_out["meets_readability_45"],
        "robustness": round(robustness, 2),
        "resilience": round(reliability, 2),
        "fpr": round(pec_out["false_positive_rate"] * 100, 2),
        "readability": pec_out["avg_readability"],
    }


def run_benchmarks(
    backbone: LLaMABackbone = None,
    output_path: str = "./benchmark_results.json",
    verbose: bool = True,
    baseline_config: Dict = None,
) -> Dict:
    """
    Run PECCAVI benchmarks. If baseline_config provided, run multiple baseline models.
    """
    report: Dict = {}

    print("\n" + "═" * 60)
    print("  BENCHMARK: PECCAVI - Watermarking & Content Authenticity")
    print("═" * 60)
    
    # If baseline_config provided, run multiple models
    if baseline_config:
        baselines = baseline_config.get("baseline_models", [])
        for baseline in baselines:
            model_name = baseline.get("name")
            model_id = baseline.get("backbone")
            backend = baseline.get("backend", "transformers")
            
            print(f"\n  Running baseline: {model_name}...")
            try:
                # Create backbone for this baseline
                model_backbone = LLaMABackbone(
                    model_name=model_id,
                    backend=backend,
                )
                pec_out = run_peccavi(model_backbone, generations=5, verbose=verbose)
                report[model_name] = _peccavi_summary(pec_out)
            except Exception as e:
                logger.error(f"Error running baseline {model_name}: {e}")
                report[model_name] = {"error": str(e)}
    else:
        # Single backbone mode
        if backbone is None:
            raise ValueError("Either backbone or baseline_config must be provided")
        pec_out = run_peccavi(backbone, generations=5, verbose=verbose)
        report["peccavi"] = _peccavi_summary(pec_out)

    _print_summary(report)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n  Results saved → {output_path}")

    return report


def _print_summary(report: Dict):
    print("\n" + "═" * 60)
    print("  PECCAVI BENCHMARK SUMMARY")
    print("═" * 60)

    for model_name, results in report.items():
        if isinstance(results, dict) and "error" in results:
            print(f"\n  {model_name}: ERROR - {results['error']}")
            continue
        
        print(f"\n  Model: {model_name}")
        print(f"  θ_final           : {results['theta_final']}")
        print(f"  Effective Score   : {results['effective_score_final']:.4f}  "
              f"({'PASS' if results['pass_retention'] else 'FAIL'} ≥0.85)")
        print(f"  Improvement       : {results['improvement_pct']:.1f}%")
        print(f"  AUC-ROC           : {results['auc_roc']:.4f}  "
              f"({'PASS' if results['pass_auc'] else 'FAIL'} ≥0.90)")
        print(f"  False Positive    : {results['false_positive_rate']:.4f}")
        print(f"  Avg Readability   : {results['avg_readability']:.2f}/5  "
              f"({'PASS' if results['pass_readability'] else 'FAIL'} ≥4.5)")
