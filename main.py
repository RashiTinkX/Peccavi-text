"""
main.py
AIISC AI Integrity & Safety Consortium
Master entry point for PECCAVI watermarking via a shared LLaMA backbone.

Usage:
    python main.py --mode eval
    python main.py --mode train
    python main.py --mode kgw
    python main.py --mode infer --prompt "Your prompt here"
    python main.py --mode eval --seed 123 --output results/run_seed123.json
"""

from __future__ import annotations
import argparse
import logging
import sys
import yaml
import os
import json
import random
import numpy as np
import torch
from typing import Optional

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("aiisc.log"),
    ],
)
logger = logging.getLogger("AIISC.main")


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    logger.info(f"Seed set to {seed}")


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def init_backbone(config_path: str = "configs/peccavi.yaml"):
    from backbone.model import LLaMABackbone
    cfg = load_config(config_path)
    model_cfg = cfg.get("model", {})
    logger.info(f"Initialising backbone: {model_cfg.get('backbone')}")
    return LLaMABackbone(
        model_name=model_cfg.get("backbone", "meta-llama/Llama-2-7b-chat-hf"),
        backend=model_cfg.get("backend", "transformers"),
        device=model_cfg.get("device", "auto"),
        load_in_4bit=model_cfg.get("load_in_4bit", True),
        api_key=model_cfg.get("api_key"),
    )


THETA_CHECKPOINT = "./results/theta_checkpoint.json"


def _load_theta() -> float:
    if os.path.exists(THETA_CHECKPOINT):
        with open(THETA_CHECKPOINT) as f:
            theta = json.load(f).get("theta", 2.0)
        logger.info(f"Loaded θ={theta:.4f} from checkpoint")
        return theta
    return 2.0


def mode_eval(backbone, args):
    from eval.benchmarks import run_benchmarks
    logger.info("Starting PECCAVI benchmark evaluation...")
    output_path = getattr(args, "output", "./results/benchmark_results.json")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    use_baselines = getattr(args, "baselines", False)
    if use_baselines:
        cfg = load_config(os.path.join(args.config_dir, "peccavi.yaml"))
        results = run_benchmarks(baseline_config=cfg, output_path=output_path, verbose=True)
    else:
        results = run_benchmarks(backbone, output_path=output_path, verbose=True)

    logger.info("Evaluation complete.")
    return results


def mode_kgw(backbone, args):
    """Run KGW baseline evaluation."""
    from eval.watermark import run_peccavi
    cfg = load_config(os.path.join(args.config_dir, "kgw_baseline.yaml"))
    wm_cfg = cfg.get("watermarking", {})
    pl_cfg = cfg.get("policy_learning", {})
    seed = getattr(args, "seed", 42)

    logger.info("Running KGW baseline evaluation...")
    summary = run_peccavi(
        backbone,
        generations=pl_cfg.get("generations", 5),
        n_paraphrases=cfg.get("agents", {}).get("scriba_n_variants", 10),
        n_eval_samples=pl_cfg.get("n_eval_samples", 100),
        verbose=True,
        theta_init=_load_theta(),
        watermark_mode="kgw",
        kgw_delta=wm_cfg.get("delta", 2.0),
        kgw_gamma=wm_cfg.get("gamma", 0.5),
        seed=seed,
    )

    output_path = getattr(args, "output", "./results/kgw_baseline.json")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        report = {"kgw_baseline": {k: v for k, v in summary.items() if k != "detailed_records"}}
        json.dump(report, f, indent=2)

    logger.info(
        f"KGW done. AUC-ROC={summary['auc_roc']:.4f} | "
        f"FPR={summary['false_positive_rate']:.4f} | "
        f"S_eff={summary['effective_score_final']:.4f}"
    )


def mode_train(backbone, args):
    _train_peccavi(backbone, args)


def _train_peccavi(backbone, args=None):
    from eval.watermark import run_peccavi
    cfg_dir = getattr(args, "config_dir", "configs") if args else "configs"
    cfg = load_config(os.path.join(cfg_dir, "peccavi.yaml"))
    pl_cfg = cfg.get("policy_learning", {})
    wm_cfg = cfg.get("watermarking", {})
    seed = getattr(args, "seed", 42) if args else 42

    logger.info("Training PECCAVI watermarking policy...")
    summary = run_peccavi(
        backbone,
        generations=pl_cfg.get("generations", 10),
        n_paraphrases=cfg.get("agents", {}).get("scriba_n_variants", 5),
        n_eval_samples=pl_cfg.get("n_eval_samples", 100),
        verbose=True,
        theta_init=wm_cfg.get("theta_init", 2.0),
        lam=pl_cfg.get("lambda_wm", 0.6),
        nu=pl_cfg.get("nu_quality", 0.4),
        alpha=pl_cfg.get("alpha", 0.05),
        seed=seed,
    )

    theta_final = summary["theta_final"]
    os.makedirs(os.path.dirname(THETA_CHECKPOINT), exist_ok=True)
    with open(THETA_CHECKPOINT, "w") as f:
        json.dump({"theta": theta_final}, f)

    logger.info(
        f"PECCAVI training done. "
        f"θ_final={theta_final} | "
        f"S_eff={summary['effective_score_final']:.4f} | "
        f"Improvement={summary['effective_score_improvement_pct']:.1f}% | "
        f"θ saved → {THETA_CHECKPOINT}"
    )


def mode_infer(backbone, args):
    prompt = args.prompt
    print("\n" + "═" * 60)
    print(f"  PROMPT: {prompt}")
    print("═" * 60)

    std_out = backbone.generate(prompt, max_new_tokens=200)
    print("\n  [Backbone Response]")
    print(f"  {std_out['text']}\n")

    from peccavi.auctor import Auctor
    from peccavi.custos import Custos
    auctor = Auctor(backbone, theta=_load_theta())
    custos = Custos(backbone)
    wm_text = auctor.generate(prompt, max_tokens=200)
    detection = custos.detect(wm_text)
    print("  [PECCAVI Watermarked Response]")
    print(f"  {wm_text}")
    print(f"\n  Z-score         : {detection['z_score']} "
          f"({'detected' if detection['is_watermarked'] else 'not detected'})")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="PECCAVI - Watermarking and Content Authenticity",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    p.add_argument(
        "--mode", required=True,
        choices=["eval", "train", "kgw", "infer"],
        help=(
            "eval  - run full PECCAVI benchmarks\n"
            "train - run policy learning over simulated generations\n"
            "kgw   - run KGW baseline evaluation\n"
            "infer - single-prompt watermarking demo"
        ),
    )
    p.add_argument("--prompt", type=str,
                   default="Explain the importance of AI safety in modern systems.")
    p.add_argument("--output", type=str, default="./results/benchmark_results.json")
    p.add_argument("--baselines", action="store_true",
                   help="Run against all baseline models defined in config")
    p.add_argument("--config-dir", type=str, default="configs")
    p.add_argument("--seed", type=int, default=42,
                   help="Random seed for reproducibility")
    return p


def main():
    parser = build_parser()
    args = parser.parse_args()

    print(" PECCAVI-TEXT")
    logger.info(f"Mode: {args.mode.upper()}")

    set_seed(args.seed)

    config_file = "kgw_baseline.yaml" if args.mode == "kgw" else "peccavi.yaml"
    backbone = init_backbone(
        config_path=os.path.join(args.config_dir, config_file)
    )

    if args.mode == "eval":
        mode_eval(backbone, args)
    elif args.mode == "train":
        mode_train(backbone, args)
    elif args.mode == "kgw":
        mode_kgw(backbone, args)
    elif args.mode == "infer":
        mode_infer(backbone, args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
