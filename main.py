"""
main.py
AIISC AI Integrity & Safety Consortium
Master entry point for PECCAVI watermarking via a shared LLaMA backbone.

Usage:
    python main.py --mode eval
    python main.py --mode train
    python main.py --mode infer --prompt "Your prompt here"

"""

from __future__ import annotations
import argparse
import logging
import sys
import yaml
import os
from typing import Optional

# Logging setup
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


# Config loader
def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


# Backbone initialisation (shared singleton)
def init_backbone(config_path: str = "configs/peccavi.yaml"):
    from backbone.model import LLaMABackbone
    cfg = load_config(config_path)
    model_cfg = cfg.get("model", {})
    logger.info(f"Initialising backbone: {model_cfg.get('backbone')}")
    return LLaMABackbone(
        model_name=model_cfg.get("backbone", "TinyLlama/TinyLlama-1.1B-Chat-v1.0"),
        backend=model_cfg.get("backend", "transformers"),
        device=model_cfg.get("device", "auto"),
        load_in_4bit=model_cfg.get("load_in_4bit", True),
        api_key=model_cfg.get("api_key"),
    )


# Mode: EVALUATE
# Runs the full PECCAVI benchmark suite
def mode_eval(backbone, args):
    from eval.benchmarks import run_benchmarks
    logger.info("Starting PECCAVI benchmark evaluation...")
    output_path = getattr(args, "output", "./results/benchmark_results.json")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Check if running with baselines
    use_baselines = getattr(args, "baselines", False)
    if use_baselines:
        cfg = load_config("configs/peccavi.yaml")
        results = run_benchmarks(baseline_config=cfg, output_path=output_path, verbose=True)
    else:
        results = run_benchmarks(backbone, output_path=output_path, verbose=True)
    
    logger.info("Evaluation complete.")
    return results


# Mode: TRAIN
# Runs PECCAVI policy learning over simulated generations
def mode_train(backbone):
    _train_peccavi(backbone)


def _train_peccavi(backbone):
    from eval.watermark import run_peccavi
    cfg = load_config("configs/peccavi.yaml")
    pl_cfg = cfg.get("policy_learning", {})
    logger.info("Training PECCAVI watermarking policy...")
    summary = run_peccavi(
        backbone,
        generations=pl_cfg.get("generations", 10),
        n_paraphrases=cfg["agents"].get("scriba_n_variants", 5),
        verbose=True,
    )
    logger.info(
        f"PECCAVI training done. "
        f"θ_final={summary['theta_final']} | "
        f"S_eff={summary['effective_score_final']:.4f} | "
        f"Improvement={summary['effective_score_improvement_pct']:.1f}%"
    )


# Mode: INFER
# Single-prompt inference demonstrating the PECCAVI pipeline
def mode_infer(backbone, args):
    prompt = args.prompt

    print("\n" + "═" * 60)
    print(f"  PROMPT: {prompt}")
    print("═" * 60)

    # Standard backbone response
    std_out = backbone.generate(prompt, max_new_tokens=200)
    print("\n  [Backbone Response]")
    print(f"  {std_out['text']}\n")

    # PECCAVI watermarked response
    from peccavi.auctor import Auctor
    from peccavi.custos import Custos
    auctor = Auctor(backbone)
    custos = Custos(backbone)
    wm_text = auctor.generate(prompt, max_tokens=200)
    detection = custos.detect(wm_text)
    print("  [PECCAVI Watermarked Response]")
    print(f"  {wm_text}")
    print(f"\n  Watermark Score : {detection['score']} "
          f"({'detected' if detection['is_watermarked'] else 'not detected'})")


# CLI argument parser
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="PECCAVI - Watermarking and Content Authenticity",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    p.add_argument(
        "--mode", required=True,
        choices=["eval", "train", "infer"],
        help=(
            "eval  - run full PECCAVI benchmarks\n"
            "train - run policy learning over simulated generations\n"
            "infer - single-prompt watermarking demo"
        ),
    )
    p.add_argument(
        "--prompt",
        type=str,
        default="Explain the importance of AI safety in modern systems.",
        help="Prompt string for --mode infer",
    )
    p.add_argument(
        "--output",
        type=str,
        default="./results/benchmark_results.json",
        help="Output path for benchmark results JSON",
    )
    p.add_argument(
        "--baselines",
        action="store_true",
        help="Run evaluation against all baseline models (GPT-4, Claude-3, etc.) defined in config",
    )
    p.add_argument(
        "--config-dir",
        type=str,
        default="configs",
        help="Directory containing YAML config files",
    )
    return p


# Entry point
def main():
    parser = build_parser()
    args = parser.parse_args()

    print(" PECCAVI-TEXT")

    logger.info(f"Mode: {args.mode.upper()}")

    backbone = init_backbone(
        config_path=os.path.join(args.config_dir, "peccavi.yaml")
    )

    if args.mode == "eval":
        mode_eval(backbone, args)

    elif args.mode == "train":
        mode_train(backbone)

    elif args.mode == "infer":
        mode_infer(backbone, args)

    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
