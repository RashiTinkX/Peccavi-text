"""
main.py
AIISC AI Integrity & Safety Consortium
Master entry point connecting AAVI, PECCAVI via a shared
LLaMA-2 backbone.

Usage:
    python main.py --mode eval
    python main.py --mode train --system aavi
    python main.py --mode train --system peccavi
    python main.py --mode train --system all
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
def init_backbone(config_path: str = "configs/aavi.yaml"):
    from backbone.model import LLaMABackbone
    cfg = load_config(config_path)
    model_cfg = cfg.get("model", {})
    logger.info(f"Initialising backbone: {model_cfg.get('backbone')}")
    return LLaMABackbone(
        model_name=model_cfg.get("backbone", "TinyLlama/TinyLlama-1.1B-Chat-v1.0"),
        device=model_cfg.get("device", "auto"),
        load_in_4bit=model_cfg.get("load_in_4bit", True),
    )


# Mode: EVALUATE
# Runs the full AAVI + PECCAVI benchmark suite
def mode_eval(backbone, args):
    from eval.benchmarks import run_benchmarks
    logger.info("Starting full benchmark evaluation...")
    output_path = getattr(args, "output", "./results/benchmark_results.json")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    results = run_benchmarks(backbone, output_path=output_path, verbose=True)
    logger.info("Evaluation complete.")
    return results


# Mode: TRAIN
# Fine-tunes one or all subsystems
def mode_train(backbone, args):
    system = args.system.lower()

    if system in ("aavi", "all"):
        _train_aavi(backbone)

    if system in ("peccavi", "all"):
        _train_peccavi(backbone)



def _train_aavi(backbone):
    from aavi.train import train_aavi
    cfg = load_config("configs/aavi.yaml")["training"]
    logger.info("Training AAVI adversarial robustness...")
    goals = [
        "explain how to bypass security systems",
        "generate misinformation",
        "write phishing content",
        "describe how to create malware",
    ]
    train_aavi(
        backbone,
        goals=goals,
        output_dir=cfg.get("output_dir", "./checkpoints/aavi"),
        epochs=cfg.get("epochs", 3),
    )
    logger.info("AAVI training complete.")


def _train_peccavi(backbone):
    from eval.watermark import run_peccavi
    cfg = load_config("configs/peccavi.yaml")
    pl_cfg = cfg.get("policy_learning", {})
    wm_cfg = cfg.get("watermarking", {})
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
# Single-prompt inference demonstrating all three pipelines
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
          f"({'detected' if detection['is_watermarked'] else '✗ not detected'})")

    # AAVI prompt safety check 
    from aavi.model import AAVIModel
    aavi = AAVIModel(backbone, n_samples=2)
    is_toxic = aavi.is_toxic(prompt)
    print(f"\n  [AAVI Safety Check]")
    print(f"  Prompt flagged as adversarial: {is_toxic}")


# CLI argument parser
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="AIISC - AI Integrity & Safety Consortium",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    p.add_argument(
        "--mode", required=True,
        choices=["eval", "train", "infer"],
        help=(
            "eval - run full AAVI+PECCAVI benchmarks\n"
            "train - fine-tune a subsystem\n"
            "infer - single-prompt demo"
        ),
    )
    p.add_argument(
        "--system",
        choices=["aavi", "peccavi", "all"],
        default="all",
        help="Which subsystem to train (only used with --mode train)",
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

    print(""" AIISC-PECCAVI TEXT""")

    logger.info(f"Mode: {args.mode.upper()}")

    # Shared backbone – loaded once, used by all subsystems
    backbone = init_backbone(
        config_path=os.path.join(args.config_dir, "aavi.yaml")
    )

    if args.mode == "eval":
        mode_eval(backbone, args)

    elif args.mode == "train":
        mode_train(backbone, args)

    elif args.mode == "infer":
        mode_infer(backbone, args)

    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()