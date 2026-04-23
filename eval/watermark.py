"""
eval/watermark.py
PECCAVI full evaluation loop.
Runs G generations, updating θ each round, and reports final metrics.
"""

from __future__ import annotations
from backbone.model import LLaMABackbone
from peccavi.praeco import Praeco
from peccavi.auctor import Auctor
from peccavi.scriba import Scriba
from peccavi.custos import Custos
from peccavi.magister import Magister
from typing import Dict
import logging

logger = logging.getLogger(__name__)


def run_peccavi(
    backbone: LLaMABackbone,
    generations: int = 10,
    n_paraphrases: int = 5,
    verbose: bool = True,
) -> Dict:
    praeco = Praeco()
    auctor = Auctor(backbone)
    scriba = Scriba(backbone, n_variants=n_paraphrases)
    custos = Custos(backbone)
    magister = Magister(backbone, theta_init=auctor.theta)

    history = []

    for gen in range(1, generations + 1):
        prompt = praeco.next_prompt()

        # 1. Generate watermarked text
        auctor.theta = magister.theta
        wm_text = auctor.generate(prompt, max_tokens=100)

        # 2. Adversarial paraphrasing
        paraphrases = scriba.paraphrase(wm_text)

        # 3. Detection
        s_eff = custos.effective_score(paraphrases)
        original_score = custos.watermark_score(wm_text)

        # 4. Policy update
        new_theta = magister.update(wm_text, s_eff)

        record = {
            "generation": gen,
            "theta": round(new_theta, 4),
            "original_score": round(original_score, 4),
            "effective_score": round(s_eff, 4),
        }
        history.append(record)

        if verbose:
            logger.info(
                f"Gen {gen:>3} | θ={new_theta:.4f} | "
                f"S_orig={original_score:.4f} | S_eff={s_eff:.4f}"
            )

    # Summary metrics
    first_eff = history[0]["effective_score"]
    last_eff = history[-1]["effective_score"]
    improvement = (last_eff - first_eff) / max(first_eff, 1e-6) * 100

    summary = {
        "theta_final": history[-1]["theta"],
        "effective_score_final": last_eff,
        "effective_score_improvement_pct": round(improvement, 2),
        "meets_85pct_retention": last_eff >= 0.85,
        "history": history,
    }
    return summary