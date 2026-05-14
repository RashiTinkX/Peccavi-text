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
from sklearn.metrics import roc_auc_score
from typing import Dict
import textstat
import logging

logger = logging.getLogger(__name__)


def readability_score(text: str) -> float:
    """Maps Flesch Reading Ease (0 - 100) to a 1-5 scale."""
    fre = textstat.flesch_reading_ease(text)
    return round(1 + (fre / 100) * 4, 2)


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

    # Main generation loop
    for gen in range(1, generations + 1):          # ← restored
        prompt = praeco.next_prompt()

        auctor.theta = magister.theta
        wm_text = auctor.generate(prompt, max_tokens=100)

        paraphrases = scriba.paraphrase(wm_text)
        s_eff = custos.effective_score(paraphrases)
        
        # Update with reference text (original prompt) for semantic fidelity
        new_theta = magister.update(wm_text, s_eff, reference_text=prompt)
        original_score = custos.watermark_score(wm_text)

        record = {
            "generation": gen,
            "theta": round(new_theta, 4),
            "original_score": round(original_score, 4),
            "effective_score": round(s_eff, 4),
            "readability": readability_score(wm_text),
        }
        history.append(record)

        if verbose:                                 # ← fixed indentation
            logger.info(
                f"Gen {gen:>3} | θ={new_theta:.4f} | "
                f"S_orig={original_score:.4f} | S_eff={s_eff:.4f}"
            )

    # AUC-ROC and False Positive Rate
    logger.info("Computing AUC-ROC and false positive rate")

    eval_prompts = praeco.batch_prompts(20)

    human_texts = [
        backbone.generate(p, max_new_tokens=100)['text']
        for p in eval_prompts
    ]
    wm_texts_eval = [
        auctor.generate(p, max_tokens=100)
        for p in eval_prompts
    ]

    labels = [0] * 20 + [1] * 20
    scores = [custos.watermark_score(t) for t in human_texts + wm_texts_eval]
    auc = roc_auc_score(labels, scores)

    threshold = 0.52
    fp = sum(1 for t in human_texts if custos.watermark_score(t) >= threshold)
    fpr = fp / len(human_texts)

    # Final summary
    first_eff = history[0]["effective_score"]
    last_eff = history[-1]["effective_score"]
    improvement = (last_eff - first_eff) / max(first_eff, 1e-6) * 100

    avg_readability = round(
        sum(r["readability"] for r in history) / len(history), 2
    )

    summary = {
        "theta_final": history[-1]["theta"],
        "effective_score_final": last_eff,
        "effective_score_improvement_pct": round(improvement, 2),
        "meets_85pct_retention": last_eff >= 0.85,
        "auc_roc": round(auc, 4),
        "false_positive_rate": round(fpr, 4),
        "meets_90pct_auc": auc >= 0.90,
        "avg_readability": avg_readability,
        "meets_readability_45": avg_readability >= 4.5,
        "history": history,
    }
    return summary