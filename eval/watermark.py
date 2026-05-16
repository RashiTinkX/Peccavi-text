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
from typing import Dict, List
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
    n_eval_samples: int = 100,
    verbose: bool = True,
    theta_init: float = 2.0,
) -> Dict:
    praeco = Praeco()
    auctor = Auctor(backbone, theta=theta_init)
    scriba = Scriba(backbone, n_variants=n_paraphrases)
    custos = Custos(backbone)
    magister = Magister(backbone, theta_init=theta_init)

    history = []
    detailed_records: List[Dict] = []

    # Main generation loop
    for gen in range(1, generations + 1):
        prompt = praeco.next_prompt()

        auctor.theta = magister.theta
        wm_text = auctor.generate(prompt, max_tokens=100)

        original_score = custos.watermark_score(wm_text)

        paraphrases = scriba.paraphrase(wm_text)
        s_eff = custos.effective_score(paraphrases)
        z_eff = custos.effective_z_score(paraphrases)

        z_threshold = 4.0
        para_z_scores = [custos.z_score(p) for p in paraphrases]
        retention = sum(1 for z in para_z_scores if z >= z_threshold) / max(len(para_z_scores), 1)

        new_theta = magister.update(wm_text, original_score, reference_text=prompt)

        record = {
            "generation": gen,
            "theta": round(new_theta, 4),
            "original_score": round(original_score, 4),
            "effective_score": round(s_eff, 4),
            "effective_z_score": round(z_eff, 4),
            "retention_rate": round(retention, 4),
            "readability": readability_score(wm_text),
        }
        history.append(record)

        detailed_records.append({
            "phase": "training",
            "generation": gen,
            "dataset": praeco.get_source(prompt),
            "prompt": prompt,
            "watermarked_text": wm_text,
            "paraphrases": paraphrases,
            "scores": {
                "s_orig": round(original_score, 4),
                "s_eff": round(s_eff, 4),
                "z_eff": round(z_eff, 4),
                "retention_rate": round(retention, 4),
                "theta": round(new_theta, 4),
                "readability": readability_score(wm_text),
            },
        })

        if verbose:
            logger.info(
                f"Gen {gen:>3} | θ={new_theta:.4f} | "
                f"S_orig={original_score:.4f} | S_eff={s_eff:.4f}"
            )

    # AUC-ROC and False Positive Rate
    logger.info("Computing AUC-ROC and false positive rate")

    eval_prompts = praeco.batch_prompts(n_eval_samples)

    baseline_texts = [
        backbone.generate(p, max_new_tokens=100)['text']
        for p in eval_prompts
    ]
    wm_texts_eval = [
        auctor.generate(p, max_tokens=100)
        for p in eval_prompts
    ]

    labels = [0] * n_eval_samples + [1] * n_eval_samples
    z_scores = [custos.z_score(t) for t in baseline_texts + wm_texts_eval]
    auc = roc_auc_score(labels, z_scores)

    z_threshold = 4.0
    fp = sum(1 for t in baseline_texts if custos.z_score(t) >= z_threshold)
    fpr = fp / len(baseline_texts)

    # Build detailed eval records (paraphrases on watermarked texts only)
    logger.info("Building detailed eval records...")
    for i, prompt in enumerate(eval_prompts):
        wm_text = wm_texts_eval[i]
        paraphrases = scriba.paraphrase(wm_text)
        s_orig = custos.watermark_score(wm_text)
        s_eff_i = custos.effective_score(paraphrases)
        z_i = custos.z_score(wm_text)
        detailed_records.append({
            "phase": "eval",
            "dataset": praeco.get_source(prompt),
            "prompt": prompt,
            "baseline_text": baseline_texts[i],
            "watermarked_text": wm_text,
            "paraphrases": paraphrases,
            "scores": {
                "s_orig": round(s_orig, 4),
                "z_score": round(z_i, 4),
                "s_eff": round(s_eff_i, 4),
                "is_watermarked": z_i >= z_threshold,
            },
        })

    # Final summary
    first_eff = history[0]["effective_score"]
    last_eff = history[-1]["effective_score"]
    improvement = (last_eff - first_eff) / max(first_eff, 1e-6) * 100

    avg_readability = round(
        sum(r["readability"] for r in history) / len(history), 2
    )
    avg_retention = round(
        sum(r["retention_rate"] for r in history) / len(history), 4
    )

    summary = {
        "theta_final": history[-1]["theta"],
        "effective_score_final": last_eff,
        "effective_score_improvement_pct": round(improvement, 2),
        "avg_retention_rate": avg_retention,
        "meets_85pct_retention": avg_retention >= 0.85,
        "auc_roc": round(auc, 4),
        "false_positive_rate": round(fpr, 4),
        "meets_90pct_auc": auc >= 0.90,
        "avg_readability": avg_readability,
        "meets_readability_45": avg_readability >= 4.5,
        "history": history,
        "detailed_records": detailed_records,
    }
    return summary
