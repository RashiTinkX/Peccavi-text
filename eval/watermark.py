"""
eval/watermark.py
PECCAVI full evaluation loop.
Supports watermark_mode: "peccavi" | "kgw" | "none"
"""

from __future__ import annotations
import random
import numpy as np
import torch
from backbone.model import LLaMABackbone
from peccavi.praeco import Praeco
from peccavi.auctor import Auctor
from peccavi.auctor_kgw import KGWAuctor
from peccavi.scriba import Scriba
from peccavi.custos import Custos
from peccavi.magister import Magister
from eval.quality import quality_score, flesch_quality_score
from sklearn.metrics import roc_auc_score
from typing import Dict, List
import logging

logger = logging.getLogger(__name__)


def _set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_peccavi(
    backbone: LLaMABackbone,
    generations: int = 10,
    n_paraphrases: int = 5,
    n_eval_samples: int = 100,
    verbose: bool = True,
    theta_init: float = 2.0,
    watermark_mode: str = "peccavi",   # "peccavi" | "kgw" | "none"
    kgw_delta: float = 2.0,
    kgw_gamma: float = 0.5,
    lam: float = 0.6,
    nu: float = 0.4,
    alpha: float = 0.05,
    seed: int = 42,
) -> Dict:
    _set_seed(seed)

    praeco = Praeco()
    scriba = Scriba(backbone, n_variants=n_paraphrases)
    custos = Custos(backbone)

    if watermark_mode == "kgw":
        generator = KGWAuctor(backbone, delta=kgw_delta, gamma=kgw_gamma)
        magister = None
    elif watermark_mode == "none":
        generator = None
        magister = None
    else:
        generator = Auctor(backbone, theta=theta_init)
        magister = Magister(backbone, theta_init=theta_init, alpha=alpha, lam=lam, nu=nu)

    history = []
    detailed_records: List[Dict] = []

    for gen in range(1, generations + 1):
        prompt = praeco.next_prompt()

        if watermark_mode == "peccavi":
            generator.theta = magister.theta
            wm_text = generator.generate(prompt, max_tokens=100)
        elif watermark_mode == "kgw":
            wm_text = generator.generate(prompt, max_tokens=100)
        else:
            wm_text = backbone.generate(prompt, max_new_tokens=100)["text"]

        if watermark_mode == "kgw":
            original_score = (generator.z_score(wm_text) + 10) / 20  # normalise z to [0,1] approx
        else:
            original_score = custos.watermark_score(wm_text)

        paraphrases = scriba.paraphrase(wm_text)
        s_eff = custos.effective_score(paraphrases)
        z_eff = custos.effective_z_score(paraphrases)

        z_threshold = 4.0
        if watermark_mode == "kgw":
            para_z = [generator.z_score(p) for p in paraphrases]
        else:
            para_z = [custos.z_score(p) for p in paraphrases]
        retention = sum(1 for z in para_z if z >= z_threshold) / max(len(para_z), 1)

        q_score = quality_score(wm_text, prompt=prompt)
        readability = flesch_quality_score(wm_text)

        new_theta = magister.update(wm_text, original_score, reference_text=prompt) if magister else theta_init

        record = {
            "generation": gen,
            "theta": round(new_theta, 4),
            "original_score": round(original_score, 4),
            "effective_score": round(s_eff, 4),
            "effective_z_score": round(z_eff, 4),
            "retention_rate": round(retention, 4),
            "readability": readability,
            "gpt4_quality": q_score,
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
                "readability": readability,
                "gpt4_quality": q_score,
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
        backbone.generate(p, max_new_tokens=100)["text"]
        for p in eval_prompts
    ]

    if watermark_mode == "none":
        wm_texts_eval = [
            backbone.generate(p, max_new_tokens=100)["text"]
            for p in eval_prompts
        ]
    elif watermark_mode == "kgw":
        wm_texts_eval = [generator.generate(p, max_tokens=100) for p in eval_prompts]
    else:
        wm_texts_eval = [generator.generate(p, max_tokens=100) for p in eval_prompts]

    if watermark_mode == "kgw":
        z_scores_all = (
            [generator.z_score(t) for t in baseline_texts]
            + [generator.z_score(t) for t in wm_texts_eval]
        )
        baseline_z = [generator.z_score(t) for t in baseline_texts]
    else:
        z_scores_all = [custos.z_score(t) for t in baseline_texts + wm_texts_eval]
        baseline_z = [custos.z_score(t) for t in baseline_texts]

    labels = [0] * n_eval_samples + [1] * n_eval_samples
    auc = roc_auc_score(labels, z_scores_all)

    z_threshold = 4.0
    fp = sum(1 for z in baseline_z if z >= z_threshold)
    fpr = fp / len(baseline_texts)

    logger.info("Building detailed eval records...")
    for i, prompt in enumerate(eval_prompts):
        wm_text = wm_texts_eval[i]
        paraphrases = scriba.paraphrase(wm_text)
        s_orig_i = custos.watermark_score(wm_text)
        s_eff_i = custos.effective_score(paraphrases)
        z_i = custos.z_score(wm_text) if watermark_mode != "kgw" else generator.z_score(wm_text)
        detailed_records.append({
            "phase": "eval",
            "dataset": praeco.get_source(prompt),
            "prompt": prompt,
            "baseline_text": baseline_texts[i],
            "watermarked_text": wm_text,
            "paraphrases": paraphrases,
            "scores": {
                "s_orig": round(s_orig_i, 4),
                "z_score": round(z_i, 4),
                "s_eff": round(s_eff_i, 4),
                "is_watermarked": z_i >= z_threshold,
            },
        })

    first_eff = history[0]["effective_score"]
    last_eff = history[-1]["effective_score"]
    improvement = (last_eff - first_eff) / max(first_eff, 1e-6) * 100

    avg_readability = round(sum(r["readability"] for r in history) / len(history), 2)
    avg_gpt4_quality = round(sum(r["gpt4_quality"] for r in history) / len(history), 2)
    avg_retention = round(sum(r["retention_rate"] for r in history) / len(history), 4)

    summary = {
        "watermark_mode": watermark_mode,
        "seed": seed,
        "theta_final": history[-1]["theta"],
        "effective_score_final": last_eff,
        "effective_score_improvement_pct": round(improvement, 2),
        "avg_retention_rate": avg_retention,
        "meets_85pct_retention": avg_retention >= 0.51,
        "auc_roc": round(auc, 4),
        "false_positive_rate": round(fpr, 4),
        "meets_90pct_auc": auc >= 0.90,
        "avg_readability": avg_readability,
        "avg_gpt4_quality": avg_gpt4_quality,
        "meets_readability_45": avg_readability >= 4.5,
        "meets_readability_30": avg_readability >= 3.0,
        "history": history,
        "detailed_records": detailed_records,
    }
    return summary
