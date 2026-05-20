"""
peccavi/magister.py
Agent: Magister – Policy Learning via REINFORCE.
Updates watermark parameter θ to maximise composite reward.
"""

from __future__ import annotations
import torch
import numpy as np
from backbone.model import LLaMABackbone
from peccavi.auctor import Auctor, _watermark_score, _context_seed
from peccavi.featurizer import N_FEATURES, FEATURE_NAMES
from typing import List, Optional
import logging
from peccavi.constants import SECRET_KEY
from bert_score import BERTScorer

logger = logging.getLogger(__name__)


def text_quality_score(text: str, backbone=None, reference_text: str = None) -> float:
    if backbone is None or not text.strip():
        return 0.5
    
    ppl_score = 0.5  # Default
    
    if backbone.backend == "transformers" and hasattr(backbone, 'model'):
        try:
            import torch, math
            enc = backbone.tokenizer(text, return_tensors="pt").to(backbone.model.device)
            with torch.no_grad():
                loss = backbone.model(**enc, labels=enc["input_ids"]).loss
            ppl = math.exp(loss.item())
            ppl_score = max(0.0, 1.0 - (ppl - 1) / 99)
        except Exception as e:
            logger.warning(f"Failed to compute perplexity: {e}")
            ppl_score = 0.5
    
    if reference_text:
        try:
            if not hasattr(text_quality_score, "_scorer"):
                text_quality_score._scorer = BERTScorer(lang="en", rescale_with_baseline=True)
            P, R, F1 = text_quality_score._scorer.score([text], [reference_text])
            bert_score_val = F1.mean().item()
            return (ppl_score + bert_score_val) / 2
        except Exception as e:
            logger.warning(f"Failed to compute BERTScore: {e}")
            return ppl_score
    return ppl_score
    
def composite_reward(
    effective_wm_score: float, quality: float,
    lam: float = 0.6, nu: float = 0.4,
    mu_ppl: float = 0.0, ppl_ratio: float = 1.0,
    rho_survival: float = 0.0, survival_score: float = 0.0,
) -> float:
    """r = λ*S_eff + ν*Q - μ*max(0,PPL_ratio-1) + ρ*survival_after_attack
    rho_survival=0 (default) reproduces the original reward with no attack term."""
    ppl_penalty = mu_ppl * max(0.0, ppl_ratio - 1.0)
    return lam * effective_wm_score + nu * quality - ppl_penalty + rho_survival * survival_score


class Magister:
    def __init__(
        self,
        backbone: LLaMABackbone,
        theta_init: float = 2.0,
        alpha: float = 0.05,
        gamma: float = 0.99,
        lam: float = 0.6,
        nu: float = 0.4,
        mu_ppl: float = 0.0,
        rho_survival: float = 0.0,
        secret_key: str = SECRET_KEY,
        adaptive: bool = False,
        theta_min: float = 0.5,
        theta_max: float = 8.0,
    ):
        self.backbone = backbone
        self.theta = theta_init
        self.alpha = alpha
        self.gamma = gamma
        self.lam = lam
        self.nu = nu
        self.mu_ppl = mu_ppl
        self.rho_survival = rho_survival
        self.secret_key = secret_key
        self.history: List[float] = []

        self.adaptive = adaptive
        self.theta_min = theta_min
        self.theta_max = theta_max
        self.w: np.ndarray = np.zeros(N_FEATURES, dtype=np.float32)

    def compute_theta(self, features: Optional[np.ndarray] = None) -> float:
        """
        θ(context) = clip(θ_base + w · features, θ_min, θ_max)

        When adaptive=False or features=None, returns the base θ unchanged —
        making this a drop-in replacement for the fixed-θ policy.

        High-entropy prompts (creative writing) → higher θ (stronger watermark).
        Low-entropy prompts (factual Q&A, code) → lower θ (gentle watermark).
        """
        if not self.adaptive or features is None:
            return float(np.clip(self.theta, self.theta_min, self.theta_max))
        raw = self.theta + float(np.dot(self.w, features))
        return float(np.clip(raw, self.theta_min, self.theta_max))

    def feature_report(self) -> dict:
        """Returns the learned weight vector for logging and paper reporting."""
        return {
            "feature_names": FEATURE_NAMES,
            "w": self.w.tolist(),
            "theta_base": round(self.theta, 4),
            "interpretation": {
                name: round(float(wi), 4)
                for name, wi in zip(FEATURE_NAMES, self.w)
            },
        }

    def _lazy_load_marianmt(self) -> None:
        """Load MarianMT EN→FR→EN models lazily on first use (CPU only)."""
        if hasattr(self, "_marian_loaded"):
            return
        try:
            from transformers import MarianMTModel, MarianTokenizer
            logger.info("Loading MarianMT EN→FR→EN for attack-aware training...")
            self._tok_en_fr = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-fr")
            self._mdl_en_fr = MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-en-fr")
            self._tok_fr_en = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-fr-en")
            self._mdl_fr_en = MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-fr-en")
            self._mdl_en_fr.eval()
            self._mdl_fr_en.eval()
            self._marian_loaded = True
            logger.info("MarianMT loaded on CPU for back-translation attack")
        except Exception as e:
            logger.warning(f"MarianMT unavailable ({e}) — rho_survival term disabled")
            self._marian_loaded = False

    def _back_translate(self, text: str) -> str:
        """EN→FR→EN back-translation using MarianMT. Runs on CPU."""
        if not getattr(self, "_marian_loaded", False) or not text.strip():
            return text
        try:
            import torch
            enc = self._tok_en_fr(
                text, return_tensors="pt", truncation=True, max_length=512, padding=True
            )
            with torch.no_grad():
                fr_ids = self._mdl_en_fr.generate(**enc, max_new_tokens=512)
            fr = self._tok_en_fr.decode(fr_ids[0], skip_special_tokens=True)
            enc2 = self._tok_fr_en(
                fr, return_tensors="pt", truncation=True, max_length=512, padding=True
            )
            with torch.no_grad():
                en_ids = self._mdl_fr_en.generate(**enc2, max_new_tokens=512)
            return self._tok_fr_en.decode(en_ids[0], skip_special_tokens=True)
        except Exception:
            return text

    def _survival_score(self, attacked_text: str) -> float:
        """Watermark survival in [0,1] after back-translation attack.

        Computes hash-based z-score on the attacked text, then applies a
        sigmoid centred at z=2.0 so the reward kicks in above the practical
        detection threshold. Returns 0 if the watermark is fully destroyed,
        approaching 1 as the signal survives strongly.
        """
        import math
        if not hasattr(self.backbone, "tokenizer"):
            return 0.0
        token_ids = self.backbone.tokenizer.encode(attacked_text)
        n = len(token_ids)
        if n == 0:
            return 0.0
        green = sum(
            1 for i, tid in enumerate(token_ids)
            if _watermark_score(tid, _context_seed(token_ids[:i], self.secret_key)) > 0.5
        )
        z = (green - n * 0.5) / max(math.sqrt(n * 0.25), 1e-6)
        return float(1.0 / (1.0 + math.exp(-(z - 2.0))))

    def _policy_gradient(self, token_ids: List[int]) -> float:
        """
        Approximate ∇_θ log p_w(x) = Σ_t g(x_t, r_t)
        over all tokens since Auctor now watermarks the full generated text.
        """
        grad = 0.0
        for i, tid in enumerate(token_ids):
            r_t = _context_seed(token_ids[:i], self.secret_key)
            g = _watermark_score(tid, r_t)
            grad += g
        return grad

    def update(
        self,
        generated_text: str,
        effective_wm_score: float,
        reference_text: str = None,
        prompt_features: Optional[np.ndarray] = None,
    ) -> float:
        """
        One REINFORCE update step. Updates both the base θ and, when
        adaptive=True, the feature weight vector w.

        Returns the updated base θ. Use compute_theta(features) to get
        the context-specific θ for the next prompt.
        """
        quality = text_quality_score(generated_text, self.backbone, reference_text)
        # Compute backbone PPL ratio for penalty term (1.0 = no cost; >1.0 penalised)
        ppl_ratio = 1.0
        if self.mu_ppl > 0.0 and self.backbone.backend == "transformers" and hasattr(self.backbone, "model"):
            try:
                import math
                ref_enc = self.backbone.tokenizer(reference_text or generated_text, return_tensors="pt").to(self.backbone.model.device)
                gen_enc = self.backbone.tokenizer(generated_text, return_tensors="pt").to(self.backbone.model.device)
                import torch
                with torch.no_grad():
                    ppl_ref = math.exp(self.backbone.model(**ref_enc, labels=ref_enc["input_ids"]).loss.item())
                    ppl_gen = math.exp(self.backbone.model(**gen_enc, labels=gen_enc["input_ids"]).loss.item())
                ppl_ratio = ppl_gen / max(ppl_ref, 1e-6)
            except Exception:
                pass
        survival_score = 0.0
        if self.rho_survival > 0.0:
            self._lazy_load_marianmt()
            if getattr(self, "_marian_loaded", False):
                attacked = self._back_translate(generated_text)
                survival_score = self._survival_score(attacked)

        reward = composite_reward(
            effective_wm_score, quality, self.lam, self.nu,
            self.mu_ppl, ppl_ratio, self.rho_survival, survival_score,
        )

        if hasattr(self.backbone, "tokenizer"):
            token_ids = self.backbone.tokenizer.encode(generated_text)
        else:
            token_ids = generated_text.split()

        self.history.append(reward)
        grad = self._policy_gradient(token_ids)

        baseline = float(np.mean(self.history[-20:])) if len(self.history) >= 5 else 0.5
        advantage = reward - baseline

        # Update base θ (same as non-adaptive policy)
        self.theta += self.alpha * grad * advantage
        self.theta = float(np.clip(self.theta, self.theta_min, self.theta_max))

        # Update feature weights w (only when adaptive and features provided)
        # ∇_w J ≈ grad * advantage * features  (REINFORCE for linear policy)
        if self.adaptive and prompt_features is not None:
            self.w += self.alpha * grad * advantage * prompt_features
            # Clip w to prevent unbounded growth; ±3 allows θ to swing ±3 units
            self.w = np.clip(self.w, -3.0, 3.0)

        logger.debug(
            f"θ_base={self.theta:.4f} | reward={reward:.4f} | "
            f"advantage={advantage:.4f}"
            + (f" | survival={survival_score:.3f}" if self.rho_survival > 0 else "")
            + (f" | w={self.w.tolist()}" if self.adaptive else "")
        )
        return self.theta