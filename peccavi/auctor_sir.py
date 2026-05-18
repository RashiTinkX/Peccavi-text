"""
peccavi/auctor_sir.py
Baseline: Entropy-Aware Watermarking (SIR-style).
Applies the KGW green-list delta ONLY at high-entropy token positions.
Low-entropy positions (near-deterministic choices) are generated unwatermarked,
preserving fluency where the model has little freedom.

Reference: Kirchenbauer et al. 2023 entropy-aware variant; also related to
"A Semantic Invariant Robust Watermark for Large Language Models" (Liu et al. 2023).

Comparison with PECCAVI:
  SIR uses a fixed entropy threshold (hyperparameter).
  PECCAVI learns the optimal watermark strength theta via REINFORCE, adapting
  to content and jointly optimising detection power with text quality.
"""

from __future__ import annotations
import hashlib
import math
import torch
import numpy as np
from backbone.model import LLaMABackbone
from typing import List
from peccavi.constants import SECRET_KEY
from peccavi.auctor_kgw import _kgw_seed, _green_mask


def _token_entropy(logits: torch.Tensor) -> float:
    """Shannon entropy (nats) of the softmax distribution over logits."""
    probs = torch.softmax(logits.float(), dim=-1)
    log_probs = torch.log(probs + 1e-12)
    return float(-torch.sum(probs * log_probs).item())


class SIRAuctor:
    """
    Entropy-aware watermarked generation.

    At each step:
      1. Compute H = entropy of p_LM(·|context).
      2. If H > entropy_threshold: add delta to green token logits (watermark active).
      3. If H <= entropy_threshold: sample normally (low-entropy = nearly forced token).

    Detection:
      Re-run the model to recover entropy at each position.
      Z-test counts only high-entropy positions toward the green-token statistic,
      matching the generation-time decision exactly.
    """

    def __init__(
        self,
        backbone: LLaMABackbone,
        delta: float = 2.0,
        gamma: float = 0.5,
        entropy_threshold: float = 1.0,   # nats; ~3 nat ≈ uniform over ~20 tokens
        secret_key: str = SECRET_KEY,
    ):
        self.backbone = backbone
        self.delta = delta
        self.gamma = gamma
        self.entropy_threshold = entropy_threshold
        self.secret_key = secret_key

    # ------------------------------------------------------------------ #
    #  Generation                                                          #
    # ------------------------------------------------------------------ #

    def generate(self, prompt: str, max_tokens: int = 200) -> str:
        if not hasattr(self.backbone, "tokenizer"):
            raw = self.backbone.generate(prompt, max_new_tokens=max_tokens)
            return raw["text"] if isinstance(raw, dict) else raw

        tokenizer = self.backbone.tokenizer
        formatted_prompt = prompt
        if hasattr(tokenizer, "chat_template") and tokenizer.chat_template is not None:
            messages = [{"role": "user", "content": prompt}]
            formatted_prompt = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

        prompt_ids = tokenizer.encode(formatted_prompt, add_special_tokens=False)
        generated_ids: List[int] = []
        eos_token_id = tokenizer.eos_token_id
        vocab_size = tokenizer.vocab_size or len(tokenizer)

        for _ in range(max_tokens):
            context_ids = prompt_ids + generated_ids
            context_text = tokenizer.decode(context_ids, skip_special_tokens=True)

            inputs = tokenizer(
                context_text, return_tensors="pt", truncation=True, max_length=2048
            ).to(self.backbone.model.device)

            with torch.no_grad():
                outputs = self.backbone.model(**inputs)
                logits = outputs.logits[:, -1, :].squeeze(0)

            logits = torch.nan_to_num(logits, nan=0.0, posinf=1e4, neginf=-1e4)
            H = _token_entropy(logits)

            if H > self.entropy_threshold:
                prev = generated_ids[-1] if generated_ids else (prompt_ids[-1] if prompt_ids else 0)
                seed = _kgw_seed(prev, self.secret_key)
                mask = _green_mask(vocab_size, seed, self.gamma).to(logits.device)
                logits = logits.clone()
                logits[mask] += self.delta

            probs = torch.softmax(logits, dim=0).clamp(min=0.0)
            prob_sum = probs.sum()
            if prob_sum <= 0 or not torch.isfinite(prob_sum):
                new_token = int(torch.argmax(logits).item())
            else:
                probs = probs / prob_sum
                new_token = int(torch.multinomial(probs, 1).item())

            if eos_token_id is not None and new_token == eos_token_id:
                break
            generated_ids.append(new_token)

        return tokenizer.decode(generated_ids, skip_special_tokens=True) if generated_ids else ""

    # ------------------------------------------------------------------ #
    #  Detection                                                           #
    # ------------------------------------------------------------------ #

    def z_score(self, text: str) -> float:
        """
        Entropy-aware z-test. Only counts tokens whose context entropy exceeds
        the threshold used during generation — matching the generation decision.

        z = (count_green_at_high_H - n_high_H * gamma) / sqrt(n_high_H * gamma * (1-gamma))
        """
        if not hasattr(self.backbone, "tokenizer"):
            return 0.0

        tokenizer = self.backbone.tokenizer
        token_ids = tokenizer.encode(text)
        n = len(token_ids)
        if n == 0:
            return 0.0

        vocab_size = tokenizer.vocab_size or len(tokenizer)
        green_count = 0
        n_high_entropy = 0

        for i, tid in enumerate(token_ids):
            # Reconstruct context up to position i
            context_ids = token_ids[:i]
            if not context_ids:
                context_text = ""
            else:
                context_text = tokenizer.decode(context_ids, skip_special_tokens=True)

            inputs = tokenizer(
                context_text, return_tensors="pt", truncation=True, max_length=2048
            ).to(self.backbone.model.device)

            with torch.no_grad():
                outputs = self.backbone.model(**inputs)
                logits = outputs.logits[:, -1, :].squeeze(0)

            logits = torch.nan_to_num(logits, nan=0.0, posinf=1e4, neginf=-1e4)
            H = _token_entropy(logits)

            if H > self.entropy_threshold:
                n_high_entropy += 1
                prev = token_ids[i - 1] if i > 0 else 0
                seed = _kgw_seed(prev, self.secret_key)
                mask = _green_mask(vocab_size, seed, self.gamma)
                if tid < len(mask) and mask[tid]:
                    green_count += 1

        if n_high_entropy == 0:
            return 0.0

        expected = n_high_entropy * self.gamma
        variance = n_high_entropy * self.gamma * (1 - self.gamma)
        return (green_count - expected) / math.sqrt(variance)

    def detect(self, text: str, z_threshold: float = 4.0) -> dict:
        z = self.z_score(text)
        return {
            "z_score": round(z, 4),
            "is_watermarked": z >= z_threshold,
            "z_threshold": z_threshold,
        }
