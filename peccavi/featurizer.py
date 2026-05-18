"""
peccavi/featurizer.py
PromptFeaturizer — extracts lightweight numerical features from a prompt
that predict how much watermarking flexibility the content allows.

Intuition:
  - High-entropy creative prompts tolerate stronger watermarking (higher θ)
    because there is more token choice at each step.
  - Low-entropy factual prompts need gentle watermarking (lower θ)
    to preserve precision and avoid distorting specific terminology.

Features (all normalised to [0, 1]):
  f0  token_entropy      — Shannon entropy of token distribution, normalised by log2(vocab)
  f1  length_norm        — token count / max_len
  f2  vocab_diversity    — unique tokens / total tokens (type-token ratio)
  f3  avg_token_len_norm — mean character length / 12

All features are bounded [0, 1] so the learned weight vector w is directly
interpretable: w[0] > 0 means "assign higher θ to high-entropy prompts".
"""

from __future__ import annotations
import math
import re
from collections import Counter
from typing import List

import numpy as np

N_FEATURES = 4
FEATURE_NAMES = ["token_entropy", "length_norm", "vocab_diversity", "avg_token_len_norm"]


class PromptFeaturizer:
    """
    Maps a raw prompt string to a fixed-size feature vector for the
    content-adaptive θ policy in Magister.
    """

    def __init__(self, max_len: int = 512):
        self.max_len = max_len

    def _tokenize(self, text: str) -> List[str]:
        return re.findall(r"\b\w+\b", text.lower())

    def extract(self, prompt: str) -> np.ndarray:
        """Returns feature vector of shape (N_FEATURES,), values in [0, 1]."""
        tokens = self._tokenize(prompt)
        n = len(tokens)
        if n == 0:
            return np.zeros(N_FEATURES, dtype=np.float32)

        counts = Counter(tokens)
        probs = np.array(list(counts.values()), dtype=np.float64) / n

        # f0: normalised Shannon entropy
        raw_h = -float(np.sum(probs * np.log2(probs + 1e-12)))
        max_h = math.log2(max(len(counts), 2))
        f0 = float(min(raw_h / max_h, 1.0))

        # f1: normalised length
        f1 = float(min(n / self.max_len, 1.0))

        # f2: type-token ratio (vocabulary diversity)
        f2 = float(len(counts) / n)

        # f3: normalised average token length (typical range 3–8 chars)
        f3 = float(min(np.mean([len(t) for t in tokens]) / 12.0, 1.0))

        return np.array([f0, f1, f2, f3], dtype=np.float32)
