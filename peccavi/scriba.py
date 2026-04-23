"""
peccavi/scriba.py
Agent: Scriba – Adversarial Paraphrasing.
Generates N paraphrased variants via lexical, syntactic, and semantic attacks.
"""

from __future__ import annotations
from backbone.model import LLaMABackbone
from backbone.generate import generate_n_responses
from typing import List

PARAPHRASE_PROMPTS = [
    "Rewrite the following text using completely different words while preserving meaning:\n\n{text}",
    "Rephrase the following in a formal academic tone:\n\n{text}",
    "Rewrite the following as if explaining to a 10-year-old:\n\n{text}",
    "Restructure the following sentences, reversing the order of ideas:\n\n{text}",
    "Replace all key nouns and verbs with synonyms in this text:\n\n{text}",
]


class Scriba:
    def __init__(self, backbone: LLaMABackbone, n_variants: int = 5):
        self.backbone = backbone
        self.n_variants = n_variants

    def paraphrase(self, text: str) -> List[str]:
        """
        Returns N adversarial paraphrases of `text`.
        Uses a mix of prompt-based transformations.
        """
        variants: List[str] = []
        prompts = PARAPHRASE_PROMPTS[:self.n_variants]
        for template in prompts:
            prompt = template.format(text=text)
            out = self.backbone.generate(
                prompt, max_new_tokens=len(text.split()) + 50, temperature=0.85
            )
            variants.append(out["text"].strip())
        return variants