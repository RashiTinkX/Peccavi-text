"""
peccavi/praeco.py
Agent: Praeco – Dynamic Prompting Orchestrator.
Manages prompt construction each PECCAVI generation round.
"""

from __future__ import annotations
from typing import List
import random

PROMPT_BANK = [
    "Summarize recent AI safety research.",
    "Explain the importance of watermarking in AI-generated content.",
    "Describe the risks of large language models in misinformation.",
    "Write a short paragraph about responsible AI deployment.",
    "Discuss how adversarial robustness improves AI reliability.",
    "Explain what content authenticity means in the context of generative AI.",
    "Describe how reinforcement learning is used in AI alignment.",
]


class Praeco:
    def __init__(self, custom_prompts: List[str] | None = None):
        self.prompts = custom_prompts if custom_prompts else PROMPT_BANK

    def next_prompt(self) -> str:
        return random.choice(self.prompts)

    def batch_prompts(self, n: int) -> List[str]:
        return random.choices(self.prompts, k=n)