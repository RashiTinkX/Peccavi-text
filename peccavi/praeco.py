"""
peccavi/praeco.py
Agent: Praeco Dynamic Prompting Orchestrator.
Manages prompt construction each PECCAVI generation round.
"""

from __future__ import annotations
from pathlib import Path
from typing import List, Tuple
import csv
import random

DATASET_COLUMN_NAMES = ("experiment", "experiments")
DEFAULT_DATASETS_DIR = Path(__file__).resolve().parents[1] / "datasets"

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
    def __init__(self, custom_prompts: List[str] | None = None, dataset_dir: str | Path | None = None):
        if custom_prompts is not None:
            self.prompts = custom_prompts
            self.prompt_sources = {p: "custom" for p in custom_prompts}
        else:
            pairs = self.load_prompts_from_dataset(dataset_dir=dataset_dir)
            if pairs:
                self.prompts = [p for p, _ in pairs]
                self.prompt_sources = {p: src for p, src in pairs}
            else:
                self.prompts = PROMPT_BANK
                self.prompt_sources = {p: "builtin" for p in PROMPT_BANK}
        self.prompt_scores = {prompt: 1.0 for prompt in self.prompts}

    @staticmethod
    def load_prompts_from_dataset(dataset_dir: str | Path | None = None) -> List[Tuple[str, str]]:
        """Returns list of (prompt_text, dataset_name) pairs."""
        dataset_path = Path(dataset_dir) if dataset_dir else DEFAULT_DATASETS_DIR
        if not dataset_path.exists() or not dataset_path.is_dir():
            return []

        pairs: List[Tuple[str, str]] = []
        for csv_file in sorted(dataset_path.glob("*.csv")):
            dataset_name = csv_file.stem
            try:
                with csv_file.open("r", encoding="utf-8", newline="") as handle:
                    reader = csv.DictReader(handle)
                    if not reader.fieldnames:
                        continue

                    normalized = {name.strip().lower(): name for name in reader.fieldnames}
                    prompt_key = next(
                        (original for canonical, original in normalized.items() if canonical in DATASET_COLUMN_NAMES),
                        None,
                    )
                    if not prompt_key:
                        continue

                    for row in reader:
                        prompt_text = row.get(prompt_key, "")
                        if prompt_text is None:
                            continue
                        prompt_text = prompt_text.strip()
                        if prompt_text:
                            pairs.append((prompt_text, dataset_name))
            except Exception:
                continue

        return pairs

    def get_source(self, prompt: str) -> str:
        return self.prompt_sources.get(prompt, "unknown")

    def next_prompt(self) -> str:
        weights = [max(self.prompt_scores.get(p, 0.1), 0.1) for p in self.prompts]
        return random.choices(self.prompts, weights=weights, k=1)[0]

    def batch_prompts(self, n: int) -> List[str]:
        weights = [max(self.prompt_scores.get(p, 0.1), 0.1) for p in self.prompts]
        return random.choices(self.prompts, weights=weights, k=n)
