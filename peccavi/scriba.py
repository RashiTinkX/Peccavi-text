"""
peccavi/scriba.py
Agent: Scriba – Adversarial Paraphrasing.
Generates N paraphrased variants via lexical, syntactic, and semantic attacks.
"""

from __future__ import annotations
from backbone.model import LLaMABackbone #replace with GPT-4 / Claude (for high-quality adversarial paraphrases
from backbone.generate import generate_n_responses
from typing import List

PARAPHRASE_PROMPTS = [
    "Rewrite the following text using completely different words while preserving meaning:\n\n{text}",
    "Rephrase the following in a formal academic tone:\n\n{text}",
    "Rewrite the following as if explaining to a 10-year-old:\n\n{text}",
    "Restructure the following sentences, reversing the order of ideas:\n\n{text}",
    "Replace all key nouns and verbs with synonyms in this text:\n\n{text}",
]



from nltk.corpus import wordnet
from transformers import MarianMTModel, MarianTokenizer
import torch
import random

# Download NLTK data if needed
from nltk import download
download('wordnet', quiet=True)
download('omw-1.4', quiet=True)

class Scriba:
    def __init__(self, backbone: LLaMABackbone, n_variants: int = 5):
        self.backbone = backbone
        self.n_variants = n_variants
        self._translation_available = False
        try:
            self._tok_en_fr = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-fr")
            self._mdl_en_fr = MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-en-fr")
            self._tok_fr_en = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-fr-en")
            self._mdl_fr_en = MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-fr-en")
            self._translation_available = True
        except Exception:
            pass
        self.paraphrase_model = None

    def lexical_attack(self, text: str) -> str:
        words = text.split()
        new_words = []
        for word in words:
            synonyms = wordnet.synsets(word)
            if synonyms:
                synonym = random.choice(synonyms[0].lemmas()).name()
                new_words.append(synonym.replace('_', ' '))
            else:
                new_words.append(word)
        return ' '.join(new_words)

    def syntactic_attack(self, text: str) -> str:
        if not self._translation_available:
            return self.lm_paraphrase(text, random.choice(PARAPHRASE_PROMPTS))
        try:
            inputs = self._tok_en_fr(text, return_tensors="pt", truncation=True, max_length=512)
            with torch.no_grad():
                fr_ids = self._mdl_en_fr.generate(**inputs)
            fr_text = self._tok_en_fr.decode(fr_ids[0], skip_special_tokens=True)
            inputs2 = self._tok_fr_en(fr_text, return_tensors="pt", truncation=True, max_length=512)
            with torch.no_grad():
                en_ids = self._mdl_fr_en.generate(**inputs2)
            return self._tok_fr_en.decode(en_ids[0], skip_special_tokens=True)
        except Exception:
            return text

    def lm_paraphrase(self, text: str, prompt_template: str) -> str:
        prompt = prompt_template.format(text=text)
        output = self.backbone.generate(prompt, max_new_tokens=120)
        return output.get("text", "").strip() or text

    def semantic_attack(self, text: str) -> str:
        if self.paraphrase_model is not None:
            return self.paraphrase_model_text(text)
        return self.lm_paraphrase(text, random.choice(PARAPHRASE_PROMPTS))

    def paraphrase_model_text(self, text: str) -> str:
        prompt = f"paraphrase: {text}"
        output = self.paraphrase_model(prompt, max_length=256, num_return_sequences=1)
        return output[0].get("generated_text", "").strip()

    def paraphrase(self, text: str) -> List[str]:
        variants = []
        techniques = [
            self.lexical_attack,
            self.syntactic_attack,
            self.semantic_attack,
            lambda t: self.lm_paraphrase(t, random.choice(PARAPHRASE_PROMPTS)),
        ]
        for _ in range(self.n_variants):
            technique = random.choice(techniques)
            variants.append(technique(text).strip())
        return variants