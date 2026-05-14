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
from transformers import pipeline
import random

# Download NLTK data if needed
from nltk import download
download('wordnet', quiet=True)
download('omw-1.4', quiet=True)

class Scriba:
    def __init__(self, backbone: LLaMABackbone, n_variants: int = 5):
        self.backbone = backbone
        self.n_variants = n_variants
        # Load translation pipeline for back-translation
        self.translator_en_fr = pipeline("translation", model="Helsinki-NLP/opus-mt-en-fr")
        self.translator_fr_en = pipeline("translation", model="Helsinki-NLP/opus-mt-fr-en")
        self.paraphrase_model = None
        try:
            self.paraphrase_model = pipeline(
                "text2text-generation",
                model="Vamsi/T5_Paraphrase_Paws",
                device=-1,
                max_length=256,
            )
        except Exception:
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
        try:
            fr_text = self.translator_en_fr(text)[0]['translation_text']
            en_text = self.translator_fr_en(fr_text)[0]['translation_text']
            return en_text
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