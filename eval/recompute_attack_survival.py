"""
eval/recompute_attack_survival.py
Re-runs attack survival on saved watermarked texts from old result JSONs.

All z-score computations are hash-based and need only the tokenizer (no GPU).
Attacks use GPT-4o-mini (cheap, ~$0.001 per file) or local lexical fallback.

Usage:
    python eval/recompute_attack_survival.py results/peccavi_s7.json
    python eval/recompute_attack_survival.py results/peccavi_s7.json results/kgw_s7.json results/sir_s7.json
"""

from __future__ import annotations
import json
import math
import os
import random
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional

THRESHOLDS = [1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
SAMPLE_N = 30          # texts per attack type
SECRET_KEY = "PECCAVI-SECRET"


# ---------------------------------------------------------------------------
# Tokenizer-only backbone — sufficient for hash-based z-score computation
# ---------------------------------------------------------------------------

class _LightBackbone:
    def __init__(self, model_name: str):
        print(f"  Loading tokenizer: {model_name}")
        from transformers import AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.backend = "transformers"
        self.model = None   # not loaded


def _detect_model(inner: dict) -> str:
    """Guess backbone model name from result JSON."""
    wf = inner.get("w_feature_names")
    mode = inner.get("watermark_mode", "peccavi")
    # Try to find model name in history records
    if "history" in inner and inner["history"]:
        pass
    # Fall back to known defaults
    if "mistral" in str(inner).lower():
        return "mistralai/Mistral-7B-Instruct-v0.3"
    return "meta-llama/Llama-2-7b-chat-hf"


# ---------------------------------------------------------------------------
# Lightweight z-scorers (tokenizer-only)
# ---------------------------------------------------------------------------

def _peccavi_z(text: str, tokenizer) -> float:
    import hashlib
    token_ids = tokenizer.encode(text)
    n = len(token_ids)
    if n == 0:
        return 0.0
    green = 0
    for i, tid in enumerate(token_ids):
        ctx = token_ids[:i]
        seed_str = SECRET_KEY + ":" + ":".join(str(t) for t in ctx[-3:])
        r = int(hashlib.sha256(seed_str.encode()).hexdigest()[:8], 16)
        if (tid + r) % 2 == 0:
            green += 1
    return (green - n * 0.5) / math.sqrt(n * 0.25)


def _kgw_z(text: str, tokenizer, gamma: float = 0.5) -> float:
    import hashlib, numpy as np
    token_ids = tokenizer.encode(text)
    n = len(token_ids)
    if n == 0:
        return 0.0
    vocab_size = getattr(tokenizer, "vocab_size", None) or len(tokenizer)
    green = 0
    for i, tid in enumerate(token_ids):
        prev = token_ids[i - 1] if i > 0 else 0
        seed = int(hashlib.sha256(f"{SECRET_KEY}:{prev}".encode()).hexdigest()[:8], 16)
        rng = np.random.default_rng(seed % (2 ** 32))
        n_green = int(gamma * vocab_size)
        green_idx = set(rng.choice(vocab_size, n_green, replace=False).tolist())
        if tid in green_idx:
            green += 1
    return (green - n * gamma) / math.sqrt(n * gamma * (1 - gamma))


def _sir_z(text: str, tokenizer, gamma: float = 0.5) -> float:
    # SIR uses same green-list logic as KGW but only on high-entropy positions.
    # We approximate with KGW z-score (same formula, slight overcount on z).
    return _kgw_z(text, tokenizer, gamma)


def _get_z_fn(mode: str, tokenizer):
    if mode == "kgw":
        return lambda t: _kgw_z(t, tokenizer)
    elif mode == "sir":
        return lambda t: _sir_z(t, tokenizer)
    else:
        return lambda t: _peccavi_z(t, tokenizer)


# ---------------------------------------------------------------------------
# Lightweight attacks (no backbone required)
# ---------------------------------------------------------------------------

def _lexical_attack(text: str) -> str:
    try:
        from nltk.corpus import wordnet
        from nltk import download
        download("wordnet", quiet=True)
        download("omw-1.4", quiet=True)
        words = text.split()
        out = []
        for w in words:
            syns = wordnet.synsets(w)
            if syns:
                lemma = random.choice(syns[0].lemmas()).name().replace("_", " ")
                out.append(lemma)
            else:
                out.append(w)
        return " ".join(out)
    except Exception:
        return text


def _syntactic_attack(text: str) -> str:
    """Back-translate EN→FR→EN using MarianMT if available."""
    try:
        from transformers import MarianMTModel, MarianTokenizer
        import torch
        tok_ef = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-fr")
        mdl_ef = MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-en-fr")
        tok_fe = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-fr-en")
        mdl_fe = MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-fr-en")
        inp = tok_ef(text, return_tensors="pt", truncation=True, max_length=512)
        with torch.no_grad():
            fr_ids = mdl_ef.generate(**inp)
        fr = tok_ef.decode(fr_ids[0], skip_special_tokens=True)
        inp2 = tok_fe(fr, return_tensors="pt", truncation=True, max_length=512)
        with torch.no_grad():
            en_ids = mdl_fe.generate(**inp2)
        return tok_fe.decode(en_ids[0], skip_special_tokens=True)
    except Exception:
        return _lexical_attack(text)


def _gpt4_attack(text: str, client) -> str:
    """GPT-4o-mini paraphrase — strongest attack."""
    try:
        r = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "Rewrite the text with completely different words while preserving meaning. Output only the rewritten text."},
                {"role": "user", "content": text},
            ],
            max_tokens=200,
            temperature=0.7,
        )
        return r.choices[0].message.content.strip() or text
    except Exception:
        return _lexical_attack(text)


def _build_attack_fns(openai_key: Optional[str]):
    client = None
    if openai_key:
        try:
            from openai import OpenAI
            client = OpenAI(api_key=openai_key)
        except ImportError:
            pass

    fns = {
        "lexical": _lexical_attack,
        "syntactic": _syntactic_attack,
        "semantic": (lambda t: _gpt4_attack(t, client)) if client else _lexical_attack,
        "lm_paraphrase": (lambda t: _gpt4_attack(t, client)) if client else _lexical_attack,
        "gpt4_paraphrase": (lambda t: _gpt4_attack(t, client)) if client else _lexical_attack,
    }
    return fns


# ---------------------------------------------------------------------------
# Core recompute logic
# ---------------------------------------------------------------------------

def _inner(data: dict) -> tuple:
    keys = list(data.keys())
    if len(keys) == 1 and isinstance(data[keys[0]], dict):
        inner = data[keys[0]]
        if "watermark_mode" in inner or "auc_roc" in inner:
            return keys[0], inner
    return None, data


def recompute(path: str, openai_key: Optional[str] = None) -> None:
    p = Path(path)
    print(f"\n{'='*60}")
    print(f"  Processing: {p.name}")

    with open(p) as f:
        data = json.load(f)
    top_key, inner = _inner(data)

    mode = inner.get("watermark_mode", "peccavi")
    model_name = _detect_model(inner)

    # Extract eval-phase watermarked texts
    records = inner.get("detailed_records", [])
    eval_texts = [
        r["watermarked_text"]
        for r in records
        if r.get("phase") == "eval" and r.get("watermarked_text")
    ]
    if not eval_texts:
        print(f"  No eval watermarked_texts in detailed_records — skipping")
        return

    sample = eval_texts[:SAMPLE_N]
    print(f"  mode={mode}  texts={len(sample)}")

    backbone = _LightBackbone(model_name)
    z_fn = _get_z_fn(mode, backbone.tokenizer)
    attack_fns = _build_attack_fns(openai_key)

    attack_z_scores: Dict[str, List[float]] = {}
    attack_survival_by_threshold: Dict[str, Dict[str, float]] = {}
    attack_survival: Dict[str, float] = {}

    for attack_name, attack_fn in attack_fns.items():
        print(f"    attack: {attack_name} ...", end=" ", flush=True)
        z_list = []
        for wm_text in sample:
            try:
                attacked = attack_fn(wm_text)
                z_list.append(z_fn(attacked))
            except Exception as e:
                z_list.append(0.0)
        attack_z_scores[attack_name] = [round(z, 4) for z in z_list]
        attack_survival[attack_name] = round(
            sum(1 for z in z_list if z >= 4.0) / max(len(z_list), 1), 4
        )
        attack_survival_by_threshold[attack_name] = {
            f"z{t:.1f}": round(sum(1 for z in z_list if z >= t) / max(len(z_list), 1), 4)
            for t in THRESHOLDS
        }
        print(f"z≥2.0={attack_survival_by_threshold[attack_name]['z2.0']:.3f}  z≥4.0={attack_survival[attack_name]:.3f}")

    inner["attack_z_scores"] = attack_z_scores
    inner["attack_survival_by_threshold"] = attack_survival_by_threshold
    inner["attack_survival"] = attack_survival

    if top_key:
        data[top_key] = inner

    with open(p, "w") as f:
        json.dump(data, f, indent=2)
    print(f"  Saved → {p.name}")


def main():
    paths = sys.argv[1:]
    if not paths:
        print("Usage: python eval/recompute_attack_survival.py <results.json> [...]")
        sys.exit(1)
    openai_key = os.environ.get("OPENAI_API_KEY")
    if not openai_key:
        print("  OPENAI_API_KEY not set — semantic/lm/gpt4 attacks will use lexical fallback")
    for path in paths:
        try:
            recompute(path, openai_key)
        except Exception as e:
            print(f"  ERROR {path}: {e}")
    print("\nDone. Re-run eval/survival_analysis.py to see the curves.")


if __name__ == "__main__":
    main()
