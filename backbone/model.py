"""
backbone/model.py
Shared backbone for AAVI, PECCAVI
Windows-compatible: works without bitsandbytes (falls back to fp16).
"""

from __future__ import annotations
import logging
import os
import sys
from typing import Optional

_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _root not in sys.path:
    sys.path.insert(0, _root)

try:
    import torch
except ImportError:
    raise ImportError("\n[AIISC] torch not found.\nRun: pip install torch\n")

try:
    from transformers import AutoTokenizer, AutoModelForCausalLM
except ImportError:
    raise ImportError("\n[AIISC] transformers not found.\nRun: pip install transformers accelerate\n")

try:
    from transformers import BitsAndBytesConfig
    import bitsandbytes  # noqa: F401
    BNB_AVAILABLE = True
except Exception:
    BNB_AVAILABLE = False

logger = logging.getLogger(__name__)


class LLaMABackbone:
    _instance: Optional["LLaMABackbone"] = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(
        self,
        model_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        device: str = "auto",
        load_in_4bit: bool = True,
    ):
        if self._initialized:
            return

        self.model_name = model_name
        self.device = device

        print(f"[AIISC] Loading tokenizer: {model_name}", flush=True)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        load_kwargs: dict = {"device_map": device, "trust_remote_code": True}

        if load_in_4bit and BNB_AVAILABLE:
            print("[AIISC] Using 4-bit quantization", flush=True)
            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
        else:
            print("[AIISC] bitsandbytes unavailable — using fp16", flush=True)
            load_kwargs["torch_dtype"] = torch.float32

        print("[AIISC] Loading model weights...", flush=True)
        self.model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)
        self.model.eval()
        self._initialized = True
        print("[AIISC] Backbone ready!", flush=True)

    @torch.no_grad()
    def generate(self, prompt, max_new_tokens=256, temperature=0.7,
                 top_p=0.9, do_sample=True, return_logits=False):
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        outputs = self.model.generate(
            **inputs, max_new_tokens=max_new_tokens, temperature=temperature,
            top_p=top_p, do_sample=do_sample, output_scores=return_logits,
            return_dict_in_generate=True,
        )
        generated_ids = outputs.sequences[0][inputs["input_ids"].shape[1]:]
        text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        result = {"text": text}
        if return_logits and hasattr(outputs, "scores"):
            result["scores"] = outputs.scores
        return result

    @torch.no_grad()
    def token_distribution(self, context):
        inputs = self.tokenizer(context, return_tensors="pt").to(self.model.device)
        logits = self.model(**inputs).logits[:, -1, :]
        return torch.softmax(logits.squeeze(0), dim=-1)

    @torch.no_grad()
    def score_response(self, prompt, response):
        full = prompt + response
        enc_full = self.tokenizer(full, return_tensors="pt").to(self.model.device)
        n_prompt = self.tokenizer(prompt, return_tensors="pt")["input_ids"].shape[1]
        logits = self.model(**enc_full).logits
        log_probs = torch.log_softmax(logits[0], dim=-1)
        target_ids = enc_full["input_ids"][0, n_prompt:]
        selected = log_probs[n_prompt - 1: n_prompt - 1 + len(target_ids)]
        return selected[range(len(target_ids)), target_ids].mean().item()
    
    @torch.no_grad()
    def tokenize(self, text: str) -> list[int]:
        return self.tokenizer.encode(text, add_special_tokens=False)