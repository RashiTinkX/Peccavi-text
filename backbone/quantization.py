"""
backbone/quantization.py
Quantization config factory used when loading LLaMA-2.
"""

from transformers import BitsAndBytesConfig
import torch


def get_4bit_config() -> BitsAndBytesConfig:
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )


def get_8bit_config() -> BitsAndBytesConfig:
    return BitsAndBytesConfig(load_in_8bit=True)


def get_fp16_config() -> dict:
    """No quantization  pure fp16 for high-VRAM setups."""
    return {"torch_dtype": torch.float16}