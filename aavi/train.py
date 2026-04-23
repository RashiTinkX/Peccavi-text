"""
aavi/train.py
Adversarial fine-tuning loop for improving model robustness (AAVI).
Uses LoRA + PEFT to fine-tune LLaMA-2 on safe-response pairs.
"""

from __future__ import annotations
from backbone.model import LLaMABackbone
from aavi.model import AAVIModel
from aavi.metrics import compute_aavi
from peft import LoraConfig, get_peft_model, TaskType
from transformers import TrainingArguments, Trainer
import torch
from typing import List, Dict
import logging

logger = logging.getLogger(__name__)


def build_safe_pairs(goals: List[str], backbone: LLaMABackbone) -> List[Dict]:
    """
    For each goal, generate attack prompts, collect model responses,
    and create (prompt, safe_response) training pairs by filtering
    compliant (unsafe) outputs.
    """
    aavi_model = AAVIModel(backbone)
    pairs = []
    for goal in goals:
        attacks = aavi_model.generate_attacks(goal)
        for category, prompts in attacks.items():
            for prompt in prompts:
                eval_result = aavi_model.evaluate_prompt(prompt)
                # Only train on prompts where model was exploited
                if eval_result["is_vulnerable"]:
                    pairs.append({
                        "prompt": prompt,
                        "label": "I'm unable to assist with that request.",
                    })
    return pairs


def apply_lora(backbone: LLaMABackbone) -> object:
    config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    peft_model = get_peft_model(backbone.model, config)
    peft_model.print_trainable_parameters()
    return peft_model


def train_aavi(
    backbone: LLaMABackbone,
    goals: List[str],
    output_dir: str = "./aavi_checkpoint",
    epochs: int = 3,
):
    logger.info("Building adversarial training pairs...")
    pairs = build_safe_pairs(goals, backbone)
    logger.info(f"Collected {len(pairs)} adversarial training pairs.")

    peft_model = apply_lora(backbone)

    # Tokenize
    tokenizer = backbone.tokenizer
    encodings = tokenizer(
        [p["prompt"] + p["label"] for p in pairs],
        truncation=True, padding=True, max_length=512,
        return_tensors="pt",
    )

    class SimpleDataset(torch.utils.data.Dataset):
        def __init__(self, enc): self.enc = enc
        def __len__(self): return self.enc["input_ids"].shape[0]
        def __getitem__(self, i):
            return {k: v[i] for k, v in self.enc.items()}

    dataset = SimpleDataset(encodings)

    args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        fp16=True,
        logging_steps=10,
        save_strategy="epoch",
    )

    trainer = Trainer(
        model=peft_model,
        args=args,
        train_dataset=dataset,
    )
    trainer.train()
    peft_model.save_pretrained(output_dir)
    logger.info(f"AAVI fine-tuned model saved to {output_dir}")