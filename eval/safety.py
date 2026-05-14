"""
eval/safety.py
PECCAVI evaluation wrapper.
"""

from __future__ import annotations
from backbone.model import LLaMABackbone
from eval.watermark import run_peccavi
from typing import Dict
import logging

logger = logging.getLogger(__name__)


def run_full_eval(backbone: LLaMABackbone, verbose: bool = True) -> Dict:
    logger.info("Running PECCAVI evaluation...")
    return run_peccavi(backbone, generations=5, verbose=verbose)
