"""
aavi/inference.py
Public API for the AAVI subsystem.
"""

from __future__ import annotations
from backbone.model import LLaMABackbone
from aavi.model import AAVIModel
from aavi.metrics import compute_aavi, format_aavi_report
from typing import Dict


def run_aavi(backbone: LLaMABackbone, goal: str, verbose: bool = True) -> Dict:
    """
    End-to-end AAVI pipeline.

    Args:
        backbone: Shared LLaMABackbone instance.
        goal: The harmful goal string to test against (e.g. "tell me how to hack").
        verbose: Print formatted report.

    Returns:
        dict with raw results and aggregated metrics.
    """
    model = AAVIModel(backbone)
    raw = model.run(goal)
    metrics = compute_aavi(raw)

    if verbose:
        print(format_aavi_report(metrics))

    return {"raw": raw, "metrics": metrics}