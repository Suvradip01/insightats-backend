"""M3 — DistilBERT (or similar) project-complexity head; heuristic fallback when weights are absent."""

from __future__ import annotations

import os
import re
from typing import List, Optional, Tuple

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from app.core.config import settings
from app.schemas.analyze import Confidence3, ProjectComplexity, ShapKeywords

COMPLEXITY_MODEL_PATH = settings.COMPLEXITY_MODEL_DIR

ADV_KW = (
    "kubernetes",
    "distributed",
    "microservices",
    "scalability",
    "architecture",
    "sharding",
    "consensus",
    "multi-region",
    "high availability",
    "latency",
    "throughput",
)
MID_KW = (
    "rest",
    "graphql",
    "docker",
    "ci/cd",
    "postgresql",
    "redis",
    "aws",
    "gcp",
    "azure",
    "api",
    "backend",
    "frontend",
    "full stack",
)


def _find_safetensors(folder: str) -> Optional[str]:
    if not os.path.isdir(folder):
        return None
    for fname in os.listdir(folder):
        if fname.endswith(".safetensors"):
            return os.path.join(folder, fname)
    return None


def _project_section_block(resume_text: str) -> Tuple[str, bool]:
    """
    Returns (resume text slice for analysis, found_explicit_section).
    Only scores a project/portfolio slice so work-experience buzzwords don't inflate M3 heuristics.
    """
    lower = resume_text.lower()
    m = re.search(
        r"(^|\n)\s*(projects?|personal projects|academic projects|key projects|portfolio)\b",
        lower,
        re.MULTILINE | re.IGNORECASE,
    )
    if m:
        start = m.start()
        return resume_text[start : start + 4000], True
    return resume_text[:800], False


def heuristic_complexity(resume_text: str) -> ProjectComplexity:
    has_repo = bool(
        re.search(r"(https?://)?([\w.]*\.)?(github|gitlab)\.com/[^\s)]+", resume_text, re.I)
    )
    block, has_heading = _project_section_block(resume_text)
    bl = block.lower()

    if not has_heading and not has_repo:
        return ProjectComplexity(
            level="Basic",
            confidence=Confidence3(basic=0.78, intermediate=0.17, advanced=0.05),
            shap_keywords=ShapKeywords(basic=["projects", "github"], intermediate=[], advanced=[]),
            plain_explanation=(
                "Heuristic (no M3 weights): no Projects/Portfolio heading and no GitHub/GitLab link "
                "found — project score stays conservative."
            ),
        )

    adv = sum(1 for k in ADV_KW if k in bl)
    mid = sum(1 for k in MID_KW if k in bl)
    has_quant = bool(re.search(r"\b\d{2,}%|\b\d+x\b|latency|ms\b|qps|rps", bl))

    if has_heading and (adv >= 2 or (adv >= 1 and has_quant)):
        level = "Advanced"
        conf = Confidence3(basic=0.07, intermediate=0.18, advanced=0.75)
        expl = "Heuristic: strong depth signals inside the project section."
        sk = ShapKeywords(advanced=["scale", "architecture"], intermediate=["docker"], basic=[])
    elif has_repo or mid >= 2 or (has_heading and mid >= 1):
        level = "Intermediate"
        conf = Confidence3(basic=0.12, intermediate=0.68, advanced=0.20)
        expl = "Heuristic: repository link and/or moderate project-section detail."
        sk = ShapKeywords(advanced=[], intermediate=["repo", "stack"], basic=[])
    else:
        level = "Basic"
        conf = Confidence3(basic=0.70, intermediate=0.22, advanced=0.08)
        expl = "Heuristic: project evidence is thin — expand with stack, scope, and outcomes."
        sk = ShapKeywords(advanced=[], intermediate=[], basic=["description", "impact"])

    return ProjectComplexity(level=level, confidence=conf, shap_keywords=sk, plain_explanation=expl)


class ComplexityRunner:
    def __init__(self) -> None:
        self.device_id = 0 if torch.cuda.is_available() else -1
        self.loaded = False
        self.model = None
        self.tokenizer = None

        weights = _find_safetensors(COMPLEXITY_MODEL_PATH) if os.path.exists(COMPLEXITY_MODEL_PATH) else None
        if not weights:
            print(
                f"[INFO] Complexity model (M3) not found at {COMPLEXITY_MODEL_PATH} — using heuristics."
            )
            return
        try:
            print(f"[LOADING] Complexity (M3) from {COMPLEXITY_MODEL_PATH}")
            device = torch.device("cuda" if self.device_id >= 0 and torch.cuda.is_available() else "cpu")
            self.tokenizer = AutoTokenizer.from_pretrained(COMPLEXITY_MODEL_PATH, use_fast=True)
            self.model = AutoModelForSequenceClassification.from_pretrained(
                COMPLEXITY_MODEL_PATH, ignore_mismatched_sizes=False
            ).to(device)
            self.model.eval()
            self.loaded = True
            print("[OK] Complexity (M3) loaded.")
        except Exception as e:
            print(f"[WARN] Complexity load failed, using heuristics: {e}")
            self.model = None
            self.tokenizer = None

    def predict(self, resume_text: str) -> ProjectComplexity:
        if not self.loaded or self.model is None or self.tokenizer is None:
            return heuristic_complexity(resume_text)

        device = next(self.model.parameters()).device
        blk, _ = _project_section_block(resume_text)
        text = blk if len(blk.strip()) > 200 else resume_text[:2500]
        enc = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        )
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            logits = self.model(**enc).logits
        probs = torch.softmax(logits, dim=-1)[0].cpu().numpy()
        if probs.shape[0] < 3:
            return heuristic_complexity(resume_text)

        p0, p1, p2 = float(probs[0]), float(probs[1]), float(probs[2])
        idx = int(np.argmax(probs))
        labels = ("Basic", "Intermediate", "Advanced")
        level = labels[idx] if idx < 3 else "Intermediate"

        conf = Confidence3(basic=p0, intermediate=p1, advanced=p2)
        top = max(p0, p1, p2)
        if level == "Advanced":
            expl = (
                f"Project write-ups read as **advanced** depth (strongest class probability ≈ {top:.0%}). "
                "You show substantial technical scope and detail."
            )
        elif level == "Intermediate":
            expl = (
                f"Project write-ups read as **intermediate** complexity (strongest class probability ≈ {top:.0%}). "
                "Solid implementation detail; add more measurable impact or scale where possible."
            )
        else:
            expl = (
                f"Project narratives look **basic** relative to strong portfolios (strongest class probability ≈ {top:.0%}). "
                "Expand stack, scope, metrics, and outcomes."
            )
        sk = ShapKeywords(
            advanced=["learned weights"] if idx == 2 else [],
            intermediate=["learned weights"] if idx == 1 else [],
            basic=["learned weights"] if idx == 0 else [],
        )
        return ProjectComplexity(level=level, confidence=conf, shap_keywords=sk, plain_explanation=expl)
