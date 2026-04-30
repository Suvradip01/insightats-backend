"""M2 — RoBERTa 3-class resume vs JD matcher + optional SHAP explainability."""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

try:
    import shap
except ImportError:
    shap = None

from app.core.config import settings
from app.schemas.analyze import BreakdownProbs, FitResult, SkillSignals
from app.services.constants import MATCHER_CLASS_LABELS

MATCHER_MODEL_PATH = settings.MATCHER_MODEL_DIR


def _find_safetensors(folder: str) -> Optional[str]:
    if not os.path.isdir(folder):
        return None
    for fname in os.listdir(folder):
        if fname.endswith(".safetensors"):
            return os.path.join(folder, fname)
    return None


class MatcherRunner:
    def __init__(self) -> None:
        self.device_id = 0 if torch.cuda.is_available() else -1
        self.loaded = False
        self.model = None
        self.tokenizer = None
        self.partial_thr = 0.18
        self.strong_thr = 0.65

        weights = _find_safetensors(MATCHER_MODEL_PATH) if os.path.exists(MATCHER_MODEL_PATH) else None
        if not weights:
            print(
                f"[WARN] Matcher model not found at {MATCHER_MODEL_PATH}\n"
                "       -> Add fine-tuned RoBERTa to backend/models/matcher_model/"
            )
            return
        try:
            print(f"[LOADING] Matcher (M2) from {MATCHER_MODEL_PATH}")
            device = torch.device("cuda" if self.device_id >= 0 and torch.cuda.is_available() else "cpu")
            self.tokenizer = AutoTokenizer.from_pretrained(MATCHER_MODEL_PATH, use_fast=True)
            self.model = AutoModelForSequenceClassification.from_pretrained(
                MATCHER_MODEL_PATH, ignore_mismatched_sizes=False
            ).to(device)
            self.model.eval()
            self.loaded = True
            print("[OK] Matcher (M2) loaded.")

            thr_path = os.path.join(MATCHER_MODEL_PATH, "thresholds.json")
            if os.path.exists(thr_path):
                with open(thr_path, "r", encoding="utf-8") as f:
                    thr = json.load(f)
                self.partial_thr = float(thr.get("partial_thr", 0.18))
                self.strong_thr = float(thr.get("strong_thr", 0.65))
                print(f"[OK] Matcher thresholds: partial>={self.partial_thr} strong>={self.strong_thr}")
        except Exception as e:
            print(f"[ERROR] Matcher load failed: {e}")
            self.model = None
            self.tokenizer = None

    def predict_probs(
        self,
        resume_text: str,
        job_description: str,
        *,
        resume_for_encoding: Optional[str] = None,
    ) -> Optional[np.ndarray]:
        if not self.loaded or self.model is None or self.tokenizer is None:
            return None
        device = next(self.model.parameters()).device
        seq_a = (resume_for_encoding or resume_text)[:3500]
        enc = self.tokenizer(
            seq_a,
            job_description[:2200],
            return_tensors="pt",
            truncation="longest_first",
            max_length=512,
        )
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            logits = self.model(**enc).logits
        return torch.softmax(logits, dim=-1)[0].cpu().numpy()

    def build_fit_result(
        self,
        resume_text: str,
        job_description: str,
        skill_signals: SkillSignals,
        *,
        resume_for_encoding: Optional[str] = None,
    ) -> FitResult:
        probs = self.predict_probs(
            resume_text, job_description, resume_for_encoding=resume_for_encoding
        )
        if probs is None:
            return FitResult(
                label="Unknown",
                verdict="Matcher not loaded",
                fit_score=0.0,
                breakdown=BreakdownProbs(),
                skill_signals=skill_signals,
                domain_override=None,
            )

        p0, p1, p2 = float(probs[0]), float(probs[1]), float(probs[2])
        raw_fit = float(np.clip(0.5 * p1 + 1.0 * p2, 0.0, 1.0))
        idx = int(np.argmax(probs))
        label = MATCHER_CLASS_LABELS[idx] if idx < len(MATCHER_CLASS_LABELS) else "Partial Fit"
        # Single source of truth: headline matches argmax class (avoids "STRONG FIT — Partial Fit").
        verdict = {
            "No Fit": "NOT A FIT",
            "Partial Fit": "PARTIAL FIT",
            "Strong Fit": "STRONG FIT",
        }.get(label, "PARTIAL FIT")

        return FitResult(
            label=label,
            verdict=verdict,
            fit_score=raw_fit,
            breakdown=BreakdownProbs(
                p_no_fit=round(p0, 4),
                p_partial=round(p1, 4),
                p_strong=round(p2, 4),
            ),
            skill_signals=skill_signals,
            domain_override=None,
        )

    def shap_explain(
        self,
        resume_text: str,
        job_description: str,
        *,
        resume_for_encoding: Optional[str] = None,
    ) -> Tuple[str, List[Any], List[Any]]:
        if not self.loaded or self.model is None or self.tokenizer is None or shap is None:
            return "SHAP unavailable (install shap or load matcher).", [], []

        body = (resume_for_encoding or resume_text)[:800]
        cross_input = f"{body} [SEP] {job_description[:400]}"
        feedback = "Model evaluation complete."
        tokens_out: List[Any] = []
        values_out: List[Any] = []

        try:
            def _shap_fn(texts):
                results = []
                dev = next(self.model.parameters()).device
                for text in texts:
                    enc = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
                    enc = {k: v.to(dev) for k, v in enc.items()}
                    with torch.no_grad():
                        logits = self.model(**enc).logits
                    pr = torch.softmax(logits, dim=-1)[0].cpu().numpy()
                    fit = float(np.clip(0.5 * pr[1] + 1.0 * pr[2], 0.0, 1.0))
                    results.append([1.0 - fit, fit])
                return results

            explainer = shap.Explainer(_shap_fn, self.tokenizer)
            shap_values = explainer([cross_input])
            tokens = shap_values.data[0]
            values = shap_values.values[0]
            if len(tokens) > 0:
                v = values[:, 1] if (hasattr(values, "ndim") and values.ndim == 2) else values
                top_positive = tokens[int(np.asarray(v).argmax())]
                top_negative = tokens[int(np.asarray(v).argmin())]
                feedback = (
                    f"The term '{top_positive}' significantly boosted your match score. "
                    f"The absence of '{top_negative}' lowered it."
                )
                tokens_out = list(tokens)
                values_out = v.tolist() if hasattr(v, "tolist") else list(v)
        except Exception as e:
            feedback = f"SHAP analysis unavailable: {e}"

        return feedback, tokens_out, values_out
