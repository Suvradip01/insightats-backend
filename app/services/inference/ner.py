"""M1 — BERT NER: grouped entities + contract mapping."""

from __future__ import annotations

import os
import re
from typing import Dict, List, Optional

import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer, pipeline

from app.core.config import settings
from app.schemas.analyze import NerEntities
from app.services.constants import ALL_NER_ENTITY_TYPES

NER_MODEL_PATH = settings.NER_MODEL_DIR

EMAIL_RE = re.compile(r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}")
_JUNK = {":", ";", "-", ".", ",", "|", "(", ")", "[", "]", "—", "–", "/", "\\", "#"}
YOE_TEXT_RE = re.compile(r"(\d+)\+?\s*(?:years?|yrs?)", re.I)
GRAD_YEAR_RE = re.compile(r"\b(20[0-2]\d|19\d{2})\b")


def _find_safetensors(folder: str) -> Optional[str]:
    if not os.path.isdir(folder):
        return None
    for fname in os.listdir(folder):
        if fname.endswith(".safetensors"):
            return os.path.join(folder, fname)
    return None


def _clean_word(word: str) -> Optional[str]:
    w = word.strip()
    if len(w) <= 1 or w in _JUNK:
        return None
    return w


class NerRunner:
    def __init__(self) -> None:
        self.device_id = 0 if torch.cuda.is_available() else -1
        self.loaded = False
        self._pipe = None

        weights = _find_safetensors(NER_MODEL_PATH) if os.path.exists(NER_MODEL_PATH) else None
        if not weights:
            print(
                f"[WARN] NER model not found at {NER_MODEL_PATH}\n"
                "       -> Add fine-tuned weights to backend/models/ner_model/"
            )
            return
        try:
            print(f"[LOADING] NER model from {NER_MODEL_PATH}")
            ner_model = AutoModelForTokenClassification.from_pretrained(
                NER_MODEL_PATH, ignore_mismatched_sizes=False
            )
            ner_tok = AutoTokenizer.from_pretrained(NER_MODEL_PATH, use_fast=True)
            self._pipe = pipeline(
                "ner",
                model=ner_model,
                tokenizer=ner_tok,
                aggregation_strategy="first",
                device=self.device_id,
            )
            self.loaded = True
            print("[OK] NER (M1) loaded.")
        except Exception as e:
            print(f"[ERROR] NER load failed: {e}")

    def extract_grouped(self, resume_text: str) -> Dict[str, List[str]]:
        if not self.loaded or not self._pipe:
            return {}

        raw = self._pipe(resume_text[:8000])
        grouped: Dict[str, List[str]] = {etype: [] for etype in ALL_NER_ENTITY_TYPES}

        for ent in raw:
            if ent.get("score", 0) < 0.50:
                continue
            group = ent.get("entity_group", "")
            if group not in grouped:
                continue
            word = _clean_word(ent.get("word", ""))
            if word is None:
                continue
            if word.lower() not in [v.lower() for v in grouped[group]]:
                grouped[group].append(word)

        emails = EMAIL_RE.findall(resume_text)
        if emails:
            existing = [v.lower() for v in grouped.get("Email Address", [])]
            for em in emails:
                if em.lower() not in existing:
                    grouped.setdefault("Email Address", []).append(em)
                    existing.append(em.lower())

        return {k: v for k, v in grouped.items() if v}

    @staticmethod
    def to_contract(grouped: Dict[str, List[str]], resume_text: str) -> NerEntities:
        def first(key: str) -> str:
            vals = grouped.get(key) or []
            return vals[0] if vals else ""

        yoe_span = first("Years of Experience") or None
        if not yoe_span:
            m = YOE_TEXT_RE.search(resume_text)
            if m:
                yoe_span = m.group(0).strip()

        grad = first("Graduation Year") or None
        if not grad:
            gm = GRAD_YEAR_RE.search(resume_text)
            if gm:
                grad = gm.group(1)

        return NerEntities(
            name=first("Name"),
            email=first("Email Address"),
            skills=list(grouped.get("Skills") or []),
            designation=first("Designation"),
            degree=first("Degree"),
            college_name=first("College Name"),
            companies=list(grouped.get("Companies worked at") or []),
            location=first("Location"),
            yoe=yoe_span,
            grad_year=grad,
        )
