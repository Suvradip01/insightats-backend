"""Derive dimension scores (0–100) from M1/M2/M3 outputs — pure functions, easy to test."""

from __future__ import annotations

import re
from typing import Dict, List, Optional, Set, Tuple

from app.services.constants import STRUCTURE_ENTITY_GROUPS

# Common tech tokens for JD keyword mining when structured skills are missing
_TECH_LEXICON = {
    "python",
    "java",
    "sql",
    "react",
    "node",
    "nodejs",
    "aws",
    "docker",
    "kubernetes",
    "api",
    "git",
    "javascript",
    "typescript",
    "c++",
    "golang",
    "go",
    "linux",
    "cloud",
    "agile",
    "django",
    "flask",
    "fastapi",
    "mongodb",
    "postgres",
    "mysql",
    "redis",
    "tensorflow",
    "pytorch",
    "nlp",
    "machine learning",
    "ml",
    "ai",
}

YOE_RESUME_RE = re.compile(
    r"(?:^|\s)(\d+)\+?\s*(?:years?|yrs?)(?:\s+of)?(?:\s+experience)?",
    re.IGNORECASE,
)
JD_YOE_RE = re.compile(
    r"(?:minimum|min\.?|at\s+least|over|more\s+than)?\s*(\d+)\+?\s*(?:years?|yrs?)",
    re.IGNORECASE,
)


def _norm_skill(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip().lower())


def jd_term_found_in_resume(term: str, resume_text: str) -> bool:
    """
    True if the JD skill term appears in the resume (word-safe where needed).
    Avoids false positives like matching 'java' inside 'javascript'.
    """
    rt = resume_text.lower()
    t = term.strip().lower()
    if not t:
        return False
    if " " in t:
        return t in rt
    # High-confusion tokens
    if t == "java":
        if "javascript" in rt or "typescript" in rt:
            if not re.search(r"(?<![a-z])java(?![a-z])", rt):
                return False
        return bool(re.search(r"(?<![a-z])java(?![a-z])", rt))
    if t == "js":
        return bool(re.search(r"\bjs\b", rt)) or "javascript" in rt
    if t == "go":
        return bool(re.search(r"\bgo\b", rt)) or "golang" in rt
    if t == "ml":
        return bool(re.search(r"\bml\b", rt)) or "machine learning" in rt
    if t == "ai":
        return bool(re.search(r"\bai\b", rt)) or "artificial intelligence" in rt
    if t == "r":
        return bool(re.search(r"\br\b", rt))
    return bool(re.search(rf"(?<![a-z0-9]){re.escape(t)}(?![a-z0-9])", rt))


def infer_skills_from_resume_text(resume_text: str) -> List[str]:
    """When M1 skills list is empty, mine common tech tokens from raw text (safe matching)."""
    found: List[str] = []
    seen: Set[str] = set()
    for tok in sorted(_TECH_LEXICON, key=len, reverse=True):
        if tok in seen:
            continue
        if jd_term_found_in_resume(tok, resume_text):
            seen.add(tok)
            found.append(tok)
        if len(found) >= 40:
            break
    return found


def expand_skill_phrases(skills: List[str]) -> List[str]:
    """Split compound NER skills: 'Java / Spring' → separate tokens."""
    out: List[str] = []
    seen: Set[str] = set()
    for s in skills:
        for part in re.split(r"[/|,]+", s):
            p = part.strip()
            if len(p) <= 1:
                continue
            pl = p.lower()
            if pl not in seen:
                seen.add(pl)
                out.append(p)
    return out if out else list(skills)


def jd_required_skills(
    job_description: str,
    mandatory: Optional[List[str]] = None,
    preferred: Optional[List[str]] = None,
) -> List[str]:
    """Union of explicit lists + tech tokens found in JD text."""
    out: List[str] = []
    seen: Set[str] = set()

    for bucket in (mandatory or [], preferred or []):
        for s in bucket:
            n = _norm_skill(s)
            if n and n not in seen:
                seen.add(n)
                out.append(s.strip())

    jd_lower = job_description.lower()
    for tok in _TECH_LEXICON:
        # Never use raw substring (e.g. "go" matches inside "MongoDB").
        if tok in seen:
            continue
        if jd_term_found_in_resume(tok, job_description):
            seen.add(tok)
            out.append(tok)

    # Bigram "machine learning"
    if "machine learning" in jd_lower and "machine learning" not in seen:
        seen.add("machine learning")
        out.append("machine learning")

    return out


def extract_resume_yoe_years(resume_text: str, ner_yoe: Optional[str]) -> Optional[int]:
    """Prefer NER Years of Experience span, else regex on full resume."""
    if ner_yoe:
        m = re.search(r"(\d+)", ner_yoe)
        if m:
            return int(m.group(1))
    for mo in YOE_RESUME_RE.finditer(resume_text):
        return int(mo.group(1))
    return None


def extract_jd_required_years(job_description: str) -> Optional[int]:
    """Largest plausible 'N years' requirement from JD."""
    best: Optional[int] = None
    for mo in JD_YOE_RE.finditer(job_description):
        v = int(mo.group(1))
        if v <= 0 or v > 40:
            continue
        if best is None or v > best:
            best = v
    return best


def _resume_token_covers_jd_term(jd_norm: str, resume_skill_norm: str) -> bool:
    """Pairwise NER/JD match — avoids 'go'⊂'mongodb', 'java'⊂'javascript', etc."""
    if jd_norm == resume_skill_norm:
        return True
    if jd_norm in ("node", "nodejs"):
        return bool(re.search(r"node\.?js|nodejs|\bnode\b", resume_skill_norm))
    # Word-boundary safe (same rules as full-resume scan)
    return jd_term_found_in_resume(jd_norm, resume_skill_norm)


def compute_skill_overlap(
    resume_skills: List[str],
    jd_skills: List[str],
) -> Tuple[List[str], List[str]]:
    """Returns (matched JD requirements, missing JD requirements)."""
    rs = [_norm_skill(s) for s in resume_skills if _norm_skill(s)]
    matched: List[str] = []
    for req in jd_skills:
        rn = _norm_skill(req)
        if not rn:
            continue
        for skill in rs:
            if _resume_token_covers_jd_term(rn, skill):
                matched.append(req.strip())
                break

    seen_m = set()
    matched_dedup: List[str] = []
    for m in matched:
        k = _norm_skill(m)
        if k not in seen_m:
            seen_m.add(k)
            matched_dedup.append(m)

    matched_lower = {_norm_skill(x) for x in matched_dedup}
    missing: List[str] = []
    miss_seen: Set[str] = set()
    for req in jd_skills:
        rn = _norm_skill(req)
        if not rn:
            continue
        covered = any(
            _resume_token_covers_jd_term(rn, ml) for ml in matched_lower
        )
        if not covered and rn not in miss_seen:
            miss_seen.add(rn)
            missing.append(req.strip())

    return matched_dedup, missing


def refine_overlap_with_resume_text(
    resume_text: str,
    resume_skills: List[str],
    jd_skills: List[str],
) -> Tuple[List[str], List[str]]:
    """
    Reconcile NER/list overlap with full resume text so skills present in the CV
    are not marked missing when M1 tokenization splits labels badly.
    """
    matched, _ = compute_skill_overlap(resume_skills, jd_skills)
    mseen = {_norm_skill(x) for x in matched}

    for req in jd_skills:
        nk = _norm_skill(req)
        if not nk or nk in mseen:
            continue
        if jd_term_found_in_resume(req, resume_text):
            matched.append(req.strip())
            mseen.add(nk)

    seen_order: List[str] = []
    seen_k: Set[str] = set()
    for x in matched:
        k = _norm_skill(x)
        if k and k not in seen_k:
            seen_k.add(k)
            seen_order.append(x.strip())

    missing: List[str] = []
    for req in jd_skills:
        nk = _norm_skill(req)
        if nk and nk not in seen_k:
            missing.append(req.strip())

    return seen_order, missing


def structure_score_from_entities(
    grouped_entities: Dict[str, List[str]],
    resume_text: str = "",
) -> int:
    """
    M1: populated NER groups / 8 × 100, blended with section-heading detection.
    PDFs and student resumes often under-fill NER; headings (Projects, Education, …) still imply structure.
    """
    n = len(STRUCTURE_ENTITY_GROUPS)
    filled = sum(1 for g in STRUCTURE_ENTITY_GROUPS if grouped_entities.get(g))
    ner_score = int(round(100 * filled / n)) if n else 0

    if not resume_text.strip():
        return ner_score

    lower = resume_text.lower()
    section_hits = sum(
        1
        for pat in (
            r"\b(summary|objective|profile)\b",
            r"\b(skills?|technical skills?)\b",
            r"\b(projects?|portfolio)\b",
            r"\b(education|academic)\b",
            r"\b(experience|employment|work history|internship)\b",
            r"\b(certificates?|certifications?)\b",
        )
        if re.search(pat, lower)
    )
    # Floor from visible IA when NER misses blocks (common on exports)
    heading_floor = 0
    if section_hits >= 5:
        heading_floor = 72
    elif section_hits >= 4:
        heading_floor = 62
    elif section_hits >= 3:
        heading_floor = 50

    # Students: no employer NER is normal; education-heavy resumes still score
    if not grouped_entities.get("Companies worked at") and re.search(
        r"\b(bachelor|master|bca|mca|b\.?tech|m\.?tech|university|institute|college)\b",
        resume_text,
        re.I,
    ):
        ner_score = max(ner_score, min(55, int(round(100 * (filled + 1) / n))))

    return min(100, max(ner_score, heading_floor))


def project_score_from_level(level: str) -> int:
    """M3 level → radar axis (blueprint: Basic=40, Intermediate=70, Advanced=95)."""
    lv = (level or "Basic").strip().lower()
    if lv.startswith("adv"):
        return 95
    if lv.startswith("inter") or lv.startswith("medium"):
        return 70
    return 40


def derive_skill_score(
    matched_count: int,
    jd_skill_count: int,
    fit_score_0_1: Optional[float],
) -> int:
    """M1/M2-style axis: mostly JD↔resume skill evidence; M2 fit is a light tie-breaker."""
    if jd_skill_count <= 0:
        base = 70 if matched_count > 0 else 45
    else:
        base = int(round(100 * min(1.0, matched_count / jd_skill_count)))
    if fit_score_0_1 is not None:
        blended = 0.82 * base + 0.18 * (fit_score_0_1 * 100)
        return int(round(max(0, min(100, blended))))
    return max(0, min(100, base))


def derive_experience_score(
    resume_yoe: Optional[int],
    jd_required_years: Optional[int],
    exp_gap: int,
    fit_score_0_1: Optional[float],
) -> int:
    """
    M1 YOE vs JD required years, penalized by exp_gap (years short of requirement).
    """
    if jd_required_years is None:
        # JD did not state a minimum tenure — do not treat missing YOE as a strong negative (students/interns).
        score = 78 if resume_yoe is not None else 66
    else:
        if resume_yoe is None:
            score = 35
        else:
            if resume_yoe >= jd_required_years:
                score = 92
            else:
                gap = jd_required_years - resume_yoe
                score = max(25, 88 - gap * 12)

    score -= min(40, exp_gap * 8)
    score = max(0, min(100, score))

    if fit_score_0_1 is not None:
        score = int(round(0.72 * score + 0.28 * (fit_score_0_1 * 100)))
        score = max(0, min(100, score))

    return score


def compute_exp_gap_years(resume_yoe: Optional[int], jd_required_years: Optional[int]) -> int:
    if jd_required_years is None or resume_yoe is None:
        return 0
    return max(0, jd_required_years - resume_yoe)
