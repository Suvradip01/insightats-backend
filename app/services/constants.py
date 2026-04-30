"""Shared NER labels and pipeline constants."""

# Fine-tuned NER model entity groups (aggregation_strategy="first" strips B-/I-).
ALL_NER_ENTITY_TYPES = [
    "Name",
    "Skills",
    "Designation",
    "Degree",
    "College Name",
    "Companies worked at",
    "Years of Experience",
    "Graduation Year",
    "Location",
    "Email Address",
    "Links",
]

# Eight fields used for structure_score (completeness of resume structure).
STRUCTURE_ENTITY_GROUPS = [
    "Name",
    "Email Address",
    "Skills",
    "Designation",
    "Degree",
    "College Name",
    "Companies worked at",
    "Location",
]

MATCHER_CLASS_LABELS = ("No Fit", "Partial Fit", "Strong Fit")
