from pydantic import BaseModel
from typing import List, Optional

class JobDescription(BaseModel):
    title: str
    description: str
    mandatory_skills: List[str] = []
    preferred_skills: List[str] = []
