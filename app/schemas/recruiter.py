from __future__ import annotations

from pydantic import BaseModel, Field


class RecruiterRegisterRequest(BaseModel):
    company: str = Field(..., min_length=2, max_length=120)
    username: str = Field(..., min_length=3, max_length=60)
    password: str = Field(..., min_length=8, max_length=200)


class RecruiterLoginRequest(BaseModel):
    username: str = Field(..., min_length=3, max_length=60)
    password: str = Field(..., min_length=8, max_length=200)


class RecruiterAuthResponse(BaseModel):
    token: str
    token_type: str = "bearer"
