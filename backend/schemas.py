from pydantic import BaseModel, EmailStr
from typing import Optional
from datetime import datetime


# Auth
class SignupRequest(BaseModel):
    email: str
    name: str
    password: str


class LoginRequest(BaseModel):
    email: str
    password: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"


class KakaoLoginRequest(BaseModel):
    access_token: str


class GoogleLoginRequest(BaseModel):
    credential: str  # Google ID token (JWT)


class UserResponse(BaseModel):
    id: int
    email: Optional[str]
    name: str
    auth_provider: str
    created_at: datetime

    class Config:
        from_attributes = True


# Plans
class PlanInfo(BaseModel):
    name: str
    price: int
    items: int
    revisions: int
    label: str
    description: str


class PurchaseRequest(BaseModel):
    plan_name: str  # basic, plus, pro


class PurchaseResponse(BaseModel):
    id: int
    plan_name: str
    price: int
    items_remaining: int
    revisions_remaining: int
    created_at: datetime

    class Config:
        from_attributes = True


# Review
class ReviewRequest(BaseModel):
    company_name: str
    question: str
    text: str


class ReviseRequest(BaseModel):
    review_id: int
    instruction: str


class ReviewResponse(BaseModel):
    id: int
    company_name: str
    question: str
    original_text: str
    result: str
    revision_count: int
    created_at: datetime

    class Config:
        from_attributes = True
