import os
import json
import httpx
from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session
from openai import OpenAI

from database import engine, get_db, Base
from models import User, Purchase, Review
from schemas import (
    SignupRequest, LoginRequest, TokenResponse, UserResponse,
    KakaoLoginRequest, GoogleLoginRequest,
    PlanInfo, PurchaseRequest, PurchaseResponse,
    ReviewRequest, ReviseRequest, ReviewResponse,
)
from auth import hash_password, verify_password, create_access_token, get_current_user

KAKAO_JS_KEY = os.environ.get("KAKAO_JS_KEY", "")
GOOGLE_CLIENT_ID = os.environ.get("GOOGLE_CLIENT_ID", "")

# Create tables
Base.metadata.create_all(bind=engine)

app = FastAPI(title="자소서 AI 첨삭 API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Plan definitions ──
PLANS = {
    "basic": PlanInfo(
        name="basic", price=9900, items=2, revisions=20,
        label="Basic", description="가볍게 시작. 한두 문항을 제대로 완성.",
    ),
    "plus": PlanInfo(
        name="plus", price=19900, items=4, revisions=40,
        label="Plus", description="자소서 1세트 완성. 가장 많이 선택하는 플랜.",
    ),
    "pro": PlanInfo(
        name="pro", price=49900, items=10, revisions=120,
        label="Pro", description="여러 기업 자소서를 한 번에 완성.",
    ),
}

# ── GPT system prompt ──
SYSTEM_PROMPT = """당신은 한국 취업 시장 전문 자소서 첨삭 AI입니다.

## 역할
- 사용자가 제출한 자기소개서 문항을 분석하고, 문장별 진단 + 구체적 대안 문장을 제시합니다.
- 지원 기업의 인재상과 핵심 가치를 자동으로 반영합니다.

## 첨삭 규칙
1. **문장별 진단**: 각 문장이 왜 약한지 / 강한지를 구체적으로 진단합니다.
2. **대안 문장 제시**: "이 문장이 약합니다"로 끝나지 않고, 바로 적용 가능한 대안 문장을 제시합니다.
3. **기업 인재상 반영**: 지원 기업의 인재상·핵심 가치를 파악하여 첨삭에 반영합니다.
4. **구조 피드백**: 전체 자소서의 흐름, 논리 구조, 서사 완성도를 평가합니다.
5. **한국어 비즈니스 문체**: 번역체가 아닌 자연스러운 한국어 비즈니스 문체를 사용합니다.

## 출력 형식 (반드시 JSON)
다음 JSON 형식으로만 응답하세요. 다른 텍스트 없이 JSON만 출력합니다:

{
  "company_analysis": {
    "company": "기업명",
    "core_values": ["인재상 키워드 1", "인재상 키워드 2", ...],
    "reflection_strategy": "이 기업 인재상을 자소서에 어떻게 반영할지 전략"
  },
  "sentence_diagnosis": [
    {
      "original": "원본 문장",
      "issue": "문제점 진단",
      "severity": "high | medium | low",
      "alternative": "대안 문장"
    }
  ],
  "structure_feedback": {
    "flow_score": 1~10,
    "logic_score": 1~10,
    "specificity_score": 1~10,
    "company_fit_score": 1~10,
    "overall_comment": "전체 구조에 대한 피드백"
  },
  "revised_full_text": "전체 수정된 자소서 텍스트",
  "summary": "핵심 개선 사항 요약 (2-3문장)"
}"""

REVISE_SYSTEM_PROMPT = """당신은 한국 취업 시장 전문 자소서 첨삭 AI입니다.
사용자가 이전에 첨삭받은 자소서에 대해 추가 수정을 요청합니다.
이전 첨삭 맥락을 유지하면서, 사용자의 수정 요청을 반영하여 다시 첨삭합니다.

## 출력 형식 (반드시 JSON)
다음 JSON 형식으로만 응답하세요:

{
  "changes_made": [
    {
      "before": "수정 전 문장",
      "after": "수정 후 문장",
      "reason": "수정 이유"
    }
  ],
  "revised_full_text": "전체 수정된 자소서 텍스트",
  "summary": "이번 수정 사항 요약"
}"""


# ──────────────────────────────────────
# Auth endpoints
# ──────────────────────────────────────

@app.get("/api/auth/oauth-config")
def oauth_config():
    return {"kakao_js_key": KAKAO_JS_KEY, "google_client_id": GOOGLE_CLIENT_ID}


@app.post("/api/auth/kakao", response_model=TokenResponse)
def kakao_login(req: KakaoLoginRequest, db: Session = Depends(get_db)):
    # Verify token with Kakao API
    with httpx.Client() as client:
        res = client.get(
            "https://kapi.kakao.com/v2/user/me",
            headers={"Authorization": f"Bearer {req.access_token}"},
        )
    if res.status_code != 200:
        raise HTTPException(status_code=401, detail="카카오 인증에 실패했습니다.")

    kakao_data = res.json()
    kakao_id = str(kakao_data["id"])
    kakao_account = kakao_data.get("kakao_account", {})
    profile = kakao_account.get("profile", {})
    nickname = profile.get("nickname", "카카오 사용자")
    email = kakao_account.get("email")

    # Find existing user or create new
    user = db.query(User).filter(User.kakao_id == kakao_id).first()
    if not user:
        # Check if email already exists (link accounts)
        if email:
            user = db.query(User).filter(User.email == email).first()
            if user:
                user.kakao_id = kakao_id
                db.commit()

        if not user:
            user = User(
                email=email,
                name=nickname,
                kakao_id=kakao_id,
                auth_provider="kakao",
            )
            db.add(user)
            db.commit()
            db.refresh(user)

    token = create_access_token(user.id)
    return TokenResponse(access_token=token)


@app.post("/api/auth/google", response_model=TokenResponse)
def google_login(req: GoogleLoginRequest, db: Session = Depends(get_db)):
    # Verify Google access token via userinfo endpoint
    with httpx.Client() as client:
        res = client.get(
            "https://www.googleapis.com/oauth2/v3/userinfo",
            headers={"Authorization": f"Bearer {req.credential}"},
        )
    if res.status_code != 200:
        raise HTTPException(status_code=401, detail="구글 인증에 실패했습니다.")

    google_data = res.json()
    google_id = google_data.get("sub")
    if not google_id:
        raise HTTPException(status_code=401, detail="구글 사용자 정보를 가져올 수 없습니다.")

    email = google_data.get("email")
    name = google_data.get("name", "구글 사용자")

    # Find existing user or create new
    user = db.query(User).filter(User.google_id == google_id).first()
    if not user:
        if email:
            user = db.query(User).filter(User.email == email).first()
            if user:
                user.google_id = google_id
                db.commit()

        if not user:
            user = User(
                email=email,
                name=name,
                google_id=google_id,
                auth_provider="google",
            )
            db.add(user)
            db.commit()
            db.refresh(user)

    token = create_access_token(user.id)
    return TokenResponse(access_token=token)


@app.post("/api/auth/signup", response_model=TokenResponse)
def signup(req: SignupRequest, db: Session = Depends(get_db)):
    if db.query(User).filter(User.email == req.email).first():
        raise HTTPException(status_code=400, detail="이미 가입된 이메일입니다.")
    if len(req.password) < 6:
        raise HTTPException(status_code=400, detail="비밀번호는 6자 이상이어야 합니다.")

    user = User(
        email=req.email,
        name=req.name,
        password_hash=hash_password(req.password),
        auth_provider="email",
    )
    db.add(user)
    db.commit()
    db.refresh(user)

    token = create_access_token(user.id)
    return TokenResponse(access_token=token)


@app.post("/api/auth/login", response_model=TokenResponse)
def login(req: LoginRequest, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.email == req.email).first()
    if not user or not user.password_hash or not verify_password(req.password, user.password_hash):
        raise HTTPException(status_code=401, detail="이메일 또는 비밀번호가 올바르지 않습니다.")

    token = create_access_token(user.id)
    return TokenResponse(access_token=token)


@app.get("/api/auth/me", response_model=UserResponse)
def me(user: User = Depends(get_current_user)):
    return user


# ──────────────────────────────────────
# Plan endpoints
# ──────────────────────────────────────

@app.get("/api/plans", response_model=list[PlanInfo])
def list_plans():
    return list(PLANS.values())


@app.post("/api/plans/purchase", response_model=PurchaseResponse)
def purchase_plan(req: PurchaseRequest, user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    plan = PLANS.get(req.plan_name)
    if not plan:
        raise HTTPException(status_code=400, detail="존재하지 않는 플랜입니다.")

    # 실제 결제 연동은 추후 구현 (토스페이먼츠 등)
    # 여기서는 구매 기록만 생성
    purchase = Purchase(
        user_id=user.id,
        plan_name=plan.name,
        price=plan.price,
        items_remaining=plan.items,
        revisions_remaining=plan.revisions,
    )
    db.add(purchase)
    db.commit()
    db.refresh(purchase)
    return purchase


@app.get("/api/plans/my", response_model=list[PurchaseResponse])
def my_plans(user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    purchases = (
        db.query(Purchase)
        .filter(Purchase.user_id == user.id)
        .filter((Purchase.items_remaining > 0) | (Purchase.revisions_remaining > 0))
        .order_by(Purchase.created_at.desc())
        .all()
    )
    return purchases


# ──────────────────────────────────────
# Review endpoints (GPT API)
# ──────────────────────────────────────

def get_openai_client():
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="OpenAI API 키가 설정되지 않았습니다. 서버 관리자에게 문의하세요.")
    return OpenAI(api_key=api_key)


@app.post("/api/review", response_model=ReviewResponse)
def create_review(req: ReviewRequest, user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    # Find active purchase with remaining items
    purchase = (
        db.query(Purchase)
        .filter(Purchase.user_id == user.id, Purchase.items_remaining > 0)
        .order_by(Purchase.created_at.asc())
        .first()
    )
    if not purchase:
        raise HTTPException(status_code=403, detail="사용 가능한 플랜이 없습니다. 플랜을 구매해주세요.")

    # 800자 기준 문항 소진 계산
    char_count = len(req.text)
    items_needed = max(1, (char_count + 799) // 800)
    if purchase.items_remaining < items_needed:
        raise HTTPException(
            status_code=403,
            detail=f"문항 첨삭이 부족합니다. 필요: {items_needed}개, 남은 횟수: {purchase.items_remaining}개",
        )

    # Call GPT
    client = get_openai_client()
    user_message = f"""## 지원 기업: {req.company_name}
## 자소서 질문: {req.question}
## 자소서 내용:
{req.text}"""

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        temperature=0.7,
        response_format={"type": "json_object"},
    )

    result_text = response.choices[0].message.content

    # Deduct items
    purchase.items_remaining -= items_needed
    db.commit()

    # Save review
    review = Review(
        user_id=user.id,
        purchase_id=purchase.id,
        company_name=req.company_name,
        question=req.question,
        original_text=req.text,
        result=result_text,
    )
    db.add(review)
    db.commit()
    db.refresh(review)
    return review


@app.post("/api/review/revise", response_model=ReviewResponse)
def revise_review(req: ReviseRequest, user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    review = db.query(Review).filter(Review.id == req.review_id, Review.user_id == user.id).first()
    if not review:
        raise HTTPException(status_code=404, detail="첨삭 기록을 찾을 수 없습니다.")

    # Check revision quota
    purchase = (
        db.query(Purchase)
        .filter(Purchase.user_id == user.id, Purchase.revisions_remaining > 0)
        .order_by(Purchase.created_at.asc())
        .first()
    )
    if not purchase:
        raise HTTPException(status_code=403, detail="수정 횟수가 소진되었습니다.")

    client = get_openai_client()
    user_message = f"""## 이전 첨삭 결과:
{review.result}

## 원본 자소서:
{review.original_text}

## 지원 기업: {review.company_name}
## 자소서 질문: {review.question}

## 사용자 수정 요청:
{req.instruction}"""

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": REVISE_SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        temperature=0.7,
        response_format={"type": "json_object"},
    )

    result_text = response.choices[0].message.content

    # Update review
    review.result = result_text
    review.revision_count += 1
    purchase.revisions_remaining -= 1
    db.commit()
    db.refresh(review)
    return review


@app.get("/api/review/history", response_model=list[ReviewResponse])
def review_history(user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    reviews = (
        db.query(Review)
        .filter(Review.user_id == user.id)
        .order_by(Review.created_at.desc())
        .all()
    )
    return reviews


# ── Serve static files ──
STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")

@app.get("/")
def serve_index():
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))

@app.get("/review")
def serve_review():
    return FileResponse(os.path.join(STATIC_DIR, "review.html"))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
