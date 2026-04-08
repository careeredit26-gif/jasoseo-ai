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
import base64
import uuid
from models import User, Purchase, Review, PaymentOrder
from schemas import (
    SignupRequest, LoginRequest, TokenResponse, UserResponse,
    KakaoLoginRequest, GoogleLoginRequest,
    PlanInfo, PurchaseRequest, PurchaseResponse,
    PaymentCreateRequest, PaymentConfirmRequest,
    ReviewRequest, ReviseRequest, ReviewResponse,
)
from auth import hash_password, verify_password, create_access_token, get_current_user

KAKAO_JS_KEY = os.environ.get("KAKAO_JS_KEY", "")
KAKAO_REST_KEY = os.environ.get("KAKAO_REST_KEY", "")
GOOGLE_CLIENT_ID = os.environ.get("GOOGLE_CLIENT_ID", "")
TOSS_CLIENT_KEY = os.environ.get("TOSS_CLIENT_KEY", "")
TOSS_SECRET_KEY = os.environ.get("TOSS_SECRET_KEY", "")

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

# ── GPT system prompt (v3.0 modular) ──
# Load prompt modules from prompt/ directory.
# Each numbered file is a self-contained module that gets composed into one
# comprehensive system prompt at startup.
PROMPT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "prompt")

_PROMPT_FILE_MAP = {
    "core":          "01_core_system.json",
    "question":      "02_question_engine.json",
    "company":       "03_company_db.json",
    "company_b":     "03b_company_db_expanded.json",
    "industry":      "03c_industry_framework.json",
    "finance":       "03d_finance.json",
    "it_game":       "03e_it_game_content.json",
    "manufacturing": "03f_manufacturing.json",
    "consumer":      "03g_consumer_logistics.json",
    "public":        "03h_public_sector.json",
    "writing":       "04_writing_engine.json",
    "diagnosis":     "05_diagnosis_scoring.json",
    "interview":     "06_interview_bridge.json",
    "output":        "07_output_format.json",
}


def _load_prompt_modules():
    modules = {}
    for key, filename in _PROMPT_FILE_MAP.items():
        path = os.path.join(PROMPT_DIR, filename)
        with open(path, "r", encoding="utf-8") as f:
            modules[key] = json.load(f)
    return modules


PROMPT_MODULES = _load_prompt_modules()


def _section(*lines):
    """Helper: join non-empty lines with newlines."""
    return "\n".join(str(l) for l in lines if l is not None and l != "")


def build_system_prompt():
    """Compose the comprehensive system prompt from all v3.0 modules."""
    core = PROMPT_MODULES["core"]
    qe   = PROMPT_MODULES["question"]
    cd   = PROMPT_MODULES["company"]
    we   = PROMPT_MODULES["writing"]
    ds   = PROMPT_MODULES["diagnosis"]
    ib   = PROMPT_MODULES["interview"]
    of   = PROMPT_MODULES["output"]

    parts = []

    # ─── ROLE & CORE PHILOSOPHY ───
    parts.append(core["role"])
    cp = core.get("core_philosophy", {})
    if cp:
        parts.append(_section(
            "\n## 핵심 철학",
            cp.get("principle", ""),
            cp.get("evaluation_mindset", ""),
            cp.get("differentiation_rule", ""),
        ))

    # ─── EVALUATOR SIMULATION ───
    es = core.get("evaluator_simulation", {})
    if es:
        parts.append("\n## 평가자 시뮬레이션")
        parts.append(es.get("instruction", ""))
        for stage, desc in (es.get("reading_pattern") or {}).items():
            parts.append(f"- {stage}: {desc}")
        if es.get("instant_reject_signals"):
            parts.append("\n### 즉시 탈락 신호")
            for s in es["instant_reject_signals"]:
                parts.append(f"- {s}")
        if es.get("positive_signals"):
            parts.append("\n### 긍정 신호")
            for s in es["positive_signals"]:
                parts.append(f"- {s}")

    # ─── CHARACTER COUNT STRATEGY ───
    ccs = core.get("character_count_strategy", {})
    if ccs:
        parts.append("\n## 글자수 전략")
        parts.append(ccs.get("instruction", ""))
        for sname, sval in (ccs.get("strategies") or {}).items():
            if isinstance(sval, dict):
                desc = sval.get("structure") or sval.get("description") or ""
                parts.append(f"- **{sname}**: {desc}")
            else:
                parts.append(f"- **{sname}**: {sval}")

    # ─── STEP 1: QUESTION CLASSIFICATION ───
    parts.append("\n## STEP 1: 질문 유형 판별")
    parts.append(qe.get("classification_instruction", ""))
    for type_name, info in qe.get("types", {}).items():
        parts.append(f"\n#### {type_name}")
        parts.append(f"- 평가 의도: {info.get('evaluator_intent', '')}")
        scores = info.get("what_evaluator_actually_scores", {})
        if scores.get("S_tier"):
            parts.append(f"- S등급: {scores['S_tier']}")
        if scores.get("C_tier"):
            parts.append(f"- C등급: {scores['C_tier']}")
        struct = info.get("required_structure", {})
        if struct:
            parts.append(f"- 필수 구조: {' → '.join(f'{k}({v})' for k, v in struct.items())}")
        fm = info.get("fatal_mistakes", [])
        if fm:
            parts.append(f"- 치명적 실수: {'; '.join(fm)}")
    cqh = qe.get("complex_question_handling", {})
    if cqh:
        parts.append(f"\n### 복합 문항 처리\n{cqh.get('instruction', '')}")

    # ─── STEP 2: COMPANY ANALYSIS ───
    parts.append("\n## STEP 2: 기업 분석")
    parts.append(cd.get("instruction", ""))

    # 2a) Flat company DBs (03 + 03b — same shape)
    def _render_flat_companies(db):
        for name, info in (db.get("companies") or {}).items():
            gv = ', '.join(info.get("group_values", []))
            parts.append(f"\n**{name}** [{gv}]")
            parts.append(f"  {info.get('evaluation_style', '')}")
            for div_name, div_info in (info.get("divisions") or {}).items():
                parts.append(
                    f"  - {div_name}: {div_info.get('what_they_want', '')} "
                    f"(톤: {div_info.get('tone_guide', '')})"
                )

    _render_flat_companies(cd)

    cb = PROMPT_MODULES["company_b"]
    parts.append(f"\n### 확장 대기업 그룹 DB — {cb.get('description', '')}")
    _render_flat_companies(cb)

    # 2b) Sector-grouped DBs (03d–03h)
    def _join(v):
        if isinstance(v, list):
            return '; '.join(str(x) for x in v)
        return str(v) if v is not None else ''

    def _render_sector_db(db, label):
        parts.append(f"\n### {label}")
        sc = db.get("sector_common")
        if isinstance(sc, dict):
            for k, v in sc.items():
                parts.append(f"- [공통:{k}] {_join(v)}")
        elif sc:
            parts.append(_join(sc))
        for sub_sector, comps in (db.get("companies") or {}).items():
            parts.append(f"\n#### {sub_sector}")
            if not isinstance(comps, dict):
                continue
            for cname, cinfo in comps.items():
                if not isinstance(cinfo, dict):
                    continue
                wt = cinfo.get("what_they_want", "")
                tg = cinfo.get("tone_guide", "")
                parts.append(f"- **{cname}**: {wt} (톤: {tg})")
                sh = cinfo.get("strategic_hooks")
                if sh:
                    parts.append(f"    · 전략 포인트: {_join(sh)}")

    for key, label in (
        ("finance",       "금융권 DB (은행/증권/보험/카드/자산운용)"),
        ("it_game",       "IT/게임/콘텐츠 DB"),
        ("manufacturing", "제조/화학/건설/방산/에너지/부품 DB"),
        ("consumer",      "유통/식품/뷰티/물류/항공/서비스 DB"),
        ("public",        "공기업/공공기관 DB"),
    ):
        _render_sector_db(PROMPT_MODULES[key], label)

    # 2c) Industry framework (fallback for unregistered companies)
    ind = PROMPT_MODULES["industry"]
    parts.append("\n### 산업 프레임워크 (DB 미등록 기업 대응)")
    if ind.get("usage_instruction"):
        parts.append(ind["usage_instruction"])
    for ind_name, ind_info in (ind.get("industry_profiles") or {}).items():
        if not isinstance(ind_info, dict):
            continue
        parts.append(f"\n**[{ind_name}]**")
        if ind_info.get("includes"):
            parts.append(f"  포함: {_join(ind_info['includes'])}")
        if ind_info.get("common_values"):
            parts.append(f"  공통 가치: {_join(ind_info['common_values'])}")
        if ind_info.get("key_emphasis"):
            parts.append(f"  핵심 강조: {_join(ind_info['key_emphasis'])}")
        if ind_info.get("strategic_trends"):
            parts.append(f"  최근 트렌드: {_join(ind_info['strategic_trends'])}")
        et = ind_info.get("essay_tips")
        if et:
            if isinstance(et, list):
                for t in et:
                    parts.append(f"    · {t}")
            else:
                parts.append(f"    · {et}")

    fcg = ind.get("foreign_company_guide", {})
    if fcg:
        parts.append("\n### 외국계 기업 가이드")
        if fcg.get("instruction"):
            parts.append(fcg["instruction"])
        adj = fcg.get("adjustments")
        if isinstance(adj, dict):
            for k, v in adj.items():
                parts.append(f"- {k}: {_join(v)}")
        elif isinstance(adj, list):
            for v in adj:
                parts.append(f"- {_join(v)}")

    cip = ind.get("company_inference_protocol")
    if cip:
        parts.append("\n### 기업 추론 프로토콜")
        if isinstance(cip, dict):
            for k, v in cip.items():
                parts.append(f"- {k}: {_join(v)}")
        else:
            parts.append(_join(cip))

    ucp = cd.get("unknown_company_protocol", {})
    if ucp:
        parts.append("\n### 미등록 기업 처리 프로토콜")
        for step_key, step_val in ucp.items():
            parts.append(f"- {step_key}: {step_val}")

    # ─── STEP 3: WRITING RULES ───
    parts.append("\n## STEP 3: 작성 규칙")
    parts.append("### 절대 규칙")
    for r in we.get("absolute_rules", []):
        parts.append(f"- {r}")
    parts.append("\n### 문장 규칙")
    for r in we.get("sentence_level_rules", []):
        parts.append(f"- {r}")

    # Anti-cliche engine (3 tiers)
    ace = we.get("anti_cliche_engine", {})
    if ace:
        parts.append(f"\n### 안티 클리셰 엔진\n{ace.get('instruction', '')}")
        for tier_key in ("tier_1_instant_kill", "tier_2_weak_signals", "tier_3_overused_structures"):
            tier = ace.get(tier_key, {})
            if not tier:
                continue
            parts.append(f"\n**[{tier_key}]** {tier.get('description', '')}")
            items = tier.get("expressions") or tier.get("patterns") or []
            for e in items:
                if isinstance(e, dict):
                    cliche = e.get("cliche") or e.get("pattern") or ""
                    why = e.get("why_bad", "")
                    alt = e.get("alternative_direction") or e.get("alternative") or ""
                    line = f"- '{cliche}'"
                    if why:
                        line += f" — {why}"
                    if alt:
                        line += f" / 대안: {alt}"
                    parts.append(line)
                else:
                    parts.append(f"- {e}")

    # Tone calibration (brief reference)
    tc = we.get("tone_calibration", {})
    if tc:
        parts.append(f"\n### 톤 캘리브레이션\n{tc.get('instruction', '')}")
        # tone_profile of each company is already given in STEP 2 indirectly
        # only include the formality scale summary here
        fs = tc.get("formality_scale", {})
        if isinstance(fs, dict) and fs:
            parts.append("기업별 격식도 척도(1=캐주얼 ~ 10=극도 격식):")
            for level, desc in list(fs.items())[:6]:
                parts.append(f"- {level}: {desc}")

    # Translation body detector
    tbd = we.get("translation_body_detector", {})
    if tbd:
        parts.append(f"\n### 번역체 감지\n{tbd.get('instruction', '')}")

    # ─── STEP 4: DIAGNOSIS & SCORING ───
    parts.append("\n## STEP 4: 진단 및 채점")
    ddr = ds.get("diagnosis_depth_rules", {})
    if ddr:
        parts.append(f"### 진단 깊이 (3단계)\n{ddr.get('instruction', '')}")
        parts.append(f"1) {ddr.get('level_1_what', '')}")
        parts.append(f"2) {ddr.get('level_2_why', '')}")
        parts.append(f"3) {ddr.get('level_3_how', '')}")

    sev = ds.get("severity_system", {})
    if sev:
        parts.append("\n### 심각도 체계")
        for level, info in sev.items():
            if isinstance(info, dict):
                parts.append(f"- **{level}**: {info.get('description', info.get('criteria', ''))}")
            else:
                parts.append(f"- **{level}**: {info}")

    sr = ds.get("scoring_rubric", {})
    if sr:
        parts.append(f"\n### 5차원 채점 루브릭\n{sr.get('instruction', '')}")
        for dim_name, dim_info in (sr.get("dimensions") or {}).items():
            if isinstance(dim_info, dict):
                desc = dim_info.get("description") or dim_info.get("what_it_measures", "")
                parts.append(f"- **{dim_name}**: {desc}")
            else:
                parts.append(f"- **{dim_name}**: {dim_info}")

    rfd = ds.get("red_flag_detection", {})
    if rfd:
        parts.append(f"\n### 레드플래그 감지\n{rfd.get('instruction', '')}")
        for flag_key, flag_val in rfd.items():
            if flag_key == "instruction":
                continue
            if isinstance(flag_val, list):
                for item in flag_val:
                    if isinstance(item, dict):
                        sig = item.get("signal") or item.get("description") or item.get("pattern", "")
                        parts.append(f"- [{flag_key}] {sig}")
                    else:
                        parts.append(f"- [{flag_key}] {item}")

    ip = ds.get("improvement_priority", {})
    if ip:
        parts.append(f"\n### 개선 우선순위\n{ip.get('instruction', '')}")

    # ─── STEP 5: INTERVIEW BRIDGE ───
    if ib:
        parts.append("\n## STEP 5: 면접 연계 검증")
        if ib.get("philosophy"):
            parts.append(ib["philosophy"])
        irc = ib.get("interview_readiness_check", {})
        if irc.get("instruction"):
            parts.append(f"\n{irc['instruction']}")
        dsd = ib.get("danger_sentence_detection", {})
        if dsd.get("instruction"):
            parts.append(f"\n### 위험 문장 감지\n{dsd['instruction']}")
        if ib.get("output_instruction"):
            parts.append(f"\n{ib['output_instruction']}")

    # ─── OUTPUT FORMAT ───
    parts.append("\n## 출력 형식 (반드시 JSON만 출력, 다른 텍스트 없이)")
    parts.append(json.dumps(of["initial_review"], ensure_ascii=False, indent=2))

    return "\n".join(parts)


SYSTEM_PROMPT = build_system_prompt()


# ── Revision system prompt ──
_rg = PROMPT_MODULES["writing"].get("revision_guidelines", {})
_rg_common = '\n'.join(
    f"- '{k}': {v}" for k, v in (_rg.get("common_requests") or {}).items()
)
REVISE_SYSTEM_PROMPT = f"""{PROMPT_MODULES["core"]["role"]}

{_rg.get('instruction', '')}

## 자주 요청되는 수정 유형
{_rg_common}

## 출력 형식 (반드시 JSON만 출력)
{json.dumps(PROMPT_MODULES["output"]["revision"], ensure_ascii=False, indent=2)}"""


# ──────────────────────────────────────
# Auth endpoints
# ──────────────────────────────────────

@app.get("/api/auth/oauth-config")
def oauth_config():
    return {"kakao_js_key": KAKAO_JS_KEY, "google_client_id": GOOGLE_CLIENT_ID}


@app.get("/api/auth/kakao/callback")
def kakao_callback(code: str, db: Session = Depends(get_db)):
    from fastapi.responses import HTMLResponse
    # Determine base URL
    base_url = os.environ.get("RENDER_EXTERNAL_URL", "")
    if not base_url:
        base_url = "http://localhost:8080"
    redirect_uri = f"{base_url}/api/auth/kakao/callback"

    # Exchange code for access token
    with httpx.Client() as client:
        token_res = client.post(
            "https://kauth.kakao.com/oauth/token",
            data={
                "grant_type": "authorization_code",
                "client_id": KAKAO_REST_KEY,
                "redirect_uri": redirect_uri,
                "code": code,
            },
            headers={"Content-Type": "application/x-www-form-urlencoded"},
        )
    if token_res.status_code != 200:
        return HTMLResponse("<script>alert('��카오 인증에 실패했습니다.');location.href='/';</script>")

    access_token = token_res.json().get("access_token")

    # Get user info
    with httpx.Client() as client:
        res = client.get(
            "https://kapi.kakao.com/v2/user/me",
            headers={"Authorization": f"Bearer {access_token}"},
        )
    if res.status_code != 200:
        return HTMLResponse("<script>alert('카카오 사용자 정보를 가져올 수 없습니다.');location.href='/';</script>")

    kakao_data = res.json()
    kakao_id = str(kakao_data["id"])
    kakao_account = kakao_data.get("kakao_account", {})
    profile = kakao_account.get("profile", {})
    nickname = profile.get("nickname", "카카오 사용자")
    email = kakao_account.get("email")

    user = db.query(User).filter(User.kakao_id == kakao_id).first()
    if not user:
        if email:
            user = db.query(User).filter(User.email == email).first()
            if user:
                user.kakao_id = kakao_id
                db.commit()
        if not user:
            user = User(email=email, name=nickname, kakao_id=kakao_id, auth_provider="kakao")
            db.add(user)
            db.commit()
            db.refresh(user)

    jwt_token = create_access_token(user.id)
    return HTMLResponse(f"<script>localStorage.setItem('token','{jwt_token}');location.href='/';</script>")


@app.post("/api/auth/google", response_model=TokenResponse)
def google_login(req: GoogleLoginRequest, db: Session = Depends(get_db)):
    # Verify Google ID token via tokeninfo endpoint
    with httpx.Client() as client:
        res = client.get(
            f"https://oauth2.googleapis.com/tokeninfo?id_token={req.credential}"
        )
    if res.status_code != 200:
        raise HTTPException(status_code=401, detail="구글 인증에 실패했습니다.")

    google_data = res.json()

    if GOOGLE_CLIENT_ID and google_data.get("aud") != GOOGLE_CLIENT_ID:
        raise HTTPException(status_code=401, detail="구글 인증 정보가 유효하지 않습니다.")

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
# Payment endpoints (Toss Payments)
# ──────────────────────────────────────

@app.get("/api/payment/toss-key")
def toss_client_key():
    return {"client_key": TOSS_CLIENT_KEY}


@app.post("/api/payment/create")
def create_payment(req: PaymentCreateRequest, user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    plan = PLANS.get(req.plan_name)
    if not plan:
        raise HTTPException(status_code=400, detail="존재하지 않는 플랜입니다.")

    order_id = f"order_{uuid.uuid4().hex[:16]}"
    order = PaymentOrder(
        order_id=order_id,
        user_id=user.id,
        plan_name=plan.name,
        amount=plan.price,
    )
    db.add(order)
    db.commit()

    return {
        "order_id": order_id,
        "amount": plan.price,
        "order_name": f"자소서 AI 첨삭 - {plan.label}",
        "customer_name": user.name,
        "customer_email": user.email,
    }


@app.post("/api/payment/confirm")
def confirm_payment(req: PaymentConfirmRequest, db: Session = Depends(get_db)):
    order = db.query(PaymentOrder).filter(PaymentOrder.order_id == req.order_id).first()
    if not order:
        raise HTTPException(status_code=404, detail="주문을 찾을 수 없습니다.")
    if order.status == "paid":
        raise HTTPException(status_code=400, detail="이미 처리된 결제입니다.")
    if order.amount != req.amount:
        raise HTTPException(status_code=400, detail="결제 금액이 일치하지 않습니다.")

    # Confirm with Toss Payments API
    auth = base64.b64encode(f"{TOSS_SECRET_KEY}:".encode()).decode()
    with httpx.Client() as client:
        res = client.post(
            "https://api.tosspayments.com/v1/payments/confirm",
            headers={
                "Authorization": f"Basic {auth}",
                "Content-Type": "application/json",
            },
            json={
                "paymentKey": req.payment_key,
                "orderId": req.order_id,
                "amount": req.amount,
            },
        )

    if res.status_code != 200:
        order.status = "failed"
        db.commit()
        error_msg = res.json().get("message", "결제 승인에 실패했습니다.")
        raise HTTPException(status_code=400, detail=error_msg)

    # Payment confirmed — activate plan
    order.status = "paid"
    order.payment_key = req.payment_key
    db.commit()

    plan = PLANS[order.plan_name]
    purchase = Purchase(
        user_id=order.user_id,
        plan_name=plan.name,
        price=plan.price,
        items_remaining=plan.items,
        revisions_remaining=plan.revisions,
    )
    db.add(purchase)
    db.commit()

    return {"status": "success", "plan_name": plan.name}


# ──────────────────────────────────────
# Review endpoints (GPT API)
# ──────────────────────────────────────

import re

def clean_chinese_chars(text):
    """Replace Chinese characters with Korean equivalents"""
    replacements = {
        '積累': '축적', '貢献': '기여', '貢獻': '기여', '努力': '노력',
        '經驗': '경험', '成長': '성장', '挑戰': '도전', '協業': '협업',
        '問題': '문제', '解決': '해결', '發展': '발전', '實現': '실현',
    }
    for cn, kr in replacements.items():
        text = text.replace(cn, kr)
    # Remove any remaining CJK Unified Ideographs (Chinese chars)
    text = re.sub(r'[\u4e00-\u9fff]+', '', text)
    return text


def get_llm_client():
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="Groq API 키가 설정되지 않았습니다.")
    return OpenAI(api_key=api_key, base_url="https://api.groq.com/openai/v1")


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
    client = get_llm_client()
    user_message = f"""## 지원 기업: {req.company_name}
## 자소서 질문: {req.question}
## 자소서 내용:
{req.text}"""

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        temperature=0.7,
        response_format={"type": "json_object"},
    )

    result_text = clean_chinese_chars(response.choices[0].message.content)

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

    client = get_llm_client()
    user_message = f"""## 이전 첨삭 결과:
{review.result}

## 원본 자소서:
{review.original_text}

## 지원 기업: {review.company_name}
## 자소서 질문: {review.question}

## 사용자 수정 요청:
{req.instruction}"""

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": REVISE_SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        temperature=0.7,
        response_format={"type": "json_object"},
    )

    result_text = clean_chinese_chars(response.choices[0].message.content)

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

@app.get("/companies.json")
def serve_companies():
    return FileResponse(os.path.join(STATIC_DIR, "companies.json"))

@app.get("/editing")
def serve_editing():
    return FileResponse(os.path.join(STATIC_DIR, "main.html"))

@app.get("/payment/success")
def serve_payment_success():
    return FileResponse(os.path.join(STATIC_DIR, "payment-success.html"))

@app.get("/payment/fail")
def serve_payment_fail():
    return FileResponse(os.path.join(STATIC_DIR, "payment-fail.html"))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
