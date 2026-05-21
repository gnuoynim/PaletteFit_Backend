import json
import re
from pathlib import Path

from fastapi import APIRouter, Depends, Header, HTTPException, Request

from limiter import limiter
from schemas import ToneRequest, ToneUpdateRequest
from services.ai_service import analyze_tone

router = APIRouter()

# 세션별 영속 저장 (MVP — 인증 없이 브라우저당 UUID로 격리)
DATA_DIR = Path(__file__).parent.parent / "data" / "users"
DATA_DIR.mkdir(parents=True, exist_ok=True)

_SESSION_RE = re.compile(r"^[A-Za-z0-9_-]{8,64}$")
_DEFAULT_SESSION = "default"


def get_session_id(x_session_id: str | None = Header(default=None)) -> str:
    """X-Session-Id 헤더에서 세션 ID 추출. 없으면 default 세션 사용."""
    if x_session_id and _SESSION_RE.match(x_session_id):
        return x_session_id
    return _DEFAULT_SESSION


def _tone_path(session_id: str) -> Path:
    user_dir = DATA_DIR / session_id
    user_dir.mkdir(parents=True, exist_ok=True)
    return user_dir / "tone.json"


def get_user_tone(session_id: str) -> dict | None:
    """다른 라우터에서 세션별 톤 데이터를 조회할 때 사용."""
    path = _tone_path(session_id)
    if not path.exists():
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        return None


def save_user_tone(session_id: str, data: dict) -> None:
    with open(_tone_path(session_id), "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


@router.post("/analyze")
@limiter.limit("3/minute")
async def analyze(request: Request, req: ToneRequest, session_id: str = Depends(get_session_id)):
    try:
        result = analyze_tone(req.image)
        if result.get("error"):
            return result
        # drapeImages는 용량이 크므로 저장 시 제외
        save_data = {k: v for k, v in result.items() if k != "drapeImages"}
        save_user_tone(session_id, save_data)
        # 응답에는 drapeImages 포함
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/current")
@limiter.limit("60/minute")
async def get_current_tone(request: Request, session_id: str = Depends(get_session_id)):
    data = get_user_tone(session_id)
    if data is None:
        raise HTTPException(status_code=404, detail="톤 분석 결과가 없습니다")
    return data


@router.put("/update")
@limiter.limit("10/minute")
async def update_tone(request: Request, data: ToneUpdateRequest, session_id: str = Depends(get_session_id)):
    current = get_user_tone(session_id) or {}
    current.update(data.model_dump(exclude_none=True))
    save_user_tone(session_id, current)
    return current
