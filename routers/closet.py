from fastapi import APIRouter, Depends, HTTPException, Request

from limiter import limiter
from schemas import ClothingRequest, ClothingResponse
from services.ai_service import analyze_clothing
from routers.tone import get_session_id, get_user_tone

router = APIRouter()


@router.post("/analyze", response_model=ClothingResponse)
@limiter.limit("10/minute")
async def analyze(request: Request, req: ClothingRequest, session_id: str = Depends(get_session_id)):
    user_tone = get_user_tone(session_id)
    if user_tone is None:
        raise HTTPException(status_code=400, detail="톤 분석을 먼저 진행해주세요")
    try:
        result = analyze_clothing(req.image, user_tone)
        return ClothingResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
