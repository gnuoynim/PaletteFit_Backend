"""데모 시드 — 시연 시 톤 분석/옷 업로드 없이 바로 추천 흐름을 보여주기 위한 엔드포인트."""
from fastapi import APIRouter, Depends, Request

from limiter import limiter
from routers.tone import get_session_id, save_user_tone

router = APIRouter()


# 시연용 미리 분석된 톤 (여름 쿨톤 — 발표 자리에서 가장 흔하고 보기 좋음)
_DEMO_TONE = {
    "season": "여름 쿨톤",
    "undertone": "쿨톤",
    "brightness": "라이트",
    "saturation": "소프트",
    "confidence": "높음",
    "description": "차가운 핑크·블루 계열에서 피부가 가장 맑게 보이는 라이트 쿨톤입니다. 채도가 높지 않은 부드러운 색이 어울립니다.",
    "bestColors": [
        {"hex": "#EBBCC8", "name": "더스티로즈", "reason": "혈색을 가장 자연스럽게 살림"},
        {"hex": "#E6E6FA", "name": "라벤더", "reason": "쿨톤 피부에 화사함 부여"},
        {"hex": "#B0C4DE", "name": "소프트블루", "reason": "차가운 톤과 깔끔하게 어울림"},
        {"hex": "#D8BFD8", "name": "소프트모브", "reason": "은은한 명도 대비로 얼굴이 부드러워짐"},
        {"hex": "#FFE4E1", "name": "쉘핑크", "reason": "데일리로 가장 무난한 베이스"},
        {"hex": "#C8D9E6", "name": "스카이그레이", "reason": "쿨한 인상에 청량감 추가"},
        {"hex": "#9CB4CC", "name": "더스티블루", "reason": "윤곽을 또렷하게 살림"},
        {"hex": "#F0E6F0", "name": "베이비라일락", "reason": "밝은 채도로 화사한 인상"},
    ],
    "worstColors": [
        {"hex": "#FF4500", "name": "오렌지레드", "reason": "쿨톤 피부를 칙칙하게 가라앉힘"},
        {"hex": "#FF8C00", "name": "비비드오렌지", "reason": "얼굴이 누렇게 떠 보임"},
        {"hex": "#8B4513", "name": "다크브라운", "reason": "톤과 색온도 충돌로 무겁게 보임"},
        {"hex": "#DAA520", "name": "골든로드", "reason": "노란기가 피부 결점을 부각"},
    ],
    "qualityScores": {
        "lighting":    {"score": 5, "reason": "데모 시드 — 이상적인 자연광 가정"},
        "obscuration": {"score": 5, "reason": "얼굴이 충분히 보임"},
        "makeup":      {"score": 5, "reason": "메이크업 영향 없음"},
        "resolution":  {"score": 5, "reason": "고해상도"},
        "skinSample":  {"score": 5, "reason": "피부 노출 충분"},
        "background":  {"score": 5, "reason": "배경 반사 없음"},
    },
    "scores": {
        "여름 쿨톤": {"score": 8, "reason": "쿨 계열 드레이프 4장 모두에서 피부가 더 맑고 화사하게 보임"},
        "겨울 쿨톤": {"score": 5, "reason": "선명한 색은 다소 얼굴을 압도해 차분한 여름 쿨톤이 더 적합"},
    },
    "undertoneTest": {
        "warmScore": 4,
        "coolScore": 8,
        "reason": "쿨 계열 드레이프 3장 중 다수에서 피부가 더 맑고 다크서클이 덜 도드라짐",
    },
}


# 시연용 미리 분석된 옷장 5벌
_DEMO_CLOSET = [
    {"label": "오버사이즈 베이지 셔츠", "color": "베이지", "category": "상의", "fit": "애매함", "score": 58},
    {"label": "라벤더 니트", "color": "라벤더", "category": "상의", "fit": "잘 어울림", "score": 86},
    {"label": "라이트 워싱 데님 팬츠", "color": "라이트블루", "category": "하의", "fit": "잘 어울림", "score": 82},
    {"label": "체크 트위드 자켓", "color": "쿨베이지", "category": "아우터", "fit": "잘 어울림", "score": 78},
    {"label": "오렌지 H 라인 스커트", "color": "오렌지", "category": "하의", "fit": "비추천", "score": 32},
]


@router.post("/seed")
@limiter.limit("5/minute")
async def seed_demo(request: Request, session_id: str = Depends(get_session_id)):
    """현재 세션에 데모 톤·옷장을 한 번에 세팅."""
    save_user_tone(session_id, _DEMO_TONE)
    return {
        "ok": True,
        "tone": _DEMO_TONE,
        "closet": _DEMO_CLOSET,
        "message": "데모 시드 완료 — 여름 쿨톤 + 옷장 5벌이 세팅됐어요.",
    }


@router.delete("/seed")
@limiter.limit("5/minute")
async def clear_demo(request: Request, session_id: str = Depends(get_session_id)):
    """데모 세션 초기화 — 톤 파일 삭제."""
    from routers.tone import _tone_path
    p = _tone_path(session_id)
    if p.exists():
        p.unlink()
    return {"ok": True}
