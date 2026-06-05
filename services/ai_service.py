import os
import json
import math
import base64
import binascii
from io import BytesIO
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image
from google import genai
from google.genai import types

MODEL = "gemini-2.5-flash"


_client = None


def _get_client():
    global _client
    if _client is None:
        _client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
    return _client


def resize_image_base64(base64_str: str, max_size: int = 768) -> str:
    """Resize base64 image to reduce API cost. Handles PNG/RGBA by flattening onto white."""
    if base64_str.startswith("data:"):
        base64_str = base64_str.split(",", 1)[1]

    img_data = base64.b64decode(base64_str)
    img = Image.open(BytesIO(img_data))
    img.thumbnail((max_size, max_size))

    if img.mode != "RGB":
        if img.mode == "P":
            img = img.convert("RGBA")
        if img.mode in ("RGBA", "LA"):
            bg = Image.new("RGB", img.size, (255, 255, 255))
            mask = img.split()[-1]
            bg.paste(img, mask=mask)
            img = bg
        else:
            img = img.convert("RGB")

    buffer = BytesIO()
    img.save(buffer, format="JPEG", quality=80)
    return base64.b64encode(buffer.getvalue()).decode()


def _parse_json_response(text: str) -> dict | None:
    """Extract JSON from response text."""
    text = text.strip()
    if "```" in text:
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
        text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


# Step 1 drapes: gold vs silver for warm/cool determination
UNDERTONE_DRAPES = {
    "웜톤": [
        ((212, 175, 55), "소프트골드"),
        ((240, 210, 170), "웜베이지"),
        ((230, 190, 120), "허니베이지"),
    ],
    "쿨톤": [
        ((196, 196, 204), "소프트실버"),
        ((210, 215, 230), "쿨그레이"),
        ((220, 225, 235), "아이스그레이"),
    ],
}

# 시즌별 폴백 팔레트 (AI 팔레트 생성 실패 시에만 사용)
_FALLBACK_PALETTE = {
    "봄 웜톤": {
        "best": [
            {"hex": "#FF7F50", "name": "코랄"},
            {"hex": "#FFDAB9", "name": "피치"},
            {"hex": "#FFDF80", "name": "버터옐로우"},
            {"hex": "#FFB347", "name": "살구"},
        ],
        "worst": [
            {"hex": "#000000", "name": "블랙"},
            {"hex": "#1C1C1C", "name": "차콜"},
            {"hex": "#2F3A56", "name": "콜드네이비"},
        ],
    },
    "여름 쿨톤": {
        "best": [
            {"hex": "#EBBCC8", "name": "더스티로즈"},
            {"hex": "#E6E6FA", "name": "라벤더"},
            {"hex": "#B0C4DE", "name": "소프트블루"},
            {"hex": "#D8BFD8", "name": "소프트모브"},
        ],
        "worst": [
            {"hex": "#FF4500", "name": "오렌지레드"},
            {"hex": "#FF6600", "name": "비비드오렌지"},
            {"hex": "#8B4513", "name": "다크브라운"},
        ],
    },
    "가을 웜톤": {
        "best": [
            {"hex": "#CD853F", "name": "카멜"},
            {"hex": "#6B8E23", "name": "올리브"},
            {"hex": "#D2691E", "name": "테라코타"},
            {"hex": "#A07846", "name": "브론즈브라운"},
        ],
        "worst": [
            {"hex": "#FF69B4", "name": "핫핑크"},
            {"hex": "#00FFFF", "name": "시안"},
            {"hex": "#E6E6FA", "name": "라벤더"},
        ],
    },
    "겨울 쿨톤": {
        "best": [
            {"hex": "#4169E1", "name": "로얄블루"},
            {"hex": "#DC143C", "name": "체리레드"},
            {"hex": "#BA55D3", "name": "크리스탈퍼플"},
            {"hex": "#F5F5F5", "name": "쿨화이트"},
        ],
        "worst": [
            {"hex": "#F5DEB3", "name": "휘트베이지"},
            {"hex": "#FFDAB9", "name": "피치"},
            {"hex": "#DAA520", "name": "골든로드"},
        ],
    },
}


# ── 톤 분석 프롬프트 (system_instruction + 사용자 프롬프트 분리로 캐시 가능) ──

_TONE_SYSTEM_INSTRUCTION = """당신은 퍼스널 컬러 드레이핑 전문가입니다.

공통 판단 원칙
- 색 자체의 예쁨이 아니라, 색이 얼굴(피부·윤곽)을 더 깨끗하고 조화롭게 살리는지를 봅니다.
- 1장만 보고 단정하지 말고, 같은 계열 다수에서 일관되게 반복되는 패턴을 따릅니다.
- 조명·필터·메이크업의 영향을 감안하고, 차이가 작으면 억지로 단정하지 말고 "불확실"을 사용합니다.
- 출력은 설명 없이 JSON만 반환합니다.

평가 축: 피부 맑음·화사함, 다크서클·붉은기·잡티 완화, 얼굴 윤곽 선명도, 색-얼굴 조화.

Confidence 규칙: 두 후보 점수 차 ≥3 "높음", 2 "보통", ≤1 "낮음"."""


_STEP1_PROMPT = """입력: 원본 얼굴 1장 + 웜 계열 드레이프 3장 + 쿨 계열 드레이프 3장.

Task 1) 사진 품질 6축 1~5점 평가
- lighting (조명 균일·자연광), obscuration (가림 적음), makeup (가벼움),
  resolution (피부 결 비교 가능), skinSample (이마·볼·턱 노출), background (반사 적음).

Task 2) 웜3장군 vs 쿨3장군을 종합 비교해 어느 계열이 얼굴을 더 살리는지 판단
- warmScore, coolScore 각 1~10점.
- 차이 ≤1: undertone="불확실". 사진 품질이 너무 낮아 비교 자체가 어려움: "판별불가".

반례 가드
- 웜 드레이프에서 누렇게 뜨고 쿨에서 맑아 보이면 → 쿨톤. (반대도 마찬가지)
- 4장 중 1장만 좋아 보이고 나머지는 비슷 → "불확실".

출력 JSON 형식:
{
  "qualityScores": {
    "lighting":    {"score": 1-5, "reason": "한 줄"},
    "obscuration": {"score": 1-5, "reason": "한 줄"},
    "makeup":      {"score": 1-5, "reason": "한 줄"},
    "resolution":  {"score": 1-5, "reason": "한 줄"},
    "skinSample":  {"score": 1-5, "reason": "한 줄"},
    "background":  {"score": 1-5, "reason": "한 줄"}
  },
  "undertone": "웜톤" | "쿨톤" | "불확실" | "판별불가",
  "warmScore": 1-10,
  "coolScore": 1-10,
  "confidence": "높음" | "보통" | "낮음",
  "reason": "드레이핑 근거 1~2문장"
}"""


def _step2_prompt(undertone: str, season_a: str, season_b: str, warm_score: int, cool_score: int) -> str:
    return f"""1단계 결과: undertone={undertone} (warmScore={warm_score}, coolScore={cool_score}).
입력: 원본 얼굴 + {season_a} 드레이프 4장 + {season_b} 드레이프 4장.

Task 1) 4장씩 종합 비교해 어느 시즌이 얼굴을 더 살리는지 판단.

Task 2) 결정된 시즌과 원본 얼굴을 함께 보고, **이 사람에게 맞춤 컬러 팔레트**를 생성.
- best 8개, worst 4개. 각 항목은 {{"hex":"#RRGGBB","name":"한국어 색이름","reason":"왜 이 사람에게 (안)어울리는지 1줄"}}.
- 시즌 일반 색이 아니라 이 사람의 피부 명도·채도·웜쿨 강도까지 반영해 다양하게 분포시킬 것.
- 같은 시즌이라도 사람마다 결과가 달라야 함.
- season이 "불확실"이면 personalizedPalette.best/worst를 각각 빈 배열로 둘 것.

반례 가드: 4장 중 1장만 좋아 보이고 나머지는 비슷 → "불확실"로.

출력 JSON 형식:
{{
  "scores": {{
    "{season_a}": {{"score": 1-10, "reason": "4장 종합 평가 1줄"}},
    "{season_b}": {{"score": 1-10, "reason": "4장 종합 평가 1줄"}}
  }},
  "season": "{season_a}" | "{season_b}" | "불확실",
  "brightness": "라이트" | "딥" | "불확실",
  "saturation": "클리어" | "소프트" | "불확실",
  "confidence": "높음" | "보통" | "낮음",
  "description": "왜 이 시즌인지 2~3문장",
  "personalizedPalette": {{
    "best": [{{"hex":"#RRGGBB","name":"...","reason":"..."}}, ... 총 8개],
    "worst": [{{"hex":"#RRGGBB","name":"...","reason":"..."}}, ... 총 4개]
  }}
}}"""


def _palette_hexes(items) -> list[str]:
    """팔레트 항목 리스트에서 헥스 코드만 추출 (구버전 string[] 형태도 지원)."""
    out = []
    if not isinstance(items, list):
        return out
    for it in items:
        if isinstance(it, dict):
            h = str(it.get("hex", "")).strip()
        elif isinstance(it, str):
            h = it.strip()
        else:
            continue
        if h.startswith("#") and len(h) == 7:
            out.append(h.upper())
    return out


def _palette_names(items) -> list[str]:
    """팔레트 항목 리스트에서 색상 이름만 추출 (이름 없으면 헥스 코드)."""
    out = []
    if not isinstance(items, list):
        return out
    for it in items:
        if isinstance(it, dict):
            name = str(it.get("name", "")).strip() or str(it.get("hex", "")).strip()
        elif isinstance(it, str):
            name = it.strip()
        else:
            continue
        if name:
            out.append(name)
    return out


def _normalize_palette_items(items, max_count: int) -> list[dict]:
    """AI가 반환한 팔레트 항목을 정규화 (hex/name/reason 키 존재 보장)."""
    out = []
    if not isinstance(items, list):
        return out
    for it in items[:max_count]:
        if not isinstance(it, dict):
            continue
        hex_val = str(it.get("hex", "")).strip()
        if not hex_val.startswith("#") or len(hex_val) != 7:
            continue
        out.append({
            "hex": hex_val.upper(),
            "name": str(it.get("name", "")).strip(),
            "reason": str(it.get("reason", "")).strip(),
        })
    return out


# Step 2 drapes: season-specific colors (3 per season)
SEASON_DRAPES = {
    "봄 웜톤": [
        ((255, 127, 80), "코랄"),
        ((255, 218, 185), "피치"),
        ((255, 223, 128), "버터옐로우"),
        ((255, 179, 71), "살구"),
    ],
    "여름 쿨톤": [
        ((235, 188, 200), "더스티로즈"),
        ((230, 230, 250), "라벤더"),
        ((176, 196, 222), "소프트블루"),
        ((216, 191, 216), "소프트모브"),
    ],
    "가을 웜톤": [
        ((205, 133, 63), "카멜"),
        ((107, 142, 35), "올리브"),
        ((210, 105, 30), "테라코타"),
        ((160, 120, 70), "브론즈브라운"),
    ],
    "겨울 쿨톤": [
        ((65, 105, 225), "로얄블루"),
        ((220, 20, 60), "체리레드"),
        ((186, 85, 211), "크리스탈퍼플"),
        ((245, 245, 245), "쿨화이트"),
    ],
}


def _create_drape_image(img: Image.Image, color_rgb: tuple) -> bytes:
    """Overlay a colored drape on the lower portion of the face image (fast)."""
    base = img.copy().convert("RGBA")
    w, h = base.size

    drape_top = int(h * 0.55)
    drape_h = h - drape_top

    # Solid color band
    solid = Image.new("RGBA", (w, drape_h), (*color_rgb, 220))

    # Gradient fade at top edge
    fade_h = min(int(h * 0.08), drape_h)
    for row in range(fade_h):
        alpha = int(220 * row / fade_h)
        fade_strip = Image.new("RGBA", (w, 1), (*color_rgb, alpha))
        solid.paste(fade_strip, (0, row))

    base.paste(solid, (0, drape_top), solid)

    result = base.convert("RGB")
    buf = BytesIO()
    result.save(buf, format="JPEG", quality=85)
    return buf.getvalue()


def analyze_tone(image_base64: str) -> dict:
    """Analyze personal tone using 2-step professional draping method."""
    resized = resize_image_base64(image_base64)
    img_bytes = base64.b64decode(resized)
    img = Image.open(BytesIO(img_bytes))

    # ── Step 0: Face check ──
    face_check_prompt = """이미지에 사람의 얼굴이 있는지 확인하세요.
얼굴이 없으면: {"error": true, "message": "얼굴 사진을 업로드해주세요."}
얼굴이 있으면: {"error": false}"""

    face_res = _get_client().models.generate_content(
        model=MODEL,
        contents=[
            types.Content(
                role="user",
                parts=[
                    types.Part.from_bytes(data=img_bytes, mime_type="image/jpeg"),
                    types.Part.from_text(text=face_check_prompt),
                ],
            )
        ],
        config=types.GenerateContentConfig(
            temperature=0.1,
            max_output_tokens=200,
            thinking_config=types.ThinkingConfig(thinking_budget=0),
            response_mime_type="application/json",
        ),
    )

    face_result = _parse_json_response(face_res.text)
    if face_result and face_result.get("error"):
        return face_result

    # ── Step 1: Gold vs Silver → Warm or Cool ──
    step1_parts = [
        types.Part.from_text(text="[원본 사진]:"),
        types.Part.from_bytes(data=img_bytes, mime_type="image/jpeg"),
    ]

    for tone_name, colors in UNDERTONE_DRAPES.items():
        group_label = "웜 계열" if tone_name == "웜톤" else "쿨 계열"
        total = len(colors)

        for idx, (rgb, color_label) in enumerate(colors, start=1):
            d_bytes = _create_drape_image(img, rgb)
            step1_parts.append(
                types.Part.from_text(
                    text=f"[{group_label} 드레이프 {idx}/{total} - {color_label}]:"
                )
            )
            step1_parts.append(
                types.Part.from_bytes(data=d_bytes, mime_type="image/jpeg")
            )
    step1_parts.append(types.Part.from_text(text=_STEP1_PROMPT))

    step1_res = _get_client().models.generate_content(
        model=MODEL,
        contents=[types.Content(role="user", parts=step1_parts)],
        config=types.GenerateContentConfig(
            system_instruction=_TONE_SYSTEM_INSTRUCTION,
            temperature=0.2,
            max_output_tokens=1500,
            thinking_config=types.ThinkingConfig(thinking_budget=1024),
            response_mime_type="application/json",
        ),
    )

    step1_result = _parse_json_response(step1_res.text)
    if step1_result is None:
        return {"error": True, "message": "1단계 분석에 실패했습니다. 다시 시도해주세요."}

    undertone = step1_result.get("undertone")
    quality_scores = step1_result.get("qualityScores", {})

    # ── Step 2: Season draping within the determined undertone ──
    if undertone == "웜톤":
        candidates = ["봄 웜톤", "가을 웜톤"]
        other_candidates = ["여름 쿨톤", "겨울 쿨톤"]
    elif undertone == "쿨톤":
        candidates = ["여름 쿨톤", "겨울 쿨톤"]
        other_candidates = ["봄 웜톤", "가을 웜톤"]
    else:
        return {
            "error": False,
            "season": None,
            "undertone": undertone,
            "brightness": "불확실",
            "saturation": "불확실",
            "confidence": step1_result.get("confidence", "낮음"),
            "description": step1_result.get("reason", "언더톤 판별이 불확실해 시즌 분석을 진행하지 않았습니다."),
            "bestColors": [],
            "worstColors": [],
            "scores": {},
            "qualityScores": quality_scores,
            "drapeImages": {},
            "undertoneTest": {
                "warmScore": step1_result.get("warmScore", 5),
                "coolScore": step1_result.get("coolScore", 5),
                "reason": step1_result.get("reason", ""),
        },
    }

    season_a, season_b = candidates

    step2_parts = [
        types.Part.from_text(text="[원본 사진]:"),
        types.Part.from_bytes(data=img_bytes, mime_type="image/jpeg"),
    ]

    drape_b64 = {}
    for season in candidates:
        colors = SEASON_DRAPES[season]
        total = len(colors)

        for idx, (rgb, color_label) in enumerate(colors, start=1):
            d_bytes = _create_drape_image(img, rgb)
            step2_parts.append(
                types.Part.from_text(text=f"[{season} 드레이프 {idx}/{total} - {color_label}]:")
            )
            step2_parts.append(
                types.Part.from_bytes(data=d_bytes, mime_type="image/jpeg")
            )

        # 프론트 표시용 대표 드레이프
        drape_b64[season] = base64.b64encode(
            _create_drape_image(img, colors[0][0])
        ).decode()

    for season in other_candidates:
        colors = SEASON_DRAPES[season]
        drape_b64[season] = base64.b64encode(
            _create_drape_image(img, colors[0][0])
        ).decode()

    season_a, season_b = candidates
    step2_parts.append(
        types.Part.from_text(
            text=_step2_prompt(
                undertone=undertone,
                season_a=season_a,
                season_b=season_b,
                warm_score=step1_result.get("warmScore", 5),
                cool_score=step1_result.get("coolScore", 5),
            )
        )
    )

    step2_res = _get_client().models.generate_content(
        model=MODEL,
        contents=[types.Content(role="user", parts=step2_parts)],
        config=types.GenerateContentConfig(
            system_instruction=_TONE_SYSTEM_INSTRUCTION,
            temperature=0.2,
            max_output_tokens=3000,
            thinking_config=types.ThinkingConfig(thinking_budget=1024),
            response_mime_type="application/json",
        ),
    )

    step2_result = _parse_json_response(step2_res.text)
    if step2_result is None:
        return {"error": True, "message": "2단계 분석에 실패했습니다. 다시 시도해주세요."}

    scores = step2_result.get("scores", {})
    a_score = scores.get(season_a, {}).get("score", 0)
    b_score = scores.get(season_b, {}).get("score", 0)

    # 점수 차 기반 보정
    if abs(a_score - b_score) <= 1:
        final_season = None
        final_brightness = "불확실"
        final_saturation = "불확실"
        final_confidence = "낮음"
    else:
        final_season = step2_result.get("season", season_a)
        final_brightness = step2_result.get("brightness", "불확실")
        final_saturation = step2_result.get("saturation", "불확실")
        final_confidence = step2_result.get("confidence", "보통")

    # AI가 생성한 개인 맞춤 팔레트 우선, 실패 시 시즌별 폴백 사용
    palette = step2_result.get("personalizedPalette", {}) or {}
    best_colors = _normalize_palette_items(palette.get("best"), max_count=8)
    worst_colors = _normalize_palette_items(palette.get("worst"), max_count=4)

    if final_season and (not best_colors or not worst_colors):
        fallback = _FALLBACK_PALETTE.get(final_season, {"best": [], "worst": []})
        if not best_colors:
            best_colors = [{**c, "reason": ""} for c in fallback.get("best", [])]
        if not worst_colors:
            worst_colors = [{**c, "reason": ""} for c in fallback.get("worst", [])]

    excluded_seasons = {
        other_season: {
            "reason": f"1단계에서 {undertone}으로 판별되어 2단계 비교 대상에서 제외"
        }
        for other_season in other_candidates
    }

    result = {
        "season": final_season,
        "undertone": undertone,
        "brightness": final_brightness,
        "saturation": final_saturation,
        "confidence": final_confidence,
        "description": step2_result.get("description", ""),
        "bestColors": best_colors,
        "worstColors": worst_colors,
        "scores": scores,
        "excludedSeasons": excluded_seasons,
        "qualityScores": quality_scores,
        "drapeImages": drape_b64,
        "undertoneTest": {
            "warmScore": step1_result.get("warmScore", 5),
            "coolScore": step1_result.get("coolScore", 5),
            "reason": step1_result.get("reason", ""),
        },
    }

    return result


# =========================================================
# OpenCV + AI 하이브리드 옷 궁합 분석
# =========================================================

# 카테고리별 얼굴 근접도 가중치
_CATEGORY_WEIGHT = {
    "상의": 1.0, "아우터": 0.9, "원피스": 1.0,
    "하의": 0.7, "신발": 0.5, "악세서리": 0.4, "기타": 0.5,
}

# 색상 이름 팔레트 (RGB)
_COLOR_PALETTE = [
    ("블랙", (0, 0, 0)), ("화이트", (255, 255, 255)), ("그레이", (128, 128, 128)),
    ("네이비", (25, 35, 80)), ("블루", (60, 110, 220)), ("스카이블루", (135, 206, 235)),
    ("라벤더", (200, 170, 230)), ("퍼플", (128, 0, 128)),
    ("핑크", (255, 182, 193)), ("로즈핑크", (255, 150, 160)), ("핫핑크", (255, 105, 180)),
    ("살몬핑크", (250, 128, 114)), ("더스티핑크", (210, 150, 150)), ("라이트핑크", (255, 200, 200)),
    ("레드", (220, 20, 60)), ("버건디", (128, 0, 32)),
    ("코랄", (255, 153, 102)), ("오렌지", (255, 140, 0)),
    ("머스타드", (218, 165, 32)), ("옐로우", (255, 235, 120)),
    ("아이보리", (255, 245, 220)), ("베이지", (222, 200, 160)),
    ("카멜", (193, 154, 107)), ("브라운", (139, 69, 19)),
    ("카키", (107, 142, 35)), ("올리브", (128, 128, 0)),
    ("민트", (152, 255, 152)), ("그린", (34, 139, 34)), ("틸", (0, 128, 128)),
]

# 카테고리별 ROI (y_start%, y_end%, x_start%, x_end%)
_CATEGORY_ROI = {
    "상의": (0.05, 0.60, 0.10, 0.90),
    "아우터": (0.05, 0.75, 0.08, 0.92),
    "원피스": (0.05, 0.90, 0.12, 0.88),
    "하의": (0.40, 0.95, 0.10, 0.90),
    "신발": (0.55, 1.00, 0.05, 0.95),
    "악세서리": (0.15, 0.85, 0.15, 0.85),
    "기타": (0.08, 0.92, 0.08, 0.92),
}

# ── CIELAB 헬퍼 ──

def _rgb_to_lab(rgb: Tuple[int, int, int]) -> Tuple[float, float, float]:
    """RGB -> CIELAB (정규화된 L*a*b* 스케일)"""
    pixel = np.array([[[rgb[2], rgb[1], rgb[0]]]], dtype=np.uint8)
    lab = cv2.cvtColor(pixel, cv2.COLOR_BGR2LAB)[0][0]
    # OpenCV LAB: L=[0,255], a,b=[0,255] centered 128 → 표준 스케일로 변환
    return float(lab[0]) * 100.0 / 255.0, float(lab[1]) - 128.0, float(lab[2]) - 128.0


def _delta_e(rgb1: Tuple[int, int, int], rgb2: Tuple[int, int, int]) -> float:
    """CIE76 Delta-E (CIELAB 유클리드 거리)"""
    l1, a1, b1 = _rgb_to_lab(rgb1)
    l2, a2, b2 = _rgb_to_lab(rgb2)
    return math.sqrt((l1 - l2) ** 2 + (a1 - a2) ** 2 + (b1 - b2) ** 2)


# 사전 계산된 LAB 팔레트 (색 매칭 가속)
_COLOR_PALETTE_LAB = [(name, _rgb_to_lab(rgb)) for name, rgb in _COLOR_PALETTE]


def _is_skin_color(rgb: Tuple[int, int, int]) -> bool:
    """HSV + YCrCb 듀얼 체크로 피부색 판별"""
    pixel_bgr = np.array([[[rgb[2], rgb[1], rgb[0]]]], dtype=np.uint8)
    # HSV 체크
    hsv = cv2.cvtColor(pixel_bgr, cv2.COLOR_BGR2HSV)[0][0]
    h, s, v = float(hsv[0]), float(hsv[1]), float(hsv[2])
    hsv_ok = (0 <= h <= 25) and (20 <= s <= 180) and (80 <= v <= 255)
    # YCrCb 체크
    ycrcb = cv2.cvtColor(pixel_bgr, cv2.COLOR_BGR2YCrCb)[0][0]
    cr, cb = float(ycrcb[1]), float(ycrcb[2])
    ycrcb_ok = (133 <= cr <= 173) and (77 <= cb <= 127)
    return hsv_ok and ycrcb_ok


def _strip_b64_header(data: str) -> str:
    if "," in data and data.strip().startswith("data:"):
        return data.split(",", 1)[1]
    return data


def _b64_to_bgr(image_base64: str) -> np.ndarray:
    raw = base64.b64decode(_strip_b64_header(image_base64))
    arr = np.frombuffer(raw, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("이미지 디코딩 실패")
    return img


def _rgb_to_hex(rgb: Tuple[int, int, int]) -> str:
    return f"#{rgb[0]:02X}{rgb[1]:02X}{rgb[2]:02X}"


def _hex_to_rgb(hex_color: str) -> Tuple[int, int, int]:
    h = hex_color.lstrip("#").upper()
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)


def _nearest_color_name(rgb: Tuple[int, int, int]) -> str:
    """LAB Delta-E 기반 가장 가까운 색상명 매칭"""
    lab = _rgb_to_lab(rgb)
    best_name, best_d = "기타", float("inf")
    for name, ref_lab in _COLOR_PALETTE_LAB:
        d = math.sqrt((lab[0] - ref_lab[0]) ** 2 + (lab[1] - ref_lab[1]) ** 2 + (lab[2] - ref_lab[2]) ** 2)
        if d < best_d:
            best_d, best_name = d, name
    return best_name


def _classify_value(v: float) -> str:
    if v >= 190: return "light"
    if v <= 90: return "deep"
    return "mid"


def _classify_chroma(s: float) -> str:
    return "clear" if s >= 145 else "soft"


def _classify_undertone(h: float, rgb: Tuple[int, int, int]) -> str:
    """OpenCV HSV hue(0-179) 기반 언더톤 분류 — 완화된 버전"""
    if max(rgb) - min(rgb) < 18:
        return "neutral"
    # 0-25: 레드~오렌지~옐로우 (warm)
    # 26-44: 옐로우그린 (warm)
    # 45-49: 그린 경계 (neutral)
    # 50-135: 그린~시안~블루~바이올렛 (cool)
    # 136-164: 바이올렛~마젠타 경계 (neutral — 이전에 warm으로 잘못 분류됨)
    # 165-179: 딥 레드/마젠타 (warm)
    if (0 <= h <= 25) or (165 <= h <= 179):
        return "warm"
    if 26 <= h <= 44:
        return "warm"
    if 50 <= h <= 135:
        return "cool"
    return "neutral"


# ── OpenCV 색상 특징 추출 ──

def _extract_color_features(image_base64: str, category: str = "") -> Dict[str, Any]:
    """카테고리 기반 ROI + GrabCut + KMeans + 피부색 필터 + 가중 평균 대표색"""
    img = _b64_to_bgr(image_base64)
    h, w = img.shape[:2]

    # ── 1) 카테고리별 ROI 크롭 ──
    roi = _CATEGORY_ROI.get(category, (0.08, 0.92, 0.08, 0.92))
    y1, y2 = int(h * roi[0]), int(h * roi[1])
    x1, x2 = int(w * roi[2]), int(w * roi[3])
    cropped = img[y1:y2, x1:x2]

    # ── 2) GrabCut 전경 분리 ──
    ch2, cw2 = cropped.shape[:2]
    mask = np.zeros((ch2, cw2), np.uint8)
    rect = (max(1, int(cw2 * 0.06)), max(1, int(ch2 * 0.06)),
            max(2, int(cw2 * 0.88)), max(2, int(ch2 * 0.88)))
    bgd, fgd = np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64)
    try:
        cv2.grabCut(cropped, mask, rect, bgd, fgd, 5, cv2.GC_INIT_WITH_RECT)
        fg_mask = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 1, 0).astype("uint8")
    except Exception:
        fg_mask = np.zeros((ch2, cw2), dtype=np.uint8)
        cv2.ellipse(fg_mask, (cw2 // 2, ch2 // 2), (int(cw2 * 0.35), int(ch2 * 0.4)), 0, 0, 360, 1, -1)

    # ── 3) KMeans 클러스터링 (5~7개) ──
    pixels = cropped[fg_mask > 0] if np.any(fg_mask > 0) else cropped.reshape(-1, 3)
    if len(pixels) < 20:
        pixels = cropped.reshape(-1, 3)
    pixels = np.float32(pixels)
    n_clusters = min(7, max(3, len(pixels) // 10))
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 80, 0.1)
    _, labels, centers = cv2.kmeans(pixels, n_clusters, None, criteria, 8, cv2.KMEANS_PP_CENTERS)
    counts = np.bincount(labels.flatten(), minlength=n_clusters)
    total_px = float(counts.sum())

    # ── 4) 클러스터별 RGB + 비중 계산 ──
    clusters = []
    for i in range(n_clusters):
        bgr = tuple(map(int, centers[i]))
        rgb = (int(bgr[2]), int(bgr[1]), int(bgr[0]))
        weight = counts[i] / total_px
        clusters.append((rgb, weight))
    clusters.sort(key=lambda x: -x[1])  # 비중 내림차순

    # ── 5) 필터링: 피부색 / 너무 어두움 / 너무 밝음 / 무채색 제거 ──
    def _is_valid_clothing(rgb: Tuple[int, int, int]) -> bool:
        r, g, b = rgb
        brightness = (r + g + b) / 3.0
        diff = max(r, g, b) - min(r, g, b)
        # 너무 어둡거나 밝은 클러스터 제외 (순수 검/흰)
        if brightness < 25 or brightness > 245:
            return False
        # 피부색 제외
        if _is_skin_color(rgb):
            return False
        return True

    def _is_chromatic(rgb: Tuple[int, int, int]) -> bool:
        r, g, b = rgb
        diff = max(r, g, b) - min(r, g, b)
        brightness = (r + g + b) / 3.0
        if brightness < 30 or brightness > 240:
            return diff > 30
        return diff > 25

    valid = [(rgb, w) for rgb, w in clusters if _is_valid_clothing(rgb)]
    if not valid:
        valid = clusters  # 폴백: 필터링 결과 없으면 전체 사용

    # ── 6) 가중 평균 대표색 (상위 2~3개 클러스터) ──
    top_n = valid[:3]
    total_w = sum(w for _, w in top_n)
    if total_w > 0:
        avg_r = sum(rgb[0] * w for rgb, w in top_n) / total_w
        avg_g = sum(rgb[1] * w for rgb, w in top_n) / total_w
        avg_b = sum(rgb[2] * w for rgb, w in top_n) / total_w
        weighted_rgb = (int(round(avg_r)), int(round(avg_g)), int(round(avg_b)))
    else:
        weighted_rgb = clusters[0][0]

    # 대표 색상명은 가중 평균에서, secondary는 두 번째 클러스터에서
    dominant = weighted_rgb
    secondary = valid[1][0] if len(valid) > 1 else valid[0][0]

    # 유채색/무채색 구분 — 무채색이면 그대로 사용 (블랙, 그레이, 화이트 옷도 있으므로)
    if not _is_chromatic(dominant):
        # 무채색 옷: 가장 큰 클러스터를 그대로 사용
        dominant = clusters[0][0]

    # ── 7) HSV 통계 ──
    dom_bgr = np.array([[[dominant[2], dominant[1], dominant[0]]]], dtype=np.uint8)
    dom_hsv = cv2.cvtColor(dom_bgr, cv2.COLOR_BGR2HSV)[0][0]
    avg_h, avg_s, avg_v = float(dom_hsv[0]), float(dom_hsv[1]), float(dom_hsv[2])

    return {
        "dominantColor": _nearest_color_name(dominant),
        "dominantHex": _rgb_to_hex(dominant),
        "secondaryColor": _nearest_color_name(secondary),
        "secondaryHex": _rgb_to_hex(secondary),
        "value": _classify_value(avg_v),
        "chroma": _classify_chroma(avg_s),
        "undertoneHint": _classify_undertone(avg_h, dominant),
    }


# ── 코드 기반 적합도 점수 ──

def _min_color_dist(target_hex: str, candidates: List[str]) -> Optional[float]:
    """Delta-E(LAB) 기반 최소 색상 거리"""
    if not candidates:
        return None
    target_rgb = _hex_to_rgb(target_hex)
    dists = []
    for c in candidates:
        try:
            cand_rgb = _hex_to_rgb(c)
            dists.append(_delta_e(target_rgb, cand_rgb))
        except Exception:
            continue
    return min(dists) if dists else None


def _score_fit(features: Dict, category: str, user_tone: Optional[Dict]) -> Dict:
    item_hex = features["dominantHex"]
    item_ut = features["undertoneHint"]
    item_val = features["value"]
    item_chr = features["chroma"]

    if not user_tone:
        return {"score": 60, "fit": "애매함", "reasonTags": ["사용자 톤 정보 없음"]}

    weight = _CATEGORY_WEIGHT.get(category, 0.5)
    score = 50.0
    tags: List[str] = []

    user_ut = user_tone.get("undertone", "neutral")
    best = _palette_hexes(user_tone.get("bestColors", []))
    worst = _palette_hexes(user_tone.get("worstColors", []))

    # 1) best/worst 색상 거리
    best_d = _min_color_dist(item_hex, best)
    worst_d = _min_color_dist(item_hex, worst)

    # Delta-E 기준: ~2.3 = JND, ~10 = 비슷, ~25 = 눈에 띄게 다름
    if worst_d is not None and worst_d < 25:
        score -= 35
        tags.append("피해야 할 색상과 유사")
    elif best_d is not None and best_d < 22:
        score += 30
        tags.append("어울리는 색상과 유사")

    # 2) 언더톤
    ut_map = {"웜톤": "warm", "쿨톤": "cool"}
    user_ut_en = ut_map.get(user_ut, user_ut)

    if user_ut_en == "neutral" or item_ut == "neutral":
        score += 5
        tags.append("중립 계열")
    elif user_ut_en == item_ut:
        score += 20
        tags.append("언더톤 조화")
    else:
        penalty = 18 if category in {"상의", "원피스", "아우터"} else 8
        score -= penalty
        tags.append("언더톤 불일치")

    # 3) 명도
    user_val = user_tone.get("value", "mid")
    if user_val == item_val:
        score += 12
        tags.append("명도 조화")
    elif abs(["light", "mid", "deep"].index(user_val) - ["light", "mid", "deep"].index(item_val)) == 1:
        score += 4
        tags.append("명도 무난")
    else:
        score -= 8
        tags.append("명도 차이")

    # 4) 채도
    user_chr = user_tone.get("chroma", "soft")
    if user_chr == item_chr:
        score += 10
        tags.append("채도 조화")
    else:
        score -= 6
        tags.append("채도 차이")

    # 가중치 적용
    score = 50 + (score - 50) * weight
    score = max(0, min(100, round(score)))

    fit = "잘 어울림" if score >= 75 else ("애매함" if score >= 50 else "비추천")

    return {"score": score, "fit": fit, "reasonTags": tags, "altColors": best[:3]}


# ── AI: 카테고리 분류 (옷 종류만 판단) ──

def _classify_category_ai(image_b64: str) -> Dict[str, str]:
    """이미지에서 옷의 카테고리와 아이템명만 판단. 색상은 판단하지 않음."""
    img_bytes = base64.b64decode(_strip_b64_header(image_b64))
    prompt = """이미지에서 옷(의류)의 종류를 판단하세요.

규칙:
- 옷의 카테고리와 구체적인 아이템명만 판단하세요.
- 색상은 판단하지 마세요.
- 사람이 입고 있는 경우, 가장 눈에 띄는 의류 아이템 1개를 선택하세요.
- 옷만 있는 사진이면 해당 옷을 판단하세요.

카테고리 목록: 상의, 하의, 아우터, 원피스, 신발, 악세서리, 기타

JSON 형식: {"category": "카테고리", "label": "구체적 아이템명 (예: 니트, 슬랙스, 트렌치코트, 플리츠스커트)"}"""

    res = _get_client().models.generate_content(
        model=MODEL,
        contents=[types.Content(role="user", parts=[
            types.Part.from_bytes(data=img_bytes, mime_type="image/jpeg"),
            types.Part.from_text(text=prompt),
        ])],
        config=types.GenerateContentConfig(
            temperature=0.1, max_output_tokens=200,
            thinking_config=types.ThinkingConfig(thinking_budget=0),
            response_mime_type="application/json",
        ),
    )
    result = _parse_json_response(res.text)
    if not result:
        return {"category": "기타", "label": "기타 아이템"}
    cat = result.get("category", "기타")
    if cat not in _CATEGORY_WEIGHT:
        cat = "기타"
    return {
        "category": cat,
        "label": result.get("label", "기타 아이템"),
    }


# ── AI: 색상 검증 (OpenCV 추출 결과를 이미지와 대조) ──

def _verify_color_ai(image_b64: str, cv_color: str, cv_hex: str, category: str, label: str) -> Dict[str, str]:
    """OpenCV가 추출한 색상을 AI가 실제 이미지와 대조해서 검증/보정."""
    img_bytes = base64.b64decode(_strip_b64_header(image_b64))
    color_names = ", ".join(name for name, _ in _COLOR_PALETTE)
    prompt = f"""이 이미지의 {category}({label})의 실제 색상을 판단하세요.

컴퓨터 비전이 이 옷의 색상을 "{cv_color}" ({cv_hex})로 추출했습니다.

당신의 역할:
1. 이미지를 직접 보고 옷 자체의 실제 색상을 판단하세요.
2. 컴퓨터 비전 결과가 맞는지 확인하세요.
3. 틀렸다면 올바른 색상으로 수정하세요.

주의:
- 피부색, 배경색, 그림자는 무시하고 옷 원단의 색상만 판단하세요.
- 조명 영향을 고려해 옷의 본래 색상을 추정하세요.
- 검정 옷이 조명 때문에 회색이나 네이비로 보일 수 있음에 주의하세요.

색상은 반드시 아래 목록 중 하나:
{color_names}

JSON 형식: {{"color": "위 목록 중 하나", "corrected": true/false, "reason": "판단 근거 한 줄"}}"""

    res = _get_client().models.generate_content(
        model=MODEL,
        contents=[types.Content(role="user", parts=[
            types.Part.from_bytes(data=img_bytes, mime_type="image/jpeg"),
            types.Part.from_text(text=prompt),
        ])],
        config=types.GenerateContentConfig(
            temperature=0.1, max_output_tokens=200,
            thinking_config=types.ThinkingConfig(thinking_budget=0),
            response_mime_type="application/json",
        ),
    )
    result = _parse_json_response(res.text)
    if not result:
        return {"color": cv_color, "corrected": False}
    return {
        "color": result.get("color", cv_color),
        "corrected": result.get("corrected", False),
    }


# ── AI: 에이전트형 브리핑 생성 ──

def _generate_feedback_ai(features: Dict, ai_meta: Dict, fit: Dict, user_tone: Optional[Dict]) -> Dict[str, str]:
    tone_ctx = ""
    if user_tone:
        best_names = _palette_names(user_tone.get("bestColors", []))
        worst_names = _palette_names(user_tone.get("worstColors", []))
        tone_ctx = f"""
[사용자 톤]
- 시즌: {user_tone.get("season", "알 수 없음")}
- 언더톤: {user_tone.get("undertone", "")}
- 어울리는 색상: {", ".join(best_names)}
- 피해야 할 색상: {", ".join(worst_names)}"""

    alt_names = [_nearest_color_name(_hex_to_rgb(c)) for c in fit.get("altColors", []) if c]
    score = fit.get("score", 50)
    fit_label = fit.get("fit", "애매함")

    prompt = f"""당신은 PaletteFit의 옷 분석 결과 해설자입니다. 옷의 톤 궁합 분석 결과를 사용자에게 자연어로 풀어줍니다 (코디 추천 아님).
분석 결과를 단순 설명하지 말고, 사용자가 다음에 무엇을 하면 좋을지까지 제안하세요.
{tone_ctx}

[옷 분석]
- 카테고리: {ai_meta.get("category")} ({ai_meta.get("label")})
- 주요 색상: {features.get("dominantColor")} ({features.get("dominantHex")})
- 보조 색상: {features.get("secondaryColor")}
- 명도: {features.get("value")}, 채도: {features.get("chroma")}, 언더톤: {features.get("undertoneHint")}

[판정]
- 적합도: {fit_label} (점수: {score}/100)
- 판단 근거: {", ".join(fit.get("reasonTags", []))}
- 추천 대체 색상: {", ".join(alt_names)}

[출력 규칙]
- headline: 이 아이템이 잘 맞는지 한 줄 판단 (예: "톤은 맞지만 베스트는 아니에요", "이 아이템은 지금 톤에 잘 맞아요")
- reason: 왜 그렇게 판단했는지 핵심 이유 1문장 (색·채도·명도·톤 거리 등)
- tip: 이 아이템 색상에 대한 톤 관점의 짧은 메모 1문장 (코디 조합이나 매치 제안 금지)
- nextAction: 사용자가 다음에 할 수 있는 행동 제안 1문장 (예: "옷장에 추가됐으니 비슷한 색의 다른 옷도 확인해보세요" — 코디 추천 금지)
- 문장은 짧고 단정하게. 블로그체, 과한 감탄 금지.
- 반드시 반말(~해요, ~에요) 체로 작성.

JSON 형식:
{{"headline": "한 줄 판단", "reason": "핵심 이유", "tip": "톤 관점 메모", "nextAction": "다음 행동 제안"}}"""

    res = _get_client().models.generate_content(
        model=MODEL,
        contents=[types.Content(role="user", parts=[types.Part.from_text(text=prompt)])],
        config=types.GenerateContentConfig(
            temperature=0.4, max_output_tokens=350,
            thinking_config=types.ThinkingConfig(thinking_budget=0),
            response_mime_type="application/json",
        ),
    )
    result = _parse_json_response(res.text)
    if not result:
        return {
            "headline": "분석은 완료했지만 설명 생성에 실패했어요.",
            "reason": "", "tip": "", "nextAction": "",
        }
    return {
        "headline": result.get("headline", ""),
        "reason": result.get("reason", ""),
        "tip": result.get("tip", ""),
        "nextAction": result.get("nextAction", ""),
    }


# ── 최종 오케스트레이터 ──

def analyze_clothing(image_base64: str, user_tone: dict | None = None) -> dict:
    """
    개선된 파이프라인:
    1) 이미지 리사이즈
    2) AI: 카테고리 분류 (먼저!)
    3) OpenCV: 카테고리별 ROI 기반 색상 추출
    4) AI 색상 교차 검증
    5) 코드: 퍼스널 톤 적합도 점수 계산
    6) AI: 자연어 설명 생성
    """
    # 0) 이미지 리사이즈 (한 번만)
    resized_b64 = resize_image_base64(image_base64)

    # 1) AI 카테고리 분류 (옷 종류만, 먼저 실행)
    try:
        ai_meta = _classify_category_ai(resized_b64)
    except Exception:
        ai_meta = {"category": "기타", "label": "기타 아이템"}

    # 2) 카테고리 기반 OpenCV 색상 추출
    try:
        features = _extract_color_features(resized_b64, category=ai_meta["category"])
    except Exception:
        return {
            "category": ai_meta["category"], "label": ai_meta["label"],
            "color": "알 수 없음", "colorHex": "#888888",
            "fit": "애매함", "score": 50,
            "headline": "이미지를 분석하지 못했어요. 옷이 잘 보이는 사진으로 다시 시도해주세요.",
            "reason": "", "tip": "", "nextAction": "", "reasonTags": [],
        }

    # 3) AI 색상 검증 — OpenCV 결과를 이미지와 대조해서 보정
    try:
        color_check = _verify_color_ai(
            resized_b64,
            cv_color=features["dominantColor"],
            cv_hex=features["dominantHex"],
            category=ai_meta["category"],
            label=ai_meta["label"],
        )
        ai_color_name = color_check.get("color", "")
        if ai_color_name and ai_color_name != features["dominantColor"]:
            # AI가 보정한 색상의 RGB 찾기
            ai_rgb = None
            for name, rgb in _COLOR_PALETTE:
                if name == ai_color_name:
                    ai_rgb = rgb
                    break
            if ai_rgb:
                features["dominantColor"] = ai_color_name
                features["dominantHex"] = _rgb_to_hex(ai_rgb)
                ai_bgr = np.array([[[ai_rgb[2], ai_rgb[1], ai_rgb[0]]]], dtype=np.uint8)
                ai_hsv = cv2.cvtColor(ai_bgr, cv2.COLOR_BGR2HSV)[0][0]
                features["undertoneHint"] = _classify_undertone(float(ai_hsv[0]), ai_rgb)
                features["value"] = _classify_value(float(ai_hsv[2]))
                features["chroma"] = _classify_chroma(float(ai_hsv[1]))
    except Exception:
        pass  # 색상 검증 실패 시 OpenCV 결과 그대로 사용

    # 4) 코드 기반 점수 계산
    if user_tone and "value" not in user_tone:
        season = user_tone.get("season", "")
        if "봄" in season or "여름" in season:
            user_tone["value"] = "light"
        else:
            user_tone["value"] = "deep"
        if "봄" in season or "겨울" in season:
            user_tone["chroma"] = "clear"
        else:
            user_tone["chroma"] = "soft"

    fit_result = _score_fit(features, ai_meta["category"], user_tone)

    # 5) AI 에이전트형 브리핑 생성
    try:
        feedback = _generate_feedback_ai(features, ai_meta, fit_result, user_tone)
    except Exception:
        feedback = {"headline": "분석은 완료했지만 설명 생성에 실패했어요.", "reason": "", "tip": "", "nextAction": ""}

    return {
        "category": ai_meta["category"],
        "label": ai_meta["label"],
        "color": features["dominantColor"],
        "colorHex": features["dominantHex"],
        "secondaryColor": features.get("secondaryColor", ""),
        "secondaryHex": features.get("secondaryHex", ""),
        "fit": fit_result["fit"],
        "score": fit_result["score"],
        "reasonTags": fit_result["reasonTags"],
        "headline": feedback["headline"],
        "reason": feedback["reason"],
        "tip": feedback["tip"],
        "nextAction": feedback["nextAction"],
        "altColors": fit_result.get("altColors", []),
    }


_AGENT_DECIDE_SYSTEM = """당신은 PaletteFit AI 스타일 에이전트입니다.

사용자가 보낸 사진/메시지를 보고 어떤 도구를 호출할지 신중하게 결정합니다.
도구 선택 시 항상 reasoning과 confidence를 함께 출력하세요 — 사용자에게 판단 과정이 노출됩니다.

판단 원칙:
1. 사진의 주요 피사체가 무엇인지 먼저 식별 (얼굴 vs 옷 vs 둘 다 vs 둘 다 아님).
2. 사용자 메시지가 의도를 명시했다면 (예: "이 옷 어울려?", "내 톤 알려줘") 그것을 우선.
3. 메시지가 비어있거나 모호하면 사진의 피사체로 판단.
4. 사진 품질이 매우 낮거나(흐릿/너무 어두움/얼굴·옷 식별 불가) 판단 근거가 부족하면 request_clarification.
5. 사용자가 옷장에 이미 분석한 옷이 있고 새 옷이 점수가 낮게 나올 가능성이 보이면 find_better_color도 고려.

confidence 기준:
- "높음": 피사체가 명확하고 의도가 분명
- "보통": 약간의 모호함은 있으나 가장 합리적인 선택이 분명
- "낮음": 모호하지만 그래도 진행 — 사용자에게 결과 후 확인 옵션 줄 것"""


def agent_decide(message: str, image_b64: str, has_tone: bool, user_tone: dict | None, history: list, closet_context: str = "") -> dict:
    """AI Agent: Gemini function calling으로 이미지 의도를 판단한다."""

    tone_status = "미분석"
    if user_tone:
        tone_status = f"분석 완료 ({user_tone.get('season', '알수없음')}, undertone={user_tone.get('undertone', '')})"

    user_context = f"[사용자 상태]\n- 퍼스널 톤: {tone_status}\n"
    if closet_context:
        user_context += f"- {closet_context}\n"

    contents = []
    for h in history[-12:]:
        role = "user" if h["role"] == "user" else "model"
        contents.append(types.Content(role=role, parts=[types.Part.from_text(text=h["content"])]))

    resized = resize_image_base64(image_b64)
    img_bytes = base64.b64decode(resized)
    user_msg = message.strip() if message else "(메시지 없음 — 사진만 업로드)"
    parts = [
        types.Part.from_text(text=user_context),
        types.Part.from_bytes(data=img_bytes, mime_type="image/jpeg"),
        types.Part.from_text(text=f"[사용자 메시지]\n{user_msg}\n\n위 사진과 메시지를 보고 가장 적합한 도구를 reasoning과 confidence와 함께 호출하세요."),
    ]
    contents.append(types.Content(role="user", parts=parts))

    # 모든 도구가 reasoning + confidence를 강제로 받게 함 → 판단 과정이 항상 응답에 담김
    decision_params = types.Schema(
        type="OBJECT",
        properties={
            "reasoning": types.Schema(
                type="STRING",
                description="이 도구를 선택한 이유 — 사진에서 본 것과 사용자 의도를 1~2문장으로 (사용자에게 표시됨)",
            ),
            "confidence": types.Schema(
                type="STRING",
                enum=["높음", "보통", "낮음"],
                description="판단 확신도",
            ),
        },
        required=["reasoning", "confidence"],
    )

    func_decls = [
        types.FunctionDeclaration(
            name="analyze_tone",
            description="얼굴/셀카 사진으로 퍼스널 컬러 톤을 진단합니다. 사진의 주요 피사체가 사람 얼굴일 때 사용.",
            parameters=decision_params,
        ),
    ]
    if has_tone:
        func_decls.append(types.FunctionDeclaration(
            name="analyze_clothing",
            description="옷/의류 사진과 사용자 퍼스널 톤의 궁합을 분석합니다. 사진의 주요 피사체가 옷 단독(행거·평면샷·쇼핑몰 컷)일 때 사용.",
            parameters=decision_params,
        ))
        func_decls.append(types.FunctionDeclaration(
            name="find_better_color",
            description="사용자가 '이 옷의 더 잘 어울리는 색을 찾아줘' 같이 색상 대안을 명시적으로 요청할 때 사용.",
            parameters=decision_params,
        ))

    func_decls.append(types.FunctionDeclaration(
        name="request_clarification",
        description="사진이 흐림/너무 어두움/주제 불분명/얼굴과 옷이 모두 없음 등 분석 자체가 어려울 때, 사용자에게 어떤 사진을 다시 올려달라고 요청합니다.",
        parameters=types.Schema(
            type="OBJECT",
            properties={
                "reasoning": types.Schema(type="STRING", description="왜 분석이 어려운지 1문장"),
                "confidence": types.Schema(type="STRING", enum=["높음", "보통", "낮음"]),
                "ask": types.Schema(type="STRING", description="사용자에게 보낼 안내 메시지 (어떤 사진을 다시 올려달라고)"),
            },
            required=["reasoning", "confidence", "ask"],
        ),
    ))

    response = _get_client().models.generate_content(
        model=MODEL,
        contents=contents,
        config=types.GenerateContentConfig(
            system_instruction=_AGENT_DECIDE_SYSTEM,
            temperature=0.1,
            max_output_tokens=600,
            tools=[types.Tool(function_declarations=func_decls)],
            thinking_config=types.ThinkingConfig(thinking_budget=512),
        ),
    )

    if response.candidates and response.candidates[0].content.parts:
        for part in response.candidates[0].content.parts:
            if hasattr(part, "function_call") and part.function_call:
                fc = part.function_call
                args = dict(fc.args) if fc.args else {}
                return {
                    "intent": fc.name,
                    "reasoning": str(args.get("reasoning", "")).strip(),
                    "confidence": str(args.get("confidence", "보통")).strip(),
                    "ask": str(args.get("ask", "")).strip(),
                }

    return {
        "intent": "chat",
        "reasoning": "이미지에서 명확한 의도를 읽지 못했습니다.",
        "confidence": "낮음",
        "text": (response.text or "").strip(),
    }


# =========================================================
# Reflection·Tool Chaining (Level 3 패턴)
# =========================================================

_AGENT_REFLECT_SYSTEM = """당신은 PaletteFit 에이전트의 후속 액션 결정자(Reflector)입니다.
방금 옷 궁합 분석이 끝난 직후, 사용자가 추가로 알면 좋을 정보를 자율적으로 판단합니다.

판단 원칙:
1. 결과가 비추천(점수 < 40)이면 → 거의 항상 find_better_color (사용자가 다음에 뭘 입어야 할지 막막함)
2. 결과가 잘 어울림(점수 >= 65)이고 옷장에 같은 카테고리·비슷한 색상이 이미 있으면 → suggest_closet_match (중복 구매 방지 안내)
3. 결과가 애매함(40~64)이면 → no_chain (사용자 결정에 맡김 — 자동 권유는 과함)
4. 옷장 정보가 없거나 비어있으면 → suggest_closet_match 절대 X
5. 확신이 없거나 한 도구로 분명히 떨어지지 않으면 → no_chain (자동화는 신중히)

체이닝 안 하는 게 더 자연스러우면 주저하지 말고 no_chain. 모든 결과에 무언가 권유할 필요 없음.

출력은 반드시 JSON. reasoning은 사용자에게 thinking 카드로 노출됨."""


def agent_reflect(clothing_result: dict, user_tone: dict | None, closet_summary: str) -> dict:
    """옷 분석 결과를 보고 후속 도구 호출 여부를 자율 결정.
    Returns: {"chain": bool, "tool": "find_better_color"|"suggest_closet_match"|"none", "reasoning": str}
    """
    season = (user_tone or {}).get("season", "알수없음")
    prompt = f"""[방금 옷 분석 결과]
- 카테고리: {clothing_result.get('category', '?')}
- 라벨: {clothing_result.get('label', '?')}
- 색상: {clothing_result.get('color', '?')} ({clothing_result.get('colorHex', '?')})
- 적합도: {clothing_result.get('fit', '?')}
- 점수: {clothing_result.get('score', 0)}/100

[사용자 톤] {season}

[옷장 요약]
{closet_summary or '(옷장 비어있음)'}

위 결과를 보고 다음 중 하나로 자동 진행할까요?
- find_better_color: 비슷한 핏의 더 잘 받는 색 즉시 제시 (낮은 점수에 유용)
- suggest_closet_match: 옷장의 비슷한 옷을 보여주고 중복 고려 안내 (잘 받는 옷이지만 옷장에 비슷한 게 있을 때)
- none: 체이닝 없이 사용자 결정에 맡김 (애매하거나 확신 없을 때)

출력 JSON:
{{
  "chain": true/false,
  "tool": "find_better_color" | "suggest_closet_match" | "none",
  "reasoning": "왜 이 결정을 했는지 1줄 (사용자에게 보임)"
}}"""

    try:
        res = _get_client().models.generate_content(
            model=MODEL,
            contents=[types.Content(role="user", parts=[types.Part.from_text(text=prompt)])],
            config=types.GenerateContentConfig(
                system_instruction=_AGENT_REFLECT_SYSTEM,
                temperature=0.1,
                max_output_tokens=300,
                thinking_config=types.ThinkingConfig(thinking_budget=0),
                response_mime_type="application/json",
            ),
        )
        parsed = _parse_json_response(res.text) or {}
        chain = bool(parsed.get("chain", False))
        tool = str(parsed.get("tool", "none")).strip()
        if tool not in ("find_better_color", "suggest_closet_match", "none"):
            tool = "none"
        if not chain or tool == "none":
            chain = False
            tool = "none"
        return {
            "chain": chain,
            "tool": tool,
            "reasoning": str(parsed.get("reasoning", "")).strip() or "추가 도움 필요 없음",
        }
    except Exception as e:
        return {"chain": False, "tool": "none", "reasoning": f"reflect 실패: {str(e)[:60]}"}


def find_alternative_colors_for_item(clothing_result: dict, user_tone: dict | None) -> list[dict]:
    """옷의 dominantHex와 사용자 베스트 팔레트를 비교해 비슷한 색 중 잘 받는 색 3개 추천.
    순수 계산 — Gemini 호출 없음.
    Returns: [{"hex": ..., "name": ..., "reason": "..."}, ...] (최대 3개)
    """
    if not user_tone:
        return []

    item_hex = clothing_result.get("colorHex", "")
    if not item_hex or not item_hex.startswith("#"):
        return []

    # altColors가 _score_fit에서 이미 계산돼 있으면 우선 활용
    alt_hexes = clothing_result.get("altColors", []) or []

    # 없으면 user의 bestColors에서 직접 계산
    if not alt_hexes:
        best_palette = user_tone.get("bestColors", [])
        alt_hexes = _palette_hexes(best_palette)

    if not alt_hexes:
        return []

    # item_hex와의 LAB 거리 기준으로 정렬 — 비슷할수록 "대안"으로 적합
    try:
        item_rgb = _hex_to_rgb(item_hex)
        scored = []
        for h in alt_hexes:
            try:
                rgb = _hex_to_rgb(h)
                d = _delta_e(item_rgb, rgb)
                scored.append((h, d))
            except Exception:
                continue
        scored.sort(key=lambda x: x[1])
        top3 = scored[:3]
    except Exception:
        top3 = [(h, 0.0) for h in alt_hexes[:3]]

    # 사용자 베스트 팔레트에서 이름 찾기 (있으면 사용, 없으면 휴리스틱 이름)
    best_palette = user_tone.get("bestColors", []) if user_tone else []
    name_by_hex = {}
    for item in best_palette:
        if isinstance(item, dict):
            h = str(item.get("hex", "")).upper()
            n = str(item.get("name", "")).strip()
            if h and n:
                name_by_hex[h] = n

    item_name = clothing_result.get("color", "이 색")
    out = []
    for h, d in top3:
        h_up = h.upper()
        name = name_by_hex.get(h_up) or _nearest_color_name(_hex_to_rgb(h))
        if d < 12:
            reason = f"{item_name}과 비슷한 톤대인데 더 잘 받는 색"
        elif d < 25:
            reason = f"{item_name}과 분위기는 비슷하지만 톤이 더 어울림"
        else:
            reason = f"{item_name} 대신 시도해볼 만한 추천 색"
        out.append({"hex": h_up, "name": name, "reason": reason})
    return out


def find_closet_match_for_item(clothing_result: dict, closet_items: list) -> list[dict]:
    """옷장에서 같은 카테고리 + 색 거리(Delta-E) 가까운 옷 찾기.
    순수 계산 — Gemini 호출 없음.
    closet_items: 백엔드로 들어온 ClosetItem 리스트 (label, color, category, fit, score)
    Returns: [{"label": ..., "color": ..., "category": ..., "fit": ..., "score": int}, ...] (최대 3개)
    """
    if not closet_items:
        return []

    item_category = clothing_result.get("category", "")
    item_hex = clothing_result.get("colorHex", "")
    item_color_name = (clothing_result.get("color") or "").strip()

    matches = []
    for ci in closet_items:
        cat = getattr(ci, "category", "") if not isinstance(ci, dict) else ci.get("category", "")
        if cat != item_category:
            continue
        # 옷장 아이템에 hex가 없을 수 있음 (요약 모델은 label·color만 가짐).
        # 색 이름이 같으면 비슷하다고 보고 후보로 포함.
        ci_color = getattr(ci, "color", "") if not isinstance(ci, dict) else ci.get("color", "")
        ci_label = getattr(ci, "label", "") if not isinstance(ci, dict) else ci.get("label", "")
        ci_fit = getattr(ci, "fit", "") if not isinstance(ci, dict) else ci.get("fit", "")
        ci_score = getattr(ci, "score", 0) if not isinstance(ci, dict) else ci.get("score", 0)

        # 색 이름이 겹치면 후보 (느슨한 기준)
        if ci_color and item_color_name and (ci_color in item_color_name or item_color_name in ci_color):
            matches.append({
                "label": ci_label,
                "color": ci_color,
                "category": cat,
                "fit": ci_fit,
                "score": ci_score,
            })

    # 점수 높은 순으로 정렬, 최대 3개
    matches.sort(key=lambda m: m.get("score", 0), reverse=True)
    return matches[:3]


def chat_stream(message: str, history: list, use_closet: bool, user_tone: dict | None = None, closet_context: str = "", last_item_context: str = ""):
    """Stream chat response for style recommendations."""
    system_prompt = """당신은 PaletteFit의 톤·옷장 어시스턴트입니다. 사용자가 자신의 퍼스널 톤·분석한 옷·옷장에 대해 묻는 질문에 답하는 것이 유일한 역할입니다.

[역할 제한 — 반드시 지킬 것]
- 코디 조합 제안 금지 (예: "X 셔츠 + Y 슬랙스 + Z 신발").
- 외출 룩·OOTD·출근룩·데이트룩 같은 코디 추천 금지.
- "이렇게 입어보세요" 같은 스타일링 지시 금지.
- 사용자가 직접 코디 추천을 요청해도 정중히 거절하고 톤·옷장 정보로 답을 돌릴 것.
  예: "코디 조합은 직접 골라보시는 게 좋아요. 대신 어떤 색이 잘 받는지는 알려드릴 수 있어요."

[허용되는 답변 유형]
- 톤 정보 조회 ("내 톤이 뭐였더라?" → 저장된 시즌·언더톤·신뢰도 그대로 알려줌)
- 어울리는/피할 색 설명 ("라벤더는 나한테 어울려?" → 사용자 베스트 색과 비교)
- 옷장 정보 조회 ("내 옷장에서 가장 잘 받는 옷이 뭐야?" → 점수 높은 순으로 알려줌)
- 분석 결과 해석 ("이 옷이 왜 점수가 낮아?" → 색·톤 거리 근거로 설명)
- 일반 톤 지식 ("쿨톤이 피해야 할 색은?" → 사실 정보)

[답변 톤]
- 자연스러운 대화체 (한두 문단).
- 사용자 데이터(톤·옷장·직전 분석 옷)를 명시적으로 인용 — 일반론 금지.
- 모르는 정보(옷장에 없는 옷 등)는 "옷장에 그 정보는 없어요" 라고 솔직히.
- 불릿·번호·이모지 금지. 4~7줄.

[환각 금지 — 절대 규칙]
- 데이터로 주어지지 않은 정보(특히 날짜·시간·숫자·이름)는 절대 추측·창작 금지.
- 사용자가 "언제 분석했어?" 처럼 시점을 물으면, [사용자 톤]의 analyzed_at 값이 있을 때만 그 값을 그대로 사용. 없으면 "정확한 분석 시점은 기록되어 있지 않아요" 라고 답.
- "최근에", "며칠 전에" 같은 모호한 시간 표현도 데이터 근거 없으면 사용 금지.

[좋은 답변 예시]
Q: "내 옷장에서 가장 잘 받는 옷이 뭐야?"
A: 지금 옷장에 8벌이 있고, 그중 잘 받는 색으로 분류된 옷은 라벤더 니트(86점)와 라이트 워싱 데님 팬츠(82점)예요. 둘 다 여름 쿨톤에 잘 어울리는 부드러운 색이라 점수가 높게 나왔어요. 더 자세히 보고 싶은 아이템이 있으면 옷장 페이지에서 확인하실 수 있어요.

Q: "라벤더는 나한테 어울려?"
A: 네, 잘 받는 편이에요. 라벤더는 당신의 베스트 컬러 중 하나로 분석돼 있고, 옷장에 있는 라벤더 니트도 86점이 나왔어요. 비슷한 톤대의 소프트모브나 베이비라일락도 잘 어울릴 색이에요.

[피해야 할 답변]
- "라벤더 니트에 그레이 슬랙스 매치하면 잘 어울려요" (코디 추천 — 금지)
- "출근에는 X 입고 데이트에는 Y 입으세요" (시나리오 추천 — 금지)"""

    if user_tone:
        tone_lines = [
            f"시즌: {user_tone.get('season', '알 수 없음')}",
            f"어울리는 색상: {', '.join(_palette_names(user_tone.get('bestColors', [])))}",
        ]
        if user_tone.get("analyzed_at"):
            tone_lines.append(f"analyzed_at: {user_tone['analyzed_at']}")
        system_prompt += "\n\n[사용자 톤]\n" + "\n".join(tone_lines)

    if last_item_context:
        system_prompt += f"\n\n[방금 분석한 옷 — 우선 활용]\n{last_item_context}"

    if closet_context:
        system_prompt += f"\n\n{closet_context}"

    contents = [
        types.Content(role="user", parts=[types.Part.from_text(text=system_prompt)]),
        types.Content(role="model", parts=[types.Part.from_text(text="네, 알겠습니다. 톤·옷장 정보 기반 Q&A 어시스턴트로 답변하겠습니다.")]),
    ]

    for h in history[-16:]:
        role = "user" if h["role"] == "user" else "model"
        contents.append(
            types.Content(role=role, parts=[types.Part.from_text(text=h["content"])])
        )

    contents.append(
        types.Content(role="user", parts=[types.Part.from_text(text=message)])
    )

    return _get_client().models.generate_content_stream(
        model=MODEL,
        contents=contents,
        config=types.GenerateContentConfig(
            temperature=0.7,
            max_output_tokens=2000,
            thinking_config=types.ThinkingConfig(thinking_budget=0),
        ),
    )
