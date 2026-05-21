from typing import List, Optional, Union

from pydantic import BaseModel, Field


# ── 톤 분석 ──

class ColorItem(BaseModel):
    """개인 맞춤 색상 (헥스 + 한국어 이름 + 사용자 맞춤 이유)."""
    hex: str = Field(pattern=r"^#[0-9A-Fa-f]{6}$")
    name: str = ""
    reason: str = ""


class ToneRequest(BaseModel):
    image: str = Field(..., min_length=100, description="Base64 인코딩된 얼굴 이미지")


class QualityScore(BaseModel):
    score: int = Field(ge=1, le=5)
    reason: str = ""


class QualityScores(BaseModel):
    lighting: Optional[QualityScore] = None
    obscuration: Optional[QualityScore] = None
    makeup: Optional[QualityScore] = None
    resolution: Optional[QualityScore] = None
    skinSample: Optional[QualityScore] = None
    background: Optional[QualityScore] = None


class UndertoneTest(BaseModel):
    warmScore: int = Field(ge=1, le=10)
    coolScore: int = Field(ge=1, le=10)
    reason: str = ""


class SeasonScore(BaseModel):
    score: int = Field(ge=1, le=10)
    reason: str = ""


class ToneResponse(BaseModel):
    season: str
    undertone: str
    brightness: str
    saturation: str
    confidence: str
    description: str = ""
    bestColors: List[ColorItem] = []
    worstColors: List[ColorItem] = []
    scores: dict[str, SeasonScore] = {}
    qualityScores: Optional[QualityScores] = None
    drapeImages: Optional[dict[str, str]] = None
    undertoneTest: Optional[UndertoneTest] = None


class ToneErrorResponse(BaseModel):
    error: bool = True
    message: str


class ToneUpdateRequest(BaseModel):
    season: Optional[str] = None
    undertone: Optional[str] = None
    brightness: Optional[str] = None
    saturation: Optional[str] = None
    bestColors: Optional[List[Union[ColorItem, str]]] = None
    worstColors: Optional[List[Union[ColorItem, str]]] = None


# ── 옷 궁합 분석 ──

class ClothingRequest(BaseModel):
    image: str = Field(..., min_length=100, description="Base64 인코딩된 옷 이미지")


class ClothingResponse(BaseModel):
    category: str
    label: str
    color: str
    colorHex: str = Field(pattern=r"^#[0-9A-Fa-f]{6}$")
    secondaryColor: Optional[str] = None
    secondaryHex: Optional[str] = None
    fit: str = Field(description="잘 어울림 / 애매함 / 비추천")
    score: int = Field(ge=0, le=100)
    reasonTags: List[str] = []
    suggestion: str = ""
    matchingRecommendation: str = ""
    altColors: List[str] = []


# ── 채팅 ──

class ChatMessage(BaseModel):
    role: str = Field(pattern=r"^(user|assistant)$")
    content: str


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=2000)
    useCloset: bool = False
    history: List[ChatMessage] = []
