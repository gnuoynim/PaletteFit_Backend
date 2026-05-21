import asyncio
import json

from fastapi import APIRouter, Depends, Request
from fastapi.responses import StreamingResponse

from limiter import limiter
from pydantic import BaseModel, Field
from typing import List, Optional

from services.ai_service import (
    analyze_tone,
    analyze_clothing,
    chat_stream,
    agent_decide,
    agent_reflect,
    find_alternative_colors_for_item,
    find_closet_match_for_item,
)
from routers.tone import get_session_id, get_user_tone, save_user_tone

router = APIRouter()


class AgentMessage(BaseModel):
    role: str = Field(pattern=r"^(user|assistant)$")
    content: str
    msgType: str = "text"


class ClosetItem(BaseModel):
    """프론트에서 보내는 분석된 옷 요약 정보"""
    label: str = ""
    color: str = ""
    category: str = ""
    fit: str = ""
    score: int = 0


class AgentRequest(BaseModel):
    message: str = Field(default="", max_length=2000)
    image: Optional[str] = None
    imageIntent: Optional[str] = None  # "tone" / "clothing" / None(AI 판단)
    history: List[AgentMessage] = []
    closetMemory: List[ClosetItem] = []  # 이전에 분석한 옷 목록
    lastClothing: Optional[ClosetItem] = None  # 방금 분석한 옷 (있으면 챗 컨텍스트에서 우선 활용)


def _sse(event_type: str, data: dict) -> str:
    return f"data: {json.dumps({'type': event_type, **data}, ensure_ascii=False)}\n\n"


@router.post("")
@limiter.limit("15/minute")
async def agent_chat(request: Request, req: AgentRequest, session_id: str = Depends(get_session_id)):
    async def generate():
        user_tone = get_user_tone(session_id)
        has_tone = user_tone is not None
        history_for_ai = [{"role": m.role, "content": m.content} for m in req.history if m.msgType == "text"]

        # ── 옷장 기억 컨텍스트 구성 (통계 요약 + 상위/하위 항목) ──
        closet_context = ""
        if req.closetMemory:
            total = len(req.closetMemory)
            good = sum(1 for it in req.closetMemory if it.fit == "잘 어울림")
            meh = sum(1 for it in req.closetMemory if it.fit == "애매함")
            bad = sum(1 for it in req.closetMemory if it.fit == "비추천")
            match_rate = round((good / total) * 100) if total else 0

            sorted_items = sorted(req.closetMemory, key=lambda it: it.score, reverse=True)
            items_desc = [
                f"- {item.color} {item.label}({item.category}): {item.fit} {item.score}점"
                for item in sorted_items
            ]
            closet_context = (
                f"[옷장 요약]\n"
                f"총 {total}벌 · 잘 어울림 {good}벌 · 애매함 {meh}벌 · 비추천 {bad}벌 · 톤 적합도 {match_rate}%\n\n"
                f"[옷장 전체 — 점수 높은 순]\n"
                + "\n".join(items_desc)
            )

        # ── 방금 분석한 옷 컨텍스트 (있으면 챗에 우선 주입) ──
        last_item_context = ""
        if req.lastClothing and (req.lastClothing.label or req.lastClothing.color):
            lc = req.lastClothing
            last_item_context = f"{lc.color} {lc.label} ({lc.category}) — 궁합 {lc.fit}, {lc.score}점. 사용자가 이 옷을 중심으로 답변을 기대합니다."

        # ── 이미지가 있는 경우: 항상 AI가 이미지를 보고 판단 ──
        if req.image:
            yield _sse("thinking", {"step": "perceive", "label": "사진 분석 중", "detail": "주요 피사체와 사용자 의도를 파악하고 있어요"})
            try:
                decision = await asyncio.to_thread(
                    agent_decide, req.message, req.image, has_tone, user_tone, history_for_ai, closet_context
                )
            except Exception as e:
                decision = {
                    "intent": "analyze_tone" if not has_tone else "analyze_clothing",
                    "reasoning": f"판단 중 오류가 발생해 기본 동작으로 진행합니다 ({str(e)[:60]})",
                    "confidence": "낮음",
                }

            raw_intent = decision.get("intent", "chat")
            reasoning = decision.get("reasoning", "")
            confidence = decision.get("confidence", "보통")

            # 의도 → 사람 친화적 라벨 매핑
            intent_label = {
                "analyze_tone": "퍼스널 톤 분석",
                "analyze_clothing": "옷 궁합 분석",
                "find_better_color": "더 잘 받는 색 찾기",
                "request_clarification": "사진 다시 요청",
                "chat": "대화로 이어가기",
            }.get(raw_intent, raw_intent)

            yield _sse("thinking", {
                "step": "decide",
                "label": f"판단 완료 → {intent_label}",
                "detail": reasoning or "(이유 없음)",
                "confidence": confidence,
            })

            if raw_intent == "analyze_tone":
                intent = "tone"
            elif raw_intent == "analyze_clothing":
                intent = "clothing"
            elif raw_intent == "find_better_color":
                yield _sse("text", {"content": reasoning or "어떤 옷을 기준으로 더 잘 받는 색을 찾아드릴까요? 옷 사진을 올려주세요."})
                yield _sse("actions", {"content": "", "buttons": [
                    {"label": "옷 사진 올리기", "action": "clothing"},
                ]})
                yield "data: [DONE]\n\n"
                return
            elif raw_intent == "request_clarification":
                ask = decision.get("ask") or "사진을 다시 올려주세요. 얼굴이 잘 보이는 셀카 또는 옷이 또렷하게 보이는 사진이 좋아요."
                yield _sse("text", {"content": ask})
                buttons = [{"label": "얼굴 사진 다시 올리기", "action": "upload_tone"}]
                if has_tone:
                    buttons.append({"label": "옷 사진 다시 올리기", "action": "clothing"})
                yield _sse("actions", {"content": "", "buttons": buttons})
                yield "data: [DONE]\n\n"
                return
            else:
                text = decision.get("text") or reasoning or "이 사진으로 무엇을 하고 싶으신가요? 톤 분석 또는 옷 궁합 분석이 가능해요."
                yield _sse("text", {"content": text})
                if not has_tone:
                    yield _sse("actions", {"content": "", "buttons": [
                        {"label": "얼굴 사진 올리기", "action": "upload_tone"},
                    ]})
                yield "data: [DONE]\n\n"
                return

            # ── 톤 분석 ──
            if intent == "tone":
                yield _sse("thinking", {
                    "step": "act",
                    "label": "드레이핑 시뮬레이션 시작",
                    "detail": "웜/쿨 6장 → 시즌 8장 드레이프를 사진에 합성해 비교합니다",
                })

                try:
                    result = await asyncio.to_thread(analyze_tone, req.image)
                    if result.get("error"):
                        yield _sse("text", {"content": result.get("message", "분석에 실패했어요. 얼굴이 잘 보이는 사진으로 다시 시도해주세요.")})
                        yield "data: [DONE]\n\n"
                        return

                    save_data = {k: v for k, v in result.items() if k != "drapeImages"}
                    save_user_tone(session_id, save_data)
                    user_tone = save_data
                    has_tone = True

                    yield _sse("tone_result", {
                        "content": "",
                        "data": {
                            "season": result.get("season", ""),
                            "undertone": result.get("undertone", ""),
                            "brightness": result.get("brightness", ""),
                            "saturation": result.get("saturation", ""),
                            "confidence": result.get("confidence", ""),
                            "description": result.get("description", ""),
                            "bestColors": result.get("bestColors", []),
                            "worstColors": result.get("worstColors", []),
                        },
                    })

                    season = result.get("season", "")
                    yield _sse("text", {"content": f"{season}으로 분석됐어요. 이제 옷 사진을 올리면 톤이랑 맞는지 바로 봐드릴게요."})
                    yield _sse("actions", {"content": "", "buttons": [
                        {"label": "옷 사진 분석", "action": "clothing"},
                        {"label": "내 옷장 보기", "action": "open_closet"},
                    ]})

                except Exception as e:
                    yield _sse("text", {"content": f"분석 중 오류가 발생했어요: {str(e)[:100]}"})

                yield "data: [DONE]\n\n"
                return

            # ── 옷 궁합 분석 ──
            elif intent == "clothing":
                if not has_tone:
                    yield _sse("text", {"content": "옷 분석을 하려면 먼저 퍼스널 톤 분석이 필요해요! 얼굴 사진을 올려주세요."})
                    yield _sse("actions", {"content": "", "buttons": [
                        {"label": "얼굴 사진 올리기", "action": "upload_tone"},
                    ]})
                    yield "data: [DONE]\n\n"
                    return

                yield _sse("thinking", {
                    "step": "act",
                    "label": "옷 궁합 분석 시작",
                    "detail": f"옷 색상 추출 → {user_tone.get('season', '')} 톤과의 거리 계산 (CIELAB Delta-E)",
                })

                try:
                    result = await asyncio.to_thread(analyze_clothing, req.image, user_tone)

                    # 분석 결과 카드
                    yield _sse("clothing_result", {"content": "", "data": result})

                    score = result.get("score", 50)
                    label = result.get("label", "이 아이템")
                    color = result.get("color", "")

                    # ── Reflection 단계: AI가 결과 보고 후속 도구 호출 여부 자율 결정 ──
                    yield _sse("thinking", {
                        "step": "reflect",
                        "label": "후속 도움 판단 중",
                        "detail": "분석 결과를 보고 자동으로 이어가면 좋을 게 있는지 검토",
                    })
                    reflection = await asyncio.to_thread(agent_reflect, result, user_tone, closet_context)
                    chained_tool = reflection.get("tool", "none") if reflection.get("chain") else "none"

                    chain_label = {
                        "find_better_color": "→ 대안 색 자동 제시",
                        "suggest_closet_match": "→ 옷장 중복 안내",
                        "none": "→ 체이닝 없이 사용자 결정에 맡김",
                    }.get(chained_tool, "→ 체이닝 없음")
                    yield _sse("thinking", {
                        "step": "decide",
                        "label": f"reflect 완료 {chain_label}",
                        "detail": reflection.get("reasoning", ""),
                    })

                    # ── 체이닝 분기 ──
                    if chained_tool == "find_better_color":
                        alts = find_alternative_colors_for_item(result, user_tone)
                        if alts:
                            yield _sse("alt_colors", {
                                "content": "",
                                "data": {
                                    "label": label,
                                    "color": color,
                                    "colorHex": result.get("colorHex", ""),
                                    "alternatives": alts,
                                },
                            })
                            yield _sse("text", {"content": f"이 {color} {label} 대신 비슷한 핏의 잘 받는 색 {len(alts)}가지를 골라봤어요. 위 중에서 마음에 드는 게 있으면 참고하세요."})
                        else:
                            # 대안 추출 실패 시 폴백
                            yield _sse("text", {"content": f"이 {color} {label}은 톤과 충돌이 좀 있어요. 다른 색을 시도해보세요."})
                        yield _sse("actions", {"content": "", "buttons": [
                            {"label": "다른 옷 분석", "action": "clothing"},
                            {"label": "옷장에 저장", "action": "save_to_closet"},
                        ]})

                    elif chained_tool == "suggest_closet_match":
                        closet_matches = find_closet_match_for_item(result, req.closetMemory or [])
                        if closet_matches:
                            yield _sse("closet_match", {
                                "content": "",
                                "data": {
                                    "label": label,
                                    "matches": closet_matches,
                                },
                            })
                            yield _sse("text", {"content": f"옷장에 비슷한 {color} 옷이 {len(closet_matches)}벌 있어요. 중복일 수 있으니 한 번 더 생각해보세요."})
                        else:
                            yield _sse("text", {"content": "잘 받는 아이템이에요."})
                        yield _sse("actions", {"content": "", "buttons": [
                            {"label": "옷장에 저장", "action": "save_to_closet"},
                            {"label": "내 옷장 보기", "action": "open_closet"},
                            {"label": "다른 옷 분석", "action": "clothing"},
                        ]})

                    else:
                        # 체이닝 없음 — 점수에 맞는 한 줄 평 + 기본 액션
                        if score < 40:
                            yield _sse("text", {"content": f"이 {color} {label}은 톤과 충돌이 좀 있어요."})
                        elif score < 65:
                            yield _sse("text", {"content": f"애매한 정도예요. 다른 아이템과 함께 신중히 매치하는 게 좋아요."})
                        else:
                            yield _sse("text", {"content": f"잘 받는 아이템이에요."})
                        yield _sse("actions", {"content": "", "buttons": [
                            {"label": "옷장에 저장", "action": "save_to_closet"},
                            {"label": "다른 옷 분석", "action": "clothing"},
                            {"label": "내 옷장 보기", "action": "open_closet"},
                        ]})

                except Exception as e:
                    yield _sse("text", {"content": f"옷 분석에 실패했어요: {str(e)[:100]}"})

                yield "data: [DONE]\n\n"
                return

        # ── 텍스트만: 톤이 없으면 안내, 있으면 스타일 채팅 ──
        if not req.message.strip():
            yield _sse("text", {"content": "안녕하세요! 무엇을 도와드릴까요?"})
            yield "data: [DONE]\n\n"
            return

        if not has_tone:
            yield _sse("text", {"content": "옷 톤 궁합을 봐드리려면 먼저 퍼스널 톤 분석이 필요해요. 얼굴 사진을 올려주세요."})
            yield _sse("actions", {"content": "", "buttons": [
                {"label": "얼굴 사진 올리기", "action": "upload_tone"},
                {"label": "톤 직접 선택하기", "action": "select_tone"},
            ]})
            yield "data: [DONE]\n\n"
            return

        # ── 스타일 채팅 스트리밍 ──
        def _run_chat():
            return list(chat_stream(
                message=req.message,
                history=history_for_ai,
                use_closet=False,
                user_tone=user_tone,
                closet_context=closet_context,
                last_item_context=last_item_context,
            ))

        chunks = await asyncio.to_thread(_run_chat)

        for chunk in chunks:
            if chunk.text:
                yield _sse("text_chunk", {"content": chunk.text})

        yield "data: [DONE]\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")
