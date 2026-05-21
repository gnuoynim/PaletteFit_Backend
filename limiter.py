"""IP 기반 요청 빈도 제한 — 공개 배포 시 비용 폭탄 방지.

라우터 어디서든 `from limiter import limiter` 로 import 후 `@limiter.limit("N/minute")` 로 사용.
엔드포인트 함수는 반드시 `request: Request` 파라미터를 받아야 함 (slowapi가 IP 추출에 사용).
"""
from slowapi import Limiter
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)
