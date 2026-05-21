import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded

from limiter import limiter
from routers import tone, closet, agent, demo

load_dotenv()

app = FastAPI(title="PaletteFit API")

# Rate limiter — IP 기반, 라우터 데코레이터로 사용
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# CORS: 환경변수로 허용 도메인 설정 (쉼표 구분)
_default_origins = "http://localhost:3000,http://localhost:3001"
_origins = os.getenv("ALLOWED_ORIGINS", _default_origins).split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in _origins],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "OPTIONS"],
    allow_headers=["*"],
)

app.include_router(tone.router, prefix="/api/tone", tags=["tone"])
app.include_router(closet.router, prefix="/api/closet", tags=["closet"])
app.include_router(agent.router, prefix="/api/agent", tags=["agent"])
app.include_router(demo.router, prefix="/api/demo", tags=["demo"])


@app.get("/health")
async def health():
    return {"status": "ok"}
