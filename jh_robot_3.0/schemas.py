# schemas.py
from pydantic import BaseModel, Field
from typing import List, Optional, Literal

class CardIn(BaseModel):
    rank: int = Field(..., ge=2, le=14)
    suit: int = Field(..., ge=0, le=3)

class OpponentIn(BaseModel):
    seat: int
    alive: bool
    seen: bool
    stack: int = Field(..., ge=0)
    bet: int = Field(0, ge=0)

class DecideRequest(BaseModel):
    base_bet: int = Field(..., ge=1, le=50)
    pot_total: int = Field(..., ge=0)
    round: int = Field(..., ge=0, le=20)
    alive_players: int = Field(..., ge=1, le=9)

    # ===== 我方 =====
    my_seen: bool
    my_stack: int = Field(..., ge=0)
    my_bet: int = Field(..., ge=0)

    # ⭐ 核心字段：最低下注倍数（服务端已算好）
    min_mul: int = Field(..., ge=1, le=20)

    my_cards: Optional[List[CardIn]] = None

    # ===== 对手 =====
    opponents: List[OpponentIn]

class DecideResponse(BaseModel):
    action: Literal["FOLD", "PK", "BET"]
    bet_mul: Optional[int] = None
    pk_target_seat: Optional[int] = None
    prob: float
