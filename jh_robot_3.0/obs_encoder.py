# obs_encoder.py
from typing import Dict, Any, List, Optional, Tuple
import numpy as np

def _card_vec(card: Optional[Tuple[int, int]]) -> Tuple[float, float]:
    if card is None:
        return 0.0, 0.0
    r, s = card
    return r / 14.0, s / 3.0

def encode_obs(obs: Dict[str, Any]) -> np.ndarray:
    """
    统一使用 min_mul 口径（最低下注倍数）
    """
    base_bet = float(obs["base_bet"])
    pot = float(obs["pot_total"])
    rnd = float(obs["round"])
    alive_players = float(obs["alive_players"])

    me = obs["me"]
    my_seen = 1.0 if me["seen"] else 0.0
    my_stack = float(me["stack"])
    my_bet = float(me["bet"])

    # ⭐ 核心：min_mul（不是 min_call）
    min_mul = float(me["min_mul"])

    # ========= 归一化 =========
    bb = max(1.0, base_bet)
    pot_n = pot / (bb * 200.0)
    stack_n = my_stack / (bb * 200.0)
    bet_n = my_bet / (bb * 200.0)
    min_mul_n = min_mul / 20.0

    feats: List[float] = []

    feats += [
        base_bet / 50.0,
        pot_n,
        rnd / 20.0,
        alive_players / 9.0,
        my_seen,
        stack_n,
        bet_n,
        min_mul_n,
    ]

    # ========= 手牌 =========
    cards = me.get("cards")
    if (not my_seen) or (not cards):
        for _ in range(3):
            feats += [0.0, 0.0]
    else:
        for c in cards[:3]:
            r, s = int(c["rank"]), int(c["suit"])
            feats += [r / 14.0, s / 3.0]

    # ========= 对手（最多 8 个） =========
    opps = [o for o in obs["opponents"] if o["alive"]]
    opps.sort(key=lambda x: x["seat"])
    opps = opps[:8]

    for i in range(8):
        if i < len(opps):
            o = opps[i]
            feats += [
                1.0,
                1.0 if o["seen"] else 0.0,
                float(o["stack"]) / (bb * 200.0),
                float(o.get("bet", 0)) / (bb * 200.0),
            ]
        else:
            feats += [0.0, 0.0, 0.0, 0.0]

    return np.asarray(feats, dtype=np.float32)

def obs_dim() -> int:
    # 8 + 手牌 6 + 对手 32 = 46
    return 46
