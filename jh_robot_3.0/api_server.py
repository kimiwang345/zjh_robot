# api_server.py
from fastapi import FastAPI
import torch

from schemas import DecideRequest, DecideResponse
from obs_encoder import encode_obs, obs_dim
from model_hier import PolicyValueNetHier
from ppo_hier import sample_hier
from policy_mask import build_action_mask
from action_codec import decode_action


app = FastAPI()


# =========================
# 加载模型
# =========================
net = PolicyValueNetHier(obs_dim())
net.load_state_dict(torch.load("weights/zjh_policy.pt", map_location="cpu"))
net.eval()


@app.get("/zjh/ai/ping")
def ping():
    return {"status": "ok"}


@app.post("/zjh/ai/decide", response_model=DecideResponse)
def decide(req: DecideRequest):
    """
    推理接口（Hier PPO）
    - 输入：服务端传 min_mul（已按你规则计算好）
    - 输出：BET/PK/FOLD；BET 时带 bet_mul
    """
    # ===== 构造 obs（与训练一致）=====
    obs = {
        "base_bet": req.base_bet,
        "pot_total": req.pot_total,
        "round": req.round,
        "alive_players": req.alive_players,
        "me": {
            "seen": req.my_seen,
            "stack": req.my_stack,
            "bet": req.my_bet,
            "min_mul": req.min_mul,
            "cards": (
                [{"rank": c.rank, "suit": c.suit} for c in req.my_cards]
                if req.my_seen and req.my_cards else None
            )
        },
        "opponents": [o.dict() for o in req.opponents]
    }

    # ===== 构建 mask（强牌会自动禁 FOLD）=====
    mask_np = build_action_mask(obs)
    mask = torch.tensor(mask_np, dtype=torch.float32).unsqueeze(0)

    # ===== 编码 obs =====
    x = torch.tensor(encode_obs(obs), dtype=torch.float32).unsqueeze(0)

    # ===== 模型推理 + 分头采样 =====
    with torch.no_grad():
        fold_logits, play_logits, _ = net(x)

    act, logp = sample_hier(
        fold_logits,
        play_logits,
        mask,
        temperature=1.0
    )

    dec = decode_action(int(act.item()))
    prob = float(torch.exp(logp).item())

    # ===== 返回 =====
    if dec.kind == "BET":
        return DecideResponse(
            action="BET",
            bet_mul=dec.bet_mul,
            prob=prob
        )

    if dec.kind == "PK":
        # 简化：挑第一个存活对手；你也可以改成“筹码最多/最少/随机”等策略
        target_seat = req.opponents[0].seat if req.opponents else None
        return DecideResponse(
            action="PK",
            pk_target_seat=target_seat,
            prob=prob
        )

    return DecideResponse(
        action="FOLD",
        prob=prob
    )
