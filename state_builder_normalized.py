"""
state_builder_normalized.py
===========================

职责：
- 从 ZJHEnvMulti2to9 构建【归一化状态】
- 消除 ante / stack / pot 的尺度影响
- 与 A30Agent / config_a30_hybrid 完全一致

原则：
- 所有金额类特征尽量“相对化”
- 关键规则变量（ante / round / alive_cnt）保留原值或轻归一
"""

import numpy as np


# ============================================================
# 常量
# ============================================================

EPS = 1e-6


# ============================================================
# 主入口
# ============================================================

def build_state(env):
    """
    返回：
        np.ndarray shape = (49,)
    必须与 env.state_dim 完全一致
    """

    hero = env.players[0]

    # --------------------------------------------------------
    # 1. Hero 手牌信息（不归一化类型，只归一化 rank）
    # --------------------------------------------------------

    t = env.hero_hand_type / 5.0          # 0~1
    hi = env.hero_hi / 14.0               # 0~1
    mid = env.hero_mid / 14.0
    lo = env.hero_lo / 14.0
    strength = env.hero_strength          # 已是 0~1

    # --------------------------------------------------------
    # 2. 人数 / 轮数
    # --------------------------------------------------------

    round_norm = env.round_index / max(1.0, env.max_round)
    num_players_norm = env.num_players / env.max_players
    alive_norm = env.alive_cnt / max(1.0, env.num_players)

    # --------------------------------------------------------
    # 3. 下注尺度（关键）
    # --------------------------------------------------------

    ante = env.ante_bb

    # 当前最低下注倍数（unit）
    current_unit = env.current_bet_unit

    # hero 最低需要下注的 unit
    min_unit = env._min_required_unit(hero)

    # --------------------------------------------------------
    # 4. Hero 金额相关（全部相对化）
    # --------------------------------------------------------

    # hero 剩余筹码 / 初始筹码
    hero_stack_norm = hero["stack_bb"] / max(env.starting_stack_bb, EPS)

    # hero 已投入 / 初始筹码
    hero_contrib_norm = hero["contrib_bb"] / max(env.starting_stack_bb, EPS)

    # hero stack / 场均 stack
    avg_stack = _avg_alive_stack(env)
    hero_vs_avg = hero["stack_bb"] / max(avg_stack, EPS)

    # pot / 场均 stack
    pot_norm = env.pot_bb / max(avg_stack, EPS)

    # --------------------------------------------------------
    # 5. Hero 状态
    # --------------------------------------------------------

    hero_seen = float(hero["has_seen"])

    # --------------------------------------------------------
    # 6. 拼装 Hero 状态（14 维）
    # --------------------------------------------------------

    state = [
        t,
        hi,
        mid,
        lo,
        strength,
        round_norm,
        num_players_norm,
        alive_norm,
        ante / (ante + 10.0),        # 轻归一（防止 ante=10 放大）
        pot_norm,
        current_unit / 10.0,          # unit ∈ [1,10]
        min_unit / 10.0,
        hero_seen,
        hero_vs_avg,
    ]

    # --------------------------------------------------------
    # 7. 对手状态（每人 5 维 × 8 = 40）
    # --------------------------------------------------------

    for idx in range(1, env.max_players):
        if idx < env.num_players:
            p = env.players[idx]
            if p["alive"]:
                state.extend([
                    1.0,                                   # alive
                    float(p["has_seen"]),                  # seen
                    p["stack_bb"] / max(avg_stack, EPS),   # stack / avg
                    p["contrib_bb"] / max(avg_stack, EPS), # contrib / avg
                    float(p["last_act"]) / 5.0,            # act type
                ])
            else:
                state.extend([0.0, 0.0, 0.0, 0.0, 0.0])
        else:
            state.extend([0.0, 0.0, 0.0, 0.0, 0.0])

    assert len(state) == 54, f"State dim mismatch: {len(state)} != 54"

    return np.array(state, dtype=np.float32)


# ============================================================
# Utils
# ============================================================

def _avg_alive_stack(env):
    total = 0.0
    cnt = 0
    for p in env.players[:env.num_players]:
        if p["alive"]:
            total += p["stack_bb"]
            cnt += 1
    return total / max(1, cnt)
