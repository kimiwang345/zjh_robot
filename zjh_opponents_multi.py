import random
from ZJHEnvMulti2to9 import OppAct, evaluate_hand

# ==========================================================
# 牌力评估（0~1，保持原粗粒度） 对手系统（PSRO）
# ==========================================================

def strength(cards):
    t, ranks = evaluate_hand(cards)
    return (t + ranks[0] / 14.0) / 6.0


# ==========================================================
# BB 工具函数
# ==========================================================

def avg_stack_bb(env):
    s = 0.0
    c = 0
    for p in env.players[:env.num_players]:
        if p["alive"]:
            s += p["stack_bb"]
            c += 1
    return s / max(1, c)


def stack_bucket(bb):
    if bb < 15:
        return "short"
    if bb < 60:
        return "mid"
    return "deep"


def min_required_unit(env, p):
    return env._min_required_unit(p)


def min_required_pay(env, p):
    u = min_required_unit(env, p)
    return u * env.ante_bb * (2 if p["has_seen"] else 1)


def can_bet(env, p):
    return p["stack_bb"] >= min_required_pay(env, p)


def safe_pk(env):
    return env.round_index > 0


# ==========================================================
# Base Class
# ==========================================================

class BaseOpponent:
    def decide(self, env, idx):
        raise NotImplementedError


# ==========================================================
# 1. NitOpponent（紧弱）
# ==========================================================

class NitOpponent(BaseOpponent):

    def decide(self, env, idx):
        p = env.players[idx]
        s = strength(p["cards"])
        bucket = stack_bucket(p["stack_bb"])

        # 筹码不足
        if not can_bet(env, p):
            return (OppAct.COMPARE_ALL, None) if safe_pk(env) else (OppAct.FOLD, None)

        umin = min_required_unit(env, p)

        # short：非常保守
        if bucket == "short":
            if s < 0.55:
                return OppAct.FOLD, None
            return OppAct.BET, umin

        # mid
        if s < 0.35:
            return OppAct.FOLD, None
        if s < 0.65:
            return OppAct.BET, umin
        return OppAct.BET, min(umin + 1, 10)


# ==========================================================
# 2. LooseOpponent（松）
# ==========================================================

class LooseOpponent(BaseOpponent):

    def decide(self, env, idx):
        p = env.players[idx]
        bucket = stack_bucket(p["stack_bb"])

        if not can_bet(env, p):
            return OppAct.COMPARE_ALL, None

        umin = min_required_unit(env, p)

        if bucket == "short":
            return OppAct.BET, umin

        if random.random() < 0.65:
            return OppAct.BET, umin
        return OppAct.BET, min(umin + random.randint(1, 3), 10)


# ==========================================================
# 3. AggroOpponent（激进）
# ==========================================================

class AggroOpponent(BaseOpponent):

    def decide(self, env, idx):
        p = env.players[idx]
        s = strength(p["cards"])
        bucket = stack_bucket(p["stack_bb"])

        if not can_bet(env, p):
            return OppAct.COMPARE_ALL, None

        umin = min_required_unit(env, p)

        if bucket == "short":
            if s > 0.6 and safe_pk(env):
                return OppAct.PK, None
            return OppAct.BET, umin

        r = random.random()
        if r < 0.5:
            return OppAct.BET, min(umin + random.randint(1, 3), 10)
        if r < 0.75 and safe_pk(env):
            return OppAct.PK, None
        return OppAct.BET, umin


# ==========================================================
# 4. FishOpponent（鱼）
# ==========================================================

class FishOpponent(BaseOpponent):

    def decide(self, env, idx):
        p = env.players[idx]

        if not can_bet(env, p):
            return OppAct.FOLD, None

        umin = min_required_unit(env, p)

        if random.random() < 0.8:
            return OppAct.BET, umin
        return OppAct.BET, min(umin + 1, 10)


# ==========================================================
# 5. ManiacOpponent（疯子）
# ==========================================================

class ManiacOpponent(BaseOpponent):

    def decide(self, env, idx):
        p = env.players[idx]

        if not can_bet(env, p):
            return OppAct.COMPARE_ALL, None

        umin = min_required_unit(env, p)

        if random.random() < 0.6:
            return OppAct.BET, min(umin + random.randint(2, 5), 10)

        if safe_pk(env):
            return OppAct.PK, None

        return OppAct.BET, umin


# ==========================================================
# 6. TAGOpponent（紧凶）
# ==========================================================

class TAGOpponent(BaseOpponent):

    def decide(self, env, idx):
        p = env.players[idx]
        s = strength(p["cards"])

        if not can_bet(env, p):
            return OppAct.COMPARE_ALL, None

        umin = min_required_unit(env, p)

        if s < 0.35:
            return OppAct.FOLD, None
        if s < 0.65:
            return OppAct.BET, umin
        return OppAct.BET, min(umin + random.randint(1, 2), 10)


# ==========================================================
# 7. GTOOpponent（GTO-lite）
# ==========================================================

class GTOOpponent(BaseOpponent):

    def decide(self, env, idx):
        p = env.players[idx]
        s = strength(p["cards"])

        if not can_bet(env, p):
            return OppAct.COMPARE_ALL, None

        umin = min_required_unit(env, p)

        if s < 0.3:
            return OppAct.FOLD, None
        if s < 0.6:
            return OppAct.BET, umin
        return OppAct.BET, min(umin + random.choice([1, 2, 3]), 10)
