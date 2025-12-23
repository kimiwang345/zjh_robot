import random
from ZJHEnvMulti2to9 import OppAct, evaluate_hand


# ==========================================================
# 牌力评估（0~1）
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


# ==========================================================
# Base Class（BB-aware）
# ==========================================================

class BaseOpponent:
    """
    BB-aware 对手基类
    """

    def decide(self, env, idx):
        raise NotImplementedError


# ==========================================================
# 1. NitOpponent（BB-aware 紧弱）
# ==========================================================

class NitOpponent(BaseOpponent):

    def decide(self, env, idx):
        p = env.players[idx]
        s = strength(p["cards"])

        need = env.max_bet_bb - p["bet_bb"]
        stack_bb = p["stack_bb"]
        ante = env.ante_bb
        bucket = stack_bucket(stack_bb)

        # short stack：极度保守
        if bucket == "short":
            if need > 0.3 * stack_bb:
                return OppAct.FOLD, None
            return OppAct.CALL, None

        # mid stack
        if bucket == "mid":
            if s < 0.35:
                if need > ante:
                    return OppAct.FOLD, None
                return OppAct.CALL, None

            if s < 0.65:
                return OppAct.CALL, None

            return OppAct.BET, random.choice([2, 3])

        # deep stack
        if s < 0.4:
            return OppAct.CALL, None
        if s < 0.75:
            return OppAct.CALL, None
        return OppAct.BET, random.choice([2, 3])


# ==========================================================
# 2. LooseOpponent（BB-aware 松）
# ==========================================================

class LooseOpponent(BaseOpponent):

    def decide(self, env, idx):
        p = env.players[idx]
        s = strength(p["cards"])

        need = env.max_bet_bb - p["bet_bb"]
        stack_bb = p["stack_bb"]
        bucket = stack_bucket(stack_bb)

        # short stack：call-heavy
        if bucket == "short":
            if need > 0.6 * stack_bb:
                return OppAct.FOLD, None
            return OppAct.CALL, None

        # mid stack
        if bucket == "mid":
            if random.random() < 0.75:
                return OppAct.CALL, None
            return OppAct.BET, random.randint(2, 4)

        # deep stack
        r = random.random()
        if r < 0.65:
            return OppAct.CALL, None
        if r < 0.85:
            return OppAct.BET, random.randint(3, 6)
        return OppAct.PK, None


# ==========================================================
# 3. AggroOpponent（BB-aware 激进）
# ==========================================================

class AggroOpponent(BaseOpponent):

    def decide(self, env, idx):
        p = env.players[idx]
        s = strength(p["cards"])
        stack_bb = p["stack_bb"]
        bucket = stack_bucket(stack_bb)

        # short stack：避免自杀
        if bucket == "short":
            if s > 0.55:
                return OppAct.PK, None
            return OppAct.CALL, None

        # mid stack
        if bucket == "mid":
            r = random.random()
            if r < 0.4:
                return OppAct.BET, random.randint(3, 6)
            if r < 0.8:
                return OppAct.CALL, None
            return OppAct.PK, None

        # deep stack
        r = random.random()
        if r < 0.55:
            return OppAct.BET, random.randint(5, 8)
        if r < 0.85:
            return OppAct.PK, None
        return OppAct.CALL, None


# ==========================================================
# 4. FishOpponent（BB-aware 鱼）
# ==========================================================

class FishOpponent(BaseOpponent):

    def decide(self, env, idx):
        p = env.players[idx]
        stack_bb = p["stack_bb"]
        bucket = stack_bucket(stack_bb)

        # short stack：乱 call
        if bucket == "short":
            if random.random() < 0.85:
                return OppAct.CALL, None
            return OppAct.FOLD, None

        # mid stack
        if random.random() < 0.85:
            return OppAct.CALL, None
        return OppAct.BET, 2

        # deep stack（基本不 PK）


# ==========================================================
# 5. ManiacOpponent（BB-aware 疯子）
# ==========================================================

class ManiacOpponent(BaseOpponent):

    def decide(self, env, idx):
        p = env.players[idx]
        stack_bb = p["stack_bb"]
        bucket = stack_bucket(stack_bb)

        # short stack：少乱来
        if bucket == "short":
            if random.random() < 0.7:
                return OppAct.CALL, None
            return OppAct.PK, None

        # mid stack
        if random.random() < 0.6:
            return OppAct.BET, random.randint(4, 7)
        return OppAct.PK, None

        # deep stack
        if random.random() < 0.7:
            return OppAct.BET, random.randint(6, 10)
        return OppAct.PK, None


# ==========================================================
# 6. TAGOpponent（BB-aware 紧凶）
# ==========================================================

class TAGOpponent(BaseOpponent):

    def decide(self, env, idx):
        p = env.players[idx]
        s = strength(p["cards"])
        stack_bb = p["stack_bb"]
        bucket = stack_bucket(stack_bb)

        # short stack
        if bucket == "short":
            if s > 0.6:
                return OppAct.PK, None
            return OppAct.CALL, None

        # mid stack
        if s < 0.35:
            return OppAct.FOLD, None
        if s < 0.65:
            return OppAct.CALL, None
        return OppAct.BET, random.choice([3, 4, 5])

        # deep stack
        if s < 0.4:
            return OppAct.CALL, None
        return OppAct.BET, random.choice([3, 4, 5])


# ==========================================================
# 7. GTOOpponent（BB-aware GTO-lite）
# ==========================================================

class GTOOpponent(BaseOpponent):

    def decide(self, env, idx):
        p = env.players[idx]
        s = strength(p["cards"])
        stack_bb = p["stack_bb"]
        bucket = stack_bucket(stack_bb)

        # short stack：偏保守
        if bucket == "short":
            if s > 0.6:
                return OppAct.PK, None
            return OppAct.CALL, None

        # mid stack
        if s < 0.3:
            return OppAct.FOLD, None if random.random() < 0.6 else (OppAct.CALL, None)
        if s < 0.6:
            return OppAct.CALL, None
        return OppAct.BET, random.choice([2, 3, 4])

        # deep stack
        if s < 0.35:
            return OppAct.CALL, None
        if s < 0.7:
            return OppAct.CALL, None
        return OppAct.BET, random.choice([3, 4, 5])
