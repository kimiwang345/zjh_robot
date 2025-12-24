# ZJHEnvMulti2to9.py
import random
import math
import numpy as np
from enum import Enum


# ============================================================
# 对手动作类型
# ============================================================

class OppAct(Enum):
    FOLD = 0
    LOOK = 1
    BET = 2        # param = unit (1~10)
    PK = 3
    COMPARE_ALL = 4


# ============================================================
# 牌型评估
# ============================================================

def evaluate_hand(cards):
    """
    返回:
      type: 5 豹子, 4 顺金, 3 金花, 2 顺子, 1 对子, 0 高牌
      ranks: [hi, mid, lo]
    """
    ranks = sorted([c[0] for c in cards], reverse=True)
    suits = [c[1] for c in cards]
    r1, r2, r3 = ranks

    is_flush = len(set(suits)) == 1
    is_straight = (r1 == r2 + 1 == r3 + 2) or (r1 == 14 and r2 == 3 and r3 == 2)

    if r1 == r2 == r3:
        return 5, ranks
    if is_flush and is_straight:
        return 4, ranks
    if is_flush:
        return 3, ranks
    if is_straight:
        return 2, ranks
    if r1 == r2 or r2 == r3:
        return 1, ranks
    return 0, ranks


def compare_hands(c1, c2):
    t1, r1 = evaluate_hand(c1)
    t2, r2 = evaluate_hand(c2)
    if t1 != t2:
        return 1 if t1 > t2 else -1
    return 1 if r1 > r2 else -1


# ============================================================
# ZJHEnvMulti2to9（最终版）
# ============================================================

class ZJHEnvMulti2to9:

    def __init__(self,
                 min_players=2,
                 max_players=9,
                 max_round=20):

        self.min_players = min_players
        self.max_players = max_players
        self.max_round = max_round

        self.action_dim = 14
        self.state_dim = 54

        self.reset()

    # ========================================================
    # 初始化
    # ========================================================

    def _sample_env_params(self):
        self.num_players = random.randint(self.min_players, self.max_players)

        # 初始筹码（BB，LogUniform）
        lo, hi = math.log(20), math.log(300)
        self.starting_stack_bb = math.exp(random.uniform(lo, hi))

        # ante = 下注基数
        self.ante_bb = random.choice([0.5, 1.0, 2.0])

    def _init_players(self):
        self.players = []
        for _ in range(self.num_players):
            self.players.append({
                "cards": None,
                "stack_bb": self.starting_stack_bb,
                "contrib_bb": 0.0,
                "alive": True,
                "has_seen": False,
                "last_act": 0,
            })

        while len(self.players) < self.max_players:
            self.players.append({
                "cards": None,
                "stack_bb": 0.0,
                "contrib_bb": 0.0,
                "alive": False,
                "has_seen": False,
                "last_act": 0,
            })

    def _deal_cards(self):
        deck = [(r, s) for r in range(2, 15) for s in range(4)]
        random.shuffle(deck)
        for i in range(self.num_players):
            self.players[i]["cards"] = deck[i * 3:(i + 1) * 3]

    # ========================================================
    # Hero 牌力缓存
    # ========================================================

    def _cache_hero_hand(self):
        hero = self.players[0]
        t, ranks = evaluate_hand(hero["cards"])
        hi, mid, lo = ranks
        self.hero_hand_type = t
        self.hero_hi = hi
        self.hero_mid = mid
        self.hero_lo = lo
        self.hero_strength = (t + hi / 14.0) / 6.0

    # ========================================================
    # reset
    # ========================================================

    def reset(self):
        self._sample_env_params()
        self._init_players()
        self._deal_cards()

        self.round_index = 0
        self.done = False

        self.pot_bb = 0.0
        self.current_bet_unit = 1
        self.first_round_open_unit = False

        self.alive_cnt = self.num_players

        for p in self.players[:self.num_players]:
            p["stack_bb"] -= self.ante_bb
            p["contrib_bb"] = self.ante_bb
            self.pot_bb += self.ante_bb

        self.hero_contrib_bb = self.ante_bb

        self._cache_hero_hand()

        return self._get_state()

    # ========================================================
    # 对手绑定
    # ========================================================

    def bind_opponents(self, opponents):
        """
        opponents: list, len = max_players
        index 0 必须是 None
        """
        self.opponents = opponents

    # ========================================================
    # 规则核心
    # ========================================================

    def _min_required_unit(self, player):
        """
        计算玩家当前最小允许下注倍数
        """
        unit = self.current_bet_unit

        # 看牌倍数规则
        if player["has_seen"]:
            if not self.first_round_open_unit:
                unit *= 2
        else:
            if self.first_round_open_unit:
                unit = max(1, unit // 2)

        return max(1, min(unit, 10))

    def _do_bet(self, idx, unit):
        p = self.players[idx]

        min_unit = self._min_required_unit(p)
        unit = max(min_unit, min(10, unit))

        pay = unit * self.ante_bb
        if p["has_seen"]:
            pay *= 2

        if p["stack_bb"] < pay:
            return False

        p["stack_bb"] -= pay
        p["contrib_bb"] += pay
        self.pot_bb += pay

        self.current_bet_unit = unit
        if self.round_index == 0:
            self.first_round_open_unit = True

        p["last_act"] = 2
        return True

    # ========================================================
    # PK / COMPARE
    # ========================================================

    def _pk_one(self):
        hero_cards = self.players[0]["cards"]
        alive = [i for i in range(1, self.num_players)
                 if self.players[i]["alive"]]

        if not alive:
            return self._reward_win()

        opp = random.choice(alive)
        res = compare_hands(hero_cards, self.players[opp]["cards"])
        return self._reward_win() if res >= 0 else self._reward_lose()

    def _compare_all(self):
        hero_cards = self.players[0]["cards"]
        for i in range(1, self.num_players):
            p = self.players[i]
            if p["alive"]:
                if compare_hands(hero_cards, p["cards"]) < 0:
                    return self._reward_lose()
        return self._reward_win()

    # ========================================================
    # reward
    # ========================================================

    def _reward_win(self):
        self.done = True
        return self.pot_bb - self.hero_contrib_bb

    def _reward_lose(self):
        self.done = True
        return -self.hero_contrib_bb

    # ========================================================
    # step
    # ========================================================

    def step(self, action):
        if self.done:
            return self._get_state(), 0.0, True, {}

        hero = self.players[0]

        # -------- Hero 行为 --------
        if action == 0:  # FOLD
            hero["alive"] = False
            self.alive_cnt -= 1
            return self._get_state(), self._reward_lose(), True, {}

        elif action == 1:  # LOOK
            hero["has_seen"] = True

        elif 2 <= action <= 11:  # BET
            unit = action - 1
            self._do_bet(0, unit)

        elif action == 12:  # PK
            return self._get_state(), self._pk_one(), True, {}

        elif action == 13:  # COMPARE
            return self._get_state(), self._compare_all(), True, {}

        # -------- 对手轮流行动 --------
        if hasattr(self, "opponents"):
            for i in range(1, self.num_players):
                p = self.players[i]
                if not p["alive"]:
                    continue

                opp = self.opponents[i]
                if opp is None:
                    continue

                act, param = opp.decide(self, i)

                if act == OppAct.FOLD:
                    p["alive"] = False
                    self.alive_cnt -= 1
                    continue

                if act == OppAct.LOOK:
                    p["has_seen"] = True
                    continue

                if act == OppAct.BET:
                    self._do_bet(i, param)
                    continue

                if act == OppAct.PK:
                    return self._get_state(), self._pk_one(), True, {}

                if act == OppAct.COMPARE_ALL:
                    return self._get_state(), self._compare_all(), True, {}

        # -------- 轮数推进 --------
        self.round_index += 1
        if self.round_index >= self.max_round:
            return self._get_state(), self._compare_all(), True, {}

        if self.alive_cnt == 1:
            return self._get_state(), self._reward_win(), True, {}

        return self._get_state(), 0.0, False, {}

    # ========================================================
    # State
    # ========================================================

    def _get_state(self):
        """
        原始 state（如你用 state_builder_normalized 可忽略）
        """
        hero = self.players[0]

        state = [
            float(self.hero_hand_type),
            float(self.hero_hi),
            float(self.hero_mid),
            float(self.hero_lo),
            float(self.hero_strength),
            float(self.round_index),
            float(self.num_players),
            float(self.alive_cnt),
            float(self.ante_bb),
            float(self.pot_bb),
            float(self.current_bet_unit),
            float(hero["has_seen"]),
            float(hero["stack_bb"]),
            float(hero["contrib_bb"]),
        ]

        for i in range(1, self.max_players):
            if i < self.num_players:
                p = self.players[i]
                state.extend([
                    float(p["alive"]),
                    float(p["has_seen"]),
                    float(p["stack_bb"]),
                    float(p["contrib_bb"]),
                    float(p["last_act"]),
                ])
            else:
                state.extend([0, 0, 0, 0, 0])

        return np.array(state, dtype=np.float32)
