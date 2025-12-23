# ZJHEnvMulti14_8p.py
import numpy as np
import random
from enum import Enum


# ============================================================
# 对手动作类型
# ============================================================

class OppAct(Enum):
    FOLD = 0
    CALL = 1
    LOOK = 2
    BET = 3        # param = 倍数（2~10）
    PK = 4
    COMPARE_ALL = 5


# ============================================================
# 牌型评估（保持原规则不变）
# ============================================================

def evaluate_hand(cards):
    """
    cards: [(rank, suit), ...]  rank: 2~14, suit: 0~3
    返回: (hand_type, [hi, mid, lo])
      hand_type:
        5: 豹子
        4: 顺金
        3: 金花
        2: 顺子
        1: 对子
        0: 高牌
    """
    ranks = sorted([c[0] for c in cards], reverse=True)
    suits = [c[1] for c in cards]
    r1, r2, r3 = ranks

    is_flush = (len(set(suits)) == 1)
    is_straight = (r1 == r2 + 1 == r3 + 2) or (r1 == 14 and r2 == 3 and r3 == 2)

    if r1 == r2 == r3:
        return 5, ranks          # 豹子
    if is_flush and is_straight:
        return 4, ranks          # 顺金
    if is_flush:
        return 3, ranks          # 金花
    if is_straight:
        return 2, ranks          # 顺子
    if r1 == r2 or r2 == r3:
        return 1, ranks          # 对子
    return 0, ranks              # 高牌


def compare_hands(c1, c2):
    """1: c1赢, -1: c2赢"""
    t1, r1 = evaluate_hand(c1)
    t2, r2 = evaluate_hand(c2)
    if t1 != t2:
        return 1 if t1 > t2 else -1
    return 1 if r1 > r2 else -1


# ============================================================
# 多人（3~8人）+ 14 动作 扎金花环境（高性能版）
# Hero = players[0]
# ============================================================

class ZJHEnvMulti14_8p:

    def __init__(self,
                 starting_stack=100,   # 初始筹码
                 ante=1,              # 底注
                 max_round=20,        # 最多回合数
                 min_players=3,       # 最少玩家数
                 max_players=8):      # 最多玩家数

        self.starting_stack = starting_stack
        self.ante = ante
        self.max_round = max_round
        self.min_players = min_players
        self.max_players = max_players

        # 14 动作：
        # 0 FOLD, 1 CALL, 2 LOOK,
        # 3~11 BET_2X~BET_10X, 12 PK_ONE, 13 COMPARE_ALL
        self.action_dim = 14

        # 状态：hero 14 维 + 每个对手 5 维 × 7 = 49
        self.state_dim = 49

        # 缓存字段（避免重复遍历）
        self.max_bet = 0.0         # 当前所有存活玩家的最大下注
        self.alive_cnt = 0         # 当前存活玩家数量（含 Hero）

        # Hero 牌力缓存（整局固定）
        self.hero_hand_type = 0.0
        self.hero_hi = 0.0
        self.hero_mid = 0.0
        self.hero_lo = 0.0
        self.hero_strength = 0.0

        self.reset()

    # ------------------ 玩家初始化 & 发牌 ------------------

    def _init_players(self):
        self.num_players = random.randint(self.min_players, self.max_players)
        self.players = []
        for _ in range(self.num_players):
            self.players.append(dict(
                cards=None,
                stack=self.starting_stack,
                bet=0.0,
                alive=True,
                has_seen=False,
                last_act=0,
            ))
        # 补足到 max_players 个座位（空位）
        while len(self.players) < self.max_players:
            self.players.append(dict(
                cards=None,
                stack=0.0,
                bet=0.0,
                alive=False,
                has_seen=False,
                last_act=0,
            ))

    def _deal_cards(self):
        deck = [(r, s) for r in range(2, 15) for s in range(4)]
        random.shuffle(deck)
        for i in range(self.num_players):
            self.players[i]["cards"] = deck[i * 3:(i + 1) * 3]

    # ------------------ Hero 牌力缓存 ------------------

    def _cache_hero_hand(self):
        """
        一局只算一次 Hero 牌力，之后直接复用。
        """
        hero = self.players[0]
        t, ranks = evaluate_hand(hero["cards"])
        hi, mid, lo = ranks
        strength = (t + hi / 14.0) / 6.0

        self.hero_hand_type = float(t)
        self.hero_hi = float(hi)
        self.hero_mid = float(mid)
        self.hero_lo = float(lo)
        self.hero_strength = float(strength)

    # ------------------ 全局下注缓存维护 ------------------

    def _recompute_max_bet(self):
        """
        当有玩家弃牌时，如果弃牌的是最大下注者，需要重新扫一遍。
        只在 FOLD 场景下调用，频率很低。
        """
        max_b = 0.0
        for p in self.players[:self.num_players]:
            if p["alive"] and p["bet"] > max_b:
                max_b = p["bet"]
        self.max_bet = float(max_b)

    # ------------------ 状态编码 ------------------

    def _get_state(self):
        hero = self.players[0]

        # 使用缓存好的 Hero 牌力信息
        t = self.hero_hand_type
        hi = self.hero_hi
        mid = self.hero_mid
        lo = self.hero_lo
        strength = self.hero_strength

        # max_bet / alive_cnt 用缓存
        max_bet = self.max_bet
        alive_cnt = float(self.alive_cnt)

        state = [
            t,                      # 0  hero_hand_type
            hi,                     # 1
            mid,                    # 2
            lo,                     # 3
            strength,               # 4  牌力强度
            float(self.round_index),    # 5
            float(self.num_players),    # 6
            alive_cnt,                  # 7
            float(self.ante),           # 8
            float(self.pot),            # 9
            float(max_bet),             # 10
            float(int(hero["has_seen"])),  # 11
            float(hero["stack"]),       # 12
            float(hero["bet"]),         # 13
        ]

        # 对手（1~7）
        for idx in range(1, self.max_players):
            if idx < self.num_players:
                p = self.players[idx]
                if p["alive"]:
                    state.extend([
                        1.0,                         # alive
                        float(int(p["has_seen"])),   # seen
                        float(p["stack"]),           # stack
                        float(p["bet"]),             # bet
                        float(p["last_act"]),        # last_act
                    ])
                else:
                    state.extend([0.0, 0.0, 0.0, 0.0, float(p["last_act"])])
            else:
                state.extend([0.0, 0.0, 0.0, 0.0, 0.0])

        return np.array(state, dtype=np.float32)

    # ------------------ 奖励 ------------------

    def _reward_win(self):
        return float(self.pot - self.hero_contrib)

    def _reward_lose(self):
        return float(-self.hero_contrib)

    # ------------------ PK & 通比 ------------------

    def _compare_all(self):
        hero_cards = self.players[0]["cards"]
        win = True
        for i in range(1, self.num_players):
            p = self.players[i]
            if not p["alive"]:
                continue
            if compare_hands(hero_cards, p["cards"]) < 0:
                win = False
                break
        self.done = True
        return self._reward_win() if win else self._reward_lose()

    def _pk_one(self):
        alive_idx = [i for i in range(1, self.num_players)
                     if self.players[i]["alive"]]
        if not alive_idx:
            self.done = True
            return self._reward_win()
        opp = random.choice(alive_idx)
        hero_cards = self.players[0]["cards"]
        opp_cards = self.players[opp]["cards"]
        res = compare_hands(hero_cards, opp_cards)
        self.done = True
        return self._reward_win() if res >= 0 else self._reward_lose()

    # ------------------ 对手绑定 ------------------

    def bind_opponents(self, opponents):
        """
        opponents: 长度 = max_players
        index=0 为 None（Hero），1~7 为对手实例
        """
        self.opponents = opponents

    # ------------------ reset ------------------

    def reset(self):
        self._init_players()
        self._deal_cards()

        self.pot = 0.0
        self.hero_contrib = 0.0
        self.round_index = 0
        self.done = False

        # 前注
        for i in range(self.num_players):
            p = self.players[i]
            p["stack"] -= self.ante
            p["bet"] = 0.0
            p["alive"] = True
            p["has_seen"] = False
            p["last_act"] = 0
            self.pot += self.ante
            if i == 0:
                self.hero_contrib += self.ante

        # 缓存全局信息
        self.alive_cnt = self.num_players
        self.max_bet = 0.0   # 所有人 bet = 0
        self._cache_hero_hand()

        return self._get_state()

    # ------------------ Hero 下注工具 ------------------

    def _hero_bet(self, mult):
        hero = self.players[0]
        amount = mult * self.ante
        if hero["has_seen"]:
            amount *= 2

        max_bet = self.max_bet
        desired = max(max_bet, hero["bet"]) + amount
        add = max(0.0, desired - hero["bet"])
        add = min(add, hero["stack"])

        if add <= 0.0:
            return

        hero["stack"] -= add
        hero["bet"] += add
        hero["last_act"] = 3
        self.pot += add
        self.hero_contrib += add

        if hero["bet"] > self.max_bet:
            self.max_bet = hero["bet"]

    # ------------------ 敌人执行逻辑动作 ------------------

    def _enemy_do(self, idx, act: OppAct, param):
        p = self.players[idx]
        if not p["alive"]:
            return 0.0  # no-op

        p["last_act"] = 0

        max_bet = self.max_bet
        need = max_bet - p["bet"]

        if act == OppAct.FOLD:
            if p["alive"]:
                p["alive"] = False
                p["last_act"] = 1
                self.alive_cnt -= 1
                # 如果弃牌玩家可能是最大下注者，重算 max_bet
                self._recompute_max_bet()
            return 0.0

        if act == OppAct.CALL:
            call_amt = min(max(0.0, need), p["stack"])
            if call_amt > 0.0:
                p["stack"] -= call_amt
                p["bet"] += call_amt
                p["last_act"] = 2
                self.pot += call_amt
                if p["bet"] > self.max_bet:
                    self.max_bet = p["bet"]
            return 0.0

        if act == OppAct.LOOK:
            p["has_seen"] = True
            p["last_act"] = 4
            return 0.0

        if act == OppAct.BET:
            mult = max(2, min(10, int(param)))  # 限制倍数2~10
            amt = mult * self.ante
            if p["has_seen"]:
                amt *= 2

            desired = max(max_bet, p["bet"]) + amt
            add = max(0.0, desired - p["bet"])
            add = min(add, p["stack"])

            if add > 0.0:
                p["stack"] -= add
                p["bet"] += add
                p["last_act"] = 3
                self.pot += add
                if p["bet"] > self.max_bet:
                    self.max_bet = p["bet"]
            return 0.0

        if act == OppAct.PK:
            return -1.0   # 触发 PK_ONE

        if act == OppAct.COMPARE_ALL:
            return -2.0   # 触发通比

        return 0.0

    # ------------------ 敌人轮流行动 ------------------

    def _simulate_opponents(self):
        for i in range(1, self.num_players):
            p = self.players[i]
            if not p["alive"]:
                continue

            if not hasattr(self, "opponents") or self.opponents[i] is None:
                continue

            act, param = self.opponents[i].decide(self, i)
            res = self._enemy_do(i, act, param)

            if res == -1.0:  # PK
                rew = self._pk_one()
                return True, rew
            if res == -2.0:  # 通比
                rew = self._compare_all()
                return True, rew

            # 若只剩 hero
            if self.alive_cnt == 1 and self.players[0]["alive"]:
                self.done = True
                return True, self._reward_win()

        return False, 0.0

    # ------------------ step ------------------

    def step(self, action):
        if self.done:
            return self._get_state(), 0.0, True, {}

        hero = self.players[0]
        info = {}

        # 0: FOLD
        if action == 0:
            if hero["alive"]:
                hero["alive"] = False
                self.alive_cnt -= 1
                self._recompute_max_bet()
            self.done = True
            return self._get_state(), self._reward_lose(), True, info

        # 1: CALL
        if action == 1:
            maxb = self.max_bet
            need = max(0.0, maxb - hero["bet"])
            add = min(need, hero["stack"])
            if add > 0.0:
                hero["stack"] -= add
                hero["bet"] += add
                hero["last_act"] = 2
                self.pot += add
                self.hero_contrib += add
                if hero["bet"] > self.max_bet:
                    self.max_bet = hero["bet"]

        # 2: LOOK
        elif action == 2:
            hero["has_seen"] = True
            hero["last_act"] = 4

        # 3~11: BET_2X ~ BET_10X
        elif 3 <= action <= 11:
            mapping = {
                3: 2,
                4: 3,
                5: 4,
                6: 5,
                7: 6,
                8: 7,
                9: 8,
                10: 9,
                11: 10,
            }
            mult = mapping[action]
            self._hero_bet(mult)

        # 12: PK_ONE
        elif action == 12:
            rew = self._pk_one()
            return self._get_state(), rew, True, info

        # 13: COMPARE_ALL
        elif action == 13:
            rew = self._compare_all()
            return self._get_state(), rew, True, info

        # 敌人行动
        done, rew = self._simulate_opponents()
        if done:
            return self._get_state(), rew, True, info

        # 轮数+1，超出则通比
        self.round_index += 1
        if self.round_index >= self.max_round:
            rew = self._compare_all()
            return self._get_state(), rew, True, info

        return self._get_state(), 0.0, False, info
