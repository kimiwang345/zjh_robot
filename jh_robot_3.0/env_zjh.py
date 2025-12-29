# env_zjh.py
import random
from dataclasses import dataclass
from typing import List, Optional

from cards import Card, hand_strength
from config.config_loader import HotConfig


@dataclass
class Player:
    seat: int
    alive: bool = True
    seen: bool = False
    stack: int = 0
    bet: int = 0
    cards: Optional[List[Card]] = None


class ZJHEnv:
    def __init__(self, num_players: int, seed: int = 42):
        self.rng = random.Random(seed)
        self.num_players = num_players

        self.opp_cfg = HotConfig("config/opponent_config.json")
        self.opponent_tier = "weak"

    # =========================
    # 对手强度设置
    # =========================
    def set_opponent_tier(self, tier: str):
        self.opponent_tier = tier

    # =========================
    # 重置牌局
    # =========================
    def reset(self, base_bet: int, init_stack: int):
        self.base_bet = base_bet
        self.round = 0

        self.players = [
            Player(seat=i, stack=init_stack)
            for i in range(self.num_players)
        ]

        # 发牌
        deck = [Card(r, s) for r in range(2, 15) for s in range(4)]
        self.rng.shuffle(deck)
        for p in self.players:
            p.cards = [deck.pop(), deck.pop(), deck.pop()]
            p.bet = base_bet
            p.stack -= base_bet

        self.pot = base_bet * self.num_players
        self.current_seat = 0  # AI 永远先手

        return self.observe(0)

    # =========================
    # 生成 min_mul（训练简化版）
    # =========================
    def _calc_min_mul(self, player: Player) -> int:
        # 训练环境中：看牌 2，没看牌 1
        return 2 if player.seen else 1

    # =========================
    # 对手策略
    # =========================
    def _opponent_action(self, player: Player):
        cfg = self.opp_cfg.get()["tiers"][self.opponent_tier]
        r = self.rng.random()

        if r < cfg["fold_rate"]:
            return ("FOLD", None)

        if r < cfg["fold_rate"] + cfg["pk_rate"]:
            return ("PK", None)

        # BET
        min_mul = self._calc_min_mul(player)
        max_ratio = cfg["max_bet_ratio"]
        max_mul = int(min_mul * max_ratio)

        if not player.seen:
            max_mul = min(max_mul, 10)
        else:
            max_mul = min(max_mul, 20)
            if max_mul % 2 == 1:
                max_mul -= 1

        bet_mul = self.rng.randint(min_mul, max(max_mul, min_mul))
        return ("BET", bet_mul)

    # =========================
    # 观测
    # =========================
    def observe(self, seat: int):
        me = self.players[seat]

        return {
            "base_bet": self.base_bet,
            "pot_total": self.pot,
            "round": self.round,
            "alive_players": sum(p.alive for p in self.players),
            "me": {
                "seen": me.seen,
                "stack": me.stack,
                "bet": me.bet,
                "min_mul": self._calc_min_mul(me),
                "cards": (
                    [{"rank": c.rank, "suit": c.suit} for c in me.cards]
                    if me.seen else None
                )
            },
            "opponents": [
                {
                    "seat": p.seat,
                    "alive": p.alive,
                    "seen": p.seen,
                    "stack": p.stack,
                    "bet": p.bet
                }
                for p in self.players
                if p.seat != seat and p.alive
            ]
        }

    # =========================
    # 执行动作
    # =========================
    def step(self, seat: int, action: str, bet_mul: Optional[int] = None, pk_target: Optional[int] = None):
        player = self.players[seat]

        if action == "FOLD":
            player.alive = False

        elif action == "BET":
            cost = bet_mul * self.base_bet
            cost = min(cost, player.stack)
            player.stack -= cost
            player.bet += cost
            self.pot += cost

        elif action == "PK":
            # 简化：随机选一个活人
            targets = [p for p in self.players if p.alive and p.seat != seat]
            if targets:
                target = self.rng.choice(targets)
                if hand_strength(player.cards) >= hand_strength(target.cards):
                    target.alive = False
                else:
                    player.alive = False

        # ===== 轮到下一个玩家（含对手自动行动）=====
        self.round += 1
        self.current_seat = (seat + 1) % self.num_players

        while self.current_seat != 0:
            opp = self.players[self.current_seat]
            if opp.alive:
                act, mul = self._opponent_action(opp)
                if act == "FOLD":
                    opp.alive = False
                elif act == "BET":
                    cost = mul * self.base_bet
                    cost = min(cost, opp.stack)
                    opp.stack -= cost
                    opp.bet += cost
                    self.pot += cost
                elif act == "PK":
                    targets = [p for p in self.players if p.alive and p.seat != opp.seat]
                    if targets:
                        target = self.rng.choice(targets)
                        if hand_strength(opp.cards) >= hand_strength(target.cards):
                            target.alive = False
                        else:
                            opp.alive = False

            self.current_seat = (self.current_seat + 1) % self.num_players

        done = sum(p.alive for p in self.players) <= 1 or self.round >= 20
        winner = None
        if done:
            alive = [p for p in self.players if p.alive]
            winner = alive[0].seat if alive else None

        return done, winner
