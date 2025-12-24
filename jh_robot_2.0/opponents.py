from __future__ import annotations
import random
from typing import Dict, Any, List

class BaseOpponent:
    def __init__(self, rng: random.Random):
        self.rng = rng

    def act(self, obs: Dict[str, Any], legal: List[Dict[str, Any]]) -> Dict[str, Any]:
        raise NotImplementedError

class TightOpponent(BaseOpponent):
    """
    稳健：低倍跟，遇到压力更倾向弃牌；少PK
    """
    def act(self, obs, legal):
        # 先可选 SEE（有概率看牌）
        for a in legal:
            if a["type"] == "SEE" and self.rng.random() < 0.35:
                return a

        if obs["force_allin_or_fold"]:
            # 仅 FOLD / ALL_IN
            return {"type": "FOLD"} if self.rng.random() < 0.8 else {"type": "ALL_IN"}

        # 找最低 BET
        bet_actions = [a for a in legal if a["type"] == "BET"]
        pk_actions = [a for a in legal if a["type"] == "PK"]

        if bet_actions:
            bet_actions.sort(key=lambda x: x["mul"])
            min_bet = bet_actions[0]
            # 压力大时更容易弃牌
            if obs["last_bet_mul"] >= 8 and self.rng.random() < 0.5:
                return {"type": "FOLD"}
            # 偶尔小加注
            if self.rng.random() < 0.15 and len(bet_actions) >= 3:
                return bet_actions[min(2, len(bet_actions)-1)]
            return min_bet

        # 无法 BET 就 PK 或弃牌
        if pk_actions and self.rng.random() < 0.1:
            return self.rng.choice(pk_actions)
        return {"type": "FOLD"}

class AggroOpponent(BaseOpponent):
    """
    激进：更喜欢抬倍、PK
    """
    def act(self, obs, legal):
        for a in legal:
            if a["type"] == "SEE" and self.rng.random() < 0.55:
                return a

        if obs["force_allin_or_fold"]:
            return {"type": "ALL_IN"} if self.rng.random() < 0.7 else {"type": "FOLD"}

        bet_actions = [a for a in legal if a["type"] == "BET"]
        pk_actions = [a for a in legal if a["type"] == "PK"]

        if pk_actions and self.rng.random() < 0.25:
            return self.rng.choice(pk_actions)

        if bet_actions:
            bet_actions.sort(key=lambda x: x["mul"])
            # 更倾向中高倍
            pick = int(0.7 * (len(bet_actions)-1))
            if self.rng.random() < 0.25:
                pick = len(bet_actions)-1
            return bet_actions[pick]

        return {"type": "FOLD"}

class MixedOpponent(BaseOpponent):
    """
    混合：介于两者之间
    """
    def act(self, obs, legal):
        for a in legal:
            if a["type"] == "SEE" and self.rng.random() < 0.45:
                return a

        if obs["force_allin_or_fold"]:
            return {"type": "FOLD"} if self.rng.random() < 0.6 else {"type": "ALL_IN"}

        bet_actions = [a for a in legal if a["type"] == "BET"]
        pk_actions = [a for a in legal if a["type"] == "PK"]

        if pk_actions and self.rng.random() < 0.15:
            return self.rng.choice(pk_actions)

        if bet_actions:
            bet_actions.sort(key=lambda x: x["mul"])
            # 60% 最低跟，40% 小加注
            if self.rng.random() < 0.6:
                return bet_actions[0]
            return bet_actions[min(1, len(bet_actions)-1)]

        return {"type": "FOLD"}
