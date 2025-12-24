import torch
import numpy as np
from typing import Any, Dict, Optional, List, Tuple

from ZJHEnvMulti2to9 import ZJHEnvMulti2to9
from agent_a30 import A30Agent
from state_builder_normalized import build_state


# ============================================================
# Action Mask（与训练一致，含兜底）
# ============================================================

def build_action_mask(env: ZJHEnvMulti2to9) -> np.ndarray:
    mask = [1] * env.action_dim
    hero = env.players[0]

    # 第一轮禁止 PK
    if env.round_index == 0:
        mask[12] = 0

    # 已看牌不能再 LOOK（注意：你环境 action=1 是 LOOK）
    if hero["has_seen"]:
        mask[1] = 0

    # 最低下注倍数 & 最低支付金额
    min_unit = env._min_required_unit(hero)
    min_pay = min_unit * env.ante_bb * (2 if hero["has_seen"] else 1)

    # 筹码不足 → 禁止所有 BET（action 2..11 对应 BET_1..BET_10）
    if hero["stack_bb"] < min_pay:
        for a in range(2, 12):
            mask[a] = 0
    else:
        for a in range(2, 12):
            unit = a - 1
            if unit < min_unit:
                mask[a] = 0

    # 兜底：至少允许 FOLD
    if sum(mask) == 0:
        mask[0] = 1

    return np.array(mask, dtype=np.float32)


# ============================================================
# AI Service（线上推理版：decide(payload) 一步到位）
# ============================================================

class ZJHAIService:
    """
    用途：
      - 线上实时决策（HTTP / WS）
      - 每次 decide(payload) 都以 payload 为准覆盖 env 状态
      - 与训练 100% 同构（state_builder_normalized + action mask）

    注意：
      - 一个实例可以服务多次请求，但每次都会覆盖 env
      - 如果你要做“同一局多次 step 推进”，请用 step()，并不要每次都传 payload 覆盖
    """

    def __init__(self, model_path: str, device: str = "cpu"):
        self.device = torch.device(device)

        self.env = ZJHEnvMulti2to9()

        self.agent = A30Agent(
            state_dim=self.env.state_dim,   # 54
            action_dim=self.env.action_dim,
            device=self.device,
        )
        self.agent.load(model_path)
        self.agent.epsilon = 0.0
        # 你的 A30Agent 用的是 target_q_net 命名
        if hasattr(self.agent, "set_eval"):
            self.agent.set_eval()
        else:
            # 兜底
            if hasattr(self.agent, "q_net"):
                self.agent.q_net.eval()
            if hasattr(self.agent, "target_q_net"):
                self.agent.target_q_net.eval()

    # --------------------------------------------------------
    # 工具：把 cards 统一转成 List[Tuple[int,int]]
    # --------------------------------------------------------

    @staticmethod
    def _normalize_cards(cards: Any) -> Optional[List[Tuple[int, int]]]:
        """
        允许：
          - [(14,0),(13,1),(12,2)]
          - [[14,0],[13,1],[12,2]]
        返回：
          - [(14,0),(13,1),(12,2)]
        """
        if cards is None:
            return None
        if not isinstance(cards, (list, tuple)):
            raise ValueError(f"cards must be list/tuple, got {type(cards)}")

        out: List[Tuple[int, int]] = []
        for c in cards:
            if not isinstance(c, (list, tuple)) or len(c) != 2:
                raise ValueError(f"invalid card item: {c}, expected [rank,suit] or (rank,suit)")
            r = int(c[0])
            s = int(c[1])
            out.append((r, s))

        return out

    # --------------------------------------------------------
    # 外部状态 → env（核心）
    # --------------------------------------------------------

    def load_external_state(self, external_state: Dict[str, Any]) -> None:
        env = self.env

        if "ante_bb" not in external_state:
            raise ValueError("missing required field: ante_bb")
        if "players" not in external_state:
            raise ValueError("missing required field: players")

        # ---------- 基础参数 ----------
        env.ante_bb = float(external_state["ante_bb"])
        env.round_index = int(external_state.get("round_index", 0))
        env.current_bet_unit = int(external_state.get("current_bet_unit", 1))
        env.first_round_open_unit = bool(external_state.get("first_round_open_unit", False))

        players_in = external_state["players"]
        if not isinstance(players_in, list) or len(players_in) < 2:
            raise ValueError("players must be a list with at least 2 players (hero + opponents)")

        env.num_players = len(players_in)

        # ---------- players ----------
        env.players = []
        for i, p in enumerate(players_in):
            if not isinstance(p, dict):
                raise ValueError(f"players[{i}] must be object/dict")

            cards = self._normalize_cards(p.get("cards"))
            # hero 必须有 cards
            if i == 0 and (cards is None or len(cards) != 3):
                raise ValueError("hero (players[0]) must provide 3 cards")

            env.players.append({
                "cards": cards,
                "stack_bb": float(p.get("stack_bb", 0.0)),
                "bet_bb": float(p.get("bet_bb", 0.0)),
                "alive": bool(p.get("alive", True)),
                "has_seen": bool(p.get("has_seen", False)),
                "last_act": int(p.get("last_act", 0)),
                "contrib_bb": float(p.get("bet_bb", 0.0)),
            })

        # ---------- 补齐空位 ----------
        while len(env.players) < env.max_players:
            env.players.append({
                "cards": None,
                "stack_bb": 0.0,
                "bet_bb": 0.0,
                "alive": False,
                "has_seen": False,
                "last_act": 0,
                "contrib_bb": 0,
            })

        # ---------- 全局缓存字段（必须重算） ----------
        env.alive_cnt = sum(1 for p in env.players[:env.num_players] if p["alive"])
        env.max_bet_bb = max((p["bet_bb"] for p in env.players[:env.num_players] if p["alive"]), default=0.0)
        env.pot_bb = sum(p["bet_bb"] for p in env.players[:env.num_players])

        # ---------- Hero 牌力缓存 ----------
        env._cache_hero_hand()

    # --------------------------------------------------------
    # AI 决策（✅接收 payload，一步到位）
    # --------------------------------------------------------

    def decide(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        payload：外部传入的牌局状态快照（见 load_external_state）
        返回：
          {
            "action": int,
            "action_name": str,
            "mask": [...],
          }
        """
        # 1) 先用 payload 覆盖 env
        self.load_external_state(payload)

        # 2) 与训练一致：build_state + mask
        state = build_state(self.env)
        mask = build_action_mask(self.env)

        # 3) 选动作
        action = self.agent.select_action(state, mask)

        return {
            "action": int(action),
            "action_name": self._action_to_name(int(action)),
            "mask": mask.tolist(),
        }

    # --------------------------------------------------------
    # 可选：服务端推进一步（调试 / 模拟）
    # --------------------------------------------------------

    def step(self, action: int):
        return self.env.step(action)

    # --------------------------------------------------------
    # 工具
    # --------------------------------------------------------

    @staticmethod
    def _action_to_name(action: int) -> str:
        if action == 0:
            return "FOLD"
        if action == 1:
            return "LOOK"
        if 2 <= action <= 11:
            return f"BET_{action-1}"
        if action == 12:
            return "PK"
        if action == 13:
            return "COMPARE_ALL"
        return "UNKNOWN"
