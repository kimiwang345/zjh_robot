# buddy_ai.py
# ------------------------------------------------------------
# “陪玩”AI（规则 + 概率 + 记忆 + 情绪 + 可控输赢）
#
# 目标：
# 1) 像真人：不固定套路、会抓鸡、会情绪化、会根据回合/下注压力调整
# 2) 可控输赢：want_win_rate ∈ [0,1]，越大越倾向“打赢”，越小越倾向“打输”
# 3) 可直接线上接入：输入 obs + legal_actions，输出一个 action dict
#
# 依赖：
# - 可选：zjh_hand_eval.compare_hands / hand_rank（如果你有更细的牌力评估）
# - 否则只用 obs["hand_rank"] (0~5) 也能工作
# ------------------------------------------------------------

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import math
import random
import time


Action = Dict[str, Any]


# ---------------------------
# 可调参数（你后续可按线上感觉调）
# ---------------------------
@dataclass
class BuddyAIConfig:
    seed: int = 42

    # 行为噪声：越大越“人”
    base_noise_sigma: float = 0.12   # 推荐 0.08 ~ 0.20
    random_action_rate: float = 0.02 # 纯随机动作概率（更像手滑）

    # “抓鸡”倾向：桌子越多人，抓鸡更少；但仍保留少量
    bluff_base: float = 0.22         # 基础抓鸡概率上限（会被各种修正）
    bluff_min: float = 0.03

    # “看牌”倾向：不是每次都秒看
    see_prob_round1: float = 0.35
    see_prob_round2: float = 0.25
    see_prob_late: float = 0.15

    # 下注倍率策略：偏“人类”的少数档位，而不是每次最大
    bet_style_aggressive: float = 0.65  # 越大越偏加注
    bet_style_conservative: float = 0.35

    # 情绪模型：连输/被PK输会更激进
    emotion_decay: float = 0.92  # 每局衰减
    tilt_gain_loss: float = 0.18
    tilt_gain_pk_loss: float = 0.28
    calm_gain_win: float = 0.10

    # 防“脚本”：同样局面不总做同一动作
    anti_determinism: float = 0.15  # 0~0.3

    # 当 want_win_rate 要求“偏输”时，降低PK和大注的程度
    lose_mode_pk_penalty: float = 0.55
    lose_mode_bigbet_penalty: float = 0.45

    # 当 want_win_rate 要求“偏赢”时，提高PK和大注的程度
    win_mode_pk_bonus: float = 0.40
    win_mode_bigbet_bonus: float = 0.35

    # PK触发：越后期越喜欢PK（尤其2人）
    pk_round_boost: float = 0.06  # 每回合加成

    # 多人桌 PK 更谨慎
    pk_multi_penalty: float = 0.22

    # 允许在牌小也PK（你明确要求：顺子以下也要有PK时机）
    pk_low_hand_floor: float = 0.08  # 高牌也可能PK（特别是后期/想赢/对手弱）

    # 允许 all-in 的“人味”概率（仅在合法时）
    allin_spice: float = 0.06


@dataclass
class BuddyAIMemory:
    # 简化记忆：只存一个对手 seat（2人桌）或全局风格估计
    # aggression_est: 0~1 越高表示对手越激进（更爱下注/PK）
    aggression_est: float = 0.5

    # 记录最近若干局结果，用于控制目标胜率
    recent_results: List[int] = field(default_factory=list)  # 1 win, 0 lose
    recent_max: int = 200

    # 情绪（tilt）：[-1, +1]，越大越“上头”
    tilt: float = 0.0

    # 反确定性：记录上一手动作（避免连着做同一件事）
    last_action_type: Optional[str] = None

    def push_result(self, win: bool):
        self.recent_results.append(1 if win else 0)
        if len(self.recent_results) > self.recent_max:
            self.recent_results.pop(0)

    def win_rate(self) -> float:
        if not self.recent_results:
            return 0.5
        return sum(self.recent_results) / len(self.recent_results)


# ============================================================
# 主类：BuddyAI
# ============================================================
class BuddyAI:
    """
    使用方式：
        ai = BuddyAI()
        action = ai.decide(obs, legal_actions, want_win_rate=0.5)
        # action: {"type":"BET","mul":4} / {"type":"PK","target":1} / ...
    """

    def __init__(self, cfg: Optional[BuddyAIConfig] = None):
        self.cfg = cfg or BuddyAIConfig()
        self.rng = random.Random(self.cfg.seed)
        self.mem = BuddyAIMemory()

    # ---------------------------
    # 公开 API：每局结束后喂结果（用于长期控制想赢/想输）
    # ---------------------------
    def on_episode_end(self, hero_win: bool, reason: str = ""):
        # 更新胜率历史
        self.mem.push_result(hero_win)

        # 更新情绪：输更上头，赢更冷静
        self.mem.tilt *= self.cfg.emotion_decay
        if hero_win:
            self.mem.tilt -= self.cfg.calm_gain_win
        else:
            self.mem.tilt += self.cfg.tilt_gain_loss

        # clamp
        self.mem.tilt = max(-1.0, min(1.0, self.mem.tilt))

    # ---------------------------
    # 公开 API：对手动作反馈（可选）
    # 你如果线上能喂对手动作统计，会更像人
    # ---------------------------
    def observe_opponent_action(self, action: Action):
        t = action.get("type", "")
        # 粗略更新对手激进度
        if t == "PK":
            self.mem.aggression_est = min(1.0, self.mem.aggression_est + 0.06)
        elif t == "BET":
            mul = int(action.get("mul", 1))
            if mul >= 8:
                self.mem.aggression_est = min(1.0, self.mem.aggression_est + 0.04)
            else:
                self.mem.aggression_est = min(1.0, self.mem.aggression_est + 0.02)
        elif t == "FOLD":
            self.mem.aggression_est = max(0.0, self.mem.aggression_est - 0.03)

    # ---------------------------
    # 核心：决策函数
    # ---------------------------
    def decide(
        self,
        obs: Dict[str, Any],
        legal_actions: List[Action],
        want_win_rate: float = 0.5,
    ) -> Action:
        want_win_rate = float(max(0.0, min(1.0, want_win_rate)))

        # 纯随机动作噪声（像真人）
        if legal_actions and self.rng.random() < self.cfg.random_action_rate:
            return self.rng.choice(legal_actions)

        # 提取常用字段（尽量兼容你 env.get_obs() 的结构）
        me = obs.get("self", {}) or {}
        round_idx = int(obs.get("round", 1))
        alive_count = int(obs.get("alive_count", obs.get("alive_players", 2)))
        num_players = int(obs.get("num_players", alive_count))
        last_mul = int(obs.get("last_bet_mul", 1))
        min_mul = int(obs.get("min_bet_mul", max(1, last_mul)))
        force_allin_or_fold = bool(obs.get("force_allin_or_fold", False))
        has_seen = bool(me.get("has_seen", obs.get("has_looked", False)))
        stack = int(me.get("stack", me.get("chips", 0)))

        # 牌力：0~5（高牌/对子/顺子/金花/顺金/炸弹）
        hand_rank = int(obs.get("hand_rank", 0))
        hand_strength = hand_rank / 5.0

        # 当前长期实际胜率，用于“调制”
        actual_wr = self.mem.win_rate()

        # 计算当前局“该偏赢还是偏输”的强度：[-1, +1]
        # >0 代表需要更强（更偏赢），<0 代表要降智（更偏输）
        # 比如 want=0.5，实际=0.7，则 bias=-0.2（降智）
        bias = want_win_rate - actual_wr
        bias = max(-0.35, min(0.35, bias))  # 限制幅度，避免太假
        # 再叠加情绪：tilt>0 更激进（更偏PK/大注）
        tilt = self.mem.tilt

        # 噪声（反确定性）：高斯扰动
        noise = self.rng.gauss(0.0, self.cfg.base_noise_sigma)
        anti = (self.rng.random() - 0.5) * 2.0 * self.cfg.anti_determinism

        # ===============
        # 1) 处理 SEE（看牌）
        # ===============
        if any(a.get("type") == "SEE" for a in legal_actions):
            # 回合越早越可能看牌，但不会总是看
            if round_idx <= 1:
                p_see = self.cfg.see_prob_round1
            elif round_idx <= 2:
                p_see = self.cfg.see_prob_round2
            else:
                p_see = self.cfg.see_prob_late

            # 想赢更容易看牌；想输稍微拖
            p_see += 0.18 * max(0.0, bias)
            p_see -= 0.12 * max(0.0, -bias)
            p_see += 0.10 * max(0.0, tilt)  # 上头更想看
            p_see += noise * 0.10

            p_see = max(0.02, min(0.85, p_see))
            if self.rng.random() < p_see:
                act = {"type": "SEE"}
                self.mem.last_action_type = "SEE"
                return act
            # 不看也没关系，继续走下注/PK逻辑

        # 若强制 allin_or_fold
        if force_allin_or_fold:
            # 想赢：更倾向 ALL_IN；想输：更倾向 FOLD（但保留人味随机）
            fold = _find_action(legal_actions, "FOLD")
            allin = _find_action(legal_actions, "ALL_IN")

            if not allin and fold:
                self.mem.last_action_type = "FOLD"
                return fold
            if not fold and allin:
                self.mem.last_action_type = "ALL_IN"
                return allin

            if fold and allin:
                p_allin = 0.50 + 0.90 * max(0.0, bias) - 0.70 * max(0.0, -bias) + 0.20 * tilt + noise * 0.20
                p_allin = max(0.05, min(0.95, p_allin))
                act = allin if self.rng.random() < p_allin else fold
                self.mem.last_action_type = act["type"]
                return act

            # 兜底
            self.mem.last_action_type = "FOLD"
            return {"type": "FOLD"}

        # ===============
        # 2) 计算 PK / BET / FOLD 的概率权重
        # ===============

        # --- PK 基础概率（允许顺子以下也 PK）---
        # 牌力越大越想PK；回合越大越想PK；2人桌更想PK；多人桌更谨慎
        pk_base = (
            0.10
            + 0.55 * hand_strength
            + self.cfg.pk_round_boost * max(0, round_idx - 1)
        )

        # 低牌也要有PK“时机”：给一个地板
        pk_base = max(pk_base, self.cfg.pk_low_hand_floor)

        # 多人桌惩罚：人数越多越不PK（但后期仍可能PK）
        if num_players >= 3:
            pk_base -= self.cfg.pk_multi_penalty * (num_players - 2) / 7.0

        # 对手越激进，我方越倾向PK（尤其后期）
        pk_base += 0.18 * (self.mem.aggression_est - 0.5)

        # 想赢/想输调制
        pk_base += self.cfg.win_mode_pk_bonus * max(0.0, bias)
        pk_base -= self.cfg.lose_mode_pk_penalty * max(0.0, -bias)

        # 情绪（上头更想PK）
        pk_base += 0.22 * max(0.0, tilt)

        # 噪声
        pk_base += noise * 0.20 + anti * 0.10

        pk_base = max(0.0, min(0.95, pk_base))

        # --- FOLD 概率 ---
        # 牌力越弱越想弃；last_mul 越大压力越大越想弃；但上头会减少弃牌
        fold_base = (
            0.20
            + 0.35 * (1.0 - hand_strength)
            + 0.18 * _pressure_from_last_mul(last_mul)
        )
        fold_base += 0.35 * max(0.0, -bias)   # 想输更容易弃
        fold_base -= 0.25 * max(0.0, bias)    # 想赢更少弃
        fold_base -= 0.18 * max(0.0, tilt)    # 上头更少弃
        fold_base += noise * 0.15 - anti * 0.08
        fold_base = max(0.0, min(0.90, fold_base))

        # --- BET 概率（剩余项）---
        bet_base = max(0.0, 1.0 - pk_base - fold_base)

        # 归一化（避免数值抖动）
        s = pk_base + fold_base + bet_base
        if s <= 1e-6:
            # 兜底：优先BET
            pk_p, bet_p, fold_p = 0.10, 0.70, 0.20
        else:
            pk_p, bet_p, fold_p = pk_base / s, bet_base / s, fold_base / s

        # ===============
        # 3) 根据合法动作集合做最终选择
        # ===============

        pk_actions = [a for a in legal_actions if a.get("type") == "PK"]
        bet_actions = [a for a in legal_actions if a.get("type") == "BET"]
        fold_action = _find_action(legal_actions, "FOLD")
        allin_action = _find_action(legal_actions, "ALL_IN")

        # 防止“总做同一动作类型”
        if self.mem.last_action_type is not None:
            if self.mem.last_action_type == "PK":
                pk_p *= (1.0 - self.cfg.anti_determinism)
            elif self.mem.last_action_type == "BET":
                bet_p *= (1.0 - self.cfg.anti_determinism * 0.6)
            elif self.mem.last_action_type == "FOLD":
                fold_p *= (1.0 - self.cfg.anti_determinism * 0.4)

            # 重新归一化
            s2 = pk_p + bet_p + fold_p
            if s2 > 1e-6:
                pk_p, bet_p, fold_p = pk_p / s2, bet_p / s2, fold_p / s2

        # 如果没有某类动作，转移概率
        if not pk_actions:
            bet_p += pk_p * 0.75
            fold_p += pk_p * 0.25
            pk_p = 0.0
        if not bet_actions:
            pk_p += bet_p * 0.60
            fold_p += bet_p * 0.40
            bet_p = 0.0
        if fold_action is None:
            # 没有FOLD则把fold权重分给 bet/pk
            bet_p += fold_p * 0.60
            pk_p += fold_p * 0.40
            fold_p = 0.0

        # 重新归一化
        s3 = pk_p + bet_p + fold_p
        if s3 <= 1e-6:
            # 兜底：BET 最小倍数
            act = _pick_bet(bet_actions, min_mul=min_mul, last_mul=last_mul, bias=bias, tilt=tilt, cfg=self.cfg, rng=self.rng)
            self.mem.last_action_type = act.get("type")
            return act
        pk_p, bet_p, fold_p = pk_p / s3, bet_p / s3, fold_p / s3

        # 特殊：偶尔 ALL_IN（像真人情绪/上头/装）
        if allin_action is not None:
            p_allin = self.cfg.allin_spice
            p_allin += 0.10 * max(0.0, bias)       # 想赢更可能allin
            p_allin -= 0.08 * max(0.0, -bias)      # 想输少点allin（否则太假）
            p_allin += 0.10 * max(0.0, tilt)       # 上头更可能allin
            p_allin += 0.06 * _pressure_from_last_mul(last_mul)
            p_allin += noise * 0.08
            p_allin = max(0.0, min(0.35, p_allin))

            if self.rng.random() < p_allin:
                self.mem.last_action_type = "ALL_IN"
                return allin_action

        # 抽样选择类型
        u = self.rng.random()
        if u < pk_p:
            act = _pick_pk(pk_actions, obs=obs, bias=bias, tilt=tilt, cfg=self.cfg, rng=self.rng)
            self.mem.last_action_type = act.get("type")
            return act

        if u < pk_p + bet_p:
            act = _pick_bet(bet_actions, min_mul=min_mul, last_mul=last_mul, bias=bias, tilt=tilt, cfg=self.cfg, rng=self.rng)
            self.mem.last_action_type = act.get("type")
            return act

        # fold
        if fold_action is not None:
            self.mem.last_action_type = "FOLD"
            return fold_action

        # 最终兜底：BET or PK
        if bet_actions:
            act = _pick_bet(bet_actions, min_mul=min_mul, last_mul=last_mul, bias=bias, tilt=tilt, cfg=self.cfg, rng=self.rng)
            self.mem.last_action_type = act.get("type")
            return act
        if pk_actions:
            act = _pick_pk(pk_actions, obs=obs, bias=bias, tilt=tilt, cfg=self.cfg, rng=self.rng)
            self.mem.last_action_type = act.get("type")
            return act

        self.mem.last_action_type = "FOLD"
        return {"type": "FOLD"}


# ============================================================
# 选择器：BET / PK / 目标 seat
# ============================================================

def _find_action(legal: List[Action], t: str) -> Optional[Action]:
    for a in legal:
        if a.get("type") == t:
            return a
    return None


def _pressure_from_last_mul(last_mul: int) -> float:
    # last_mul 越大压力越大（0~1）
    # 1->0.0, 4->0.2, 8->0.45, 12->0.65, 20->1.0
    x = max(1, min(20, int(last_mul)))
    return math.log(x) / math.log(20)


def _pick_bet(
    bet_actions: List[Action],
    min_mul: int,
    last_mul: int,
    bias: float,
    tilt: float,
    cfg: BuddyAIConfig,
    rng: random.Random,
) -> Action:
    """
    从合法 BET 动作里挑一个倍数：
    - 人类常用：min、min+1/2、略高、偶尔大注
    - 想赢：更偏高倍；想输：更偏保守/跟注
    - 上头：更偏高倍
    """
    if not bet_actions:
        return {"type": "FOLD"}

    # 收集所有可选倍数
    muls = sorted({int(a.get("mul", min_mul)) for a in bet_actions})
    if not muls:
        return bet_actions[0]

    # 基础“目标倍数”倾向
    # style ∈ [0,1]：越大越激进
    style = 0.50
    style += 0.55 * max(0.0, bias)           # 想赢更激进
    style -= 0.45 * max(0.0, -bias)          # 想输更保守
    style += 0.35 * max(0.0, tilt)           # 上头更激进
    style += rng.gauss(0.0, 0.12)            # 噪声
    style = max(0.0, min(1.0, style))

    # 生成几个候选档位（更像真人）
    # 1) 跟注/最小
    # 2) 小加：min+1/2
    # 3) 中加：靠近 last_mul + 2~4
    # 4) 大加：靠近 max
    candidates: List[int] = []
    candidates.append(min_mul)

    # 小加（如果存在）
    for d in (1, 2, 3, 4):
        if (min_mul + d) in muls:
            candidates.append(min_mul + d)
            break

    # 中加（靠近 last_mul 的上方）
    for target in (last_mul + 2, last_mul + 4, last_mul + 6):
        near = _nearest_ge(muls, target)
        if near is not None:
            candidates.append(near)
            break

    # 大加（靠近最大）
    candidates.append(muls[-1])

    # 去重并确保存在
    candidates = [m for i, m in enumerate(candidates) if m in muls and m not in candidates[:i]]
    if not candidates:
        # 兜底：随机合法
        pick = rng.choice(muls)
        return {"type": "BET", "mul": pick}

    # 赋权：style 越大越偏后面（大注）
    # 用一个简单的幂权重分布
    weights = []
    for i in range(len(candidates)):
        # i 越大表示越激进
        # style=0 -> 偏 i=0，style=1 -> 偏最后
        center = style * (len(candidates) - 1)
        w = math.exp(-((i - center) ** 2) / (2 * 0.9 ** 2))
        weights.append(w)

    # 想输时，抑制最大档位
    if bias < 0:
        weights[-1] *= (1.0 - cfg.lose_mode_bigbet_penalty * min(1.0, -bias * 3.0))
    # 想赢时，提升最大档位
    if bias > 0:
        weights[-1] *= (1.0 + cfg.win_mode_bigbet_bonus * min(1.0, bias * 3.0))

    # 归一化抽样
    s = sum(weights)
    if s <= 1e-9:
        chosen = candidates[0]
    else:
        r = rng.random() * s
        acc = 0.0
        chosen = candidates[-1]
        for m, w in zip(candidates, weights):
            acc += w
            if r <= acc:
                chosen = m
                break

    return {"type": "BET", "mul": int(chosen)}


def _nearest_ge(sorted_vals: List[int], x: int) -> Optional[int]:
    for v in sorted_vals:
        if v >= x:
            return v
    return None


def _pick_pk(
    pk_actions: List[Action],
    obs: Dict[str, Any],
    bias: float,
    tilt: float,
    cfg: BuddyAIConfig,
    rng: random.Random,
) -> Action:
    """
    从合法 PK 里选目标：
    - 2人桌：只有一个 target
    - 多人桌：更像真人地“挑软柿子”或“挑激进者”
    """
    if not pk_actions:
        return {"type": "FOLD"}

    # 多人：用 opponents 估计“软柿子”
    opps = obs.get("opponents", []) or []
    opp_map = {int(o.get("seat")): o for o in opps if "seat" in o}

    # 给每个 target 打分
    scored: List[Tuple[float, Action]] = []
    for a in pk_actions:
        t = int(a.get("target", -1))
        o = opp_map.get(t, {})
        alive = bool(o.get("alive", True))
        if not alive:
            continue

        # “软柿子”：没看牌、筹码少、下注少
        o_seen = 1.0 if o.get("has_seen", False) else 0.0
        o_stack = float(o.get("stack", 0))
        o_bet = float(o.get("bet_total", 0))

        soft = 0.0
        soft += (1.0 - o_seen) * 0.45
        soft += (1.0 - min(1.0, o_stack / 5000.0)) * 0.25
        soft += (1.0 - min(1.0, o_bet / 5000.0)) * 0.20

        # 想赢：更倾向挑软；想输：更可能挑硬（但不要太离谱）
        score = soft
        score += 0.35 * max(0.0, bias)
        score -= 0.15 * max(0.0, -bias)

        # 上头：更随意
        score += 0.10 * max(0.0, tilt)

        # 噪声
        score += rng.gauss(0.0, 0.08)

        scored.append((score, a))

    if not scored:
        return pk_actions[0]

    scored.sort(key=lambda x: x[0], reverse=True)

    # 不一定总选最高：用 softmax 抽样，更像真人
    top = scored[: min(5, len(scored))]
    scores = [s for s, _ in top]
    mx = max(scores)
    temp = 1.0 + 0.6 * max(0.0, tilt) + 0.5 * max(0.0, -bias)  # 想输/上头更随机
    ps = [math.exp((s - mx) / max(0.15, temp)) for s in scores]
    ssum = sum(ps)
    r = rng.random() * ssum
    acc = 0.0
    for p, (_, act) in zip(ps, top):
        acc += p
        if r <= acc:
            return act
    return top[0][1]


# ============================================================
# 便捷函数：直接给你一个“纯函数式”入口（无状态）
# 如果你不想维护 AI 对象，也可用它（但更不像真人）
# ============================================================
_default_ai: Optional[BuddyAI] = None

def decide_action(
    obs: Dict[str, Any],
    legal_actions: List[Action],
    want_win_rate: float = 0.5,
    seed: int = 42,
) -> Action:
    global _default_ai
    if _default_ai is None or _default_ai.cfg.seed != seed:
        _default_ai = BuddyAI(BuddyAIConfig(seed=seed))
    return _default_ai.decide(obs, legal_actions, want_win_rate=want_win_rate)
