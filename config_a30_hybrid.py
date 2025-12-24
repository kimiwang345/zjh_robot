"""
A30 Hybrid Config
=================

适用范围：
- 扎金花（ZJHEnvMulti2to9）
- 正确下注规则（无 CALL / 有最小下注倍数）
- Action Mask 强约束
- PSRO-5 Self-Play
- A30Agent（Mask-aware DQN）

本配置文件的目标是：
1. 保证训练稳定
2. 保证规则一致
3. 保证可线上迁移（ante=5 / 10 / 20）
"""

import math


# ============================================================
# 1. Environment Config
# ============================================================

class EnvConfig:
    # 玩家数量
    MIN_PLAYERS = 2
    MAX_PLAYERS = 9

    # 最大轮数（到达后强制 COMPARE_ALL）
    MAX_ROUND = 20

    # ante（BB）：下注基数（不是德州意义的前注）
    # 训练时做 domain randomization
    ANTE_BB_CANDIDATES = [0.5, 1.0, 2.0]

    # 初始筹码（BB，LogUniform）
    STACK_BB_MIN = 20
    STACK_BB_MAX = 300

    # 是否允许第一轮看牌后 1 倍下注
    FIRST_ROUND_SEEN_CAN_BET_ONE = True


# ============================================================
# 2. State Config
# ============================================================

class StateConfig:
    """
    状态维度必须与 ZJHEnvMulti2to9.state_dim 完全一致
    """

    STATE_DIM = 49

    # 是否使用归一化（推荐保持 True）
    NORMALIZE_STACK = True
    NORMALIZE_POT = True

    # 防止除零
    EPS = 1e-6


# ============================================================
# 3. Action Config
# ============================================================

class ActionConfig:
    """
    动作语义（必须与 env / agent / mask 完全一致）
    """

    ACTION_DIM = 14

    ACTION_MEANINGS = {
        0: "FOLD",
        1: "LOOK",
        2: "BET_1",
        3: "BET_2",
        4: "BET_3",
        5: "BET_4",
        6: "BET_5",
        7: "BET_6",
        8: "BET_7",
        9: "BET_8",
        10: "BET_9",
        11: "BET_10",
        12: "PK",
        13: "COMPARE_ALL",
    }


# ============================================================
# 4. Agent (A30) Config
# ============================================================

class AgentConfig:
    # 网络结构
    HIDDEN_DIM = 256

    # 折扣因子
    GAMMA = 0.99

    # 学习率
    LR = 3e-4

    # epsilon-greedy
    EPSILON_START = 1.0
    EPSILON_END = 0.05
    EPSILON_DECAY_STEPS = 300_000

    # target network
    TARGET_UPDATE_INTERVAL = 5_000

    # 梯度裁剪
    MAX_GRAD_NORM = 10.0


# ============================================================
# 5. Replay Buffer (PER)
# ============================================================

class ReplayBufferConfig:
    # 容量
    CAPACITY = 300_000

    # Prioritized Experience Replay
    PER_ALPHA = 0.6
    PER_BETA_START = 0.4
    PER_BETA_FRAMES = 500_000

    # 训练起始阈值
    MIN_BUFFER_SIZE = 10_000

    # batch size
    BATCH_SIZE = 64


# ============================================================
# 6. Training Config
# ============================================================

class TrainingConfig:
    # 总训练步数（可无限跑）
    MAX_STEPS = 50_000_000

    # 模型保存
    SAVE_INTERVAL_STEPS = 20_000
    SAVE_DIR = "models"

    # 日志
    LOG_INTERVAL_EPISODES = 50

    # 是否启用 CUDA
    USE_CUDA = True


# ============================================================
# 7. PSRO Config
# ============================================================

class PSROConfig:
    """
    Population-based Self-Play
    """

    # PSRO 轮数（例如 PSRO-5）
    POPULATION_SIZE = 5

    # 对手池是否动态扩展
    DYNAMIC_POOL = True

    # 新策略加入池的条件（step）
    ADD_NEW_POLICY_INTERVAL = 2_000_000


# ============================================================
# 8. Evaluation Config
# ============================================================

class EvalConfig:
    # 每次评估局数
    NUM_EPISODES = 2_000

    # 是否禁用探索
    DETERMINISTIC = True

    # 是否打印关键行为
    VERBOSE = False


# ============================================================
# 9. Reward Shaping（可选）
# ============================================================

class RewardConfig:
    """
    默认保持最原始 reward：
      win  = pot - hero_contrib
      lose = -hero_contrib

    下面是可选增强项（当前建议先关闭）
    """

    ENABLE_SHAPING = False

    # 第一轮拖延惩罚
    STEP_PENALTY = -0.001

    # 第一轮弃弱牌奖励
    WEAK_HAND_FOLD_BONUS = 0.01


# ============================================================
# 10. Utils
# ============================================================

def sample_starting_stack_bb():
    """
    LogUniform 采样初始筹码（BB）
    """
    lo = math.log(EnvConfig.STACK_BB_MIN)
    hi = math.log(EnvConfig.STACK_BB_MAX)
    return math.exp(lo + (hi - lo) * math.random())
