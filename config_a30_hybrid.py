from dataclasses import dataclass
import torch


@dataclass
class A30ConfigHybrid:

    state_dim: int = 54
    action_dim: int = 14
    gamma: float = 0.99

    total_episodes: int = 500_000
    max_steps_per_episode: int = 200

    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    has_gpu: bool = torch.cuda.is_available()

    batch_size: int = 1024 if torch.cuda.is_available() else 256
    learn_interval: int = 1 if torch.cuda.is_available() else 2

    # ↓ 稍微降一点，配合正则化更稳
    lr: float = 8e-5 if torch.cuda.is_available() else 5e-5
    weight_decay: float = 0.0

    buffer_capacity: int = 600_000 if torch.cuda.is_available() else 300_000

    per_alpha: float = 0.6
    per_beta_start: float = 0.4
    per_beta_end: float = 1.0
    per_beta_anneal_steps: int = 400_000

    target_update_freq: int = 1000 if torch.cuda.is_available() else 2000

    eps_start: float = 0.05
    eps_end: float = 0.01
    eps_decay_episodes: int = 400_000

    # ↓ 稍微收紧梯度，防止 a11 残留震荡
    max_grad_norm: float = 4.0 if torch.cuda.is_available() else 8.0
    grad_accumulate_steps: int = 2 if torch.cuda.is_available() else 1

    log_interval: int = 5000
    eval_interval: int = 20000

    seed: int = 42

    # ↓ PSRO5 建议稍微放宽，保留多样性
    min_pop_score: float = -20.0

    eval_episodes: int = 5_000
    eval_envs: int = 32
