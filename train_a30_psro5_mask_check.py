# train_a30_psro5_mask_check.py
"""
PSRO5 Action Mask Validation Script

目标：
- 验证 a11 正则化是否生效
- 检查 STD 是否下降
- 检查 Nit / TAG / GTO EV 是否回升
- 不引入 NN Opponent
- 不做 population 晋级
"""

import os
import json
import time
import random
import signal
from typing import Any

import numpy as np
import torch

from config_a30_hybrid import A30ConfigHybrid
from agent_a30 import a30Agent

from ZJHEnvMulti2to9 import ZJHEnvMulti2to9

from zjh_opponents_multi import (
    NitOpponent,
    LooseOpponent,
    AggroOpponent,
    GTOOpponent,
    FishOpponent,
    ManiacOpponent,
)

# ============================================================
# Utils
# ============================================================

MODELS_DIR = "models"
CHECKPOINT_PATH = os.path.join(MODELS_DIR, "a30_psro5_mask_check.pth")


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


class StopFlag:
    def __init__(self):
        self.stop = False


def install_signal_handlers(flag: StopFlag):
    def _handler(sig, frame):
        flag.stop = True
        print("\n[SAFE-STOP] signal received, will stop after current episode...")
    signal.signal(signal.SIGINT, _handler)
    signal.signal(signal.SIGTERM, _handler)


# ============================================================
# Env / Opponent Sampling (Validation Only)
# ============================================================

def make_train_env():
    env = ZJHEnvMulti2to9(
        max_round=20,
        min_players=2,
        max_players=9,
    )

    opponents = [None]
    for _ in range(1, env.max_players):
        opp = random.choices(
            population=[
                NitOpponent(),
                LooseOpponent(),
                AggroOpponent(),
                GTOOpponent(),
            ],
            weights=[0.25, 0.35, 0.25, 0.15],
            k=1,
        )[0]
        opponents.append(opp)

    env.opponents = opponents
    return env


# ============================================================
# Eval (Exploit-only, Mask-aware)
# ============================================================

def make_eval_env(opponent_cls):
    env = ZJHEnvMulti2to9(
        max_round=20,
        min_players=2,
        max_players=9,
    )
    opponents = [None]
    for _ in range(1, env.max_players):
        opponents.append(opponent_cls())
    env.opponents = opponents
    return env


@torch.no_grad()
def eval_exploit_only(
    agent: a30Agent,
    opponent_cls,
    episodes: int,
    n_envs: int,
    device: str,
):
    envs = [make_eval_env(opponent_cls) for _ in range(n_envs)]
    states = [env.reset() for env in envs]

    ep_returns = []
    cur_returns = [0.0] * n_envs
    finished = 0

    t0 = time.time()

    while finished < episodes:
        actions = [agent.select_action(states[i]) for i in range(n_envs)]

        for i, env in enumerate(envs):
            ns, r, done, _ = env.step(actions[i])
            cur_returns[i] += float(r)

            if done:
                ep_returns.append(cur_returns[i])
                cur_returns[i] = 0.0
                finished += 1
                if finished >= episodes:
                    break
                ns = env.reset()
            states[i] = ns

    arr = np.asarray(ep_returns, dtype=np.float32)
    mean = float(arr.mean())
    std = float(arr.std())
    speed = finished / max(time.time() - t0, 1e-6)

    return mean, std, speed


# ============================================================
# Training Loop (Validation)
# ============================================================

def train():
    ensure_dir(MODELS_DIR)

    cfg = A30ConfigHybrid()
    cfg.total_episodes = 80_000           # 验证轮：够看趋势
    cfg.eval_interval = 20_000

    print("[INIT] PSRO5 Mask Validation")
    agent = a30Agent(cfg)

    if os.path.exists(CHECKPOINT_PATH):
        print(f"[RESUME] load {CHECKPOINT_PATH}")
        agent.load(CHECKPOINT_PATH)

    stop_flag = StopFlag()
    install_signal_handlers(stop_flag)

    total_steps = 0
    last_hb = time.time()
    hb_interval = 20.0

    for ep in range(1, cfg.total_episodes + 1):
        if stop_flag.stop:
            break

        env = make_train_env()
        state = env.reset()
        ep_ret = 0.0

        for t in range(cfg.max_steps_per_episode):
            action = agent.select_action(state)
            ns, r, done, _ = env.step(action)
            agent.store_transition(state, action, r, ns, done)
            state = ns
            ep_ret += float(r)
            total_steps += 1

            if total_steps % cfg.learn_interval == 0:
                agent.learn(total_steps)

            now = time.time()
            if now - last_hb >= hb_interval:
                print(
                    f"[TRAIN] ep={ep} t={t} steps={total_steps} ep_ev={ep_ret:+.2f}"
                )
                last_hb = now

            if done:
                break

        # ================== Evaluation ==================
        if ep % cfg.eval_interval == 0:
            print(f"\n[EVAL @ EP {ep}]")

            for name, cls in [
                ("Fish", FishOpponent),
                ("Maniac", ManiacOpponent),
                ("Aggro", AggroOpponent),
                ("Loose", LooseOpponent),
                ("GTO", GTOOpponent),
                ("Nit", NitOpponent),
            ]:
                mean, std, speed = eval_exploit_only(
                    agent,
                    cls,
                    episodes=3000,
                    n_envs=32,
                    device=cfg.device,
                )
                print(
                    f"VS {name:<7} | EV={mean:+7.2f} | STD={std:7.2f} | speed={speed:.1f} eps/s"
                )

            agent.save(CHECKPOINT_PATH, total_steps, 0.0)
            print("[CHECKPOINT] saved\n")

    print("[DONE] Mask validation finished")


if __name__ == "__main__":
    train()
