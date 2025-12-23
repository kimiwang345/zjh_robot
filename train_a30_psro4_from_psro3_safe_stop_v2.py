# train_a30_psro4_from_psro3_safe_stop_v2_1.py
import os
import re
import json
import time
import random
import signal
from dataclasses import asdict
from typing import List, Dict, Any

import numpy as np
import torch

from config_a30_hybrid import A30ConfigHybrid
from agent_a30 import a30Agent

from ZJHEnvMulti2to9 import ZJHEnvMulti2to9
from nn_opponent_wrapper import NNOpponentA30

from zjh_opponents_multi import (
    NitOpponent,
    TAGOpponent,
    GTOOpponent,
)

# ============================================================
# Paths
# ============================================================
MODELS_DIR = "models"

POP3_DIR = "population_psro3"
POP3_INDEX_PATH = os.path.join(POP3_DIR, "pop_index.json")

POP4_DIR = "population_psro4"
POP4_INDEX_PATH = os.path.join(POP4_DIR, "pop_index.json")

PSRO3_BEST = os.path.join(MODELS_DIR, "a30_psro3_best.pth")

PSRO4_LATEST = os.path.join(MODELS_DIR, "a30_psro4_latest.pth")
PSRO4_BEST   = os.path.join(MODELS_DIR, "a30_psro4_best.pth")
PSRO4_FINAL  = os.path.join(MODELS_DIR, "a30_psro4_final.pth")
PSRO4_META   = os.path.join(MODELS_DIR, "a30_psro4_meta.json")

# ============================================================
# Utils
# ============================================================
def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def safe_load_json(path: str, default: Any):
    if not os.path.exists(path):
        return default
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default

# ============================================================
# Safe Stop
# ============================================================
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
# Eval (Exploit-only Gate)
# ============================================================
def make_env_for_eval(opponent_cls):
    env = ZJHEnvMulti2to9(
        max_round=20,
        min_players=2,
        max_players=9
    )
    opponents = [None]
    for _ in range(1, env.max_players):
        opponents.append(opponent_cls())
    env.bind_opponents(opponents)
    return env

@torch.no_grad()
def eval_exploit_only_vs_rule(
    agent: a30Agent,
    opponent_cls,
    episodes: int,
    n_envs: int,
    device: str,
    heartbeat_sec: float = 10.0,
):
    net = agent.eval_net.to(device)
    net.eval()

    envs = [make_env_for_eval(opponent_cls) for _ in range(n_envs)]
    states = [env.reset() for env in envs]

    ep_returns = []
    cur_returns = [0.0] * n_envs
    finished = 0

    t0 = time.time()
    last_hb = t0

    while finished < episodes:
        st = torch.tensor(np.asarray(states), dtype=torch.float32, device=device)
        actions = torch.argmax(net(st), dim=1).tolist()

        next_states = states[:]
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
            next_states[i] = ns
        states = next_states

        now = time.time()
        if now - last_hb >= heartbeat_sec:
            speed = finished / max(now - t0, 1e-6)
            print(f"[GATE-EVAL][{opponent_cls.__name__}] {finished}/{episodes} eps speed={speed:.2f} eps/s")
            last_hb = now

    arr = np.asarray(ep_returns, dtype=np.float32)
    return float(arr.mean())

# ============================================================
# Gate thresholds (v2.1)
# ============================================================
GATE_MIN_GTO = -6.0
GATE_MIN_TAG = -6.0
GATE_MIN_NIT = -10.0

# ============================================================
# Training
# ============================================================
def train():
    ensure_dir(MODELS_DIR)
    ensure_dir(POP4_DIR)

    cfg = A30ConfigHybrid()
    print(f"[INIT] PSRO-4(v2.1) boot from {PSRO3_BEST}")

    agent = a30Agent(cfg)
    meta = safe_load_json(PSRO4_META, {})

    start_ep = int(meta.get("episode", 1))
    best_mean_ev = float(meta.get("best_mean_ev", -1e9))
    total_steps = int(meta.get("total_steps", 0))

    if os.path.exists(PSRO4_LATEST):
        print(f"[RESUME] found {PSRO4_LATEST}")
        total_steps, _ = agent.load(PSRO4_LATEST)
    else:
        agent.load(PSRO3_BEST)
        agent.save(PSRO4_LATEST, total_steps=0, eps=0.0)

    stop_flag = StopFlag()
    install_signal_handlers(stop_flag)

    # ===== heartbeat control =====
    last_hb = time.time()
    hb_interval = 30.0

    print("[DEBUG] entered training loop")

    for ep in range(start_ep, cfg.total_episodes + 1):
        if stop_flag.stop:
            break

        env = ZJHEnvMulti2to9()
        opponents = [None]

        # ---- force anchors ----
        opponents.append(NNOpponentA30(weight_path=PSRO3_BEST, device="cpu"))
        opponents.append(NNOpponentA30(weight_path=PSRO3_BEST, device="cpu"))

        nn_ratio = 0.55

        for _ in range(3, env.max_players):
            if random.random() < nn_ratio:
                opponents.append(NNOpponentA30(weight_path=PSRO3_BEST, device="cpu"))
            else:
                opponents.append(random.choice([GTOOpponent, TAGOpponent, NitOpponent])())

        env.bind_opponents(opponents)

        state = env.reset()
        ep_ret = 0.0

        for t in range(cfg.max_steps_per_episode):
            action = agent.select_action(state, eps=0.0)
            ns, r, done, _ = env.step(action)
            agent.store_transition(state, action, r, ns, done)
            state = ns
            ep_ret += float(r)
            total_steps += 1

            if total_steps % cfg.learn_interval == 0:
                agent.learn(total_steps)

            # ===== TRAIN HEARTBEAT =====
            now = time.time()
            if now - last_hb >= hb_interval:
                print(
                    f"[TRAIN-HB] ep={ep} t={t} steps={total_steps} "
                    f"ep_ev={ep_ret:+.2f} nn_ratio={nn_ratio:.2f}"
                )
                last_hb = now

            if done:
                break

        # ===== gate eval =====
        if ep % cfg.eval_interval == 0:
            gto_ev = eval_exploit_only_vs_rule(agent, GTOOpponent, 6000, 32, cfg.device)
            tag_ev = eval_exploit_only_vs_rule(agent, TAGOpponent, 6000, 32, cfg.device)
            nit_ev = eval_exploit_only_vs_rule(agent, NitOpponent, 6000, 32, cfg.device)

            gate_ok = (
                gto_ev >= GATE_MIN_GTO and
                tag_ev >= GATE_MIN_TAG and
                nit_ev >= GATE_MIN_NIT
            )

            print(
                f"[GATE v2.1][EP {ep}] "
                f"GTO={gto_ev:+.2f} TAG={tag_ev:+.2f} NIT={nit_ev:+.2f} ok={gate_ok}"
            )

            if gate_ok and ep_ret > best_mean_ev:
                best_mean_ev = ep_ret
                agent.save(PSRO4_BEST, total_steps, 0.0)
                print(f"[BEST] updated at ep={ep} ev={ep_ret:+.2f}")

    agent.save(PSRO4_FINAL, total_steps, 0.0)
    print("[DONE] PSRO-4 v2.1 finished")

if __name__ == "__main__":
    train()
