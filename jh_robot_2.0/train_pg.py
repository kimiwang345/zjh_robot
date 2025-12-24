from __future__ import annotations
import argparse
import random
from typing import Dict, Any, List
from collections import deque

import numpy as np
import torch
import torch.optim as optim

from zjh_constants import GameConfig
from zjh_env import ZJHEnv
from opponents import TightOpponent, AggroOpponent, MixedOpponent
from agent_policy import PolicyNet, obs_to_vector, encode_actions, sample_action

# ======================================================
# Opponent pool
# ======================================================
def pick_opponent(rng: random.Random):
    r = rng.random()
    if r < 0.34:
        return TightOpponent(rng)
    if r < 0.67:
        return MixedOpponent(rng)
    return AggroOpponent(rng)

# ======================================================
# Run one episode
# ======================================================
def run_episode(env: ZJHEnv, model: PolicyNet, device: str, rng: random.Random):
    obs = env.get_obs(env.hero_seat)

    logps: List[torch.Tensor] = []

    # 对手策略
    opp_policies = {
        s: pick_opponent(rng)
        for s in range(env.cfg.num_players)
        if s != env.hero_seat
    }

    done = False
    total_reward = 0.0

    while not done:
        acting = env.acting_seat

        if acting == env.hero_seat:
            obs = env.get_obs(env.hero_seat)
            legal = env.legal_actions(env.hero_seat)
            enc = encode_actions(legal)
            ov = obs_to_vector(obs)

            a_idx, logp = sample_action(model, ov, enc, device=device)
            action = enc.actions[a_idx]

            logps.append(logp)

            _, r, done, _ = env.step(env.hero_seat, action)
            total_reward += r
        else:
            oobs = env.get_obs(acting)
            legal = env.legal_actions(acting)
            action = opp_policies[acting].act(oobs, legal)
            _, r, done, _ = env.step(acting, action)
            total_reward += r

    return total_reward, logps

# ======================================================
# Main training
# ======================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=5000)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--num_players", type=int, default=6)
    ap.add_argument("--init_stack", type=int, default=200)
    ap.add_argument("--base", type=int, default=1)
    ap.add_argument("--device", type=str, default="cpu")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    cfg = GameConfig(
        num_players=args.num_players,
        base=args.base,
        init_stack=args.init_stack,
        seed=args.seed
    )

    env = ZJHEnv(cfg)
    env.reset(seed=args.seed, hero_seat=0)

    obs_dim = len(obs_to_vector(env.get_obs(0)))
    model = PolicyNet(obs_dim=obs_dim).to(args.device)
    opt = optim.Adam(model.parameters(), lr=args.lr)

    # baseline（滑动平均）
    baseline = 0.0
    beta = 0.98

    # ===== 新增：最近100局统计 =====
    recent_rewards = deque(maxlen=100)

    for ep in range(1, args.episodes + 1):
        env.reset(seed=rng.randrange(1_000_000), hero_seat=0)

        R, logps = run_episode(env, model, args.device, rng)

        # baseline 更新
        baseline = beta * baseline + (1 - beta) * R
        adv = R - baseline

        # Policy Gradient loss
        if logps:
            loss = torch.zeros(1, device=args.device)
            for lp in logps:
                loss += -adv * lp
            loss = loss / len(logps)

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
        else:
            loss = torch.tensor(0.0)

        # ===== 记录 reward =====
        recent_rewards.append(R)

        # ===== 每 100 局打印一次统计 =====
        if ep % 100 == 0:
            avg100 = sum(recent_rewards) / len(recent_rewards)
            win_rate = sum(1 for r in recent_rewards if r > 0) / len(recent_rewards) * 100.0

            print(
                f"[ep {ep}] "
                f"r={R:.1f} base={baseline:.1f} adv={adv:.1f} loss={loss.item():.4f} | "
                f"avg100={avg100:.2f} win_rate={win_rate:.1f}%"
            )

        # 定期保存
        if ep % 2000 == 0:
            torch.save(model.state_dict(), "policy_zjh.pt")
            print("[save] policy_zjh.pt")

    torch.save(model.state_dict(), "policy_zjh.pt")
    print("[done] saved policy_zjh.pt")

if __name__ == "__main__":
    main()
