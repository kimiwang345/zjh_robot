from __future__ import annotations
import argparse
import random
import time
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from zjh_env import ZJHEnv
from zjh_constants import GameConfig

from agent_dqn import (
    QNet,
    ReplayBuffer,
    obs_to_vector,
    encode_actions,
    select_action_epsilon_greedy,
)

from buddy_ai import decide_action_buddy, BuddyAIConfig


# ======================================================
# Utils
# ======================================================
def linear_decay(step, start, end, decay_steps):
    if step >= decay_steps:
        return end
    return start + (end - start) * (step / decay_steps)


# ======================================================
# Train
# ======================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=30000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device(args.device)

    # ======================================================
    # Game Config（2 人桌示例，可自行改）
    # ======================================================
    cfg = GameConfig(
        num_players=2,
        init_stack=5000,
        base=10,
        max_rounds=20,
        seed=args.seed,
    )

    env = ZJHEnv(cfg)

    hero_seat = 0

    # ======================================================
    # BuddyAI（训练期对手）
    # ======================================================
    buddy_cfg = BuddyAIConfig(
        mode="neutral",   # 训练期一定是 neutral
        strength=0.4,     # 给学习空间
        seed=123,
    )

    # ======================================================
    # DQN
    # ======================================================
    obs_dim = len(obs_to_vector(env.reset(seed=args.seed, hero_seat=hero_seat)))

    qnet = QNet(obs_dim).to(device)
    tgt_qnet = QNet(obs_dim).to(device)
    tgt_qnet.load_state_dict(qnet.state_dict())
    tgt_qnet.eval()

    optimizer = optim.Adam(qnet.parameters(), lr=1e-3)

    buffer = ReplayBuffer(capacity=50000, seed=args.seed)

    gamma = 0.99
    batch_size = 64
    start_learn = 1000
    target_sync = 1000

    # epsilon
    eps_start = 0.9
    eps_end = 0.05
    eps_decay = 15000

    # stats
    win_hist = deque(maxlen=100)
    reward_hist = deque(maxlen=100)

    global_step = 0

    # ======================================================
    # Training Loop
    # ======================================================
    for ep in range(1, args.episodes + 1):
        obs = env.reset(seed=args.seed + ep, hero_seat=hero_seat)
        done = False

        ep_reward = 0.0
        hero_win = 0

        while not done:
            seat = env.acting_seat
            obs_s = env.get_obs(seat)
            legal = env.legal_actions(seat)

            # ---------------- HERO ----------------
            if seat == hero_seat:
                obs_vec = obs_to_vector(obs_s)
                enc = encode_actions(legal)

                eps = linear_decay(
                    global_step,
                    eps_start,
                    eps_end,
                    eps_decay,
                )

                a_idx = select_action_epsilon_greedy(
                    qnet,
                    obs_vec,
                    enc,
                    eps,
                    device,
                    rng,
                )
                action = enc.actions[a_idx]

            # ---------------- OPPONENT (BuddyAI) ----------------
            else:
                action = decide_action_buddy(
                    obs_s,
                    legal,
                    buddy_cfg,
                )

            # step
            obs2, reward, done, info = env.step(seat, action)

            # ---------------- Store transition (HERO only) ----------------
            if seat == hero_seat:
                obs2_vec = obs_to_vector(obs2)
                legal2 = env.legal_actions(hero_seat)
                enc2 = encode_actions(legal2)

                buffer.push(
                    obs_vec,
                    enc.mask,
                    a_idx,
                    reward,
                    obs2_vec,
                    enc2.mask,
                    done,
                )

                ep_reward += reward

            global_step += 1

            # ---------------- Learn ----------------
            if len(buffer) >= start_learn:
                (
                    s,
                    m,
                    a,
                    r,
                    s2,
                    m2,
                    d,
                ) = buffer.sample(batch_size)

                s = torch.tensor(s, device=device)
                m = torch.tensor(m, device=device)
                a = torch.tensor(a, device=device)
                r = torch.tensor(r, device=device)
                s2 = torch.tensor(s2, device=device)
                m2 = torch.tensor(m2, device=device)
                d = torch.tensor(d, device=device)

                q = qnet(s)                       # (B, MAX_ACTIONS)
                q = q.gather(1, a.unsqueeze(1)).squeeze(1)

                with torch.no_grad():
                    q2 = tgt_qnet(s2)
                    q2[m2 < 0.5] = -1e18
                    q2_max = q2.max(dim=1)[0]
                    target = r + gamma * (1 - d) * q2_max

                loss = nn.MSELoss()(q, target)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # sync target
                if global_step % target_sync == 0:
                    tgt_qnet.load_state_dict(qnet.state_dict())

        # ======================================================
        # Episode End
        # ======================================================
        if env.winner_seat == hero_seat:
            hero_win = 1

        win_hist.append(hero_win)
        reward_hist.append(ep_reward)

        if ep % 100 == 0:
            win_rate = sum(win_hist) / len(win_hist)
            avg_r = sum(reward_hist) / len(reward_hist)
            print(
                f"[EP {ep:6d}] "
                f"win_rate={win_rate:.3f} "
                f"avg100_r={avg_r:.3f} "
                f"eps={eps:.3f} "
                f"buffer={len(buffer)}"
            )

    # ======================================================
    # Save
    # ======================================================
    torch.save(qnet.state_dict(), "dqn_hero.pth")
    print("Training finished, model saved to dqn_hero.pth")


if __name__ == "__main__":
    main()
