from __future__ import annotations
import argparse
import random
from collections import deque

import numpy as np
import torch
import torch.optim as optim

from zjh_constants import GameConfig
from zjh_env import ZJHEnv
from opponents import TightOpponent, AggroOpponent, MixedOpponent

from agent_dqn import (
    obs_to_vector, encode_actions,
    QNet, ReplayBuffer,
    select_action_epsilon_greedy,
)

def pick_opponent(rng: random.Random):
    r = rng.random()
    if r < 0.34:
        return TightOpponent(rng)
    if r < 0.67:
        return MixedOpponent(rng)
    return AggroOpponent(rng)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=50000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--base", type=int, default=1)
    ap.add_argument("--init_stack", type=int, default=200)
    ap.add_argument("--device", type=str, default="cpu")

    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--buffer_size", type=int, default=200000)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--learning_starts", type=int, default=2000)
    ap.add_argument("--train_every", type=int, default=1)
    ap.add_argument("--target_update", type=int, default=1000)

    ap.add_argument("--eps_start", type=float, default=1.0)
    ap.add_argument("--eps_end", type=float, default=0.05)
    ap.add_argument("--eps_decay_episodes", type=int, default=20000)

    ap.add_argument("--save_every", type=int, default=5000)
    ap.add_argument("--model_out", type=str, default="qnet_zjh_2p_ddqn.pt")

    args = ap.parse_args()

    rng = random.Random(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # ===== 固定 2 人桌 =====
    cfg = GameConfig(
        num_players=2,
        base=args.base,
        init_stack=args.init_stack,
        seed=args.seed
    )
    env = ZJHEnv(cfg)
    env.reset(seed=args.seed, hero_seat=0)

    obs_dim = len(obs_to_vector(env.get_obs(0)))

    q = QNet(obs_dim).to(args.device)
    q_tgt = QNet(obs_dim).to(args.device)
    q_tgt.load_state_dict(q.state_dict())
    q_tgt.eval()

    opt = optim.Adam(q.parameters(), lr=args.lr)
    rb = ReplayBuffer(args.buffer_size, seed=args.seed)

    recent_rewards = deque(maxlen=100)

    def eps_by_ep(ep: int) -> float:
        # 线性衰减
        if ep >= args.eps_decay_episodes:
            return args.eps_end
        t = ep / float(args.eps_decay_episodes)
        return args.eps_start + (args.eps_end - args.eps_start) * t

    global_step = 0

    for ep in range(1, args.episodes + 1):
        env.reset(seed=rng.randrange(1_000_000), hero_seat=0)
        opp = pick_opponent(rng)

        done = False
        ep_reward = 0.0

        while not done:
            acting = env.acting_seat

            if acting == env.hero_seat:
                obs = env.get_obs(env.hero_seat)
                legal = env.legal_actions(env.hero_seat)
                enc = encode_actions(legal)
                s = obs_to_vector(obs)

                eps = eps_by_ep(ep)
                a_idx = select_action_epsilon_greedy(q, s, enc, eps, args.device, rng)
                action = enc.actions[a_idx]

                # 执行动作（hero）
                _, r, done, _ = env.step(env.hero_seat, action)
                ep_reward += r

                # 下一状态（统一从 hero 视角取 obs）
                s2 = obs_to_vector(env.get_obs(env.hero_seat))
                legal2 = env.legal_actions(env.hero_seat)
                enc2 = encode_actions(legal2)

                rb.push(s, enc.mask, a_idx, float(r), s2, enc2.mask, bool(done))
                global_step += 1

                # ===== 训练：Double DQN =====
                if len(rb) >= args.learning_starts and (global_step % args.train_every == 0):
                    bs = args.batch_size
                    S, M, A, R, S2, M2, D = rb.sample(bs)

                    S = torch.from_numpy(S).to(args.device)
                    M = torch.from_numpy(M).to(args.device)     # (B, MAX_ACTIONS)
                    A = torch.from_numpy(A).to(args.device)     # (B,)
                    R = torch.from_numpy(R).to(args.device)     # (B,)
                    S2 = torch.from_numpy(S2).to(args.device)
                    M2 = torch.from_numpy(M2).to(args.device)
                    D = torch.from_numpy(D).to(args.device)     # (B,) 0/1

                    # Q(s,a)
                    q_all = q(S)  # (B, MAX_ACTIONS)
                    q_sa = q_all.gather(1, A.view(-1, 1)).squeeze(1)  # (B,)

                    with torch.no_grad():
                        # 1) online 网络在 s' 上选 a*
                        q2_online = q(S2)  # (B, MAX_ACTIONS)
                        q2_online = q2_online + (M2 - 1.0) * 1e9
                        a_star = q2_online.argmax(dim=1)  # (B,)

                        # 2) target 网络在 s' 上评估 Q_tgt(s', a*)
                        q2_tgt = q_tgt(S2)
                        q2_tgt = q2_tgt + (M2 - 1.0) * 1e9
                        q2_val = q2_tgt.gather(1, a_star.view(-1, 1)).squeeze(1)  # (B,)

                        target = R + (1.0 - D) * args.gamma * q2_val

                    loss = torch.mean((q_sa - target) ** 2)

                    opt.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(q.parameters(), 5.0)
                    opt.step()

                    # target network 更新
                    if global_step % args.target_update == 0:
                        q_tgt.load_state_dict(q.state_dict())

            else:
                # 对手动作
                oobs = env.get_obs(acting)
                legal = env.legal_actions(acting)
                action = opp.act(oobs, legal)
                _, r, done, _ = env.step(acting, action)
                ep_reward += r

        recent_rewards.append(ep_reward)

        # 每 100 局输出统计
        if ep % 100 == 0:
            avg100 = sum(recent_rewards) / len(recent_rewards)
            win_rate = sum(1 for x in recent_rewards if x > 0) / len(recent_rewards) * 100.0
            eps = eps_by_ep(ep)
            print(
                f"[ep {ep}] r={ep_reward:.2f} eps={eps:.3f} | "
                f"avg100={avg100:.3f} win_rate={win_rate:.1f}% | "
                f"buf={len(rb)} step={global_step}"
            )

        # 定期保存
        if ep % args.save_every == 0:
            torch.save(q.state_dict(), args.model_out)
            print(f"[save] {args.model_out}")

    torch.save(q.state_dict(), args.model_out)
    print(f"[done] saved {args.model_out}")

if __name__ == "__main__":
    main()
