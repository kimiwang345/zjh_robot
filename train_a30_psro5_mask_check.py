import os
import time
import random
import argparse
from collections import defaultdict, deque
from state_builder_normalized import build_state



import numpy as np
import torch

from ZJHEnvMulti2to9 import ZJHEnvMulti2to9
from agent_a30 import A30Agent

# 兼容两种导入方式：推荐 replay_buffer.py -> per_buffer.py
try:
    from replay_buffer import PERReplayBuffer
except Exception:
    from per_buffer import PERReplayBuffer

from zjh_opponents_multi import (
    NitOpponent,
    LooseOpponent,
    AggroOpponent,
    FishOpponent,
    TAGOpponent,
    ManiacOpponent,
    GTOOpponent,
)

def evaluate_policy(
    agent,
    env,
    opponent_pool,
    episodes=200,
    max_steps=50
):
    """
    exploit-only evaluation
    """
    old_epsilon = agent.epsilon
    agent.epsilon = 0.0  # ⭐ exploit only

    total_reward = 0.0
    total_steps = 0
    win_cnt = 0

    first_round_actions = Counter()

    for _ in range(episodes):
        env.reset()
        env.bind_opponents(
            [None] + [random.choice(opponent_pool) for _ in range(env.max_players - 1)]
        )

        done = False
        step = 0
        episode_reward = 0.0

        while not done and step < max_steps:
            state = build_state(env)
            mask = build_action_mask(env)

            action = agent.select_action(state, mask)

            # ⭐ 统计第一轮行为
            if env.round_index == 0:
                first_round_actions[action] += 1

            _, reward, done, _ = env.step(action)

            episode_reward += reward
            step += 1

        total_reward += episode_reward
        total_steps += step
        if episode_reward > 0:
            win_cnt += 1

    agent.epsilon = old_epsilon  # restore

    print("\n====== EVALUATION ======")
    print(f"episodes      : {episodes}")
    print(f"avg_reward    : {total_reward / episodes:.3f}")
    print(f"avg_steps     : {total_steps / episodes:.2f}")
    print(f"win_rate      : {win_cnt / episodes:.2%}")

    print("first round action dist:")
    total = sum(first_round_actions.values())
    for a, c in sorted(first_round_actions.items()):
        print(f"  action {a:2d}: {c / total:.2%}")

    print("========================\n")



# ============================================================
# Action Mask（必须与 ai_service / eval 完全一致）
# ============================================================

def build_action_mask(env: ZJHEnvMulti2to9):
    """
    mask shape=(14,)
      0 FOLD
      1 LOOK
      2..11 BET_1..BET_10
      12 PK
      13 COMPARE_ALL
    """
    mask = [1] * env.action_dim
    hero = env.players[0]

    # 第一轮禁止 PK（规则：第一轮不许PK）
    if env.round_index == 0:
        mask[12] = 0

    # 已看牌不能再 LOOK
    if hero["has_seen"]:
        mask[1] = 0

    # 下注最小倍数与筹码约束
    min_unit = env._min_required_unit(hero)
    min_pay = min_unit * env.ante_bb * (2 if hero["has_seen"] else 1)

    # 筹码不足：禁止所有 BET（只允许 FOLD / COMPARE_ALL）
    if hero["stack_bb"] < min_pay:
        for a in range(2, 12):
            mask[a] = 0
    else:
        # unit < min_unit 的 BET 禁止
        for a in range(2, 12):
            unit = a - 1
            if unit < min_unit:
                mask[a] = 0

    # mask 不能全 0（至少 FOLD 可用）
    if sum(mask) == 0:
        mask[0] = 1

    return np.array(mask, dtype=np.float32)


# ============================================================
# Opponent Pool / PSRO-5
# ============================================================

def build_opponent_pool():
    return [
        NitOpponent(),
        LooseOpponent(),
        AggroOpponent(),
        FishOpponent(),
        TAGOpponent(),
        ManiacOpponent(),
        GTOOpponent(),
    ]


def sample_opponents(pool, max_players):
    """
    opponents list len=max_players
      idx 0 -> None (Hero)
      idx 1.. -> opponent instance
    """
    opponents = [None]
    for _ in range(max_players - 1):
        opponents.append(random.choice(pool))
    return opponents


# ============================================================
# Checkpoints
# ============================================================

def save_checkpoint(path, agent, buffer, meta: dict):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = {
        "agent": {
            "q_net": agent.q_net.state_dict(),
            "target_q_net": agent.target_q_net.state_dict(),
            "epsilon": agent.epsilon,
            "total_steps": agent.total_steps,
        },
        "buffer": buffer,   # 直接存对象（本地训练可用）
        "meta": meta,
    }
    torch.save(payload, path)


def load_checkpoint(path, agent, device):
    ckpt = torch.load(path, map_location=device)
    agent.q_net.load_state_dict(ckpt["agent"]["q_net"])
    agent.target_q_net.load_state_dict(ckpt["agent"]["target_q_net"])
    agent.epsilon = ckpt["agent"].get("epsilon", agent.epsilon)
    agent.total_steps = ckpt["agent"].get("total_steps", 0)
    buffer = ckpt.get("buffer", None)
    meta = ckpt.get("meta", {})
    return buffer, meta


# ============================================================
# Train
# ============================================================

def train(args):
    # ---- seeds ----
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print(f"[INIT] device={device}")

    env = ZJHEnvMulti2to9(
        min_players=args.min_players,
        max_players=args.max_players,
        max_round=args.max_round
    )

    agent = A30Agent(
        state_dim=env.state_dim,
        action_dim=env.action_dim,
        device=device,
        lr=args.lr,
        gamma=args.gamma,
        epsilon_start=args.eps_start,
        epsilon_end=args.eps_end,
        epsilon_decay=args.eps_decay,
        target_update_interval=args.target_update,
    )

    # ---- replay buffer ----
    buffer = PERReplayBuffer(
        capacity=args.buffer_cap,
        alpha=args.per_alpha,
        beta_start=args.per_beta_start,
        beta_frames=args.per_beta_frames,
    )

    # ---- optional resume ----
    start_episode = 0
    total_steps = 0

    if args.resume and os.path.exists(args.resume):
        print(f"[RESUME] loading {args.resume}")
        loaded_buffer, meta = load_checkpoint(args.resume, agent, device)
        if loaded_buffer is not None:
            buffer = loaded_buffer
        start_episode = int(meta.get("episode", 0))
        total_steps = int(meta.get("total_steps", agent.total_steps))
        agent.total_steps = total_steps
        print(f"[RESUME] episode={start_episode}, total_steps={total_steps}, buffer={buffer.size()}")

    opponent_pool = build_opponent_pool()

    os.makedirs(args.model_dir, exist_ok=True)

    # ---- stats ----
    recent_rewards = deque(maxlen=200)
    recent_len = deque(maxlen=200)
    action_counter = defaultdict(int)

    print("[TRAIN] start")
    episode = start_episode

    while total_steps < args.max_steps:
        episode += 1

        env.reset()
        state = build_state(env)
        #state = env.reset()

        # 绑定对手（你的 env 现在支持 bind_opponents）
        env.bind_opponents(sample_opponents(opponent_pool, env.max_players))

        done = False
        ep_reward = 0.0
        ep_steps = 0

        while not done:
            mask = build_action_mask(env)

            # mask_check: 至少一个合法动作
            if mask.sum() <= 0:
                raise RuntimeError("mask is empty (no valid actions)")

            action = agent.select_action(state, mask)

            # mask_check: action 必须合法
            if mask[action] <= 0:
                raise RuntimeError(f"illegal action selected: action={action}, mask={mask.tolist()}")

            #next_state, reward, done, _ = env.step(action)
            _, reward, done, _ = env.step(action)
            next_state = build_state(env)

            # next_mask：用于 target Q（done 时不会用到，但仍需占位）
            if not done:
                next_mask = build_action_mask(env)
            else:
                next_mask = mask

            buffer.push(
                state,
                action,
                reward,
                next_state,
                float(done),
                mask,
                next_mask
            )


            state = next_state
            ep_reward += float(reward)
            ep_steps += 1
            total_steps += 1
            action_counter[action] += 1

            # 训练
            if buffer.size() >= args.learn_starts:
                agent.train(buffer, batch_size=args.batch_size)

            # 保存模型（step）
            if total_steps % args.save_interval == 0:
                ckpt_path = os.path.join(args.model_dir, f"a30_psro5_step_{total_steps}.pth")
                agent.save(ckpt_path)
                print(f"[SAVE] model={ckpt_path}")

                if args.save_ckpt:
                    ckpt2 = os.path.join(args.model_dir, "train_resume.pth")
                    save_checkpoint(
                        ckpt2,
                        agent,
                        buffer,
                        meta={"episode": episode, "total_steps": total_steps}
                    )
                    print(f"[SAVE] resume_ckpt={ckpt2}")

            if total_steps >= args.max_steps:
                break

        recent_rewards.append(ep_reward)
        recent_len.append(ep_steps)

        # 日志
        if episode % args.log_interval == 0:
            avg_r = sum(recent_rewards) / max(1, len(recent_rewards))
            avg_l = sum(recent_len) / max(1, len(recent_len))

            # action 分布（最近全局累计）
            total_a = sum(action_counter.values()) or 1
            top_actions = sorted(action_counter.items(), key=lambda x: -x[1])[:6]
            top_str = ", ".join([f"a{a}:{c/total_a:.1%}" for a, c in top_actions])

            print(
                f"[LOG] ep={episode} steps={total_steps} "
                f"avgR(200)={avg_r:.3f} avgLen(200)={avg_l:.2f} "
                f"buf={buffer.size()} eps={agent.epsilon:.3f} "
                f"top={top_str}"
            )

    # final save
    final_path = os.path.join(args.model_dir, f"a30_psro5_final_step_{total_steps}.pth")
    agent.save(final_path)
    print(f"[DONE] saved final model: {final_path}")


# ============================================================
# CLI
# ============================================================

def parse_args():
    p = argparse.ArgumentParser()

    # core
    p.add_argument("--max_steps", type=int, default=500_000)
    p.add_argument("--model_dir", type=str, default="models")
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--seed", type=int, default=42)

    # env
    p.add_argument("--min_players", type=int, default=2)
    p.add_argument("--max_players", type=int, default=9)
    p.add_argument("--max_round", type=int, default=20)

    # agent
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--eps_start", type=float, default=1.0)
    p.add_argument("--eps_end", type=float, default=0.05)
    p.add_argument("--eps_decay", type=int, default=300_000)
    p.add_argument("--target_update", type=int, default=5_000)

    # buffer
    p.add_argument("--buffer_cap", type=int, default=300_000)
    p.add_argument("--per_alpha", type=float, default=0.6)
    p.add_argument("--per_beta_start", type=float, default=0.4)
    p.add_argument("--per_beta_frames", type=int, default=500_000)

    # learning
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--learn_starts", type=int, default=10_000)

    # save/log
    p.add_argument("--save_interval", type=int, default=20_000)
    p.add_argument("--log_interval", type=int, default=50)
    p.add_argument("--save_ckpt", action="store_true")
    p.add_argument("--resume", type=str, default="")

    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)
