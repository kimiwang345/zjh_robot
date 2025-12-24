from __future__ import annotations
import argparse
import random
import numpy as np
import torch

from zjh_constants import GameConfig
from zjh_env import ZJHEnv
from opponents import TightOpponent, AggroOpponent, MixedOpponent
from agent_policy import PolicyNet, obs_to_vector, encode_actions, greedy_action

def pick_opponent(rng: random.Random):
    r = rng.random()
    if r < 0.34:
        return TightOpponent(rng)
    if r < 0.67:
        return MixedOpponent(rng)
    return AggroOpponent(rng)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=200)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--num_players", type=int, default=6)
    ap.add_argument("--init_stack", type=int, default=200)
    ap.add_argument("--base", type=int, default=1)
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--model", type=str, default="policy_zjh.pt")
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
    model.load_state_dict(torch.load(args.model, map_location=args.device))
    model.eval()

    rewards = []
    for i in range(args.episodes):
        env.reset(seed=rng.randrange(1_000_000), hero_seat=0)

        opp_policies = {}
        for s in range(env.cfg.num_players):
            if s != env.hero_seat:
                opp_policies[s] = pick_opponent(rng)

        done = False
        total_reward = 0.0
        while not done:
            acting = env.acting_seat
            if acting == env.hero_seat:
                obs = env.get_obs(env.hero_seat)
                legal = env.legal_actions(env.hero_seat)
                enc = encode_actions(legal)
                ov = obs_to_vector(obs)
                idx = greedy_action(model, ov, enc, device=args.device)
                action = enc.actions[idx]
                _, r, done, _ = env.step(env.hero_seat, action)
                total_reward += r
            else:
                oobs = env.get_obs(acting)
                legal = env.legal_actions(acting)
                action = opp_policies[acting].act(oobs, legal)
                _, r, done, _ = env.step(acting, action)
                total_reward += r

        rewards.append(total_reward)

    avg = float(np.mean(rewards))
    std = float(np.std(rewards))
    print(f"[eval] episodes={args.episodes} avg_reward={avg:.2f} std={std:.2f}")

if __name__ == "__main__":
    main()
