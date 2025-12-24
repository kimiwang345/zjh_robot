import argparse
import random
from collections import deque

import numpy as np
import torch

from zjh_constants import GameConfig
from zjh_env import ZJHEnv
from opponents import TightOpponent, AggroOpponent, MixedOpponent
from agent_dqn import obs_to_vector, encode_actions, QNet

def pick_opponent(rng: random.Random):
    r = rng.random()
    if r < 0.34:
        return TightOpponent(rng)
    if r < 0.67:
        return MixedOpponent(rng)
    return AggroOpponent(rng)

def greedy_action(q, state_vec, enc, device):
    with torch.no_grad():
        s = torch.tensor(state_vec, dtype=torch.float32, device=device).unsqueeze(0)
        qvals = q(s).squeeze(0)
        qvals = qvals + (torch.tensor(enc.mask, device=device) - 1.0) * 1e9
        return int(torch.argmax(qvals).item())

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=500)
    ap.add_argument("--model", type=str, required=True)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--device", type=str, default="cpu")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    cfg = GameConfig(
        num_players=2,
        base=1,
        init_stack=200,
        seed=args.seed
    )
    env = ZJHEnv(cfg)

    env.reset(seed=args.seed, hero_seat=0)
    obs_dim = len(obs_to_vector(env.get_obs(0)))

    q = QNet(obs_dim).to(args.device)
    q.load_state_dict(torch.load(args.model, map_location=args.device))
    q.eval()

    recent_rewards = deque(maxlen=100)

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

                idx = greedy_action(q, s, enc, args.device)
                action = enc.actions[idx]

                _, r, done, _ = env.step(env.hero_seat, action)
                ep_reward += r
            else:
                oobs = env.get_obs(acting)
                legal = env.legal_actions(acting)
                action = opp.act(oobs, legal)
                _, r, done, _ = env.step(acting, action)
                ep_reward += r

        recent_rewards.append(ep_reward)

        if ep % 100 == 0:
            avg100 = sum(recent_rewards) / len(recent_rewards)
            win_rate = sum(1 for x in recent_rewards if x > 0) / len(recent_rewards) * 100
            print(
                f"[eval {ep}] avg100={avg100:.3f} win_rate={win_rate:.1f}%"
            )

if __name__ == "__main__":
    main()
