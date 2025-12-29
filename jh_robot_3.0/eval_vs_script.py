from env_zjh import ZJHEnv
from obs_encoder import encode_obs
from model import PolicyValueNet
from action_codec import decode_action, encode_bet_mul, build_opp_slots, pk_slot_to_seat
from ppo import masked_categorical_sample
import torch
import random
import numpy as np

DEVICE = "cpu"

def run_eval(games=1000):
    net = PolicyValueNet(46)
    net.load_state_dict(torch.load("weights/zjh_policy.pt", map_location=DEVICE))
    net.eval()

    wins = 0
    total = 0

    for _ in range(games):
        env = ZJHEnv(num_players=5)
        obs = env.reset(base_bet=random.randint(1,50), init_stack=1000)

        done = False
        while not done:
            seat = env.current_seat
            obs_dict = obs

            # 只评估 AI 自己（seat=0），其余人随机
            if seat == 0:
                x = torch.tensor(encode_obs(obs_dict), dtype=torch.float32).unsqueeze(0)
                mask, slots = build_action_mask(obs_dict)
                mask_t = torch.tensor(mask).unsqueeze(0)

                with torch.no_grad():
                    logits, _ = net(x)
                act, _ = masked_categorical_sample(logits, mask_t, temperature=0.9)
                dec = decode_action(int(act.item()))

                if dec.kind == "FOLD":
                    done, winner = env.step(seat, "FOLD")
                elif dec.kind == "PK":
                    tgt = pk_slot_to_seat(slots, dec.pk_slot)
                    done, winner = env.step(seat, "PK", pk_target_seat=tgt)
                else:
                    done, winner = env.step(seat, "BET", bet_mul=dec.bet_mul)
            else:
                # 其他人：简单随机
                acts = env.legal_actions()
                done, winner = env.step(random.choice(acts))

            obs = env.observe(env.current_seat)

        if winner == 0:
            wins += 1
        total += 1

    print("win rate:", wins / total)

run_eval(1000)
