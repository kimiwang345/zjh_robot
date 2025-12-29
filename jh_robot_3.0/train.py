# train.py
import os
import time
import random
import numpy as np
import torch

from env_zjh import ZJHEnv
from obs_encoder import encode_obs, obs_dim
from policy_mask import build_action_mask

from model_hier import PolicyValueNetHier
from ppo_hier import sample_hier, logprob_hier

from ppo import PPOConfig, compute_gae
from config.config_loader import HotConfig
from action_codec import decode_action
from cards import Card, hand_strength


# =====================================================
# 主训练流程（Hierarchical PPO）
# =====================================================
def main():
    print("=== ZJH HIER PPO TRAIN START ===", flush=True)

    # -----------------------------
    # 基础准备
    # -----------------------------
    os.makedirs("weights", exist_ok=True)

    reward_cfg = HotConfig("config/reward_config.json")
    opp_cfg = HotConfig("config/opponent_config.json")

    env = ZJHEnv(num_players=5)
    net = PolicyValueNetHier(obs_dim())
    optimizer = torch.optim.Adam(net.parameters(), lr=3e-4)

    cfg = PPOConfig()

    STEPS_PER_UPDATE = 4000
    TOTAL_UPDATES = 300
    INIT_STACK_BB = 200

    global_step = 0
    start_time = time.time()

    # -----------------------------
    # 对手强度调度
    # -----------------------------
    def current_tier(update):
        schedule = opp_cfg.get()["schedule"]
        for s in schedule:
            if update <= s["update_lte"]:
                return s["tier"]
        return "strong"

    # =================================================
    # Update Loop
    # =================================================
    for update in range(1, TOTAL_UPDATES + 1):
        env.set_opponent_tier(current_tier(update))

        obs_buf = []
        act_buf = []
        logp_buf = []
        val_buf = []
        rew_buf = []
        done_buf = []
        mask_buf = []

        while len(obs_buf) < STEPS_PER_UPDATE:
            base_bet = random.randint(1, 50)
            init_stack = INIT_STACK_BB * base_bet
            obs = env.reset(base_bet, init_stack)

            done = False

            while not done and len(obs_buf) < STEPS_PER_UPDATE:
                me = obs["me"]
                my_seen = me["seen"]
                min_mul = me["min_mul"]
                round_id = obs["round"]

                # ========== mask / obs ==========
                mask_np = build_action_mask(obs)
                mask = torch.tensor(mask_np, dtype=torch.float32).unsqueeze(0)

                x = torch.tensor(
                    encode_obs(obs),
                    dtype=torch.float32
                ).unsqueeze(0)

                with torch.no_grad():
                    fold_logits, play_logits, value = net(x)

                # ========== 分头采样 ==========
                act, logp = sample_hier(
                    fold_logits,
                    play_logits,
                    mask,
                    temperature=1.0
                )

                act_id = int(act.item())
                dec = decode_action(act_id)

                # ========== 执行动作 ==========
                if dec.kind == "BET":
                    done, _ = env.step(0, "BET", bet_mul=dec.bet_mul)
                elif dec.kind == "PK":
                    done, _ = env.step(0, "PK")
                else:
                    done, _ = env.step(0, "FOLD")

                # =================================================
                # reward（注意：这里只保留“轻 shaping”）
                # =================================================
                r = 0.0
                cfg_r = reward_cfg.get()

                # ---- 行为成本 ----
                if dec.kind == "BET":
                    r += cfg_r["base_cost"]["bet"]
                elif dec.kind == "PK":
                    r += cfg_r["base_cost"]["pk"]

                # ---- 已看牌：弱牌小惩罚（不再决定强牌行为）----
                if my_seen and me.get("cards"):
                    cards = [Card(c["rank"], c["suit"]) for c in me["cards"]]
                    hand_cat, _ = hand_strength(cards)

                    if (
                        cfg_r["small_hand_penalty"]["enabled"]
                        and hand_cat <= cfg_r["small_hand_penalty"]["hand_cat_max"]
                    ):
                        if dec.kind == "BET":
                            aggr = dec.bet_mul / max(1.0, min_mul)
                            for rule in cfg_r["small_hand_penalty"]["aggressive_ratio"]:
                                if aggr >= rule["gte"]:
                                    r += rule["penalty"]
                                    break

                # ---- 终局收益 ----
                if done and cfg_r["terminal_reward"]["enabled"]:
                    final_stack = env.players[0].stack
                    r += (
                        (final_stack - init_stack)
                        / max(1.0, base_bet)
                        * cfg_r["terminal_reward"]["scale"]
                    )

                # ========== 存样本 ==========
                obs_buf.append(encode_obs(obs))
                act_buf.append(act_id)
                logp_buf.append(float(logp.item()))
                val_buf.append(float(value.item()))
                rew_buf.append(r)
                done_buf.append(1.0 if done else 0.0)
                mask_buf.append(mask_np)

                obs = env.observe(0)
                global_step += 1

        # =================================================
        # PPO UPDATE（Hierarchical logp）
        # =================================================
        obs_t = torch.tensor(np.asarray(obs_buf), dtype=torch.float32)
        act_t = torch.tensor(np.asarray(act_buf), dtype=torch.int64)
        old_logp_t = torch.tensor(np.asarray(logp_buf), dtype=torch.float32)
        mask_t = torch.tensor(np.asarray(mask_buf), dtype=torch.float32)

        vals = np.asarray(val_buf, dtype=np.float32)
        rews = np.asarray(rew_buf, dtype=np.float32)
        dones = np.asarray(done_buf, dtype=np.float32)

        adv, ret = compute_gae(
            rews, vals, dones, cfg.gamma, cfg.lam
        )
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        adv_t = torch.tensor(adv, dtype=torch.float32)
        ret_t = torch.tensor(ret, dtype=torch.float32)

        # ---- 多 epoch 更新 ----
        for _ in range(cfg.epochs):
            fold_logits, play_logits, values = net(obs_t)

            new_logp = logprob_hier(
                fold_logits,
                play_logits,
                mask_t,
                act_t
            )

            ratio = torch.exp(new_logp - old_logp_t)
            surr1 = ratio * adv_t
            surr2 = torch.clamp(
                ratio,
                1.0 - cfg.clip_eps,
                1.0 + cfg.clip_eps
            ) * adv_t

            policy_loss = -torch.min(surr1, surr2).mean()
            value_loss = ((values - ret_t) ** 2).mean()

            loss = policy_loss + cfg.vf_coef * value_loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), cfg.max_grad_norm)
            optimizer.step()

        # =================================================
        # LOG / SAVE
        # =================================================
        if update % 10 == 0:
            torch.save(net.state_dict(), "weights/zjh_policy.pt")
            print(
                f"[update {update}] "
                f"tier={env.opponent_tier} "
                f"steps={global_step} "
                f"time={time.time() - start_time:.1f}s",
                flush=True
            )

    torch.save(net.state_dict(), "weights/zjh_policy.pt")
    print("=== TRAIN FINISHED ===", flush=True)


if __name__ == "__main__":
    main()
