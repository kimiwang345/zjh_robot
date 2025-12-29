# ppo.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any, Tuple
import numpy as np
import torch
import torch.nn.functional as F


@dataclass
class PPOConfig:
    # ===== 折扣 & GAE =====
    gamma: float = 0.99
    lam: float = 0.95

    # ===== PPO clipping =====
    clip_eps: float = 0.2

    # ===== value loss =====
    vf_coef: float = 0.5

    # ===== 优化 =====
    lr: float = 3e-4
    max_grad_norm: float = 0.5

    # ===== PPO 训练 =====
    epochs: int = 4        # ⭐ 唯一标准字段


def masked_categorical_sample(logits: torch.Tensor, mask: torch.Tensor, temperature: float = 1.0) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    logits: [B, A], mask: [B, A] 1合法 0非法
    """
    t = max(0.05, float(temperature))
    logits = logits / t
    neg_inf = torch.tensor(-1e9, device=logits.device, dtype=logits.dtype)
    masked_logits = torch.where(mask > 0, logits, neg_inf)
    probs = F.softmax(masked_logits, dim=-1)
    dist = torch.distributions.Categorical(probs=probs)
    action = dist.sample()
    logp = dist.log_prob(action)
    return action, logp

def compute_gae(rews: np.ndarray, vals: np.ndarray, dones: np.ndarray, gamma: float, lam: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    rews, vals, dones shape: [T]
    returns: adv[T], ret[T]
    """
    T = len(rews)
    adv = np.zeros(T, dtype=np.float32)
    lastgaelam = 0.0
    for t in reversed(range(T)):
        nextnonterminal = 1.0 - float(dones[t])
        nextvalue = vals[t + 1] if t + 1 < len(vals) else 0.0
        delta = rews[t] + gamma * nextvalue * nextnonterminal - vals[t]
        lastgaelam = delta + gamma * lam * nextnonterminal * lastgaelam
        adv[t] = lastgaelam
    ret = adv + vals[:T]
    return adv, ret

def ppo_update(net, opt, obs, act, old_logp, adv, ret, mask, cfg: PPOConfig):
    """
    obs: [N, obs_dim]
    act: [N]
    old_logp: [N]
    adv: [N]
    ret: [N]
    mask: [N, A]
    """
    N = obs.size(0)
    idx = torch.randperm(N)

    for _ in range(cfg.train_epochs):
        for start in range(0, N, cfg.batch_size):
            j = idx[start:start + cfg.batch_size]
            b_obs = obs[j]
            b_act = act[j]
            b_old = old_logp[j]
            b_adv = adv[j]
            b_ret = ret[j]
            b_mask = mask[j]

            logits, value = net(b_obs)

            neg_inf = torch.tensor(-1e9, device=logits.device, dtype=logits.dtype)
            masked_logits = torch.where(b_mask > 0, logits, neg_inf)
            logp_all = F.log_softmax(masked_logits, dim=-1)
            logp = logp_all.gather(1, b_act.unsqueeze(1)).squeeze(1)

            ratio = torch.exp(logp - b_old)
            surr1 = ratio * b_adv
            surr2 = torch.clamp(ratio, 1 - cfg.clip, 1 + cfg.clip) * b_adv
            loss_pi = -torch.min(surr1, surr2).mean()

            loss_v = F.mse_loss(value, b_ret)
            # entropy（拟人）
            probs = torch.exp(logp_all)
            ent = -(probs * logp_all).sum(dim=-1).mean()

            loss = loss_pi + cfg.vf_coef * loss_v - cfg.ent_coef * ent

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), cfg.max_grad_norm)
            opt.step()
