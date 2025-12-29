# ppo_hier.py
import torch
import torch.nn.functional as F

def masked_softmax(logits, mask, dim=-1, eps=1e-8):
    # mask: 0/1
    logits = logits - logits.max(dim=dim, keepdim=True).values
    exp = torch.exp(logits) * mask
    z = exp.sum(dim=dim, keepdim=True) + eps
    return exp / z

def sample_hier(fold_logits, play_logits, mask_full, temperature=1.0):
    """
    mask_full: [B,29] float32
    action space:
      0 = FOLD
      1..28 = PLAY actions
    """
    if temperature != 1.0:
        fold_logits = fold_logits / temperature
        play_logits = play_logits / temperature

    # fold mask: allow fold? / allow play?
    allow_fold = mask_full[:, 0:1]
    allow_play = (mask_full[:, 1:].sum(dim=1, keepdim=True) > 0).float()

    mask_fold = torch.cat([allow_fold, allow_play], dim=1)  # [B,2]

    p_fold = masked_softmax(fold_logits, mask_fold, dim=1)  # [B,2]
    dist_fold = torch.distributions.Categorical(probs=p_fold)
    fold_choice = dist_fold.sample()  # 0=FOLD, 1=PLAY
    logp_fold = dist_fold.log_prob(fold_choice)  # [B]

    # 默认输出
    action = torch.zeros_like(fold_choice)
    logp = logp_fold.clone()

    # 需要 play 的样本
    idx = (fold_choice == 1).nonzero(as_tuple=False).squeeze(-1)
    if idx.numel() > 0:
        # play mask / probs
        mask_play = mask_full[idx, 1:]  # [K,28]
        p_play = masked_softmax(play_logits[idx], mask_play, dim=1)
        dist_play = torch.distributions.Categorical(probs=p_play)
        play_choice = dist_play.sample()  # 0..27 => 对应 action 1..28
        logp_play = dist_play.log_prob(play_choice)

        action[idx] = play_choice + 1
        logp[idx] = logp[idx] + logp_play

    return action, logp

def logprob_hier(fold_logits, play_logits, mask_full, actions):
    """
    用于 PPO 更新时计算 new_logp(actions)
    actions: [B] in 0..28
    """
    allow_fold = mask_full[:, 0:1]
    allow_play = (mask_full[:, 1:].sum(dim=1, keepdim=True) > 0).float()
    mask_fold = torch.cat([allow_fold, allow_play], dim=1)

    p_fold = masked_softmax(fold_logits, mask_fold, dim=1)
    logp_fold_all = torch.log(p_fold + 1e-8)  # [B,2]

    is_fold = (actions == 0)
    is_play = ~is_fold

    logp = torch.zeros_like(actions, dtype=torch.float32)

    # fold logp
    logp[is_fold] = logp_fold_all[is_fold, 0]

    # play logp = logp_fold(PLAY) + logp_play(a-1)
    if is_play.any():
        idx = is_play.nonzero(as_tuple=False).squeeze(-1)
        a = actions[idx] - 1  # 0..27
        mask_play = mask_full[idx, 1:]
        p_play = masked_softmax(play_logits[idx], mask_play, dim=1)
        logp_play_all = torch.log(p_play + 1e-8)
        logp[idx] = logp_fold_all[idx, 1] + logp_play_all[torch.arange(idx.numel()), a]

    return logp
