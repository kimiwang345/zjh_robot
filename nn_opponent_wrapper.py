import torch
import numpy as np

from agent_a30 import a30Agent
from config_a30_hybrid import A30ConfigHybrid

class NNOpponentA30:
    """
    NN Opponent wrapper for ZJHEnvMulti14_8p
    必须实现 decide(env, seat_id)
    """

    def __init__(self, weight_path, device="cpu"):
        self.device = device

        ckpt = torch.load(weight_path, map_location="cpu")
        cfg = A30ConfigHybrid(**ckpt["cfg"])
        cfg.device = device

        self.agent = a30Agent(cfg)
        self.agent.load(weight_path)
        self.agent.eval_net.to(device)
        self.agent.eval_net.eval()

    @torch.no_grad()
    def decide(self, env, seat_id):
        # ✅ ZJHEnvMulti14_8p 正确状态接口
        state = env._get_state()

        st = torch.tensor(
            state,
            dtype=torch.float32,
            device=self.device
        ).unsqueeze(0)

        q = self.agent.eval_net(st)
        action = int(torch.argmax(q, dim=1).item())

        # === 动作映射（必须与你训练时一致）===
        if action == 0:
            return 0, None  # FOLD
        elif action == 1:
            return 1, None  # CALL
        elif action == 2:
            return 2, None  # LOOK
        elif 3 <= action <= 12:
            return 3, action - 1  # BET
        else:
            return 5, None  # COMPARE_ALL
