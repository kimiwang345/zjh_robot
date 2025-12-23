# a30_selfplay_opponent.py
# Exploit-only A30 作为对手（冻结参数）

import torch
import numpy as np

class A30SelfPlayOpponent:
    def __init__(self, agent):
        self.agent = agent
        self.net = agent.eval_net
        self.device = agent.device

        self.net.eval()  # ★ Exploit-only：关闭 Noisy

    @torch.no_grad()
    def decide(self, env, idx):
        """
        env: ZJHEnvMulti14_8p
        idx: 对手座位索引
        返回：(OppAct, param)
        """
        # env 的状态编码是“Hero视角”
        # 但对手用同一状态即可（这是你环境的设计前提）
        state = env._get_state()
        st = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)

        q = self.net(st)
        action = int(torch.argmax(q, dim=1).item())

        # 将 action 映射成 OppAct
        # 你 env 中 enemy 逻辑：
        # FOLD=0, CALL=1, LOOK=2, BET=3~11, PK=12, COMPARE_ALL=13
        if action == 0:
            return env.OppAct.FOLD, None
        elif action == 1:
            return env.OppAct.CALL, None
        elif action == 2:
            return env.OppAct.LOOK, None
        elif 3 <= action <= 11:
            mult_map = {
                3: 2, 4: 3, 5: 4, 6: 5, 7: 6,
                8: 7, 9: 8, 10: 9, 11: 10
            }
            return env.OppAct.BET, mult_map[action]
        elif action == 12:
            return env.OppAct.PK, None
        elif action == 13:
            return env.OppAct.COMPARE_ALL, None

        return env.OppAct.CALL, None
