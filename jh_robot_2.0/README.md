# ZJH AI (按最新规则重写)

## 环境
Python 3.9+
pip install torch numpy

## 训练
python train_pg.py
python train_pg.py --episodes 30000


## 评估
python evaluate.py --episodes 200
python evaluate_dqn.py --episodes 500 --model qnet_zjh_2p_ddqn.pt
python evaluate_dqn.py --episodes 500 --model qnet_zjh_2p.pt

