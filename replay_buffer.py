"""
replay_buffer.py
================

统一 ReplayBuffer 入口，当前默认使用 PERReplayBuffer
"""

from per_buffer import PERReplayBuffer

__all__ = ["PERReplayBuffer"]
