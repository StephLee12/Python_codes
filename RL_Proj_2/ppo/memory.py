import torch
import torch.nn as nn
import gym

from torch.distributions import Categorical

class Memory:
    def __init__(self):
        self.actions = [] # 存储action
        self.states = [] #存储state
        self.logprobs = [] #存储取对数后的概率
        self.rewards = [] #存储reward
        self.is_done = [] #存储epoch 是否完成
    
    def clear_memory(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.is_done = []