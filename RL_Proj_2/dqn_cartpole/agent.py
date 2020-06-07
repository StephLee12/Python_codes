import gym
import math
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from net import Net

# 创建智能体
class Agent(object):
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)
        # Q值的神经网络
        self.eval_net = Net(self.state_space_dim, 256, self.action_space_dim)
        # 优化器
        self.optimizer = optim.Adam(self.eval_net.parameters(), lr=self.lr)
        # replay buffer
        self.buffer = []
        self.steps = 0
        
    def act(self, s0):
        self.steps += 1
        # epsilon greedy 计算epsilon
        epsi = self.epsi_low + (self.epsi_high-self.epsi_low) * (math.exp(-1.0 * self.steps/self.decay))
        if random.random() < epsi: #进行随机探索
            a0 = random.randrange(self.action_space_dim)
        else: #根据Q值 take action
            s0 =  torch.tensor(s0, dtype=torch.float).view(1,-1)
            a0 = torch.argmax(self.eval_net(s0)).item()
        return a0

    # 将s_t,s_(t+1),r_t,a_t这个序列存入replay buffer
    def put(self, *transition):
        if len( self.buffer)==self.capacity:
            self.buffer.pop(0)
        self.buffer.append(transition)
    
    # 学习Q值网络
    def learn(self):
        if (len(self.buffer)) < self.batch_size:
            return
        
        # 从replay buffer sample一个batch
        samples = random.sample( self.buffer, self.batch_size)
        s0, a0, r1, s1 = zip(*samples)
        s0 = torch.tensor( s0, dtype=torch.float)
        a0 = torch.tensor( a0, dtype=torch.long).view(self.batch_size, -1)
        r1 = torch.tensor( r1, dtype=torch.float).view(self.batch_size, -1)
        s1 = torch.tensor( s1, dtype=torch.float)
        
        # regression TD方法
        y_true = r1 + self.gamma * torch.max( self.eval_net(s1).detach(), dim=1)[0].view(self.batch_size, -1)
        y_pred = self.eval_net(s0).gather(1, a0)
        
        loss_fn = nn.MSELoss()
        loss = loss_fn(y_pred, y_true)
        
        # 更新Q值网络
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()