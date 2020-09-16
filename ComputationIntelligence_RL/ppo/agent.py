import torch
import torch.nn as nn
import gym

from torch.distributions import Categorical

from memory import Memory
from ac import ActorCritic

# 使用gpu
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class Agent:
    def __init__(self,input_size,output_size,hidden_size,lr,beta,gamma,update_epoch,epsilon):
        self.input_size = input_size # 即state's shape
        self.output_size = output_size # action's shape
        self.hidden_size = hidden_size
        self.lr = lr #学习率
        self.beta = beta # for optimizer
        self.gamma = gamma #衰减因子
        self.update_epoch = update_epoch #更新policy的回合数
        self.epsilon= epsilon #clip时需要

        # 创建 pi 和 pi_old
        self.policy = ActorCritic(input_size,output_size,hidden_size).to(device)
        self.old_policy = ActorCritic(input_size,output_size,hidden_size).to(device)
        # use adam optimizer to update nn's weight
        self.optimizer = torch.optim.Adam(
            self.policy.parameters(),
            lr=lr,
            betas=beta
        )
        # load policy's model
        self.old_policy.load_state_dict(self.policy.state_dict())

        self.loss = nn.MSELoss()

    # 更新网络
    def update(self,memory):
        rewards = []
        discounted_reward = 0

        # MC to calculate every state reward
        for reward,is_done in zip(reversed(memory.rewards),reversed(memory.is_done)):
            if is_done:
                discounted_reward = 0
            # 进行累加
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0,discounted_reward)
        
        # list to tensor
        rewards = torch.tensor(rewards).to(device)
        #normalize
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)

        # prepare to update policy
        old_states = torch.stack(memory.states).to(device).detach()
        old_actions = torch.stack(memory.actions).to(device).detach()
        old_logprobs = torch.stack(memory.logprobs).to(device).detach()

        # update
        for _ in range(self.update_epoch):
            # critic evaluate data sampled by old policy
            logprobs,states_value,dist_entropy = self.policy.evaluate(old_states,old_actions)

            # begin to calculate J_ppo

            # benefit of using log is that it uses minus op instead of divide
            # use log so need to do exp op after minus op
            frac = torch.exp(logprobs - old_logprobs.detach())

            # calculate advantage function
            advantages = rewards - states_value.detach()
            
            item_1 = frac * advantages
            # clip func
            item_2 = torch.clamp(frac,1-self.epsilon,1+self.epsilon) * advantages

            # calculate loss function
            loss = -torch.min(item_1,item_2) + 0.5 * self.loss(states_value,rewards) - 0.01 *dist_entropy

            # update
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()

        # load updated policy to old_policy
        self.old_policy.load_state_dict(self.policy.state_dict())

