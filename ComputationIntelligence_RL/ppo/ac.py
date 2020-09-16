import torch
import torch.nn as nn
import gym

from torch.distributions import Categorical

from memory import Memory

# 使用gpu
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class ActorCritic(nn.Module):  #ActorCritic类
    def __init__(self, input_size, output_size, hidden_size):
        super(ActorCritic, self).__init__()
        # input_size 即为 state.shaoe
        # output_size 即为 action.shape
        # hidden_size 即为每层神经元的数目
        self.actor_layer = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.Tanh(),  #tanh作为激活函数
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, output_size),
            nn.Softmax(dim=-1)  #softmax输出概率
        )

        # critic 的output size为1
        self.critic_layer = nn.Sequential(nn.Linear(input_size, hidden_size),
                                          nn.Tanh(),
                                          nn.Linear(hidden_size, hidden_size),
                                          nn.Tanh(), nn.Linear(hidden_size, 1))

    # choose an action
    def act(self, state, memory):
        # input params memory is an instance of class Memory

        # 将gym环境得到的state转换为tensor送入gpu
        state = torch.from_numpy(state).float().to(device)
        # 将state送入神经网络 得到动作的概率
        action_probs = self.actor_layer(state)
        # 依据概率分布 创建类别分布 即为不同的动作 0，1，2，3
        dist = Categorical(action_probs)
        #sample一个动作
        action = dist.sample()

        #存入memory
        memory.states.append(state)
        memory.actions.append(action)
        memory.logprobs.append(dist.log_prob(action))

        return action.item()

    # critic judge
    def evaluate(self, state, action):
        # 获得action的类别分布
        action_probs = self.actor_layer(state)
        dist = Categorical(action_probs)

        action_logporbs = dist.log_prob(action)
        dist_entropy = dist.entropy()

        # 获得state的value
        state_value = self.critic_layer(state)

        return action_logporbs, torch.squeeze(state_value), dist_entropy

