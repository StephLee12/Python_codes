import torch
import torch.nn as nn
import gym

from torch.distributions import Categorical


class Args:
    def __init__(self, env_name):
        self.env_name = env_name #要加载的环境名
        self.env = gym.make(env_name) # 加载环境

        self.input_size, self.output_size = self.set_env_params()
        self.hidden_size = 64 #每个隐层有64个units
        
        self.beta = (0.9,0.999) # optimizer
        self.gamma = 0.99 #衰减因子
        self.epsilon = 0.2 # 用于计算J_ppo
        self.lr = 0.002 #学习率
        self.max_epochs = 1500 #最多训练回合数
        self.max_timesteps = 300 #在一个回合中最大步数
        self.update_timestep = 2000 # 每2000个步长更新一次policy
        self.update_epoch = 4 # update policy 的回合数
        self.print_interval = 20 #每20个interval打印一次

    def set_env_params(self):
        return self.env.observation_space.shape[0], self.env.action_space.n

    def get_env(self):
        return self.env

    def get_all_params_dict(self):
        return {
            'input_size': self.input_size,
            'output_size': self.output_size,
            'hidden_size':self.hidden_size,
            'beta':self.beta,
            'gamma': self.gamma,
            'epsilon': self.epsilon,
            'lr': self.lr,
            'max_epochs': self.max_epochs,
            'max_timesteps':self.max_timesteps,
            'update_timestep':self.update_timestep,
            'update_epoch':self.update_epoch,
            'print_interval':self.print_interval
        }


# if __name__ == "__main__":
#     args = Args('CartPole-v0')
#     params_dict = args.get_all_params_dict()
#     print(params_dict.get('output_size'))
