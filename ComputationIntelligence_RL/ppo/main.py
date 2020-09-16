import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np 
import gym

from torch.distributions import Categorical

from memory import Memory
from agent import Agent
from ac import ActorCritic
from utils import Args

# 使用gpu
device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

# "CartPole-v0"'s observation return a 1-4 array
# the first elem is the position of the car
# the second elem is the angle of the pole with the vertical
# the third elem is the car's velocity
# the last elem is the rate change of angle
# "CartPole-v0"'s action has two option 0 for go left and 1 for go right

# "LunarLander-v2"'s observation returns a 1-8 array
# the first two comprises the coordinates
# "LunarLander-v2"'s action has four option
# 0 for do nothing
# 1 for fire left engine
# 2 for fire main engine
# 3 for fire right engine
def main(env_name):
    
    # 获取所有参数
    args = Args(env_name)
    env = args.env
    max_epochs = args.max_epochs
    max_timesteps = args.max_timesteps
    update_timestep = args.update_timestep
    print_interval = args.print_interval
    
    # 初始化memory
    memory = Memory()

    # 创建agent实例
    agent = Agent(
        input_size=args.input_size,
        output_size=args.output_size,
        hidden_size=args.hidden_size,
        lr=args.lr,
        beta=args.beta,
        gamma=args.gamma,
        update_epoch=args.update_epoch,
        epsilon=args.epsilon
    )

    reward_plot = [0] #记录每print_interval个epoch的平均reward 画图用
    timestep_count = 0 #记录步长 到update_timestep清零
    interval_reward = 0 #记录每print_interval个epoch的平均reward 后清零
    interval_timestep= 0 #记录每print_interval个epoch的平均步长 后清零

    file_name = 'RL_Proj_2/{}.txt'.format(args.env_name)

    # training loop
    for epoch in range(1,max_epochs+1):
        state = env.reset() #与env交互随机获取一个state
        # agent做出action
        for timestep in range(max_timesteps):
            timestep_count += 1

            # old policy sampling 做出action 与环境交互
            action = agent.old_policy.act(state,memory)
            state,reward,done,_ = env.step(action)
            memory.rewards.append(reward)
            memory.is_done.append(done)

            # 判断是否需要更新 policy
            if timestep_count % update_timestep == 0:
                agent.update(memory)
                memory.clear_memory()
                timestep_count = 0
            
            interval_reward += reward
            env.render()
            if done:
                break
        
        interval_timestep+= timestep

        # 每print_interval打印一次数据
        if epoch % print_interval == 0:
            interval_timestep= np.divide(interval_timestep,print_interval)
            interval_reward = np.divide(interval_reward,print_interval)

            reward_plot.append(interval_reward)

            # 储存数据
            with open(file_name,'a') as f :
                f.write(str(epoch)+' '+str(interval_timestep)+' '+str(interval_reward)+'\n')

            print('Epoch {} \t average timestep: {} \t reward: {}'.format(epoch,interval_timestep,interval_reward))

            interval_reward = 0
            interval_timestep= 0

    # 训练结束后 存储模型
    torch.save(agent.policy.state_dict(),'RL_Proj_2/{}.pth'.format(args.env_name))

    #画图
    plt.plot(reward_plot)
    plt.xlabel('Epoch = tick times {}'.format(print_interval))
    plt.ylabel('Reward')
    plt.savefig('RL_Proj_2/{}.png'.format(args.env_name))
    plt.show()
    


if __name__ == "__main__":
    env_name = [
        'CartPole-v0',
        'LunarLander-v2'
    ]
    main(env_name[0])
    