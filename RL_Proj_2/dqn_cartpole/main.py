import gym
from IPython import display
import matplotlib.pyplot as plt

from net import Net
from agent import Agent

def main():
    # 创建环境
    env = gym.make('CartPole-v0')
    # 参数
    params = {
        'gamma': 0.8, 
        'epsi_high': 0.9,
        'epsi_low': 0.05,
        'decay': 200, # epsilon衰减
        'lr': 0.001, #学习率
        'capacity': 10000, #replay buffer容量
        'batch_size': 64, #每次抽取的batch的大小
        'state_space_dim': env.observation_space.shape[0], 
        'action_space_dim': env.action_space.n   
    }
    # 创建智能体
    agent = Agent(**params)

    score = []
    mean = []

    # 训练100个回合
    for episode in range(1000):
        s0 = env.reset()
        total_reward = 1
        while True:
            env.render()
            a0 = agent.act(s0) # take action
            s1, r1, done, _ = env.step(a0)
            
            if done: #判断回合是否结束
                r1 = -1
            
            # 将序列存入replay buffer
            agent.put(s0, a0, r1, s1)
            
            if done:
                break

            total_reward += r1
            s0 = s1
            agent.learn() # 如果满足learn的条件 学习Q值网络
            
        score.append(total_reward)
        mean.append( sum(score[-100:])/100)
        
        plot(score, mean)

def plot(score, mean):
    
    display.clear_output(wait=True)
    display.display(plt.gcf())
    plt.figure(figsize=(20,10))
    plt.clf()

    plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(score)
    plt.plot(mean)
    plt.text(len(score)-1, score[-1], str(score[-1]))
    plt.text(len(mean)-1, mean[-1], str(mean[-1]))


if __name__ == '__main__':

    main()