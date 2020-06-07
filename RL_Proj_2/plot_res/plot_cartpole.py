import pandas as pd 
import matplotlib.pyplot as plt 

def pre_ppo_cartpole():
    filename = 'RL_Proj_2/plot_res/ppo_cartpole_reward.txt'

    df = pd.read_table(
        'RL_Proj_2/plot_res/ppo_cartpole.txt',
        sep=' '
    )

    df = df.iloc[:,[0,2]]

    rewards = []
    for row in range(df.shape[0]):
        with open(filename,'a') as f:
            rewards.append(df.iloc[row,1])
            f.write('{}\n'.format(df.iloc[row,1]))
    
    return rewards

def pre_ddqn_cartpole():
    filename = 'RL_Proj_2/plot_res/ddqn_cartpole_reward.txt'

    df = pd.read_table('RL_Proj_2/plot_res/ddqn_cartpole.txt')
    
    df = df.iloc[:,[0,1]]

    rewards = []
    for row in range(df.shape[0]):
        with open(filename,'a') as f:
            if (row + 1) % 20 == 0:
                mean = df.iloc[row-19:row,1].mean()
                rewards.append(mean)
                f.write('{}\n'.format(mean))
    
    return rewards

def pre_dqn_cartpole():
    filename = 'RL_Proj_2/plot_res/dqn_cartpole_reward.txt'

    df = pd.read_table(
        'RL_Proj_2/plot_res/dqn_cartpole.txt',
        sep=' ',
        header=None
    )

    rewards = []
    for row in range(df.shape[0]):
        with open(filename,'a') as f:
            if (row + 1) % 20 == 0:
                mean = df.iloc[row-19:row,0].mean()
                rewards.append(mean)
                f.write('{}\n'.format(mean))
    
    return rewards

def plot_cartpole():
    
    ppo_reward = pre_ppo_cartpole()
    dqn_reward = pre_dqn_cartpole()
    ddqn_reward = pre_ddqn_cartpole()
    

    plt.figure()
    plt.title('Implementing 3 algorithms on CartPole-v0')
    plt.plot(ppo_reward,label='PPO',color='r')
    plt.plot(dqn_reward,label='DQN',color='g')
    plt.plot(ddqn_reward,label='DDQN',color='b')
    plt.xlabel('Epoch = tick times 20')
    plt.ylabel('Rewards')
    plt.legend()
    plt.savefig('RL_Proj_2/plot_res/cartpole-v0.png')
    plt.show()

if __name__ == "__main__":
    plot_cartpole()    
