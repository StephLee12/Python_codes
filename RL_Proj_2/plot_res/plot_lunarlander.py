import pandas as pd
import matplotlib.pyplot as plt

def pre_ppo_lunarlander():

    file_name = 'RL_Proj_2/plot_res/ppo_lunarlander_reward.txt'

    df = pd.read_table(
        'RL_Proj_2/plot_res/ppo_lunarlander.txt',
        header=None,
        sep=' '
    )

    df = df.iloc[:,[0,2]]

    rewards = []
    for row in range(df.shape[0]):
        with open(file_name,'a') as f:                     
            rewards.append(df.iloc[row,1])
            f.write('{}\n'.format(df.iloc[row,1]))
    
    return rewards
    
def pre_dqn_lunarlander():
    
    file_name = 'RL_Proj_2/plot_res/dqn_lunarlander_reward.txt'

    df = pd.read_table(
        'RL_Proj_2/plot_res/dqn_lunarlander.txt',
        header=None,
        sep=' '
    )

    rewards = []
    for row in range(df.shape[0]):
        with open(file_name,'a') as f:
            rewards.append(df.iloc[row,1])
            f.write('{}\n'.format(df.iloc[row,1]))
    
    return rewards

def pre_ddqn_lunarlander():
    file_name = 'RL_Proj_2/plot_res/ddqn_lunarlander_reward.txt'

    df = pd.read_table(
        'RL_Proj_2/plot_res/ddqn_lunarlander.txt',
        header=None,
        sep = ' '
    )

    df = df.iloc[:,[0,2]]
    rewards = []
    for row in range(df.shape[0]):
        with open(file_name,'a') as f:
            if (row + 1) % 20 == 0:
                mean = df.iloc[row-19:row,1].mean()
                rewards.append(mean)
                f.write('{}\n'.format(mean))
    
    return rewards


def plot_lunarlander():
    ppo_reward = pre_ppo_lunarlander()
    dqn_reward = pre_dqn_lunarlander()
    ddqn_reward = pre_ddqn_lunarlander()

    plt.figure()
    plt.title('Implementing 3 algorithms on LunarLander-v2')
    plt.plot(ppo_reward,label='PPO',color='r')
    plt.plot(dqn_reward,label='DQN',color='g')
    plt.plot(ddqn_reward,label='DDQN',color='b')
    plt.xlabel('Epoch = tick times 20')
    plt.ylabel('Rewards')
    plt.legend()
    plt.savefig('RL_Proj_2/plot_res/lunarlander-v2.png')
    plt.show()

if __name__ == "__main__":
    plot_lunarlander()