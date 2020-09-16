"""
Deep Q network,

LunarLander-v2 example

Using:
Tensorflow: 1.0
gym: 0.8.0
"""


import gym
import  matplotlib.pyplot as plt
from DuelingDQNPrioritizedReplay import DuelingDQNPrioritizedReplay
import os

def text_save(filename, data,linelen):
    file = open(filename,'a')
    file.seek(0)
    file.truncate()  # 清空文件
    count = 0;
    for i in range(len(data)):
        s = str(data[i]).replace('[','').replace(']','')#去除[],这两行按数据不同，可以选择
        if count == linelen-1:
            s = s.replace("'", '').replace(',', '') + '\n'  #最后一个换行
            count=0
            file.write(s)
        else :
            s = s.replace("'",'').replace(',','') +'\t'   #去除单引号，逗号，每个加tab
            file.write(s)
            count=count+1
    file.close()


env = gym.make('LunarLander-v2')
# env = env.unwrapped
env.seed(1)

N_A = env.action_space.n
N_S = env.observation_space.shape[0]
MEMORY_CAPACITY = 50000
TARGET_REP_ITER = 2000
MAX_EPISODES = 3500
E_GREEDY = 0.95
E_INCREMENT = 0.00001
GAMMA = 0.99
LR = 0.0001
BATCH_SIZE = 32
HIDDEN = [400, 400]
RENDER = True

RL = DuelingDQNPrioritizedReplay(
    n_actions=N_A, n_features=N_S, learning_rate=LR, e_greedy=E_GREEDY, reward_decay=GAMMA,
    hidden=HIDDEN, batch_size=BATCH_SIZE, replace_target_iter=TARGET_REP_ITER,
    memory_size=MEMORY_CAPACITY, e_greedy_increment=E_INCREMENT,)


total_steps = 0
running_r = 0
r_scale = 1
epilist = []
landedlist=[]
rewardlist=[]
valuelist=[]
episionlist =[]
for i_episode in range(MAX_EPISODES):
    s = env.reset()  # (coord_x, coord_y, vel_x, vel_y, angle, angular_vel, l_leg_on_ground, r_leg_on_ground)
    ep_r = 0
    while True:
        if total_steps > MEMORY_CAPACITY: env.render()
        a = RL.choose_action(s)
        s_, r, done, _ = env.step(a)
        if r == -100:
            r = -30

        r /= r_scale
        ep_r += r
        RL.store_transition(s, a, r, s_)
        if total_steps > MEMORY_CAPACITY:
            RL.learn()
        if done:
            land = '| Landed' if r == 100/r_scale else '| ------'
            running_r = 0.99 * running_r + 0.01 * ep_r
            print('Epi: ', i_episode,
                  land,
                  '| Epi_R: ', round(ep_r, 2),
                  '| Running_R: ', round(running_r, 2),
                  '| Epsilon: ', round(RL.epsilon, 3))
            epilist.append(i_episode)
            if land == '| Landed' :
                landedlist.append(1)
            else :
                landedlist.append(0)
            rewardlist.append(round(ep_r, 2))
            valuelist.append(round(running_r, 2))
            episionlist.append(round(RL.epsilon, 3))
            break

        s = s_
        total_steps += 1
datalist = []
for i in range(len(epilist)):
    datalist.append(epilist[i])
    datalist.append(landedlist[i])
    datalist.append(rewardlist[i])
    datalist.append(valuelist[i])
    datalist.append(episionlist[i])
plt.plot(epilist,rewardlist ,lw = 1.5,color = 'r',label = 'reward')
plt.plot(epilist,valuelist ,lw = 1.5,color = 'b',label = 'value')
plt.xlabel('epi')
plt.ylabel('reward and value')
plt.show()
text_save('log.txt',datalist,5)
os.system("shutdown -s -t 0")
