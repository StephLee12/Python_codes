import gym
import argparse
import tensorflow.compat.v1 as tf
import numpy as np

from model import DQN

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

parser = argparse.ArgumentParser() 
parser.add_argument('-d', '--dueling', action='store_true')
parser.add_argument('-D', '--double_q', action='store_true')
parser.add_argument('-r', '--render_show', action='store_true')
parser.add_argument('-l', '--log_tensorboard', action='store_true')
parser.add_argument('-t', '--do_train', action='store_true')
args = parser.parse_args()

env = gym.make('CartPole-v0')
env = env.unwrapped
env.seed(1)

action_space = env.action_space.n # 行为数量
obs_sapce = 4 # 车的水平位置（0 为中心）、其速度、杆的角度（0 维垂直）及其角速度
render_show = args.render_show	# 是否展示画面
mean_size = 10 # 最近多少盘总reward平均值
mean_counter = 0 # 每次添加到mean array都会子增加，用于算位置
target_reward = 200 # 希望多少分的时候刷新页面并保存模型
mean_array = np.zeros([mean_size]) # reward平均值记录数组
do_save = False # 是否保存模型
save_path = 'cart-model/model.ckpt' # 模型保存路径
total_episode = 1000 # 一共玩多少
log_tensorboard = args.log_tensorboard # 是否将神经网络输出到tensorboard log中
tensorboard_log_path = 'cart-logs/' # tensorboard log 保存目录
do_train = args.do_train # 训练模式 还是 读取模型 运行模式

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
epilist =[]
meanrdlist=[]
rewardlist =[]
with tf.Session(config=config) as sess:

	dqn = DQN(s=tf.placeholder(tf.float32, [None, obs_sapce]), # 当前步的环境
				a=tf.placeholder(tf.int32, [None]), # 行为
				y=tf.placeholder(tf.float32, [None, action_space]), # 计算出的y值，用于计算loss
				s_=tf.placeholder(tf.float32, [None, obs_sapce]), # 下一步的环境
				action_space=action_space, # 行为数量
				obs_sapce=obs_sapce, # 环境数量
				layers_units=[256, 256], # 每一层隐藏层的units
				sess=sess,
				lr=1e-4, # 学习率
				gamma=0.99,
				memory_size=1000, # 记忆数量
				replace_epsioder=100, # 多少次更新一次q目标
				batch_size=64, # 每次训练多少条数据
				predict_rate=0.9, # 神经网络选择行为的几率，0.9代表90%
				dueling=args.dueling, # 是否使用dueling dqn
				double_q=args.double_q) # 是否使用double dqn
	# 如果保存tensorboard log，则输出
	if log_tensorboard:
		merged = tf.summary.merge_all()
		writer = tf.summary.FileWriter(tensorboard_log_path, sess.graph)
	# 如果是训练模式则初始化数据
	if do_train:
		sess.run(tf.global_variables_initializer())
	else:
		# 否则为运行模式，直接读取model
		dqn.restore(save_path)
	losses = 0
	for episode in range(total_episode):
		s = env.reset()
		reward_counter = 0 # 计算一局中 总共获得的reward
		last_frame_counter = 0 # 当局的帧数
		while True:
			# 如果显示游戏画面，则刷新画面
			if render_show:
				env.render()
			# 通过神经网络获得行为
			a = dqn.get_action(s)
			# 行为结束 获得环境返回值[下一步的obs，reward，是否结束，info]
			s_, r, done, _ = env.step(a)
			# 处理一下reward，obs的4位分别为车的水平位置（0 为中心）、其速度、杆的角度（0 维垂直）及其角速度
			# 保证obs的四位都为0，可以更久的将杆子保持直立
			r = -1 if done else 1 - abs(s_[0]) - abs(s_[1]) - abs(s_[2]) - abs(s_[3])
			# 增加当局的reward
			reward_counter += r
			if do_train:
				# 保存memory
				dqn.store(s, a, r, done, s_)
				# train network
				losses = dqn.train_net()
			s = s_
			# 帧数自增
			last_frame_counter += 1
			if last_frame_counter >= 200:
				done = True
			# 如果达到保存条件，则保存模型
			if do_save and do_train:
				dqn.save(save_path)
				do_save = False
			# 如果结束，打印当局数据，判断是否保存模型
			if done:
				mean_array[mean_counter % mean_size] = reward_counter
				mean_counter += 1
				# 如果mean满数据，并且达到目标，则保存模型，并且开启保存模型
				if mean_counter > mean_size and np.mean(mean_array) > target_reward and do_train:
					render_show = True
					do_save = True
				# 打印当局的数据
				epilist.append(episode)
				meanrdlist.append(np.mean(mean_array))
				rewardlist.append(reward_counter)
				print('episode: {0}, loss: {1}, reward_mean: {2}, last_frame: {3}, episode_reward: {4}'.format(episode, losses, np.mean(mean_array), last_frame_counter, reward_counter))
				break

datalist=[]
for i in range(len(epilist)):
	datalist.append(epilist[i])
	datalist.append(rewardlist[i])
	datalist.append(meanrdlist[i])
text_save('cartlog.txt',datalist,3)