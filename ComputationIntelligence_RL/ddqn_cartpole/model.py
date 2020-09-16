import tensorflow.compat.v1 as tf
import numpy as np

# 随机因素固定
np.random.seed(1)
tf.set_random_seed(1)

class DQN():
	
	def __init__(
		self,
		s,
		a,
		y,
		s_,
		action_space,
		obs_sapce,
		layers_units,
		sess,
		lr=1e-4,
		gamma=0.99,
		memory_size=4000,
		memory_couter=0,
		replace_epsioder=500,
		train_counter=0,
		batch_size=64,
		predict_rate=0.9,
		dueling=False,
		double_q=False):
		# 输入占位
		self.s = s
		self.a = a
		self.y = y
		self.s_ = s_
		self.sess = sess # tensorflow session
		self.lr = lr # 学习率
		self.gamma = gamma
		self.obs_sapce = obs_sapce # obs 形状或者数量
		self.action_space = action_space # 行为的数量
		self.predict_rate = predict_rate # 神经网络的动作选择率
		self.memory = np.zeros([memory_size, obs_sapce * 2 + 3]) # 初始化记忆库
		self.memory_size = memory_size # 记忆库大小
		self.memory_couter = memory_couter # 记忆了多少次
		self.replace_epsioder = replace_epsioder # 多少次
		self.train_counter = train_counter # 训练了多少次
		self.batch_size = batch_size # 每次训练 取多少数据
		self.layers_units = layers_units # 隐藏层的units
		self.dueling = dueling # 是否使用dueling dqn
		self.double_q = double_q # 是否使用double dqn
		self.network() # 初始化神经网络

	def _build_network(self, input, outputSize=[], name='', trainable=True):
		# 单个神经网络搭建方法
		with tf.variable_scope(name) as scope:
			hidden_layer = input
			for index in range(len(outputSize)):
				hidden_layer = tf.layers.dense(inputs=hidden_layer, units=outputSize[index], activation=tf.nn.tanh if index == len(outputSize) - 1 else tf.nn.relu, kernel_initializer=tf.random_uniform_initializer(0, 0.1), name="hidden_layer_" + str(index), trainable=trainable)
			# dueling dqn 和 普通的dqn output 是不一样的
			if self.dueling:
				A = tf.layers.dense(inputs=hidden_layer, units=self.action_space, kernel_initializer=tf.random_uniform_initializer(0, 0.1), name='hidden_layer_A_' + str(len(outputSize)), trainable=trainable)
				V = tf.layers.dense(inputs=hidden_layer, units=1, kernel_initializer=tf.random_uniform_initializer(0, 0.1), name='hidden_layer_V_' + str(len(outputSize)), trainable=trainable)
				out = (A - tf.reduce_mean(A, axis=1, keep_dims=True)) + V
			else :
				out = tf.layers.dense(inputs=hidden_layer, units=self.action_space, kernel_initializer=tf.random_uniform_initializer(0, 0.1), name='hidden_layer_out_' + str(len(outputSize)), trainable=trainable)
			return out

	def network(self):
		# 搭建神经网络
		self.q_eval = self._build_network(self.s, outputSize=self.layers_units, name='eval_q')
		self.q_target = self._build_network(self.s_, outputSize=self.layers_units, name='target_q', trainable=False)
		self.e_params, self.t_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='eval_q'), tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='target_q')
		self.replace_params = [tf.assign(t, e) for t, e in zip(self.t_params, self.e_params)]
		# 定义损失函数
		with tf.variable_scope('loss') as scope:
			self.loss = tf.reduce_mean(tf.square(self.y - self.q_eval))
		# 定义优化器
		with tf.variable_scope('train')	as scope:
			self.train = tf.train.AdamOptimizer(self.lr).minimize(self.loss, var_list=self.e_params)

	def get_action(self, s):
		# 随机一个数，根据predict_rate的比率判断是否使用神经网络预测的行为，否则随机一个行为
		if np.random.rand() < self.predict_rate:
			return np.argmax(self.sess.run(self.q_eval, feed_dict={self.s: [s]}))
		else:
			return np.random.randint(self.action_space)

	def train_net(self):
		# 训练方法，当记忆库的数据大于每次训练批量取数据的量，则开始训练
		if self.memory_couter > self.batch_size:
			# 如果第一次训练，则先使两条神经网络的参数相同，之后按replace_epsioder次 使参数传递
			if (self.train_counter >= self.replace_epsioder and self.train_counter % self.replace_epsioder == 0) or self.train_counter == 0:
				self.sess.run(self.replace_params)
			# 随机获得记忆库中的数据所在的坐标
			sample_index = np.random.choice(min(self.memory_size, self.memory_couter), self.batch_size, replace=False)
			# 取出记忆数据
			sample_s = self.memory[sample_index, :self.obs_sapce]
			sample_s_ = self.memory[sample_index, -self.obs_sapce:]
			sample_a = self.memory[sample_index, self.obs_sapce].astype(int)
			sample_r = self.memory[sample_index, self.obs_sapce + 1]
			# 神经网络计算 q现实和q目标
			q_eval, q_target = self.sess.run([self.q_eval, self.q_target], feed_dict={self.s: sample_s, self.s_: sample_s_})
			batch_size_arange = np.arange(self.batch_size, dtype=np.int32)
			# 使用double dqn和普通dqn不同
			if self.double_q:
				q_target4eval = self.sess.run(self.q_eval, feed_dict={self.s: sample_s_})
				all_train_data = q_target[batch_size_arange, np.argmax(q_target4eval, axis=1)]
			else:
				all_train_data = np.max(q_target, axis=1)
			# 算出 y值
			q_eval[batch_size_arange, sample_a] = sample_r + self.gamma * all_train_data
			# 更新神经网络
			loss, _ = self.sess.run([self.loss, self.train], feed_dict={self.s: sample_s, self.a: sample_a, self.y: q_eval, self.s_: sample_s_})
			self.train_counter += 1
			return loss

	def store(self, s, a, r, t, s_):
		# 保存记忆的方法
		index = self.memory_couter % self.memory_size
		self.memory[index, :] = np.hstack((s, [a], [r], [t], s_))
		self.memory_couter += 1

	def save(self, save_path):
		# 保存神经网络模型参数的方法
		tf.train.Saver().save(self.sess, save_path)

	def restore(self, save_path):
		# 读取神经网络模型参数的方法
		tf.train.Saver().restore(self.sess, save_path)