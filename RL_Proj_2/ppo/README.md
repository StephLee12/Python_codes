# README

👉 运行``main.py``启动训练 可以选择``CartPole-v0``环境或``LunarLander-v2``环境

👉 训练所用的参数均在``utils.py``中

👉 采用PPO算法进行训练，可以改变``utils.py``的``Args``类中的``max_epochs``属性，来改变训练的回合数

👉 ``cartpole_gym.py``和``lunarlander_gym.py``是对gym环境进行测试的脚本，对implement算法无关

👉 requirements

- ``torch``
- ``gym``
- ``matplotlib``
- ``numpy``