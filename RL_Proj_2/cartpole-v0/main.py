import numpy as np
import matplotlib.pyplot as plt
from collections import deque

import gym

import torch
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
import keras.optimizers
from keras import backend as K

from agent import Agent
from utils import Args


class RL:
    def __init__(self, env_name):
        self.env = Args(env_name).get_env()  # RL's environment
        self.args = Args(
            env_name).get_all_params_dict()  # load hyper-parameters
        self.agent = self.build_agent()  # build agent
        self.rewards = []  # store rewards

    def build_agent(self):
        return Agent(input_size=self.args.get('input_size'),
                     output_size=self.args.get('output_size'),
                     lr=self.args.get('lr'),
                     gamma=self.args.get('gamma'),
                     epilson=self.args.get('epilson'),
                     ent=self.args.get('ent'))

    # main train loop
    def train(self):
        for epoch in range(1, self.args.get('epochs') + 1):
            state = self.env.reset()  # initialization
            state = np.reshape(state,
                               [1, self.args.get('input_size')])  # reshape

            total_reward = 0  # intialize total reward as zero
            done = False  # initialize done as False

            while not done:
                self.env.render()  #show env window

                action, pi_vec = self.agent.act(state)  # take an action
                next_state, reward, done, info = self.env.step(
                    action)  # env takes one step

                reward = reward if not done else -10  # if done reward minus 10
                total_reward += reward

                next_state = np.reshape(
                    next_state,
                    [1, self.args.get('input_size')])  # reshape next state

                self.agent.store_data(state[0], action, pi_vec,
                                      reward)  # store every step data
                state = next_state  # update state

            # after one epoch is done use these data to train the model
            self.agent.train_agent()
            # record every epoch's reward
            self.rewards.append(total_reward)
            if epoch % 25 == 0:
                print('(epoch,reward)=' + str((epoch, total_reward)))

        # plot figure
        plt.plot(self.rewards)
        plt.savefig('./rewards1.png')
        plt.show()


if __name__ == "__main__":
    env_name = 'CartPole-v0'  # specify env name
    # "CartPole-v0"'s observation return a 1-4 array
    # the first elem is the position of the car
    # the second elem is the angle of the pole with the vertical
    # the third elem is the car's velocity
    # the last elem is the rate change of angle
    # "CartPole-v0"'s action has two option 0 for go left and 1 for go right
    rl_obj = RL(env_name)  # create a RL instance
    rl_obj.train()  # train
