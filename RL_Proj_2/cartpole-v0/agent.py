import numpy as np

import torch
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras.optimizers import RMSprop
from keras import backend as K
from keras.utils import to_categorical
import torch.nn.functional as F

from actor import Actor
from critic import Critic


class Agent:
    def __init__(self, input_size, output_size, lr, gamma, epilson, ent):
        self.input_size = input_size
        self.output_size = output_size
        self.actions = range(output_size)  # action list
        self.lr = lr
        self.gamma = gamma
        self.epilson = epilson  # for actor loss function
        self.ent = ent  # for actor loss function
        self.num_epochs = 10
        self.batchsize = 10

        # store data
        self.states = []
        self.actions = []
        self.pi_vecs = []  # pi(a_i|s_i)
        self.rewards = []

        # build actor and critic
        self.actor = Actor(input_size,
                           output_size,
                           lr,
                           gamma,
                           epilson,
                           ent,
                           num_layers=3,
                           hidden_size=20)
        self.critic = Critic(input_size,
                             output_size,
                             lr,
                             epilson=0.2,
                             ent=1e-3,
                             num_layers=3,
                             hidden_size=20)

    def get_batch(self):

        # get a batch
        state_arr = np.array(self.states)
        actions_arr = np.array(self.actions)
        pi_vecs_arr = np.array(self.pi_vecs)
        rewards_arr = np.array(self.rewards)

        return state_arr, actions_arr, pi_vecs_arr, rewards_arr

    def store_data(self, state, action, pi_vec, reward):
        self.states.append(state)

        action_onehot = to_categorical(action, self.output_size)
        self.actions.append(action_onehot)

        self.pi_vecs.append(pi_vec)
        self.rewards.append(reward)

    def clear_memory(self):
        # clear memory when get a batch
        self.states, self.actions, self.pi_vecs, self.rewards = [], [], [], []

    def get_state_value(self, rewards):
        # calculate state's value
        state_value = np.zeros_like(rewards)

        sum = 0
        for t in reversed(range(len(state_value))):
            # t时刻之后的reward带有衰减因子连乘 加上t时刻的reward为t时的value
            sum = sum * self.gamma + rewards[t]
            state_value[t] = sum

        # nomarlize
        state_value -= np.mean(state_value)
        state_value /= np.std(state_value)

        return np.array(state_value)

    def train_agent(self):

        # get batch
        states, actions, pi_vecs, rewards = self.get_batch()
        self.clear_memory()  # when get a batch clear the memory

        # calculate advantage
        states_value = self.get_state_value(rewards)
        baseline = self.critic.model.predict(states)
        baseline.resize(len(states_value))
        advantages = states_value - baseline

        # train
        old_preds = pi_vecs
        actor_loss = self.actor.model.fit([states, advantages, old_preds],
                                          [actions],
                                          batch_size=self.batchsize,
                                          shuffle=True,
                                          epochs=self.num_epochs,
                                          verbose=False)
        critic_loss = self.critic.model.fit([states], [states_value],
                                            batch_size=self.batchsize,
                                            shuffle=True,
                                            epochs=self.num_epochs,
                                            verbose=False)

        return actor_loss, critic_loss

    def act(self, states):

        # choose an action

        # only need state to choose an action
        # advantage and old_pi are used to loss function
        num_samples = states.shape[0]
        advantage_dummy = np.zeros((num_samples, 1))
        pi_old_dummy = np.zeros((num_samples, self.output_size))
        vec = [states, advantage_dummy, pi_old_dummy]

        # output is an distribution of action
        pi_vec = self.actor.model.predict(vec)[0]
        # p = np.nan_to_num(pi_vec)
        # p = torch.tensor(p)
        # p = F.softmax(p, dim=0)
        # p = p.numpy()

        action = np.random.choice(range(self.output_size), p=pi_vec)

        return action, pi_vec


if __name__ == "__main__":
    agent_obj = Agent(input_size=4,
                      output_size=2,
                      lr=0.001,
                      gamma=0.99,
                      epilson=0.2,
                      ent=1e-3)
