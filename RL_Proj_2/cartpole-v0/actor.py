import numpy as np
import matplotlib.pyplot as plt

import gym
from collections import deque

import torch
import tensorflow as tf
from keras.models import Sequential, Input, Model
from keras.layers import Dense
from keras.optimizers import Adam 
from keras.optimizers import RMSprop
from keras import backend as K
from keras.utils import to_categorical


class Actor:

    def __init__(self,input_size,output_size,lr,gamma,epilson,ent,num_layers=3,hidden_size=20):
        self.input_size = input_size # 输入tensor的维数
        self.output_size = output_size # 输出tensor的维数
        self.lr = lr # learning rate for gradient calculation
        self.gamma = gamma # discount
        self.epilson = epilson
        self.ent = ent # constant for entropy loss
        self.num_layers = num_layers # hidden layers number
        self.hidden_size = hidden_size  # each hidden layer units number
        self.model = self.build_network() # build network

    # calculatr J^ppo use clip instead of KL divergence
    def ppo_loss(self,advantage,old_pred):
        # input params
        # advantage : advantage function
        # old_pred : network's (using for sampling) predictions

        def loss(y_true,y_pred):

            # loss of cpi
            prob = y_true * y_pred # need to update
            old_prob = y_true * old_pred # sample network

            tmp = prob / (old_prob + 1e-10)

            elem1 = tmp * advantage # PPO min中的第一项
            # PPO min中的第二项
            elem2 = advantage * K.clip(tmp,min_value=1-self.epilson,max_value=1+self.epilson)

            loss_cpi = K.minimum(elem1,elem2)

            # loss of entropy
            loss_entropy = self.ent * (prob * K.log(prob+1e-10))

            return -K.mean(loss_cpi + loss_entropy)
        
        return loss
    
    # use keras 
    def build_network(self):

        # network input
        state = Input(shape=(self.input_size,))
        advantage = Input(shape=(1,))
        old_pred = Input(shape=(self.output_size,))

        # network structure
        x = Dense(self.hidden_size,activation='tanh')(state) # fc1
        for _ in range(self.num_layers - 1):
            x = Dense(self.hidden_size,activation='tanh')(x)
        
        # use softmax to output action's distribution
        output = Dense(self.output_size,activation='softmax',name='output')(x)

        # build the network
        model = Model(inputs=[state,advantage,old_pred],outputs=[output])
        model.compile(optimizer=Adam(learning_rate=self.lr),
                        loss=[self.ppo_loss(advantage=advantage,old_pred=old_pred)])
        
        return model


            
            