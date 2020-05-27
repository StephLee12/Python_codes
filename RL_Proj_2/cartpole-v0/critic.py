import numpy as np
from collections import deque

import tensorflow as tf
from keras.models import Sequential, Input, Model
from keras.layers import Dense
from keras.optimizers import Adam 
from keras.optimizers import RMSprop
from keras import backend as K
from keras.utils import to_categorical

class Critic:

    def __init__(self,input_size,output_size,lr,epilson=0.2,ent=1e-3,num_layers=3,hidden_size=20):
        self.input_size = input_size
        self.output_size = output_size
        self.lr = lr
        self.epilson = epilson # actually don't need this
        self.ent = ent # this too
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.model = self.build_network()
    
    
    def build_network(self):
        
        # network input
        state = Input(shape=(self.input_size,))

        # forward
        x = Dense(self.hidden_size,activation='tanh')(state)
        for _ in range(self.num_layers -1):
            x = Dense(self.hidden_size,activation='tanh')(x)
        
        output = Dense(1)(x)

        # build network
        model = Model(inputs=[state],outputs=[output])
        model.compile(optimizer=Adam(learning_rate=self.lr),loss='mse')

        return model

# if __name__ == "__main__":
#     critic_obj = Critic(
#         input_size=4,
#         output_size=2,
#         lr=0.001,
#         epilson=0.2,
#         ent=1e-3,
#         num_layers=3,
#         hidden_size=20
#     )