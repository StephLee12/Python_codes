import gym
import torch


class Args:
    def __init__(self, env_name):
        self.env_name = env_name
        self.env = gym.make(env_name)
        self.input_size, self.output_size = self.set_env_params()
        self.gamma = 0.99
        self.epilson = 0.2
        self.ent = 1e-3
        self.lr = 1e-3
        self.epochs = 2000

    def set_env_params(self):
        return self.env.observation_space.shape[0], self.env.action_space.n

    def get_env(self):
        return self.env

    def get_all_params_dict(self):
        return {
            'input_size': self.input_size,
            'output_size': self.output_size,
            'gamma': self.gamma,
            'epilson': self.epilson,
            'ent': self.ent,
            'lr': self.lr,
            'epochs': self.epochs
        }


# if __name__ == "__main__":
#     args = Args('CartPole-v0')
#     params_dict = args.get_all_params_dict()
#     print(params_dict.get('output_size'))
