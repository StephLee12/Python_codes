import gym 

if __name__ == "__main__":
    env = gym.make('LunarLander-v2')
    observation = env.reset()

    store_obs = []
    store_rew = []
    store_done = []
    store_info = []

    for _ in range(1000):
        env.render()
        #env.step(env.action_space.sample())
        action = env.action_space.sample() # take random action
        observation,reward,done,info = env.step(action)

        store_obs.append(observation)
        store_rew.append(reward)
        store_done.append(done)
        store_info.append(info)


        if done:
            observation = env.reset()
    
    env.close()

    print(store_obs)