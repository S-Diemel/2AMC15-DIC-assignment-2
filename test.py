from world import Environment
import time

# Test script to show how curriculum learning works and to test the environment

for i in range(4):

    if i == 0:
        difficulty = None
    if i == 1:
        difficulty = 0
    if i == 2:
        difficulty = 1
    if i == 3:
        difficulty = 2

    env = Environment()
    obs = env.reset(no_gui=False, seed=0, difficulty=difficulty)
    done = False

    for _ in range(50):  # Note very little steps for illustration purposes, because random agent
        
        env.render()
        action = env.action_space.sample()
        obs, reward, done = env.step(action)
        print(obs, reward, done)
        #time.sleep(3)
        if done:
            break