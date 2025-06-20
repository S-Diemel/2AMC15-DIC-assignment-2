from world import Environment
import time

# Test script to show how curriculum learning works and to test the environment

for i in range(100):

    if i == 0:
        difficulty = None
    if i == 1:
        difficulty = 0
    if i == 2:
        difficulty = 1
    if i == 3:
        difficulty = 2

    env = Environment(sigma=0.5, step_size=0.5)
    obs = env.reset(no_gui=False, seed=0, difficulty=difficulty) # , extra_obstacles=[(1,1,2,2)])
    done = False

    for _ in range(100):  # Note very little steps for illustration purposes, because random agent
        
        env.render()
        action = env.action_space.sample()
        print(action)
        obs, reward, terminated, truncated, info = env.step(action)
        print(info["env_stochasticity"])
        print(reward, done)
        time.sleep(2)
        if done:
            break