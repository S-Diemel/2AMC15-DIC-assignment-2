from world import Environment
import time

env = Environment()
obs = env.reset(no_gui=False)
done = False

for _ in range(1000):
    env.render()
    action = env.action_space.sample()
    obs, reward, done = env.step(action)
    print(obs, reward, done)
    #time.sleep(3)
    if done:
        break