from world import WarehouseEnv
import time

env = WarehouseEnv()
obs = env.reset()
done = False

for _ in range(1000):
    env.render()
    action = env.action_space.sample()
    obs, reward, done = env.step(action)
    print(obs, reward, done)
    #time.sleep(3)
    if done:
        break