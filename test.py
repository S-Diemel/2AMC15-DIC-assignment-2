from world import WarehouseEnv

env = WarehouseEnv()
obs, _ = env.reset()
done = False

for _ in range(1000):
    env.render()
    action = env.action_space.sample()
    obs, reward, done, _, _ = env.step(action)
    if done:
        break