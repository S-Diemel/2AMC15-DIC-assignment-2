from pathlib import Path
from world.environment import Environment
from agents.dqn import DQNAgent
from tqdm import trange
import argparse

def demonstrate_agent(model_path: Path):
    """
    run a demonstration of the trained agent and calc the success rate
    """
    iters = 100
    success_list =[]
    agent = DQNAgent.load(str(model_path),
                          state_size=12,
                          action_size=5, seed=60)

    # show the GUI for the first 3 evaluations
    no_gui = False
    for i in range(iters):
        success_list.append(simulate_episode(agent, 1000, no_gui, 3, 3, 0.25, 0, 0))
        if i == 2:
            no_gui = True

    print(success_list)
    print(f'success_rate: {sum(success_list)/len(success_list)}')


def simulate_episode(agent, iters, no_gui, difficulty, number_of_items, battery_drain_per_step, epsilon, sigma):
    """
    Run 1 episode of the trained agent
    """
    env = Environment(sigma=sigma)
    state, _ = env.reset(no_gui=no_gui, difficulty=difficulty, number_of_items=number_of_items, 
                         battery_drain_per_step=battery_drain_per_step)
    agent.epsilon=epsilon
    for _ in trange(iters):
        action = agent.take_action(state)
        next_state, _, terminated, truncated, _ = env.step(action)
        env.render()
        state = next_state
        if terminated:
            return True
        if truncated:
            return False
    return False


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("model", type=Path, help="Path to .pth checkpoint")
    args = p.parse_args()
    demonstrate_agent(args.model)
