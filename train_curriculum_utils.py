from world.environment import Environment
import csv
from tqdm import trange


def setup_curriculum(total_episodes, curriculum, num_evaluations_per_phase):
    """Precompute phase boundaries, parameters, and evaluation points."""
    phases = []
    start_ep = 0
    for phase in curriculum:
        percent, eps_start, eps_end, difficulty, num_items, battery_drain = phase
        end_ep = start_ep + int(percent * total_episodes) - 1
        phase_length = end_ep - start_ep + 1
        
        # Calculate evaluation points (equally spaced including start/middle/end)
        eval_points = []
        if num_evaluations_per_phase > 0:
            step = max(1, phase_length // (num_evaluations_per_phase - 1)) if num_evaluations_per_phase > 1 else phase_length
            eval_points = [start_ep + i*step for i in range(num_evaluations_per_phase)]
            eval_points = [min(ep, end_ep) for ep in eval_points]
        
        phases.append({
            'start_ep': start_ep,
            'end_ep': end_ep,
            'eps_start': eps_start,
            'eps_end': eps_end,
            'difficulty': difficulty,
            'num_items': num_items,
            'battery_drain': battery_drain,
            'eval_points': eval_points
        })
        start_ep = end_ep + 1
    
    # Extend last phase to cover any remaining episodes
    phases[-1]['end_ep'] = total_episodes - 1
    return phases


def get_curriculum_parameters(episode, phases):
    """Get params for current episode including whether to evaluate."""
    for phase in phases:
        if phase['start_ep'] <= episode <= phase['end_ep']:
            # Calculate epsilon
            decay_length = int(0.7 * (phase['end_ep'] - phase['start_ep'] + 1))
            phase_progress = min(episode - phase['start_ep'], decay_length)
            
            epsilon = phase['eps_start'] - (phase['eps_start'] - phase['eps_end']) * (phase_progress / decay_length) if decay_length > 0 else phase['eps_end']
            epsilon = max(phase['eps_end'], epsilon)
            
            # Check if this is an evaluation point
            should_evaluate = episode in phase['eval_points']
            
            return (
                phases.index(phase) + 1,    # phase_number
                epsilon,
                phase['difficulty'],
                phase['num_items'],
                phase['battery_drain'],
                should_evaluate
            )


def evaluate_agent_metrics(agent, difficulty, number_of_items, battery_drain_per_step, 
                           epsilon=0, sigma=0, eval_runs=100, eval_iters=1000, no_gui=True):
    """
    Runs `eval_runs` evaluation episodes, each with up to `eval_iters` steps.
    Returns a list of metric dicts with keys:
      'cumulative_reward', 'steps', 'terminated', 'truncated', 'ran_out_of_steps', 'percent_delivered'
    """
    metrics = []
    for _ in trange(eval_runs, desc="Evaluating agent"):
        env = Environment(sigma=sigma)
        state, _ = env.reset(no_gui=no_gui, difficulty=difficulty, number_of_items=number_of_items, 
                             battery_drain_per_step=battery_drain_per_step, difficulty_mode="eval")
        agent.epsilon = epsilon
        cumulative_reward = 0
        steps = 0
        terminated = False
        truncated = False
        for _ in range(eval_iters):
            action = agent.take_action(state)
            next_state, reward, term, trunc, _ = env.step(action)
            cumulative_reward += reward
            steps += 1
            state = next_state
            if term:
                terminated = True
                break
            if trunc:
                truncated = True
                break
        delivered = sum(env.delivered) / len(env.delivered) if hasattr(env, 'delivered') and len(env.delivered) > 0 else 0.0
        ran_out_of_steps = not terminated and not truncated
        metrics.append({
            'cumulative_reward': cumulative_reward,
            'steps': steps,
            'terminated': terminated,
            'truncated': truncated,
            'ran_out_of_steps': ran_out_of_steps,
            'percent_delivered': delivered
        })
    return metrics


def save_metrics_to_csv(metrics_by_stage, filename):
    fieldnames = [
        'phase_number', 
        'eval_stage',  # in case of 3 eval moments per phase: 0=start, 1=middle, 2=end
        'avg_cumulative_reward', 
        'avg_steps', 
        'success_rate', 
        'battery_depleted_rate', 
        'ran_out_of_steps_rate', 
        'avg_percent_delivered'
    ]
    
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        # Sort by phase then evaluation stage
        for (phase_number, eval_stage), metrics in sorted(metrics_by_stage.items()):
            n = len(metrics)
            success_rate = sum(m['terminated'] for m in metrics) / n
            battery_depleted_rate = sum(m['truncated'] and not m['terminated'] for m in metrics) / n
            ran_out_of_steps_rate = sum(m['ran_out_of_steps'] for m in metrics) / n
            avg_cum_reward = sum(m['cumulative_reward'] for m in metrics) / n
            avg_steps = sum(m['steps'] for m in metrics) / n
            avg_percent_delivered = sum(m['percent_delivered'] for m in metrics) / n
            
            writer.writerow({
                'phase_number': phase_number,
                'eval_stage': eval_stage,
                'avg_cumulative_reward': avg_cum_reward,
                'avg_steps': avg_steps,
                'success_rate': success_rate,
                'battery_depleted_rate': battery_depleted_rate,
                'ran_out_of_steps_rate': ran_out_of_steps_rate,
                'avg_percent_delivered': avg_percent_delivered
            })
