import sys
import os
import copy
import time
import json
import pandas as pd
import numpy as np
from functools import partial
from multiprocessing import Pool, Manager, freeze_support
from tqdm import tqdm

# Ensure the project root directory is discoverable
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from configs.default_config import CONFIG, NETWORK_CONFIGS
from envs.network_env import TransitNetwork
from envs.passenger_sim import PassengerSimulator
from core.ga_engine import GAEngine, worker_init_fn, _parallel_eval_wrapper
from utils.seed_manager import seed_everything


def routes_to_edgelist(routes):
    """Convert transit routes into an edge list [u, v] for direct ingestion by NetworkX."""
    edges = set()
    for route in routes:
        for i in range(len(route) - 1):
            u, v = int(route[i]), int(route[i + 1])
            edges.add(tuple(sorted((u, v))))
    return [list(e) for e in edges]


class XAILoggerEngine(GAEngine):
    """Customized GA engine tailored for logging explainable AI (XAI) dynamics without altering the core GAEngine."""

    def run_with_logger(self):
        # 1. Initialize data recording containers
        q_snapshots = {}
        transitions_log = []
        micro_slice = {}
        micro_slice_triggered = False

        t_op_start = time.perf_counter()
        population = self._initialize_population()
        self.timings['op_time'] += time.perf_counter() - t_op_start

        manager = Manager()
        fitness_cache = manager.dict()
        eval_func = partial(_parallel_eval_wrapper, simulator_instance=self.sim, shared_cache=fitness_cache)

        max_workers = self.config.get('MAX_CPU_WORKERS', 6)
        num_cores = min(os.cpu_count() or 4, max_workers, self.pop_size)
        tasks_per_child = self.config.get('MAX_TASKS_PER_CHILD', 150)
        chunk_size = self.config.get('EVAL_CHUNKSIZE', 3)

        current_best_fitness = float('inf')
        stagnation_count = 0
        reference_best_fitness = float('inf')

        with Pool(processes=num_cores, initializer=worker_init_fn, initargs=(self.base_seed,),
                  maxtasksperchild=tasks_per_child) as pool:
            fitnesses = pool.map(eval_func, population, chunksize=chunk_size)

            with tqdm(total=self.max_generations, desc="Running QL-MA Data Logger", leave=True) as pbar:
                for gen in range(self.max_generations):
                    current_best_idx = np.argmin(fitnesses)
                    if fitnesses[current_best_idx] < current_best_fitness:
                        current_best_fitness = fitnesses[current_best_idx]

                    if reference_best_fitness != float('inf'):
                        rel_improvement = (reference_best_fitness - current_best_fitness) / (
                                    reference_best_fitness + 1e-9)
                        if rel_improvement > 1e-5:
                            reference_best_fitness = current_best_fitness
                            stagnation_count = 0
                        else:
                            stagnation_count += 1
                    else:
                        reference_best_fitness = current_best_fitness

                    current_state = -1
                    if self.agent:
                        diversity = self._calculate_diversity(population)
                        is_stagnant = (stagnation_count >= 10)
                        current_state = self.agent.get_state(is_stagnant, diversity)

                        # ==========================================
                        # Logger 1: Q-Table historical snapshots
                        # ==========================================
                        if gen in [50, 250, 500, self.max_generations - 1]:
                            q_snapshots[f"Gen_{gen}"] = self.agent.q_table.tolist()

                    offspring_pool = []
                    while len(offspring_pool) < self.pop_size:
                        p1 = self._select(population, fitnesses)
                        p2 = self._select(population, fitnesses)
                        c1, c2 = self._crossover(p1, p2)
                        offspring_pool.extend([c1, c2])
                    offspring_pool = offspring_pool[:self.pop_size]

                    fitnesses_before = pool.map(eval_func, offspring_pool, chunksize=chunk_size)
                    refined_offspring = []
                    actions_chosen = []

                    # Capture the current generation's best individual for micro-slice comparison
                    best_idx_before = np.argmin(fitnesses)
                    best_ind_before = population[best_idx_before]

                    for child in offspring_pool:
                        action = self.agent.choose_action(current_state) if self.agent else 0

                        # ==========================================
                        # Logger 3: Microscopic phase transition slice (S1 triggering A3)
                        # ==========================================
                        if self.agent and current_state == 1 and action == 3 and not micro_slice_triggered:
                            micro_slice['Trigger_Gen'] = gen
                            micro_slice['State'] = 'S1'
                            micro_slice['Action'] = 'A3'
                            micro_slice['Q_Values'] = self.agent.q_table[current_state].tolist()
                            micro_slice['Before_Topology'] = routes_to_edgelist(best_ind_before['routes'])

                            # Directly apply a disruptive A3 mutation to the global best individual to capture physical topological disruption
                            mutated_best = self.memetic_ops.apply_action(3, copy.deepcopy(best_ind_before))
                            micro_slice['After_Topology'] = routes_to_edgelist(mutated_best['routes'])
                            micro_slice_triggered = True

                        mutated = self.memetic_ops.apply_action(action, child)
                        refined_offspring.append(mutated)
                        actions_chosen.append(action)

                    fitnesses_after = pool.map(eval_func, refined_offspring, chunksize=chunk_size)

                    transitions = []
                    avg_reward = 0
                    if self.agent:
                        for i in range(len(refined_offspring)):
                            rel_imp = (fitnesses_before[i] - fitnesses_after[i]) / (fitnesses_before[i] + 1e-6)
                            reward = np.tanh(rel_imp * 100.0)
                            if current_state in [0, 1]:
                                reward += 0.2 if actions_chosen[i] != 0 else -0.2
                            transitions.append((current_state, actions_chosen[i], reward))
                        avg_reward = np.mean([t[2] for t in transitions])

                    next_population = refined_offspring
                    num_elites = max(1, int(self.pop_size * self.elitism_rate))
                    elite_indices = np.argsort(fitnesses)[:num_elites]
                    elites = [copy.deepcopy(population[idx]) for idx in elite_indices]
                    next_population = next_population[:-num_elites] + elites
                    population = next_population

                    fitnesses = pool.map(eval_func, population, chunksize=chunk_size)

                    next_state = -1
                    if self.agent:
                        next_diversity = self._calculate_diversity(population)
                        next_is_stagnant = (stagnation_count >= 10)
                        next_state = self.agent.get_state(next_is_stagnant, next_diversity)

                        for s, a, r in transitions:
                            self.agent.learn(s, a, r, next_state)
                        self.agent.step_decay(next_state)

                        # ==========================================
                        # Logger 2: Complete Markov transition log
                        # ==========================================
                        # Log the majority action selected across offspring for this generation
                        maj_action = max(set(actions_chosen), key=actions_chosen.count)
                        transitions_log.append({
                            'Generation': gen,
                            'Current_State': f"S{current_state}",
                            'Action_Selected': f"A{maj_action}",
                            'Reward': round(avg_reward, 4),
                            'Next_State': f"S{next_state}"
                        })

                    pbar.update(1)

        return q_snapshots, transitions_log, micro_slice


def run_data_logger():
    CONFIG['MAX_GENERATIONS'] = 550
    # Disable early stopping to guarantee capturing full-cycle dynamics
    CONFIG['PATIENCE'] = 600

    net = 'mumford0'
    seed = 47
    seed_everything(seed)

    net_config = NETWORK_CONFIGS[net]
    CONFIG.update({
        'BASE_PATH': net_config['path'],
        'NUM_ROUTES': net_config['num_routes'],
        'MAX_TOTAL_FLEET': net_config['max_fleet'],
        'MAX_STOPS_PER_ROUTE': net_config['max_stops'],
        'MIN_STOPS_PER_ROUTE': net_config['min_stops']
    })

    network_env = TransitNetwork(base_path=net_config['path'])
    simulator = PassengerSimulator(network_env=network_env, config=CONFIG)

    print(f"\n🚀 Launching XAI Data Logger on {net.upper()} (Seed {seed})...")
    engine = XAILoggerEngine(config=CONFIG, network_env=network_env, simulator=simulator, mode='QL-MA')

    q_snaps, trans_log, m_slice = engine.run_with_logger()

    # Persist log artifacts safely to disk
    out_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'output', 'exp8_xai_logs'))
    os.makedirs(out_dir, exist_ok=True)

    with open(os.path.join(out_dir, 'q_snapshots.json'), 'w') as f:
        json.dump(q_snaps, f, indent=4)

    pd.DataFrame(trans_log).to_csv(os.path.join(out_dir, 'transitions_log.csv'), index=False)

    with open(os.path.join(out_dir, 'micro_slice.json'), 'w') as f:
        json.dump(m_slice, f, indent=4)

    print(f"\n✅ XAI data collection complete! Files saved to: {out_dir}")
    print(f"   - q_snapshots.json")
    print(f"   - transitions_log.csv")
    print(f"   - micro_slice.json")


if __name__ == '__main__':
    freeze_support()
    run_data_logger()