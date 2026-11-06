import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import copy
import time
import numpy as np
import pandas as pd
from functools import partial
from multiprocessing import Pool, Manager, freeze_support
from tqdm import tqdm

from configs.default_config import CONFIG, NETWORK_CONFIGS
from envs.network_env import TransitNetwork
from envs.passenger_sim import PassengerSimulator
from core.ga_engine import GAEngine, worker_init_fn, _parallel_eval_wrapper
from utils.seed_manager import seed_everything


def get_greedy_probs(q_vals):
    """修正：返回绝对贪婪概率，消除探索噪声，放大微小优势"""
    probs = np.zeros(4)
    probs[np.argmax(q_vals)] = 1.0
    return probs.tolist()


class SingleBranchInterventionEngine(GAEngine):

    def run_intervention(self, intervention_gen=250, total_gens=300):
        self.max_generations = total_gens
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

        best_individual = None
        current_best_fitness = float('inf')
        stagnation_count = 0
        reference_best_fitness = float('inf')

        history_baseline = []

        with Pool(processes=num_cores, initializer=worker_init_fn, initargs=(self.base_seed,),
                  maxtasksperchild=tasks_per_child) as pool:
            fitnesses = pool.map(eval_func, population, chunksize=chunk_size)
            self._last_cv = np.std(fitnesses) / (np.mean(fitnesses) + 1e-6)

            # ==========================================
            # 阶段 1: 正常基线运行
            # ==========================================
            with tqdm(total=intervention_gen, desc=f"Phase 1: Baseline to Gen {intervention_gen}", leave=False) as pbar:
                transitions = []
                for gen in range(intervention_gen):
                    current_best_idx = np.argmin(fitnesses)
                    if fitnesses[current_best_idx] < current_best_fitness:
                        current_best_fitness = fitnesses[current_best_idx]
                        best_individual = copy.deepcopy(population[current_best_idx])

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
                    a_probs = [0, 0, 0, 0]
                    if self.agent:
                        diversity = self._calculate_diversity(population)
                        is_stagnant = (stagnation_count >= 10)
                        current_state = self.agent.get_state(is_stagnant, diversity)
                        q_vals = self.agent.q_table[current_state]
                        a_probs = get_greedy_probs(q_vals)

                    history_baseline.append({
                        'Generation': gen, 'State': current_state, 'CV': self._last_cv,
                        'Prob_A0': a_probs[0], 'Prob_A1': a_probs[1],
                        'Prob_A2': a_probs[2], 'Prob_A3': a_probs[3]
                    })

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

                    for child in offspring_pool:
                        action = self.agent.choose_action(current_state) if self.agent else 0
                        mutated = self.memetic_ops.apply_action(action, child)
                        refined_offspring.append(mutated)
                        actions_chosen.append(action)

                    fitnesses_after = pool.map(eval_func, refined_offspring, chunksize=chunk_size)

                    if self.agent:
                        for i in range(len(refined_offspring)):
                            rel_imp = (fitnesses_before[i] - fitnesses_after[i]) / (fitnesses_before[i] + 1e-6)
                            reward = np.tanh(rel_imp * 100.0)
                            if current_state in [0, 1]:
                                reward += 0.2 if actions_chosen[i] != 0 else -0.2
                            transitions.append((current_state, actions_chosen[i], reward))

                    next_population = refined_offspring
                    num_elites = max(1, int(self.pop_size * self.elitism_rate))
                    elite_indices = np.argsort(fitnesses)[:num_elites]
                    elites = [copy.deepcopy(population[idx]) for idx in elite_indices]
                    next_population = next_population[:-num_elites] + elites
                    population = next_population

                    fitnesses = pool.map(eval_func, population, chunksize=chunk_size)
                    self._last_cv = np.std(fitnesses) / (np.mean(fitnesses) + 1e-6)

                    if self.agent:
                        next_diversity = self._calculate_diversity(population)
                        next_is_stagnant = (stagnation_count >= 10)
                        next_state = self.agent.get_state(next_is_stagnant, next_diversity)
                        for s, a, r in transitions:
                            self.agent.learn(s, a, r, next_state)
                        self.agent.step_decay(next_state)
                        transitions.clear()

                    pbar.update(1)

            # ==========================================
            # 阶段 2: 单一干预 (Branch A: Diversity Collapse)
            # ==========================================
            tqdm.write(f"\n⚡ [INTERVENTION] Shocking the Universe at Gen {intervention_gen}!")
            tqdm.write("  -> Running Branch A (Clone Injection / $S_1$ shock)...")
            branch_a_history = []

            pop_A = [copy.deepcopy(best_individual) for _ in range(self.pop_size)]
            fit_A = pool.map(eval_func, pop_A, chunksize=chunk_size)
            stag_A = 20
            ref_A = current_best_fitness
            agent_A = copy.deepcopy(self.agent)

            for gen in tqdm(range(intervention_gen, total_gens), desc="Branch A", leave=False):
                current_best_idx = np.argmin(fit_A)
                if fit_A[current_best_idx] < ref_A:
                    rel_imp = (ref_A - fit_A[current_best_idx]) / (ref_A + 1e-9)
                    if rel_imp > 1e-5:
                        ref_A = fit_A[current_best_idx]
                        stag_A = 0
                    else:
                        stag_A += 1
                else:
                    stag_A += 1

                cv_A = np.std(fit_A) / (np.mean(fit_A) + 1e-6)

                current_state_A = -1
                a_probs_A = [0, 0, 0, 0]
                if agent_A:
                    is_stag_A = (stag_A >= 10)
                    diversity_A = self._calculate_diversity(pop_A)
                    current_state_A = agent_A.get_state(is_stag_A, diversity_A)
                    q_vals = agent_A.q_table[current_state_A]
                    a_probs_A = get_greedy_probs(q_vals)

                branch_a_history.append({
                    'Generation': gen, 'State': current_state_A, 'CV': cv_A,
                    'Prob_A0': a_probs_A[0], 'Prob_A1': a_probs_A[1],
                    'Prob_A2': a_probs_A[2], 'Prob_A3': a_probs_A[3]
                })

                offspring_pool = []
                while len(offspring_pool) < self.pop_size:
                    p1 = self._select(pop_A, fit_A)
                    p2 = self._select(pop_A, fit_A)
                    c1, c2 = self._crossover(p1, p2)
                    offspring_pool.extend([c1, c2])
                offspring_pool = offspring_pool[:self.pop_size]

                fit_before = pool.map(eval_func, offspring_pool, chunksize=chunk_size)
                refined = []
                actions = []
                for child in offspring_pool:
                    action = agent_A.choose_action(current_state_A) if agent_A else 0
                    mutated = self.memetic_ops.apply_action(action, child)
                    refined.append(mutated)
                    actions.append(action)

                fit_after = pool.map(eval_func, refined, chunksize=chunk_size)

                transitions_A = []
                if agent_A:
                    for i in range(len(refined)):
                        rel_imp = (fit_before[i] - fit_after[i]) / (fit_before[i] + 1e-6)
                        reward = np.tanh(rel_imp * 100.0)
                        if current_state_A in [0, 1]: reward += 0.2 if actions[i] != 0 else -0.2
                        transitions_A.append((current_state_A, actions[i], reward))

                pop_A = refined
                num_elites = max(1, int(self.pop_size * self.elitism_rate))
                elite_indices = np.argsort(fit_A)[:num_elites]
                elites = [copy.deepcopy(pop_A[idx]) for idx in elite_indices]
                pop_A = pop_A[:-num_elites] + elites
                fit_A = pool.map(eval_func, pop_A, chunksize=chunk_size)

                if agent_A:
                    next_div = self._calculate_diversity(pop_A)
                    next_state = agent_A.get_state((stag_A >= 10), next_div)
                    for s, a, r in transitions_A: agent_A.learn(s, a, r, next_state)
                    agent_A.step_decay(next_state)

        return history_baseline, branch_a_history


def run_exp7_single_intervention():
    CONFIG['MAX_GENERATIONS'] = 300
    CONFIG['PATIENCE'] = 500

    net = 'mumford0'
    seed = 47
    seed_everything(seed)

    net_config = NETWORK_CONFIGS[net]
    CONFIG.update({
        'BASE_PATH': net_config['path'], 'NUM_ROUTES': net_config['num_routes'],
        'MAX_TOTAL_FLEET': net_config['max_fleet'], 'MAX_STOPS_PER_ROUTE': net_config['max_stops'],
        'MIN_STOPS_PER_ROUTE': net_config['min_stops']
    })

    network_env = TransitNetwork(base_path=net_config['path'])
    simulator = PassengerSimulator(network_env=network_env, config=CONFIG)

    engine = SingleBranchInterventionEngine(config=CONFIG, network_env=network_env, simulator=simulator, mode='QL-MA')
    print(f"\n🚀 Launching Rigorous Single-Branch Intervention on {net.upper()}...")

    base_hist, branch_a = engine.run_intervention(intervention_gen=250, total_gens=300)

    out_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'output', 'exp4_6_xai_data'))
    os.makedirs(out_dir, exist_ok=True)

    pd.DataFrame(base_hist).to_csv(os.path.join(out_dir, f'SingleBranch_{net}_Base.csv'), index=False)
    pd.DataFrame(branch_a).to_csv(os.path.join(out_dir, f'SingleBranch_{net}_Branch_A.csv'), index=False)
    print(f"✅ Real Single-Branch Data Saved!")


if __name__ == '__main__':
    freeze_support()
    run_exp7_single_intervention()