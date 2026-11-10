import os
import random
import time
import numpy as np
from typing import List, Tuple
from functools import partial
from multiprocessing import Pool
from tqdm import tqdm

from operators.memetic_tools import MemeticOperators
from core.ql_agent import qlAgent

# =====================================================================
# Global Optimization 1: Process-isolated local cache
# (Completely eliminates IPC overhead, deadlock risks, and memory leak issues)
# =====================================================================
_worker_local_cache = {}


def worker_init_fn(base_seed):
    """
    Initialize each worker subprocess. Allocates an independent random seed
    and a fully isolated local cache to each CPU core.
    """
    np.random.seed(base_seed)
    random.seed(base_seed)
    global _worker_local_cache
    _worker_local_cache = {}  # Process-local memory dictionary for ultra-fast, leak-free read/write access


def _parallel_eval_wrapper(individual, simulator_instance):
    global _worker_local_cache
    # Periodically purge the single-process cache to strictly prevent OOM system crashes
    if len(_worker_local_cache) > 3000:
        _worker_local_cache.clear()
    return simulator_instance.evaluate_fitness(individual, _worker_local_cache)


# =====================================================================
# Global Optimization 2: Fast shallow-copy function (replaces slow copy.deepcopy)
# =====================================================================
def fast_copy(ind: dict) -> dict:
    """Custom shallow-traversal copy tailored for route dictionaries, >10x faster than deepcopy."""
    return {'routes': [r[:] for r in ind.get('routes', [])]}


class GAEngine:
    """
    Multi-mode Evolutionary Engine integrating GA, RMA, SMA, and QL-MA.
    (Optimized for Extreme Performance, Memory Safety, and Absolute MDP Auditability)
    """

    def __init__(self, config: dict, network_env, simulator, mode: str = 'QL-MA'):
        self.config = config
        self.net = network_env
        self.sim = simulator
        self.mode = mode

        # Apply recommended tuned hyperparameters
        self.pop_size = config.get('POP_SIZE', 50)
        self.max_generations = config.get('MAX_GENERATIONS', 500)
        self.patience = config.get('PATIENCE', 150)

        self.elitism_rate = config.get('ELITISM_RATE', 0.1)
        self.crossover_rate = config.get('CROSSOVER_RATE', 0.8)
        self.mutation_rate = config.get('MUTATION_RATE', 0.2)
        self.base_seed = config.get('GLOBAL_SEED', 42)

        self.memetic_ops = MemeticOperators(
            config=config, sp=self.net.sp, sp_len=self.net.sp_len,
            stop_nodes_set=self.net.stop_nodes_set, od_df=self.net.od_df
        )

        self.agent = qlAgent(config=config) if self.mode == 'QL-MA' else None

        self.fitness_history = []
        self.operator_history = []
        self.q_snapshots = {}
        # Profiling time metrics initialization
        self.timings = {'eval_time': 0.0, 'op_time': 0.0, 'rl_time': 0.0}

        # Key state perception variables
        self._last_cv = 0.0
        self.cv_history = []
        self.current_dynamic_threshold = 0.0

    def _initialize_population(self) -> List[dict]:
        population = []
        num_routes = self.config.get('NUM_ROUTES', 4)
        for _ in range(self.pop_size):
            individual = {'routes': [self.memetic_ops._generate_valid_route() for _ in range(num_routes)]}
            population.append(self.memetic_ops.repair_individual(individual))
        return population

    def _calculate_diversity(self, population: List[dict]) -> float:
        unique_edges = set()
        total_edges = 0
        for ind in population:
            for r in ind.get('routes', []):
                if not r: continue
                edges = [(r[i], r[i + 1]) for i in range(len(r) - 1)]
                unique_edges.update(edges)
                total_edges += len(edges)
        return len(unique_edges) / (total_edges + 1e-6)

    def _select(self, population: List[dict], fitnesses: List[float], k: int = 3) -> dict:
        selected_indices = random.sample(range(len(population)), k)
        best_idx = min(selected_indices, key=lambda idx: fitnesses[idx])
        return fast_copy(population[best_idx])

    def _crossover(self, p1: dict, p2: dict) -> Tuple[dict, dict]:
        if random.random() > self.crossover_rate:
            return fast_copy(p1), fast_copy(p2)
        c1, c2 = fast_copy(p1), fast_copy(p2)
        r1, r2 = c1['routes'], c2['routes']
        if len(r1) > 1 and len(r2) > 1:
            pt = random.randint(1, min(len(r1), len(r2)) - 1)
            c1['routes'] = r1[:pt] + r2[pt:]
            c2['routes'] = r2[:pt] + r1[pt:]
        return c1, c2

    def run(self):
        # Operator timing: population initialization
        t_op_start = time.perf_counter()
        population = self._initialize_population()
        self.timings['op_time'] += time.perf_counter() - t_op_start

        # Bypass slow Manager().dict() by passing a lightweight wrapper function directly
        eval_func = partial(_parallel_eval_wrapper, simulator_instance=self.sim)

        max_workers = self.config.get('MAX_CPU_WORKERS', 4)
        num_cores = min(os.cpu_count() or 4, max_workers, self.pop_size)
        tasks_per_child = self.config.get('MAX_TASKS_PER_CHILD', 10)
        chunk_size = self.config.get('EVAL_CHUNKSIZE', 5)

        best_individual = None
        current_best_fitness = float('inf')
        stagnation_count = 0
        reference_best_fitness = float('inf')
        actual_gens = self.max_generations

        fit_hist = []
        op_hist = []
        q_snaps = {}
        micro_slice = {}
        micro_slice_triggered = False

        with Pool(processes=num_cores, initializer=worker_init_fn, initargs=(self.base_seed,),
                  maxtasksperchild=tasks_per_child) as pool:

            # Evaluation timing: generation 0 fitness evaluation
            t_eval_start = time.perf_counter()
            fitnesses = pool.map(eval_func, population, chunksize=chunk_size)
            self.timings['eval_time'] += time.perf_counter() - t_eval_start

            self._last_cv = np.std(fitnesses) / (np.mean(fitnesses) + 1e-6)

            with tqdm(total=self.max_generations, desc=f"Running {self.mode}", leave=False) as pbar:
                for gen in range(self.max_generations):

                    # 1. Record best fitness and evaluate global stagnation
                    current_best_idx = np.argmin(fitnesses)
                    if fitnesses[current_best_idx] < current_best_fitness:
                        current_best_fitness = fitnesses[current_best_idx]
                        best_individual = fast_copy(population[current_best_idx])

                    fit_hist.append(current_best_fitness)

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

                    if stagnation_count >= self.patience:
                        actual_gens = gen
                        break

                    # -----------------------------------------------------
                    # RL timing: Current State perception
                    # -----------------------------------------------------
                    t_rl_start = time.perf_counter()
                    current_state = -1
                    if self.agent:
                        # Maintain CV history and compute dynamic threshold
                        current_cv = self._last_cv
                        self.cv_history.append(current_cv)
                        self.current_dynamic_threshold = np.mean(self.cv_history[-50:])

                        # Determine current boolean state
                        is_div_high = current_cv >= self.current_dynamic_threshold
                        is_stagnant = (stagnation_count >= 10)

                        current_state = self.agent.get_state(is_stagnant, is_div_high)

                        # Record Q-table snapshots for XAI analysis
                        if gen in [10, 50, 100, 250, 499] or gen == self.max_generations - 1:
                            q_snaps[f"Gen_{gen}"] = self.agent.q_table.copy().tolist()
                    self.timings['rl_time'] += time.perf_counter() - t_rl_start

                    # -----------------------------------------------------
                    # Operator timing: Reproduction (selection & crossover)
                    # -----------------------------------------------------
                    t_op_start = time.perf_counter()
                    offspring_pool = []
                    while len(offspring_pool) < self.pop_size:
                        p1 = self._select(population, fitnesses)
                        p2 = self._select(population, fitnesses)
                        c1, c2 = self._crossover(p1, p2)
                        offspring_pool.extend([c1, c2])
                    offspring_pool = offspring_pool[:self.pop_size]
                    self.timings['op_time'] += time.perf_counter() - t_op_start

                    # -----------------------------------------------------
                    # Evaluation timing: Post-crossover fitness evaluation
                    # -----------------------------------------------------
                    t_eval_start = time.perf_counter()
                    fitnesses_before = pool.map(eval_func, offspring_pool, chunksize=chunk_size)
                    self.timings['eval_time'] += time.perf_counter() - t_eval_start

                    # -----------------------------------------------------
                    # Operator timing: Mutation / Local search (Memetic Ops)
                    # -----------------------------------------------------
                    t_op_start = time.perf_counter()
                    refined_offspring = []
                    actions_chosen = []

                    best_idx_before = np.argmin(fitnesses)
                    best_ind_before = population[best_idx_before]

                    for child in offspring_pool:
                        # Compatible with RMA and SMA terminology while preserving GA baseline
                        if self.mode == 'QL-MA' and self.agent:
                            action = self.agent.choose_action(current_state)
                        elif self.mode in ['MA-Random', 'RMA']:
                            action = np.random.randint(0, 4)
                        elif self.mode in ['MA-Fixed', 'SMA']:
                            action = 0
                        elif self.mode == 'GA':
                            if random.random() < self.mutation_rate:
                                action = np.random.randint(1, 4)
                            else:
                                action = -1
                        else:
                            action = -1

                        actions_chosen.append(action)

                        # Capture XAI micro-slice
                        if self.mode == 'QL-MA' and self.agent and current_state == 1 and action == 3 and gen > 100 and not micro_slice_triggered:
                            def routes_to_edges(routes):
                                edges = set()
                                for r in routes:
                                    for i in range(len(r) - 1):
                                        edges.add(tuple(sorted((int(r[i]), int(r[i + 1])))))
                                return [list(e) for e in edges]

                            micro_slice['Trigger_Gen'] = gen
                            micro_slice['State'] = 'S1'
                            micro_slice['Action'] = 'A3'
                            micro_slice['Q_Values'] = self.agent.q_table.copy().tolist()
                            micro_slice['Before_Topology'] = routes_to_edges(best_ind_before['routes'])

                            temp_mutated = self.memetic_ops.apply_action(3, fast_copy(best_ind_before))
                            micro_slice['After_Topology'] = routes_to_edges(temp_mutated['routes'])
                            micro_slice_triggered = True

                        if action != -1:
                            mutated = self.memetic_ops.apply_action(action, child)
                        else:
                            mutated = child

                        refined_offspring.append(mutated)
                    self.timings['op_time'] += time.perf_counter() - t_op_start

                    # -----------------------------------------------------
                    # Evaluation timing: Post-mutation fitness evaluation
                    # -----------------------------------------------------
                    t_eval_start = time.perf_counter()
                    fitnesses_after = pool.map(eval_func, refined_offspring, chunksize=chunk_size)
                    self.timings['eval_time'] += time.perf_counter() - t_eval_start

                    # -----------------------------------------------------
                    # RL timing: Compute reinforcement learning reward
                    # -----------------------------------------------------
                    t_rl_start = time.perf_counter()
                    transitions = []
                    avg_reward = 0
                    if self.agent:
                        for i in range(len(refined_offspring)):
                            rel_imp = (fitnesses_before[i] - fitnesses_after[i]) / (fitnesses_before[i] + 1e-6)
                            reward = np.tanh(rel_imp * 100.0)
                            # Shaped reward: encourage exploration when stagnant
                            if current_state in [0, 1]:
                                reward += 0.2 if actions_chosen[i] != 0 else -0.2
                            transitions.append((current_state, actions_chosen[i], reward))
                        avg_reward = np.mean([t[2] for t in transitions]) if transitions else 0.0

                    # Reuse evaluated elite fitnesses directly, eliminating redundant evaluations
                    num_elites = max(1, int(self.pop_size * self.elitism_rate))
                    elite_indices = np.argsort(fitnesses)[:num_elites]

                    elites = [fast_copy(population[idx]) for idx in elite_indices]
                    elite_fits = [fitnesses[idx] for idx in elite_indices]

                    population = refined_offspring[:-num_elites] + elites
                    fitnesses = fitnesses_after[:-num_elites] + elite_fits

                    self._last_cv = np.std(fitnesses) / (np.mean(fitnesses) + 1e-6)

                    # -----------------------------------------------------
                    # RL Next State deduction and Markov Decision Process (MDP) update
                    # -----------------------------------------------------
                    if self.agent:
                        # 1. Deduce next diversity boolean state using the same baseline threshold
                        next_is_div_high = self._last_cv >= self.current_dynamic_threshold

                        # 2. Deduce next stagnation boolean state
                        next_best_fitness = np.min(fitnesses)
                        temp_stagnation = stagnation_count
                        if reference_best_fitness != float('inf'):
                            rel_imp = (reference_best_fitness - next_best_fitness) / (reference_best_fitness + 1e-9)
                            if rel_imp > 1e-5:
                                temp_stagnation = 0
                            else:
                                temp_stagnation += 1
                        next_is_stagnant = (temp_stagnation >= 10)

                        # 3. Determine exact next state
                        next_state = self.agent.get_state(next_is_stagnant, next_is_div_high)

                        # Update Q-table
                        for s, a, r in transitions:
                            self.agent.learn(s, a, r, next_state)
                        self.agent.step_decay(next_state)

                        # Log MDP transition history
                        maj_action = max(set(actions_chosen), key=actions_chosen.count) if actions_chosen else -1
                        op_hist.append({
                            'Generation': gen,
                            'State': current_state,
                            'Current_State': f"S{current_state}",
                            'Action_Selected': f"A{maj_action}",
                            'Reward': round(float(avg_reward), 4),
                            'Next_State': f"S{next_state}"
                        })
                    self.timings['rl_time'] += time.perf_counter() - t_rl_start

                    # -----------------------------------------------------
                    # Progress bar display and updates
                    # -----------------------------------------------------
                    display_div = self._last_cv if self.agent else 0.0
                    pbar.set_postfix({
                        'Fit': f"{current_best_fitness:,.0f}",
                        'Stag': f"{stagnation_count}/{self.patience}",
                        'State': f"S{current_state}" if current_state != -1 else "-",
                        'Div': f"{display_div:.2f}"
                    })
                    pbar.update(1)

        return best_individual, fit_hist, op_hist, q_snaps, self.timings, actual_gens, micro_slice