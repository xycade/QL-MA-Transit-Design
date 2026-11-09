import os
import random
import time
import numpy as np
from functools import partial
from multiprocessing import Pool
from tqdm import tqdm

# 直接复用您原本写好的优秀组件
from core.ga_engine import worker_init_fn, _parallel_eval_wrapper, fast_copy
from operators.memetic_tools import MemeticOperators
from core.ql_agent import qlAgent


class CounterfactualEngine:
    """
    专门用于 Section 6.5 反事实干预实验的引擎。
    在指定代数 (COUNTERFACTUAL_GEN) 强制触发克隆注入，将种群多样性毁灭为 0。
    """

    def __init__(self, config: dict, network_env, simulator, mode: str = 'QL-MA'):
        self.config = config
        self.net = network_env
        self.sim = simulator
        self.mode = mode

        self.pop_size = config.get('POP_SIZE', 30)
        self.max_generations = config.get('MAX_GENERATIONS', 400)
        self.patience = config.get('PATIENCE', 999)  # 禁用早停
        self.base_seed = config.get('GLOBAL_SEED', 42)
        self.elitism_rate = config.get('ELITISM_RATE', 0.1)
        self.crossover_rate = config.get('CROSSOVER_RATE', 0.9)
        self.mutation_rate = config.get('MUTATION_RATE', 0.2)

        # 触发干预的代数
        self.intervention_gen = config.get('COUNTERFACTUAL_GEN', 250)

        self.memetic_ops = MemeticOperators(
            config=config, sp=self.net.sp, sp_len=self.net.sp_len,
            stop_nodes_set=self.net.stop_nodes_set, od_df=self.net.od_df
        )
        self.agent = qlAgent(config=config) if self.mode == 'QL-MA' else None

        self.cv_history = []
        self._last_cv = 0.0

    def _initialize_population(self):
        population = []
        num_routes = self.config.get('NUM_ROUTES', 4)
        for _ in range(self.pop_size):
            ind = {'routes': [self.memetic_ops._generate_valid_route() for _ in range(num_routes)]}
            population.append(self.memetic_ops.repair_individual(ind))
        return population

    def _calculate_diversity(self, population):
        unique_edges = set()
        total_edges = 0
        for ind in population:
            for r in ind.get('routes', []):
                if not r: continue
                edges = [(r[i], r[i + 1]) for i in range(len(r) - 1)]
                unique_edges.update(edges)
                total_edges += len(edges)
        return len(unique_edges) / (total_edges + 1e-6)

    def _select(self, population, fitnesses, k=3):
        selected_indices = random.sample(range(len(population)), k)
        best_idx = min(selected_indices, key=lambda idx: fitnesses[idx])
        return fast_copy(population[best_idx])

    def _crossover(self, p1, p2):
        if random.random() > self.crossover_rate: return fast_copy(p1), fast_copy(p2)
        c1, c2 = fast_copy(p1), fast_copy(p2)
        r1, r2 = c1['routes'], c2['routes']
        if len(r1) > 1 and len(r2) > 1:
            pt = random.randint(1, min(len(r1), len(r2)) - 1)
            c1['routes'] = r1[:pt] + r2[pt:]
            c2['routes'] = r2[:pt] + r1[pt:]
        return c1, c2

    def run(self):
        population = self._initialize_population()
        eval_func = partial(_parallel_eval_wrapper, simulator_instance=self.sim)

        max_workers = self.config.get('MAX_CPU_WORKERS', 5)
        num_cores = min(os.cpu_count() or 4, max_workers, self.pop_size)

        current_best_fitness = float('inf')
        best_individual = None
        reference_best_fitness = float('inf')
        stagnation_count = 0
        op_hist = []

        with Pool(processes=num_cores, initializer=worker_init_fn, initargs=(self.base_seed,)) as pool:
            fitnesses = pool.map(eval_func, population)
            self._last_cv = np.std(fitnesses) / (np.mean(fitnesses) + 1e-6)

            with tqdm(total=self.max_generations, desc="Running Counterfactual XAI") as pbar:
                for gen in range(self.max_generations):

                    # 1. 更新最佳个体
                    current_best_idx = np.argmin(fitnesses)
                    if fitnesses[current_best_idx] < current_best_fitness:
                        current_best_fitness = fitnesses[current_best_idx]
                        best_individual = fast_copy(population[current_best_idx])

                    # =====================================================
                    # 🚨 核心魔法：上帝之手强制干预！
                    # =====================================================
                    if gen == self.intervention_gen:
                        print(f"\n⚡ [God's Hand] Initiating Clone Injection at Gen {gen}!")
                        # 把整个种群替换为最佳个体的完全克隆，彻底抹杀多样性
                        population = [fast_copy(best_individual) for _ in range(self.pop_size)]
                        fitnesses = [current_best_fitness] * self.pop_size
                        self._last_cv = 0.0  # 多样性强制归零
                    # =====================================================

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

                    # 2. 状态感知
                    current_state = -1
                    if self.agent:
                        current_cv = self._last_cv
                        self.cv_history.append(current_cv)
                        dynamic_threshold = np.mean(self.cv_history[-50:]) if len(self.cv_history) > 0 else 0
                        is_div_high = current_cv >= dynamic_threshold
                        is_stagnant = (stagnation_count >= 10)
                        current_state = self.agent.get_state(is_stagnant, is_div_high)

                    # 3. 繁衍与操作
                    offspring_pool = []
                    while len(offspring_pool) < self.pop_size:
                        c1, c2 = self._crossover(self._select(population, fitnesses),
                                                 self._select(population, fitnesses))
                        offspring_pool.extend([c1, c2])
                    offspring_pool = offspring_pool[:self.pop_size]

                    fitnesses_before = pool.map(eval_func, offspring_pool)

                    refined_offspring = []
                    actions_chosen = []
                    for child in offspring_pool:
                        action = self.agent.choose_action(current_state) if self.agent else -1
                        actions_chosen.append(action)
                        refined_offspring.append(
                            self.memetic_ops.apply_action(action, child) if action != -1 else child)

                    fitnesses_after = pool.map(eval_func, refined_offspring)

                    # 4. 奖励与更新
                    transitions = []
                    avg_reward = 0
                    if self.agent:
                        for i in range(len(refined_offspring)):
                            rel_imp = (fitnesses_before[i] - fitnesses_after[i]) / (fitnesses_before[i] + 1e-6)
                            reward = np.tanh(rel_imp * 100.0)
                            if current_state in [0, 1]:
                                reward += 0.2 if actions_chosen[i] != 0 else -0.2
                            transitions.append((current_state, actions_chosen[i], reward))
                        avg_reward = np.mean([t[2] for t in transitions]) if transitions else 0.0

                    num_elites = max(1, int(self.pop_size * self.elitism_rate))
                    elite_indices = np.argsort(fitnesses)[:num_elites]
                    population = refined_offspring[:-num_elites] + [fast_copy(population[idx]) for idx in elite_indices]
                    fitnesses = fitnesses_after[:-num_elites] + [fitnesses[idx] for idx in elite_indices]

                    self._last_cv = np.std(fitnesses) / (np.mean(fitnesses) + 1e-6)

                    # 5. 日志记录
                    if self.agent:
                        next_state = self.agent.get_state(stagnation_count >= 10, self._calculate_diversity(population))
                        for s, a, r in transitions:
                            self.agent.learn(s, a, r, next_state)
                        self.agent.step_decay(next_state)

                        maj_action = max(set(actions_chosen), key=actions_chosen.count) if actions_chosen else -1
                        op_hist.append({
                            'Generation': gen,
                            'Current_State': f"S{current_state}",
                            'Action_Selected': f"A{maj_action}",
                            'Reward': round(float(avg_reward), 4),
                            'Diversity_CV': float(current_cv)  # 🚨 专属追踪字段
                        })

                    pbar.update(1)

        return best_individual, op_hist