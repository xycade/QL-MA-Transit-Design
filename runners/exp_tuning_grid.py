import os
import sys
import time
import itertools
import gc
import pandas as pd
import numpy as np
from multiprocessing import freeze_support

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from configs.default_config import CONFIG, NETWORK_CONFIGS
from utils.seed_manager import seed_everything
from envs.network_env import TransitNetwork
from envs.passenger_sim import PassengerSimulator
from core.ga_engine import GAEngine


def run_grid_search():
    # ==========================================
    # Fast Hyperparameter Tuning Configuration (Reconnaissance Mode)
    # ==========================================
    network = 'mumford0'
    num_seeds = 2   # Reduced from 3 to 2 for acceleration; sufficient to estimate mean trends
    max_gens = 100  # First 100 generations are sufficient to reveal parameter potential
    patience = 30   # Early stop if no improvement in 30 generations to discard unpromising configurations

    output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'output', 'exp_tuning'))
    os.makedirs(output_dir, exist_ok=True)

    ma_results_file = os.path.join(output_dir, "Tuning_Stage1_MA.csv")
    rl_results_file = os.path.join(output_dir, "Tuning_Stage2_RL.csv")

    base_config = CONFIG.copy()

    base_config['MAX_GENERATIONS'] = max_gens
    base_config['PATIENCE'] = patience

    base_config['MAX_CPU_WORKERS'] = 5
    base_config['MAX_TASKS_PER_CHILD'] = 5
    base_config['EVAL_CHUNKSIZE'] = 5

    net_config = NETWORK_CONFIGS[network]
    base_config.update({
        'BASE_PATH': net_config['path'], 'NUM_ROUTES': net_config['num_routes'],
        'MAX_TOTAL_FLEET': net_config['max_fleet'], 'MAX_STOPS_PER_ROUTE': net_config['max_stops'],
        'MIN_STOPS_PER_ROUTE': net_config['min_stops']
    })

    # Initialize transit network once to avoid repeated memory allocations
    network_env = TransitNetwork(base_path=base_config['BASE_PATH'])

    # ==========================================
    # Stage 1: Memetic Algorithm (MA) Parameter Tuning
    # ==========================================
    ma_grid = {
        'POP_SIZE': [30, 50, 80],
        'CROSSOVER_RATE': [0.7, 0.8, 0.9],
        'MUTATION_RATE': [0.1, 0.2, 0.3]  # All 27 parameter combinations evaluated
    }

    ma_keys = list(ma_grid.keys())
    ma_combinations = list(itertools.product(*(ma_grid[k] for k in ma_keys)))

    print("=" * 70)
    print(f"🚀 STAGE 1: RAPID MA Parameter Tuning | {len(ma_combinations)} Combinations")
    print(f"⚙️  Settings: Max Gens = {max_gens}, Patience = {patience}, Seeds = {num_seeds}")
    print("=" * 70)

    ma_results = []

    for idx, combo in enumerate(ma_combinations):
        current_config = base_config.copy()
        for k, v in zip(ma_keys, combo):
            current_config[k] = v

        print(f"\n[MA {idx + 1}/{len(ma_combinations)}] Testing: {dict(zip(ma_keys, combo))}")

        combo_fitness = []
        for seed in range(1, num_seeds + 1):
            seed_everything(42 + seed)
            current_config['GLOBAL_SEED'] = 42 + seed

            simulator = PassengerSimulator(network_env=network_env, config=current_config)
            engine = GAEngine(config=current_config, network_env=network_env, simulator=simulator, mode='MA-Random')

            best_ind, _, _, _, _, _, _ = engine.run()

            if best_ind:
                best_ind['frequencies'] = simulator.calculate_demand_responsive_frequencies(best_ind)
                metrics = simulator.calculate_performance_metrics(simulator._prepare_network_data(best_ind))
                combo_fitness.append(metrics['fitness'])

            del simulator
            del engine
            gc.collect()
            time.sleep(2)

        avg_fit = np.mean(combo_fitness) if combo_fitness else float('inf')
        print(f"  >>> Avg Fitness for this combo: {avg_fit:,.0f}")

        res_row = dict(zip(ma_keys, combo))
        res_row['Avg_Fitness'] = avg_fit
        ma_results.append(res_row)

        pd.DataFrame(ma_results).to_csv(ma_results_file, index=False)

    df_ma = pd.DataFrame(ma_results)
    best_ma_row = df_ma.loc[df_ma['Avg_Fitness'].idxmin()]

    print("\n" + "=" * 70)
    print("🏆 STAGE 1 COMPLETE. BEST MA PARAMS FOUND:")
    for k in ma_keys:
        val = best_ma_row[k]
        # Cast back to native Python types to avoid errors when Pandas converts integers to floats
        if k == 'POP_SIZE':
            val = int(val)
        else:
            val = float(val)

        print(f"   - {k}: {val}")
        base_config[k] = val
    print("=" * 70)

    # ==========================================
    # Stage 2: Reinforcement Learning (RL) Parameter Tuning
    # ==========================================
    rl_grid = {
        'RL_LEARNING_RATE': [0.1, 0.2, 0.5],
        'RL_DISCOUNT_FACTOR': [0.5, 0.8, 0.95],
        'RL_TEMPERATURE': [0.1, 0.5, 1.0]
    }

    rl_keys = list(rl_grid.keys())
    rl_combinations = list(itertools.product(*(rl_grid[k] for k in rl_keys)))

    print(f"\n🚀 STAGE 2: RAPID RL Parameter Tuning | {len(rl_combinations)} Combinations")
    print("=" * 70)

    rl_results = []

    for idx, combo in enumerate(rl_combinations):
        current_config = base_config.copy()
        for k, v in zip(rl_keys, combo):
            current_config[k] = v

        print(f"\n[RL {idx + 1}/{len(rl_combinations)}] Testing: {dict(zip(rl_keys, combo))}")

        combo_fitness = []
        for seed in range(1, num_seeds + 1):
            seed_everything(42 + seed)
            current_config['GLOBAL_SEED'] = 42 + seed

            simulator = PassengerSimulator(network_env=network_env, config=current_config)
            engine = GAEngine(config=current_config, network_env=network_env, simulator=simulator, mode='QL-MA')

            best_ind, _, _, _, _, _, _ = engine.run()

            if best_ind:
                best_ind['frequencies'] = simulator.calculate_demand_responsive_frequencies(best_ind)
                metrics = simulator.calculate_performance_metrics(simulator._prepare_network_data(best_ind))
                combo_fitness.append(metrics['fitness'])

            del simulator
            del engine
            gc.collect()
            time.sleep(2)

        avg_fit = np.mean(combo_fitness) if combo_fitness else float('inf')
        print(f"  >>> Avg Fitness for this combo: {avg_fit:,.0f}")

        res_row = dict(zip(rl_keys, combo))
        res_row['Avg_Fitness'] = avg_fit
        rl_results.append(res_row)

        pd.DataFrame(rl_results).to_csv(rl_results_file, index=False)

    df_rl = pd.DataFrame(rl_results)
    best_rl_row = df_rl.loc[df_rl['Avg_Fitness'].idxmin()]

    print("\n" + "=" * 70)
    print("🎉 ALL TUNING COMPLETE! 🎉")
    print("🏆 BEST RL PARAMS FOUND:")
    for k in rl_keys:
        val = float(best_rl_row[k])
        print(f"   - {k}: {val}")
    print(f"Results saved to: {output_dir}")
    print("=" * 70)


if __name__ == '__main__':
    freeze_support()
    run_grid_search()