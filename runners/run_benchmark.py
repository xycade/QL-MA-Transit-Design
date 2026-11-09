import os
import sys
import time
import json
import pandas as pd
import numpy as np
from multiprocessing import freeze_support

# Ensure modules from the parent directory can be imported
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from configs.default_config import CONFIG, NETWORK_CONFIGS
from utils.seed_manager import seed_everything
from envs.network_env import TransitNetwork
from envs.passenger_sim import PassengerSimulator
from core.ga_engine import GAEngine

class NumpyEncoder(json.JSONEncoder):
    """ Handle JSON serialization of NumPy data types """
    def default(self, obj):
        if isinstance(obj, np.integer): return int(obj)
        elif isinstance(obj, np.floating): return float(obj)
        elif isinstance(obj, np.ndarray): return obj.tolist()
        return super(NumpyEncoder, self).default(obj)


# Fault-tolerant write functions: prevent crashes caused by Windows file locking (e.g., opened in Excel)
def safe_append_csv(filepath, line_str):
    while True:
        try:
            with open(filepath, 'a') as f:
                f.write(line_str)
            break
        except PermissionError:
            print(f"\n⚠️ Warning: Write failed! File {filepath} is currently locked by Excel/WPS. Please close the file! Retrying in 5 seconds...")
            time.sleep(5)


def safe_save_df(df, filepath):
    while True:
        try:
            df.to_csv(filepath, index=False)
            break
        except PermissionError:
            print(f"\n⚠️ Warning: Write failed! File {filepath} is currently locked by Excel/WPS. Please close the file! Retrying in 5 seconds...")
            time.sleep(5)


def run_benchmark_experiment(test_run=True):
    # ==========================================
    # Benchmark Experiment Configuration
    # ==========================================
    networks = ['mandl', 'mumford0']
    modes = ['GA', 'RMA', 'SMA', 'QL-MA']
    total_seeds = 20

    # Inherit default parameters from default_config.py
    global_config = CONFIG.copy()

    # Dynamic experiment scale and seed control (executes full generational runs)
    if test_run:
        seed_range = range(1, 2)  # Run Seed 1 only
        print("=" * 80)
        print(f"🛠️  TEST RUN MODE | Running Seed 1 ONLY (Full {global_config['MAX_GENERATIONS']} Gens) to verify real outcomes.")
        print("=" * 80)
    else:
        seed_range = range(2, total_seeds + 1)  # Run remaining Seeds 2 to 20
        print("=" * 80)
        print(f"🚀 FULL BENCHMARK MODE | Running remaining {total_seeds - 1} Seeds (Seed 2 to {total_seeds}).")
        print(f"📊 Settings: Max Gens = {global_config['MAX_GENERATIONS']}, Patience = {global_config['PATIENCE']}")
        print("=" * 80)

    # ==========================================
    # Output Directory Initialization
    # ==========================================
    output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'output', 'exp_benchmark'))
    xai_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'output', 'exp_benchmark_xai'))
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(xai_dir, exist_ok=True)

    perf_csv = os.path.join(output_dir, "Table_Performance_Summary.csv")
    time_csv = os.path.join(output_dir, "Table_Computational_Overhead.csv")
    traj_csv = os.path.join(output_dir, "Fitness_Trajectory.csv")

    # Initialize CSV headers (write only if file does not exist, preserving header integrity on resumption)
    for path, header in [
        (perf_csv, "Network,Mode,Seed,Final_Fitness,Unmet_Demand,Total_Travel_Time,Direct_Rate,Converged_Generation\n"),
        (time_csv, "Network,Mode,Seed,Total_Time_Sec,Eval_Time_Sec,Operator_Time_Sec,RL_Time_Sec\n"),
        (traj_csv, "Network,Mode,Seed," + ",".join([f"Gen_{i}" for i in range(global_config['MAX_GENERATIONS'])]) + "\n")
    ]:
        if not os.path.exists(path):
            with open(path, 'w') as f: f.write(header)

    # ==========================================
    # Core Two-Tier Experimental Loop
    # ==========================================
    for network in networks:
        net_config = NETWORK_CONFIGS[network]
        global_config.update({
            'BASE_PATH': net_config['path'], 'NUM_ROUTES': net_config['num_routes'],
            'MAX_TOTAL_FLEET': net_config['max_fleet'], 'MAX_STOPS_PER_ROUTE': net_config['max_stops'],
            'MIN_STOPS_PER_ROUTE': net_config['min_stops']
        })

        print(f"\n[Environment] Instantiating Transit Network for {network.upper()}...")
        network_env = TransitNetwork(base_path=global_config['BASE_PATH'])
        total_demand = network_env.total_demand

        for mode in modes:
            for seed in seed_range:
                print(f"\n  >>> Executing [ {network.upper()} | {mode} | Seed: {seed}/{total_seeds} ]")

                current_seed = 42 + seed
                global_config['GLOBAL_SEED'] = current_seed
                seed_everything(current_seed)

                simulator = PassengerSimulator(network_env=network_env, config=global_config)
                engine = GAEngine(config=global_config, network_env=network_env, simulator=simulator, mode=mode)

                t_total_start = time.perf_counter()
                best_ind, fit_hist, op_hist, q_snaps, timings, actual_gens, micro_slice = engine.run()
                total_time = time.perf_counter() - t_total_start

                # ==========================================
                # Metrics Computation and Storage
                # ==========================================
                final_fitness, unmet, travel_time, direct_rate = -1, -1, -1, -1
                if best_ind:
                    best_ind['frequencies'] = simulator.calculate_demand_responsive_frequencies(best_ind)
                    metrics = simulator.calculate_performance_metrics(simulator._prepare_network_data(best_ind))
                    final_fitness = metrics['fitness']
                    unmet = metrics['unmet_passengers']
                    travel_time = metrics['total_passenger_time_cost']
                    direct_rate = metrics['passengers_0_transfer'] / total_demand

                # Persist master records using fault-tolerant write functions
                perf_line = f"{network},{mode},{current_seed},{final_fitness},{unmet},{travel_time},{direct_rate:.4f},{actual_gens}\n"
                safe_append_csv(perf_csv, perf_line)

                time_line = f"{network},{mode},{current_seed},{total_time:.2f},{timings['eval_time']:.2f},{timings['op_time']:.2f},{timings['rl_time']:.2f}\n"
                safe_append_csv(time_csv, time_line)

                # Pad trajectory length to align CSV columns
                max_gens_current = global_config['MAX_GENERATIONS']
                if len(fit_hist) < max_gens_current:
                    padding_length = max_gens_current - len(fit_hist)
                    fit_hist.extend([fit_hist[-1]] * padding_length)
                traj_line = f"{network},{mode},{current_seed}," + ",".join(
                    [f"{val:.2f}" for val in fit_hist[:max_gens_current]]) + "\n"
                safe_append_csv(traj_csv, traj_line)

                # Save route JSON topology
                if best_ind:
                    route_log_path = os.path.join(output_dir, f"Routes_{network}_{mode}_Seed_{current_seed}.json")
                    with open(route_log_path, 'w') as jf:
                        json.dump(best_ind, jf, cls=NumpyEncoder)

                # ==========================================
                # XAI Data Capture (QL-MA Only)
                # ==========================================
                if mode == 'QL-MA':
                    if op_hist:
                        df_op = pd.DataFrame(op_hist)
                        safe_save_df(df_op, os.path.join(xai_dir, f"OpHistory_{network}_Seed_{current_seed}.csv"))

                    if q_snaps:
                        q_snap_path = os.path.join(xai_dir, f"QSnapshots_{network}_Seed_{current_seed}.json")
                        with open(q_snap_path, 'w') as f:
                            json.dump(q_snaps, f, cls=NumpyEncoder, indent=4)

                    if micro_slice:
                        m_slice_path = os.path.join(xai_dir, f"MicroSlice_{network}_Seed_{current_seed}.json")
                        with open(m_slice_path, 'w') as f:
                            json.dump(micro_slice, f, cls=NumpyEncoder, indent=4)

                print(
                    f"      -> Done! Fit: {final_fitness:,.0f} | Converged Gen: {actual_gens} | Time: {total_time:.1f}s")

    print("\n✅ BENCHMARK EXECUTION COMPLETE. DATA SECURED.")

if __name__ == '__main__':
    freeze_support()
    # Set to True to run only 1 seed across all algorithms for full 500 generations to verify results.
    # Set to False once verified to automatically resume and run the remaining 19 seeds from Seed 2.
    run_benchmark_experiment(test_run=False)