import os
import sys
import time
import json
import pandas as pd
import numpy as np
from multiprocessing import freeze_support

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from configs.default_config import CONFIG
from utils.seed_manager import seed_everything
from envs.network_env import TransitNetwork
from envs.passenger_sim import PassengerSimulator
from core.ga_engine import GAEngine

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer): return int(obj)
        elif isinstance(obj, np.floating): return float(obj)
        elif isinstance(obj, np.ndarray): return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

def safe_append_csv(filepath, line_str):
    while True:
        try:
            with open(filepath, 'a') as f:
                f.write(line_str)
            break
        except PermissionError:
            print(f"\n⚠️ Warning: Write failed! File {filepath} is currently locked by another process. Retrying in 5 seconds...")
            time.sleep(5)

def safe_save_df(df, filepath):
    while True:
        try:
            df.to_csv(filepath, index=False)
            break
        except PermissionError:
            print(f"\n⚠️ Warning: Write failed! File {filepath} is currently locked by another process. Retrying in 5 seconds...")
            time.sleep(5)

def format_hxut_data(base_path):
    print("🔄 Checking and adapting HXUT data formats...")
    if os.path.exists(os.path.join(base_path, 'demand.txt')) and os.path.exists(os.path.join(base_path, 'edges.txt')):
        print("✅ Standard data files already exist. Skipping format conversion.")
        return

    df_edge = pd.read_csv(os.path.join(base_path, 'edge.csv'))
    df_edge.to_csv(os.path.join(base_path, 'edges.txt'), sep=' ', index=False, header=False)

    df_od = pd.read_csv(os.path.join(base_path, 'ODmatrix.csv'))
    max_node = max(df_od['OriginID'].max(), df_od['DestinationID'].max())

    demand_matrix = np.zeros((int(max_node) + 1, int(max_node) + 1))
    for _, row in df_od.iterrows():
        o, d, pd_val = int(row['OriginID']), int(row['DestinationID']), float(row['PD'])
        demand_matrix[o][d] = pd_val

    np.savetxt(os.path.join(base_path, 'demand.txt'), demand_matrix, fmt='%.2f', delimiter=' ')

    if os.path.exists(os.path.join(base_path, 'node.csv')):
        try:
            df_node = pd.read_csv(os.path.join(base_path, 'node.csv'))
            cols = df_node.columns.tolist()
            df_node = df_node[[cols[0], cols[1], cols[2]]]
            df_node.to_csv(os.path.join(base_path, 'coords.txt'), sep=' ', index=False, header=False)
        except Exception as e:
            print(f"⚠️ Coordinate conversion notice (can be ignored): {e}")

    print("✅ HXUT data format adaptation complete. Standard txt files generated!")


def run_hxut_case_study(test_run=True):
    # ==========================================
    # Empirical Case Study Configuration (Includes Test Run Capability)
    # ==========================================
    global_config = CONFIG.copy()
    total_seeds = 10

    # Dynamically control experimental scale (full unperturbed evolutionary cycle)
    if test_run:
        seed_range = range(1, 2)
        print("=" * 80)
        print(f"🛠️  TEST RUN MODE | Running Seed 1 ONLY (Full {global_config['MAX_GENERATIONS']} Gens) to verify HXUT real outcomes.")
        print("=" * 80)
    else:
        seed_range = range(2, total_seeds + 1)
        print("=" * 80)
        print(f"🏙️  REAL-WORLD CASE STUDY | HUAXI UNIVERSITY TOWN (HXUT)")
        print(f"📊 Running QL-MA for remaining {total_seeds - 1} Seeds (Seed 2 to {total_seeds}).")
        print("=" * 80)

    # Apply real-world operational constraints tailored for Huaxi University Town (HXUT)
    hxut_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'HXUT'))
    global_config.update({
        'BASE_PATH': hxut_path,
        'NUM_ROUTES': 10,
        'MAX_TOTAL_FLEET': 100,
        'MAX_STOPS_PER_ROUTE': 25,
        'MIN_STOPS_PER_ROUTE': 5
    })

    # Automatically adapt data formats
    format_hxut_data(hxut_path)

    # Instantiate environment
    network_env = TransitNetwork(base_path=global_config['BASE_PATH'])
    total_demand = network_env.total_demand

    # Prepare output directories
    output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'output', 'exp_casestudy_hxut'))
    xai_dir = os.path.join(output_dir, 'xai_data')
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(xai_dir, exist_ok=True)

    perf_csv = os.path.join(output_dir, "HXUT_Performance_Summary.csv")
    if not os.path.exists(perf_csv):
        with open(perf_csv, 'w') as f:
            f.write("Seed,Final_Fitness,Unmet_Demand,Total_Travel_Time,Direct_Rate,Converged_Generation,Total_Time_Sec\n")

    best_overall_fitness = float('inf')
    best_overall_seed = -1

    # ==========================================
    # Core Empirical Loop (QL-MA Only)
    # ==========================================
    for seed in seed_range:
        print(f"\n  >>> Executing [ HXUT | QL-MA | Seed: {seed}/{total_seeds} ]")

        current_seed = 42 + seed
        global_config['GLOBAL_SEED'] = current_seed
        seed_everything(current_seed)

        simulator = PassengerSimulator(network_env=network_env, config=global_config)
        engine = GAEngine(config=global_config, network_env=network_env, simulator=simulator, mode='QL-MA')

        t_total_start = time.perf_counter()
        best_ind, fit_hist, op_hist, q_snaps, timings, actual_gens, micro_slice = engine.run()
        total_time = time.perf_counter() - t_total_start

        # Compute performance metrics
        if best_ind:
            best_ind['frequencies'] = simulator.calculate_demand_responsive_frequencies(best_ind)
            metrics = simulator.calculate_performance_metrics(simulator._prepare_network_data(best_ind))

            final_fitness = metrics['fitness']
            unmet = metrics['unmet_passengers']
            travel_time = metrics['total_passenger_time_cost']
            direct_rate = metrics['passengers_0_transfer'] / total_demand

            if final_fitness < best_overall_fitness:
                best_overall_fitness = final_fitness
                best_overall_seed = current_seed

            perf_line = f"{current_seed},{final_fitness},{unmet},{travel_time},{direct_rate:.4f},{actual_gens},{total_time:.2f}\n"
            safe_append_csv(perf_csv, perf_line)

            route_log_path = os.path.join(output_dir, f"HXUT_Routes_Seed_{current_seed}.json")
            with open(route_log_path, 'w') as jf:
                json.dump(best_ind, jf, cls=NumpyEncoder)

            if op_hist:
                df_op = pd.DataFrame(op_hist)
                safe_save_df(df_op, os.path.join(xai_dir, f"HXUT_OpHistory_Seed_{current_seed}.csv"))
            if q_snaps:
                with open(os.path.join(xai_dir, f"HXUT_QSnapshots_Seed_{current_seed}.json"), 'w') as f:
                    json.dump(q_snaps, f, cls=NumpyEncoder, indent=4)

        print(f"      -> Done! Fit: {final_fitness:,.0f} | Unmet: {unmet} | Time: {total_time:.1f}s")

    print("\n" + "=" * 80)
    print("✅ CASE STUDY EXECUTION COMPLETE.")
    if not test_run:
        print(f"🏆 BEST ROUTE PLAN FOUND IN SEED: {best_overall_seed} (Fitness: {best_overall_fitness:,.0f})")
        print(f"👉 Please use 'HXUT_Routes_Seed_{best_overall_seed}.json' for ArcGIS spatial visualization!")
    print("=" * 80)


if __name__ == '__main__':
    freeze_support()
    # Set to True to verify convergence quality across a full 500-generation run on a single seed.
    # Set to False once verified to run the remaining 9 seeds automatically from Seed 2 onwards.
    run_hxut_case_study(test_run=True)