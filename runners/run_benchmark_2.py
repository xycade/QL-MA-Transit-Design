import os
import sys
import time
import json
import pandas as pd
import numpy as np
from multiprocessing import freeze_support

# 确保能导入父目录的模块
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from configs.default_config import CONFIG, NETWORK_CONFIGS
from utils.seed_manager import seed_everything
from envs.network_env import TransitNetwork
from envs.passenger_sim import PassengerSimulator
from core.ga_engine import GAEngine


class NumpyEncoder(json.JSONEncoder):
    """ 处理 Numpy 数据类型的 JSON 序列化 """

    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)


# 🛡️ 防弹写入函数：防止 Windows 文件占用导致崩溃
def safe_append_csv(filepath, line_str):
    while True:
        try:
            with open(filepath, 'a') as f:
                f.write(line_str)
            break
        except PermissionError:
            print(f"\n⚠️ 警告: 写入失败！文件 {filepath} 正在被 Excel/WPS 占用。请关闭该文件！5秒后自动重试...")
            time.sleep(5)


def safe_save_df(df, filepath):
    while True:
        try:
            df.to_csv(filepath, index=False)
            break
        except PermissionError:
            print(f"\n⚠️ 警告: 写入失败！文件 {filepath} 正在被 Excel/WPS 占用。请关闭该文件！5秒后自动重试...")
            time.sleep(5)


def run_benchmark_experiment(test_run=True):
    # ==========================================
    # 🚀 实验基础配置
    # ==========================================
    networks = ['mandl', 'mumford0']
    modes = ['GA', 'RMA', 'SMA', 'QL-MA']
    total_seeds = 20

    # 全盘继承 default_config.py 中的黄金参数
    global_config = CONFIG.copy()

    # 动态实验规模与种子控制
    if test_run:
        seed_range = range(1, 2)  # 只跑 Seed 1
        print("=" * 80)
        print(f"🛠️  TEST RUN MODE | Running Seed 1 ONLY (Full {global_config['MAX_GENERATIONS']} Gens).")
        print("=" * 80)
    else:
        seed_range = range(1, total_seeds + 1)  # 从 1 到 20 (断点逻辑会自动跳过已跑完的)
        print("=" * 80)
        print(f"🚀 FULL BENCHMARK MODE | Scanning and running up to {total_seeds} Seeds.")
        print(f"📊 Settings: Max Gens = {global_config['MAX_GENERATIONS']}, Patience = {global_config['PATIENCE']}")
        print("=" * 80)

    # ==========================================
    # 📁 输出目录初始化
    # ==========================================
    output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'output', 'exp_benchmark'))
    xai_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'output', 'exp_benchmark_xai'))
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(xai_dir, exist_ok=True)

    perf_csv = os.path.join(output_dir, "Table_Performance_Summary.csv")
    time_csv = os.path.join(output_dir, "Table_Computational_Overhead.csv")
    traj_csv = os.path.join(output_dir, "Fitness_Trajectory.csv")

    # 初始化表头
    for path, header in [
        (perf_csv, "Network,Mode,Seed,Final_Fitness,Unmet_Demand,Total_Travel_Time,Direct_Rate,Converged_Generation\n"),
        (time_csv, "Network,Mode,Seed,Total_Time_Sec,Eval_Time_Sec,Operator_Time_Sec,RL_Time_Sec\n"),
        (traj_csv,
         "Network,Mode,Seed," + ",".join([f"Gen_{i}" for i in range(global_config['MAX_GENERATIONS'])]) + "\n")
    ]:
        if not os.path.exists(path):
            with open(path, 'w') as f: f.write(header)

    # ==========================================
    # 🔍 核心优化：断点续传扫描 (智能跳过)
    # ==========================================
    completed_tasks = set()
    if os.path.exists(perf_csv):
        try:
            df_existing = pd.read_csv(perf_csv)
            for _, row in df_existing.iterrows():
                completed_tasks.add((row['Network'], row['Mode'], int(row['Seed'])))
            print(f"🔍 [断点扫描] 检测到 {len(completed_tasks)} 个已完成的实验，启动时将自动跳过！\n")
        except Exception as e:
            print(f"⚠️ 历史记录解析异常，将忽略断点: {e}")

    # ==========================================
    # 🏃 核心双层实验循环
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
                current_seed = 42 + seed

                # 🛡️ 拦截器：如果在已完成集合中，直接跳过！
                if (network, mode, current_seed) in completed_tasks:
                    print(f"  ⏭️ [跳过] {network.upper()} | {mode} | Seed: {current_seed} 已完成。")
                    continue

                print(
                    f"\n  >>> Executing [ {network.upper()} | {mode} | Seed: {seed}/{total_seeds} (Actual Seed: {current_seed}) ]")

                global_config['GLOBAL_SEED'] = current_seed
                seed_everything(current_seed)

                simulator = PassengerSimulator(network_env=network_env, config=global_config)
                engine = GAEngine(config=global_config, network_env=network_env, simulator=simulator, mode=mode)

                t_total_start = time.perf_counter()
                best_ind, fit_hist, op_hist, q_snaps, timings, actual_gens, micro_slice = engine.run()
                total_time = time.perf_counter() - t_total_start

                # ==========================================
                # 📈 指标计算与存储
                # ==========================================
                final_fitness, unmet, travel_time, direct_rate = -1, -1, -1, -1
                if best_ind:
                    best_ind['frequencies'] = simulator.calculate_demand_responsive_frequencies(best_ind)
                    metrics = simulator.calculate_performance_metrics(simulator._prepare_network_data(best_ind))
                    final_fitness = metrics['fitness']
                    unmet = metrics['unmet_passengers']
                    travel_time = metrics['total_passenger_time_cost']
                    direct_rate = metrics['passengers_0_transfer'] / total_demand

                # 使用防弹函数保存主表
                perf_line = f"{network},{mode},{current_seed},{final_fitness},{unmet},{travel_time},{direct_rate:.4f},{actual_gens}\n"
                safe_append_csv(perf_csv, perf_line)

                time_line = f"{network},{mode},{current_seed},{total_time:.2f},{timings['eval_time']:.2f},{timings['op_time']:.2f},{timings['rl_time']:.2f}\n"
                safe_append_csv(time_csv, time_line)

                # 补齐轨迹长度以对齐 CSV 列
                max_gens_current = global_config['MAX_GENERATIONS']
                if len(fit_hist) < max_gens_current:
                    padding_length = max_gens_current - len(fit_hist)
                    fit_hist.extend([fit_hist[-1]] * padding_length)
                traj_line = f"{network},{mode},{current_seed}," + ",".join(
                    [f"{val:.2f}" for val in fit_hist[:max_gens_current]]) + "\n"
                safe_append_csv(traj_csv, traj_line)

                # 路线 JSON 拓扑保存
                if best_ind:
                    route_log_path = os.path.join(output_dir, f"Routes_{network}_{mode}_Seed_{current_seed}.json")
                    with open(route_log_path, 'w') as jf:
                        json.dump(best_ind, jf, cls=NumpyEncoder)

                # ==========================================
                # 🧠 XAI 数据捕获 (仅限 QL-MA)
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
    # 💡 确保这里是 False！这样就会执行完整的 range(1, 21)，并自动过滤掉已完成的！
    run_benchmark_experiment(test_run=False)