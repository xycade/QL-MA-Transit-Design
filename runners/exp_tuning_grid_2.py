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


def run_stage2_grid_search():
    # ==========================================
    # 🚀 极速调参实验配置 (仅 Stage 2)
    # ==========================================
    network = 'mumford0'
    num_seeds = 2  # [提速] 2 个种子足以评估均值趋势
    max_gens = 100  # [提速] 前 100 代足以看出参数潜力
    patience = 30  # [提速] 30代不下降直接淘汰

    output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'output', 'exp_tuning'))
    os.makedirs(output_dir, exist_ok=True)

    rl_results_file = os.path.join(output_dir, "Tuning_Stage2_RL.csv")

    base_config = CONFIG.copy()

    # 🛡️ 性能与安全平衡设置
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

    # ==========================================
    # 🏆 注入 Stage 1 找到的最优 MA 参数
    # ==========================================
    base_config['POP_SIZE'] = 80
    base_config['CROSSOVER_RATE'] = 0.9
    base_config['MUTATION_RATE'] = 0.2

    print("=" * 70)
    print("⏭️  SKIPPING STAGE 1... Using Best MA Params: POP_SIZE=80, CROSSOVER_RATE=0.9")
    print("=" * 70)

    # 网络环境只初始化一次
    network_env = TransitNetwork(base_path=base_config['BASE_PATH'])

    # ==========================================
    # Stage 2: RL 参数调优 (带断点续传功能)
    # ==========================================
    rl_grid = {
        'RL_LEARNING_RATE': [0.1, 0.2, 0.5],
        'RL_DISCOUNT_FACTOR': [0.5, 0.8, 0.95],
        'RL_TEMPERATURE': [0.1, 0.5, 1.0]
    }

    rl_keys = list(rl_grid.keys())
    all_rl_combinations = list(itertools.product(*(rl_grid[k] for k in rl_keys)))
    total_combos = len(all_rl_combinations)

    # 🚨 断点续传核心逻辑
    rl_results = []
    completed_combos = set()

    if os.path.exists(rl_results_file):
        print(f"🔄 检测到历史记录文件: {rl_results_file}")
        try:
            existing_df = pd.read_csv(rl_results_file)
            rl_results = existing_df.to_dict('records')  # 将旧数据载入内存

            # 提取已经完成的组合（保留4位小数以防浮点数精度匹配失败）
            for _, row in existing_df.iterrows():
                combo = tuple(round(float(row[k]), 4) for k in rl_keys)
                completed_combos.add(combo)
            print(f"✅ 成功读取 {len(completed_combos)} 组已完成的参数组合。")
        except Exception as e:
            print(f"⚠️ 读取历史记录失败，将重新开始。错误信息: {e}")

    # 过滤出还没跑的组合
    remaining_combinations = []
    for combo in all_rl_combinations:
        combo_rounded = tuple(round(float(val), 4) for val in combo)
        if combo_rounded not in completed_combos:
            remaining_combinations.append(combo)

    completed_count = total_combos - len(remaining_combinations)

    print(f"\n🚀 STAGE 2: RAPID RL Parameter Tuning | {len(remaining_combinations)} Remaining / {total_combos} Total")
    print(f"⚙️  Settings: Max Gens = {max_gens}, Patience = {patience}, Seeds = {num_seeds}")
    print("=" * 70)

    if len(remaining_combinations) == 0:
        print("🎉 所有参数组合均已跑完，无需继续！")
    else:
        for idx, combo in enumerate(remaining_combinations):
            current_config = base_config.copy()
            for k, v in zip(rl_keys, combo):
                current_config[k] = v

            current_progress = completed_count + idx + 1
            print(f"\n[RL {current_progress}/{total_combos}] Testing: {dict(zip(rl_keys, combo))}")

            combo_fitness = []
            for seed in range(1, num_seeds + 1):
                seed_everything(42 + seed)
                current_config['GLOBAL_SEED'] = 42 + seed

                simulator = PassengerSimulator(network_env=network_env, config=current_config)
                # 确保这里用的是我们新定的名字 QL-MA
                engine = GAEngine(config=current_config, network_env=network_env, simulator=simulator, mode='QL-MA')

                best_ind, _, _, _, _, _, _ = engine.run()

                if best_ind:
                    best_ind['frequencies'] = simulator.calculate_demand_responsive_frequencies(best_ind)
                    metrics = simulator.calculate_performance_metrics(simulator._prepare_network_data(best_ind))
                    combo_fitness.append(metrics['fitness'])

                # 🛡️ 强制垃圾回收与冷却休眠
                del simulator
                del engine
                gc.collect()
                time.sleep(2)

            avg_fit = np.mean(combo_fitness) if combo_fitness else float('inf')
            print(f"  >>> Avg Fitness for this combo: {avg_fit:,.0f}")

            res_row = dict(zip(rl_keys, combo))
            res_row['Avg_Fitness'] = avg_fit
            rl_results.append(res_row)  # 追加新结果到总列表

            # 直接覆盖保存完整的列表，这比 'a' 追加模式更安全，能保证表头不乱
            pd.DataFrame(rl_results).to_csv(rl_results_file, index=False)

    # 最终结算，找出 27 组里最好的一组
    if len(rl_results) > 0:
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
    run_stage2_grid_search()