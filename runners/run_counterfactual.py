import os
import sys
import pandas as pd
from multiprocessing import freeze_support

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from configs.default_config import CONFIG, NETWORK_CONFIGS
from utils.seed_manager import seed_everything
from envs.network_env import TransitNetwork
from envs.passenger_sim import PassengerSimulator

# 🚨 引入我们刚刚新建的专属引擎
from core.counterfactual_engine import CounterfactualEngine


def run_counterfactual():
    global_config = CONFIG.copy()

    # 强制实验配置
    global_config['MAX_GENERATIONS'] = 400
    global_config['PATIENCE'] = 999
    global_config['COUNTERFACTUAL_GEN'] = 100
    global_config['GLOBAL_SEED'] = 99
    global_config['MAX_CPU_WORKERS'] = 10  # 放开算力，跑得快点

    out_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'output', 'exp_counterfactual'))
    os.makedirs(out_dir, exist_ok=True)

    net_config = NETWORK_CONFIGS['mumford0']
    global_config.update({
        'BASE_PATH': net_config['path'], 'NUM_ROUTES': net_config['num_routes'],
        'MAX_TOTAL_FLEET': net_config['max_fleet'], 'MAX_STOPS_PER_ROUTE': net_config['max_stops'],
        'MIN_STOPS_PER_ROUTE': net_config['min_stops']
    })

    print("🚀 启动反事实干预实验 (Counterfactual Clone Injection) ...")
    seed_everything(global_config['GLOBAL_SEED'])

    network_env = TransitNetwork(base_path=global_config['BASE_PATH'])
    simulator = PassengerSimulator(network_env=network_env, config=global_config)

    # 使用新引擎
    engine = CounterfactualEngine(config=global_config, network_env=network_env, simulator=simulator, mode='QL-MA')

    # 运行
    _, op_hist = engine.run()

    # 保存日志
    df_op = pd.DataFrame(op_hist)
    csv_path = os.path.join(out_dir, 'OpHistory_Counterfactual.csv')
    df_op.to_csv(csv_path, index=False)
    print(f"✅ 反事实数据已安全保存至: {csv_path}")


if __name__ == '__main__':
    freeze_support()
    run_counterfactual()