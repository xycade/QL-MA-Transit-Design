import os
import sys
import pandas as pd
from multiprocessing import freeze_support

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from configs.default_config import CONFIG, NETWORK_CONFIGS
from utils.seed_manager import seed_everything
from envs.network_env import TransitNetwork
from envs.passenger_sim import PassengerSimulator

# Import the dedicated counterfactual engine
from core.counterfactual_engine import CounterfactualEngine


def run_counterfactual():
    global_config = CONFIG.copy()

    # Mandatory experimental configuration
    global_config['MAX_GENERATIONS'] = 400
    global_config['PATIENCE'] = 999
    global_config['COUNTERFACTUAL_GEN'] = 100
    global_config['GLOBAL_SEED'] = 99
    global_config['MAX_CPU_WORKERS'] = 10  # Allocate more CPU workers for faster execution

    out_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'output', 'exp_counterfactual'))
    os.makedirs(out_dir, exist_ok=True)

    net_config = NETWORK_CONFIGS['mumford0']
    global_config.update({
        'BASE_PATH': net_config['path'], 'NUM_ROUTES': net_config['num_routes'],
        'MAX_TOTAL_FLEET': net_config['max_fleet'], 'MAX_STOPS_PER_ROUTE': net_config['max_stops'],
        'MIN_STOPS_PER_ROUTE': net_config['min_stops']
    })

    print("🚀 Launching Counterfactual Clone Injection Experiment...")
    seed_everything(global_config['GLOBAL_SEED'])

    network_env = TransitNetwork(base_path=global_config['BASE_PATH'])
    simulator = PassengerSimulator(network_env=network_env, config=global_config)

    # Instantiate dedicated counterfactual engine
    engine = CounterfactualEngine(config=global_config, network_env=network_env, simulator=simulator, mode='QL-MA')

    # Execute run
    _, op_hist = engine.run()

    # Persist logs
    df_op = pd.DataFrame(op_hist)
    csv_path = os.path.join(out_dir, 'OpHistory_Counterfactual.csv')
    df_op.to_csv(csv_path, index=False)
    print(f"✅ Counterfactual data successfully saved to: {csv_path}")


if __name__ == '__main__':
    freeze_support()
    run_counterfactual()