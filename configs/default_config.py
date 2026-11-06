"""
Default configuration file for the QL-MA BNDFS Project.
Contains all hyperparameters for the network environments, evolutionary engine, and XRL agent.
Uses relative paths referencing the standardized './data/' directory.
"""

# Dictionary containing structural constraints for the 3 benchmark networks
NETWORK_CONFIGS = {
    'mandl': {
        "path": "./data/mandl",  # 15 nodes, small network
        "num_routes": 4,
        "max_fleet": 99,
        "max_stops": 15,
        "min_stops": 3
    },
    'mumford0': {
        "path": "./data/mumford0",  # 30 nodes, medium network
        "num_routes": 12,
        "max_fleet": 150,
        "max_stops": 20,
        "min_stops": 5
    }
}

# Base hyperparameter configuration dictionary
CONFIG = {
    # --- 1. Global Setup ---
    'GLOBAL_SEED': 42,
    'MAX_CPU_WORKERS': 4,             # For a 12-thread CPU, 9 workers push utilization to ~75%-85%
    'MAX_TASKS_PER_CHILD': 10,       # Drastically reduces process restart overhead (restarts only every 150 tasks)
    'EVAL_CHUNKSIZE': 3,              # Balances memory spikes and IPC overhead

    # --- 2. Evolutionary Engine Parameters ---
    'POP_SIZE': 30,
    'MAX_GENERATIONS': 500,
    'ELITISM_RATE': 0.1,
    'CROSSOVER_RATE': 0.8,
    'MUTATION_RATE': 0.2,  # Crucial for Standard GA baseline exploration

    # --- 3. Passenger Simulation Parameters ---
    'MAX_TRANSFERS': 2,
    'TRANSFER_PENALTY_TIME': 5,
    'ALPHA_UNMET_DEMAND': 10000,
    'BUS_CAPACITY': 50,
    'MIN_FREQUENCY': 2,
    'MAX_FREQUENCY': 99,

    # --- 4. Reinforcement Learning (QL-MA) Parameters ---
    'RL_LEARNING_RATE': 0.2,
    'RL_DISCOUNT_FACTOR': 0.9,
    'RL_EPSILON_START': 1.0,
    'RL_EPSILON_END': 0.01,
    'RL_EPSILON_DECAY': 0.985,
    'RL_STAGNATION_THRESHOLD': 0.0001,
    'RL_TEMPERATURE': 0.5,
    'RL_DIVERSITY_THRESHOLD': 0.05
}