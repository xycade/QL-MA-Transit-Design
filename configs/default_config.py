"""
Default configuration file for the QL-MA Transit Network Design Project.
Contains all hyperparameters for the network environments, evolutionary engine, and Q-learning agent.
Uses relative paths referencing the standardized './data/' directory.
"""

# Dictionary containing structural constraints for the benchmark networks
NETWORK_CONFIGS = {
    'mandl': {
        "path": "./data/mandl",  # 15 nodes, small regional network
        "num_routes": 4,
        "max_fleet": 99,
        "max_stops": 15,
        "min_stops": 3
    },
    'mumford0': {
        "path": "./data/mumford0",  # 30 nodes, medium regional network
        "num_routes": 12,
        "max_fleet": 150,
        "max_stops": 20,
        "min_stops": 5
    }
    # 🚨 Note: mumford1 (70 nodes) is intentionally excluded to perfectly align with the "regional/micro-transit" scope.
}

# Base hyperparameter configuration dictionary
CONFIG = {
    # ==========================================
    # 1. Global Performance & Security Setup
    # ==========================================
    'GLOBAL_SEED': 42,
    'MAX_CPU_WORKERS': 10,             # Safe threshold to prevent CPU thermal throttling and BSOD
    'MAX_TASKS_PER_CHILD': 10,         # Aggressive garbage collection to prevent memory leaks in multiprocess pools
    'EVAL_CHUNKSIZE': 3,              # Balances IPC (Inter-Process Communication) overhead and memory usage

    # ==========================================
    # 2. Evolutionary Engine Parameters (MA Infrastructure)
    # ==========================================
    'POP_SIZE': 30,                   # 🏆 [Stage 1] Optimal Pareto balance between spatial search capacity and agility
    'MAX_GENERATIONS': 500,           # Official benchmark full lifecycle
    'PATIENCE': 100,                  # Early stopping patience
    'ELITISM_RATE': 0.1,
    'CROSSOVER_RATE': 0.9,            # 🏆 [Stage 1] Optimal rate for robust genetic propagation
    'MUTATION_RATE': 0.2,             # Baseline mutation rate reserved explicitly for standard GA baseline

    # ==========================================
    # 3. Passenger Simulation Parameters (BNDFS Objective)
    # ==========================================
    'MAX_TRANSFERS': 2,
    'TRANSFER_PENALTY_TIME': 5,
    'ALPHA_UNMET_DEMAND': 10000,      # Strict penalty for incomplete network coverage
    'BUS_CAPACITY': 50,
    'MIN_FREQUENCY': 2,
    'MAX_FREQUENCY': 99,

    # ==========================================
    # 4. Reinforcement Learning (QL-MA) Meta-Controller Parameters
    # ==========================================
    'RL_LEARNING_RATE': 0.1,          # 🏆 [Stage 2] Optimal alpha: Stable Q-table updates
    'RL_DISCOUNT_FACTOR': 0.95,       # 🏆 [Stage 2] Optimal gamma: Visionary long-term delayed reward tracking
    'RL_TEMPERATURE': 0.1,            # 🏆 [Stage 2] Optimal kappa: Highly greedy exploitation of the Q-table
    'RL_EPSILON_START': 1.0,
    'RL_EPSILON_END': 0.01,
    'RL_EPSILON_DECAY': 0.985,        # Smoothly decays epsilon, shifting from exploration to Q-guided exploitation
    'RL_STAGNATION_THRESHOLD': 0.0001,
    'RL_DIVERSITY_THRESHOLD': 0.015   # 🚨 Crucial calibration: Awakens S0 & S2 states by accurately sensing phenotypic convergence
}