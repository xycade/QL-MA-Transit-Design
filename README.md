# An Inherently Interpretable Q-learning Based Memetic Algorithm for Regional Transit Network Optimization

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Reproducibility Guarantee](https://img.shields.io/badge/Reproducibility-100%25-brightgreen.svg)]()

> **Official Implementation** of the QL-MA framework. This repository contains the complete source code, benchmark datasets, real-world case study data (Huaxi University Town, HXUT), and all automated visualization scripts used in the manuscript.

## 🌟 Reproducibility & Transparency Guarantee
To ensure **absolute reproducibility** and effectively address any concerns regarding algorithmic transparency or data integrity, this repository is structurally designed with the following rigid mechanisms:
1. **Deterministic Execution**: All random seeds across NumPy, Python `random`, and Multiprocessing workers are strictly locked (Base Seed: 42, configured in `configs/default_config.py`). 
2. **Glass-box Auditing**: Every single decision made by the RL meta-controller is tracked. The `output/` directory logs the Markov Transition History (`OpHistory.csv`), recording the exact State, Action, and empirical Reward for every generation across all runs.
3. **Automated Visualization Engine**: The figures in the manuscript (Sections 5 & 6) are not manually illustrated. They are strictly and directly generated from the raw `.csv` and `.json` execution logs via the scripts provided in the `analysis/` folder.

## 📁 Repository Structure
*   `configs/`: Hyperparameter settings and structural constraints (`default_config.py`).
*   `core/`: Core evolutionary engine and Tabular Q-learning meta-controller, including the physically isolated `counterfactual_engine.py`.
*   `envs/`: Transit network graphical environments and bi-level passenger simulation.
*   `analysis/`: Python scripts for automated manuscript figure generation.
*   `runners/`: One-click execution scripts for benchmarks, case studies, and counterfactual interventions.
*   `output/`: Pre-computed experimental logs and parsed topologies (JSON/CSV) for direct auditing.
*   `data/`: Underlying network graphs (Mandl, Mumford0) and real-world HXUT spatial nodes/OD matrices.

## 🛠️ Installation

# Clone the repository
git clone https://github.com/YourName/QL-MA-Transit-Design.git
cd QL-MA-Transit-Design

# Install dependencies (strictly version-locked)
pip install -r requirements.txt

🚀 How to Reproduce the Experiments
1. Benchmark Testing (Mandl & Mumford0)
To reproduce the optimization results, structural feasibility, and convergence tracking (Section 5). This script automatically aggregates 20 independent seeds.

python runners/run_benchmark.py
Note: The script utilizes safe multi-processing and supports breakpoint continuation. Results (Fitness metrics, topologies, and XAI logs) will be automatically saved to output/exp_benchmark/.

2. Real-world Case Study (HXUT)
To reproduce the Huaxi University Town micro-transit optimization (Section 7):

python runners/run_casestudy_hxut.py

3. Causal Verification: Counterfactual Intervention
To strictly prove the agent's causal reasoning without contaminating the main benchmark engine, we utilize a physically isolated engine (core/counterfactual_engine.py). This engine artificially executes a 'Clone Injection' at Generation 250 to force a diversity crash (Section 6.5).

python runners/run_counterfactual.py

4. Generate Manuscript Figures (Automated Analysis)
To verify that all plots and analytical conclusions in the manuscript are rigorously derived from the raw experimental data, we provide a suite of independent visualization scripts in the analysis/ folder.
📊 Section 5: Optimization Performance

# Generates Table 2 metrics and Figure 5 (Convergence Trajectories with 95% CI)
python analysis/plot_section5_benchmark.py
🧠 Section 6: Interpretability Analysis (XAI)
The Explainable AI (XAI) analysis is decoupled into 5 independent scripts, mapping directly to the manuscript's logical flow. Running these scripts will parse the OpHistory and QSnapshots logs to generate high-resolution PDF vectors:

# 1. State Space Deconstruction (Section 6.1)
# Visualizes how population diversity and stagnation are clustered into 4 semantic quadrants.
python analysis/plot_6_1_state_space.py

# 2. Policy Determinism Heatmap (Section 6.2)
# Generates the 4x4 Q-Table heatmap, explicitly decoding the agent's scale-adaptive rules.
python analysis/plot_6_2_q_table.py

# 3. Value-Function Causality Tracing (Section 6.3)
# Constructs a Markov Transition Graph from execution logs, showing action-driven state flows.
python analysis/plot_6_3_markov_graph.py

# 4. Physical Topology Validation (Section 6.4)
# Visualizes the 'Micro-slice' on the Mandl network, showing the exact physical edges severed and reconnected by the heuristic operators (Before vs. After).
python analysis/plot_6_4_mandl_topology.py

# 5. Counterfactual Intervention Raster Plot (Section 6.5)
# Plots the neuromorphic event raster showing the agent's immediate causal reflex (burst of A3 Explorer) in response to the artificial 'Clone Injection'.
python analysis/plot_6_5_counterfactual.py
