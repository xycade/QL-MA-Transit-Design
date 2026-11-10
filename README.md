# An Inherently Interpretable Q-learning Based Memetic Algorithm for Regional Transit Network Optimization

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Reproducibility Guarantee](https://img.shields.io/badge/Reproducibility-100%25-brightgreen.svg)]()

> **Official Implementation** of the QL-MA framework. This repository provides the complete source code, raw datasets, pre-computed execution logs, and automated visualization scripts to ensure absolute algorithmic transparency and reproducibility for the manuscript.

---

## 🌟 Reproducibility & Transparency Guarantee
To address the "black-box" dilemma and ensure strict accountability, this repository is structurally designed with the following rigid mechanisms:
1. **Deterministic Execution**: All random seeds across NumPy, Python `random`, and Multiprocessing workers are strictly locked (`GLOBAL_SEED = 42`). 
2. **Glass-box Auditing**: Every decision made by the RL meta-controller is tracked. The `output/` directory contains the raw Markov Transition Histories (`OpHistory.csv`), recording the exact State, Action, and empirical Reward for every generation across all runs.
3. **Automated Visualization Engine**: The figures presented in the manuscript (Sections 5 & 6) are **not manually illustrated**. They are strictly and directly generated from the raw `.csv` and `.json` execution logs via the provided `analysis/` scripts.

---

## 📁 Comprehensive Repository Structure

Our codebase is highly decoupled and organized into the following modular directories:

*   **`data/`**: Standardized network datasets. Contains three sub-directories:
    *   `mandl/`: Small-scale benchmark network (15 nodes).
    *   `mumford0/`: Medium-scale benchmark network (30 nodes).
    *   `HXUT/`: Real-world micro-transit network of Huaxi University Town (44 nodes).
    *   *Note: Each network strictly comprises three standardized files: `node.csv` (coordinates), `edge.csv` (feasible road segments), and `ODmatrix.csv` (passenger demand).*
*   **`configs/`**: `default_config.py` (Global hyperparameter configurations and physical constraints).
*   **`core/`**: Metaheuristic and RL engine components (`ga_engine.py`, `ql_agent.py`, `counterfactual_engine.py`).
*   **`envs/`**: Transit network graph builder and lower-level passenger assignment simulation.
*   **`operators/`**: Physical routing modification tools (crossover and the four semantic operators).
*   **`utils/`**: Seed management to ensure deterministic execution across multi-processing.
*   **`runners/`**: One-click execution scripts for the four experimental phases.
*   **`analysis/`**: Python scripts for automated manuscript figure generation (XAI & Performance).
*   **`output/`**: Stores all computational outcomes (categorized into tuning, benchmark, XAI, case study, and counterfactual logs).

---

## 🚀 Experimental Workflow (How to Reproduce)
The repository runs on standard Python 3.8+ with common scientific libraries (`numpy`, `pandas`, `networkx`, `matplotlib`, `seaborn`). No complex environment setup is required. 

The experiments in this study are divided into **Four Sequential Phases**. You can reproduce them from scratch by running the scripts in the `runners/` directory:

### Phase 1: Hyperparameter Tuning (Manuscript Appendix C)
Executes a two-stage grid search to calibrate the memetic infrastructure and RL meta-controller.
```bash
python runners/exp_tuning_grid.py
```
### Phase 2: Benchmark Network Testing (Manuscript Section 5)
Evaluates QL-MA against traditional baselines (SGA, RMA, SMA) on the Mandl and Mumford0 networks across 20 independent seeds.
```bash
python runners/run_benchmark.py
```
### Phase 3: Real-World Case Study (Manuscript Section 7)
Applies QL-MA to optimize the micro-transit network of Huaxi University Town (HXUT) under realistic physical constraints.
```bash
python runners/run_casestudy_hxut.py
```
### Phase 4: Counterfactual Intervention (Manuscript Section 6.5)
To strictly prove the agent's causal reasoning without contaminating the main benchmark engine, this script utilizes the isolated counterfactual_engine.py to artificially force a diversity crash at Generation 250.
```bash
python runners/run_counterfactual.py
```
## 📊 Fast-track XAI Auditing (Generate Manuscript Figures)
Note for Reviewers: To facilitate the review process, all raw execution logs from the 4 phases have been pre-computed and stored in the output/ directory. You can directly execute the scripts below to independently verify and generate the exact high-resolution PDF figures presented in the manuscript. All generated figures will be saved in output/analysis_results/.

### Section 5: Optimization Performance
```bash
# Generates Table 2 metrics and Figure 5 (Convergence Trajectories with 95% CI)
python analysis/plot_5_benchmark.py
```
### Section 6 & 7: Interpretability Analysis (XAI) & Case Study
```bash
# Figure 6: Semantic State Space Deconstruction (State densities and consensus actions)
python analysis/plot_6_1_state_space.py

# Figure 7: Policy Determinism Heatmap (Softmax empirical action preferences)
python analysis/plot_6_2_q_table.py

# Figure 8: Value-Function Causality Tracing (Markov Transition Graphs)
python analysis/plot_6_3_markov_graph.py

# Figure 9: Physical Topology Validation (Micro-slices on the Mandl network)
python analysis/plot_6_4_mandl_topology.py

# Figure 10: Counterfactual Intervention Raster Plot (Neuromorphic spike in A3 Explorer)
python analysis/plot_6_5_counterfactual.py

# Parses the best route configuration of HXUT for GIS mapping (Section 7)
python analysis/parse_7_gis.py
```
