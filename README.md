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

