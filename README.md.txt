# An Inherently Interpretable Q-learning Based Memetic Algorithm for Regional Transit Network Optimization

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Reproducibility Guarantee](https://img.shields.io/badge/Reproducibility-100%25-brightgreen.svg)]()

> **Official Implementation** of the QL-MA framework. This repository contains the complete source code, benchmark datasets, real-world case study data (HXUT), and all visualization scripts used in the manuscript.

## 🌟 Reproducibility & Transparency Guarantee
To ensure **absolute reproducibility** and complete algorithmic transparency, this repository is designed with the following mechanisms:
1. **Deterministic Seeds**: All random seeds across NumPy, Python `random`, and Multiprocessing workers are strictly locked (Base Seed: 42). 
2. **Glass-box Auditing**: Every single decision made by the RL meta-controller is tracked. The `output/` directory logs the Markov Transition History (`OpHistory.csv`), recording the State, Action, and Reward for every generation.
3. **Automated Visualization**: The figures in the manuscript (Section 5 & 6) are not manually drawn. They are directly generated from the raw `.csv` execution logs via the scripts in the `analysis/` folder.

## 🛠️ Installation
```bash
# Clone the repository
git clone https://github.com/YourName/QL-MA-Transit-Design.git
cd QL-MA-Transit-Design

# Install dependencies (strictly version-locked)
pip install -r requirements.txt

## 🚀 How to Reproduce the Experiments
1. Benchmark Testing (Mandl & Mumford0)
To reproduce the optimization results and computational time profiling (Section 5):
python runners/run_benchmark.py
Note: The script utilizes multi-processing. Results (Fitness metrics, topologies, and XAI logs) will be automatically saved to output/exp_benchmark/.

2. Real-world Case Study (HXUT)
To reproduce the Huaxi University Town micro-transit optimization (Section 7):
python runners/run_casestudy_hxut.py

3. Generate Manuscript Figures (XAI & Performance)
To verify that our figures are rigorously derived from raw experimental data, run:
# Generate Figure 5 (Convergence Trajectories) & Table 2
python analysis/plot_section5_benchmark.py

# Generate Figure 6 (Q-Table Heatmaps) & Figure 8 (Phase Transitions)
python analysis/plot_section6_xai.py