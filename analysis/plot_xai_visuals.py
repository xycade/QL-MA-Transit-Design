import os
import json
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import FancyArrowPatch


# ==========================================
# 0. Global Style and Typography Settings (SCI Publication Standard)
# ==========================================
def set_academic_style():
    sns.set_theme(style="white", context="paper")
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 12,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 11,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "axes.linewidth": 1.0,
    })


# Action labels and color palette (professional, restrained scheme)
ACTIONS = ['A0', 'A1', 'A2', 'A3']
ACTION_LABELS = ['A0 (Exploiter)', 'A1 (Mutator)', 'A2 (Reformer)', 'A3 (Explorer)']
ACTION_COLORS = {'A0': '#2E8B57', 'A1': '#4682B4', 'A2': '#DAA520', 'A3': '#CD5C5C'}


# ==========================================
# Helper Function: Locate Log Directory and Output Directory
# ==========================================
def get_directories():
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    log_dir = os.path.join(base_dir, 'output', 'exp8_xai_logs')
    out_dir = os.path.join(base_dir, 'output', 'analysis_results')

    # Dynamic search fallback
    if not os.path.exists(log_dir):
        for root, dirs, files in os.walk(base_dir):
            if 'transitions_log.csv' in files:
                log_dir = root
                break

    os.makedirs(out_dir, exist_ok=True)
    return log_dir, out_dir


# ==========================================
# Helper Function: Robust Row-wise Min-Max
# ==========================================
def robust_row_norm(q_table):
    q_arr = np.array(q_table)
    q_norm = np.zeros_like(q_arr)
    for r in range(q_arr.shape[0]):
        rmin, rmax = np.min(q_arr[r]), np.max(q_arr[r])
        if rmax - rmin > 1e-6:
            q_norm[r] = (q_arr[r] - rmin) / (rmax - rmin)
        else:
            q_norm[r] = 0.0
    return q_norm


# ==========================================
# Task 1: Plot Q-value Evolution Heatmaps (带灰色边框优化版)
# ==========================================
def plot_q_evolution(log_dir, out_dir):
    file_path = os.path.join(log_dir, 'q_snapshots.json')
    if not os.path.exists(file_path):
        print(f"❌ File not found: {file_path}")
        return

    with open(file_path, 'r') as f:
        q_data = json.load(f)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    titles = ['(a) Early Stage', '(b) Mid Stage', '(c) Late Stage']

    stages = list(q_data.keys())[:3]

    for i, stage in enumerate(stages):
        q_table = q_data[stage]
        # 使用强壮的归一化函数，防止全 0 行导致颜色异常
        q_norm = robust_row_norm(q_table)

        ax = axes[i]
        sns.heatmap(q_norm, annot=q_norm, fmt=".2f", cmap="Blues",
                    cbar=(i == 2), ax=ax, vmin=0, vmax=1,
                    annot_kws={"size": 11},
                    cbar_kws={'label': 'Normalized Q-value Preference'} if i == 2 else None)

        # ==============================================
        # 核心优化：为每个热力图子图强制添加灰色的细边框
        # ==============================================
        for _, spine in ax.spines.items():
            spine.set_visible(True)  # 强制显示边框
            spine.set_color('#95a5a6')  # 采用高级的浅岩灰色
            spine.set_linewidth(1.2)  # 边框粗细适中

        # 为了让格子内部也稍微清晰一点，也可以稍微加一点格子线(可选)
        # ax.axhline(y=0, color='#95a5a6',linewidth=2.4)
        # ax.axhline(y=4, color='#95a5a6',linewidth=2.4)
        # ax.axvline(x=0, color='#95a5a6',linewidth=2.4)
        # ax.axvline(x=4, color='#95a5a6',linewidth=2.4)

        ax.set_title(f"{titles[i]} ({stage.replace('_', ' ')})", fontweight='bold', pad=10)
        ax.set_xticks(np.arange(4) + 0.5)
        ax.set_xticklabels(ACTIONS)
        ax.set_xlabel('Heuristic Operators')

        ax.set_yticks(np.arange(4) + 0.5)
        if i == 0:
            ax.set_yticklabels(['S0', 'S1', 'S2', 'S3'], rotation=0)
            ax.set_ylabel('Semantic States')
        else:
            ax.set_yticklabels([])

    plt.tight_layout()
    save_path = os.path.join(out_dir, 'Task1_Q_Evolution.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ Task 1: Q-value Evolution saved to {save_path}")
    plt.close()

# ==========================================
# Task 3: Plot Microscopic Phase Transition Slice Analysis
# ==========================================
def plot_micro_slice(log_dir, out_dir):
    slice_path = os.path.join(log_dir, 'micro_slice.json')
    snap_path = os.path.join(log_dir, 'q_snapshots.json')

    if not os.path.exists(slice_path) or not os.path.exists(snap_path):
        print(f"❌ Files not found for Task 3.")
        return

    with open(slice_path, 'r') as f:
        m_data = json.load(f)
    with open(snap_path, 'r') as f:
        q_data = json.load(f)

    # 关键修复：从 Gen_250 中提取真实的 S1 状态 Q 值并归一化，替换掉原始记录的初始全0值
    raw_q_S1 = q_data['Gen_250'][1]
    norm_q_S1 = robust_row_norm([raw_q_S1])[0]

    edges_before = m_data['Before_Topology']
    edges_after = m_data['After_Topology']

    fig, axes = plt.subplots(1, 3, figsize=(16, 5), gridspec_kw={'width_ratios': [1, 1.2, 1.2]})

    # --- Subplot 1: Q-value Confidence Bar Chart ---
    ax1 = axes[0]
    colors = [ACTION_COLORS[f'A{i}'] for i in range(4)]
    bars = ax1.bar(ACTIONS, norm_q_S1, color=colors, edgecolor='black', alpha=0.85)

    ax1.set_title(f"(a) Q-values in State {m_data['State']}\n(Learned Gap)", fontweight='bold')
    ax1.set_ylabel('Normalized Q-value Preference', fontweight='bold')

    # 计算 A3(索引3) 与其余三个动作的最大值之间的差距 (Action Gap)
    max_idx = 3  # 强制高亮 A3 (Explorer)
    second_max = np.max(norm_q_S1[:3])
    gap = norm_q_S1[max_idx] - second_max

    ax1.annotate(f'Action Gap\n= +{gap:.2f}',
                 xy=(max_idx, norm_q_S1[max_idx]),
                 xytext=(max_idx - 1.5, norm_q_S1[max_idx] * 0.9),
                 arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=7),
                 fontsize=11, fontweight='bold', ha='center',
                 bbox=dict(boxstyle="round", fc="w", ec="gray", alpha=0.9))
    ax1.set_ylim(0, 1.15)

    # Extract network nodes to generate fixed layout coordinates
    G_all = nx.Graph()
    G_all.add_edges_from(edges_after)
    # 使用 kamada_kawai_layout，对于交通网络这种图通常结构更清晰美观
    pos_net = nx.kamada_kawai_layout(G_all)

    # --- Subplot 2: Physical Topology Before Mutation ---
    ax2 = axes[1]
    G_before = nx.Graph()
    G_before.add_edges_from(edges_before)

    nx.draw_networkx_nodes(G_before, pos_net, node_size=60, node_color='#7F8C8D', ax=ax2)
    nx.draw_networkx_edges(G_before, pos_net, edge_color='#BDC3C7', width=1.2, ax=ax2)
    ax2.set_title(f"(b) Before (Topological Lock-in)", fontweight='bold')
    ax2.axis('off')

    # --- Subplot 3: Physical Topology After Mutation (Highlighting New Edges) ---
    ax3 = axes[2]
    G_after = nx.Graph()
    G_after.add_edges_from(edges_after)

    old_edges = list(G_before.edges())
    # 严格判断哪些边是 A3 新引入的 (无向图需正反检查)
    new_edges = []
    for u, v in G_after.edges():
        if not G_before.has_edge(u, v):
            new_edges.append((u, v))

    nx.draw_networkx_nodes(G_after, pos_net, node_size=60, node_color='#7F8C8D', ax=ax3)
    nx.draw_networkx_edges(G_after, pos_net, edgelist=old_edges, edge_color='#BDC3C7', width=1.2, ax=ax3)
    # 高亮 A3 执行后新增的边：红色、加粗、虚线
    nx.draw_networkx_edges(G_after, pos_net, edgelist=new_edges, edge_color=ACTION_COLORS['A3'],
                           width=3.0, style='dashed', ax=ax3)

    ax3.set_title(f"(c) After {m_data['Action']} (Forced Detour Executed at Gen {m_data['Trigger_Gen']})",
                  fontweight='bold')
    ax3.axis('off')

    plt.tight_layout()
    save_path = os.path.join(out_dir, 'Task3_Micro_Slice.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ Task 3: Micro-slice Analysis saved to {save_path}")
    plt.close()


if __name__ == "__main__":
    set_academic_style()
    log_dir, out_dir = get_directories()

    print(f"📂 Reading log directory: {log_dir}")
    print(f"💾 Image save directory: {out_dir}\n")

    plot_q_evolution(log_dir, out_dir)
    plot_micro_slice(log_dir, out_dir)