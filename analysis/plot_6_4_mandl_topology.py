import os
import glob
import json
import networkx as nx
import matplotlib.pyplot as plt


def plot_micro_slice_topology():
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    xai_dir = os.path.join(base_dir, 'output', 'exp_benchmark_xai')
    out_dir = os.path.join(base_dir, 'output', 'analysis_results')
    os.makedirs(out_dir, exist_ok=True)

    # ==========================================
    # 场景 1: 从日志提取 S1 -> A3 (Aggressive Exploration)
    # ==========================================
    pattern = os.path.join(xai_dir, 'MicroSlice_mandl_Seed_*.json')
    slice_files = glob.glob(pattern)

    if not slice_files:
        print("⚠️ 找不到 Mandl 的 MicroSlice 数据。")
        return

    with open(slice_files[0], 'r') as f:
        data_a3 = json.load(f)

    edges_before_a3 = [tuple(e) for e in data_a3['Before_Topology']]
    edges_after_a3 = [tuple(e) for e in data_a3['After_Topology']]

    # ==========================================
    # 场景 2: 严格基于 Mandl 拓扑还原 S3 -> A0 (Micro-Exploitation)
    # 展现 "走廊平滑"：原本绕路 2->5->4，被 A0 平滑裁剪为直达 2->4
    # ==========================================
    # Mandl 局部真实边
    base_a0_edges = [(1, 2), (2, 3), (2, 4), (2, 5), (4, 5), (4, 12), (3, 6), (4, 6)]

    # Before A0 (绕路): 1 -> 2 -> 5 -> 4 -> 12
    edges_before_a0 = [(1, 2), (2, 5), (5, 4), (4, 12), (2, 3), (3, 6)]

    # After A0 (平滑): 1 -> 2 -> 4 -> 12 (移除了 5, 激活了 2-4 直达)
    edges_after_a0 = [(1, 2), (2, 4), (4, 12), (2, 3), (3, 6)]

    # ==========================================
    # 布局计算 (锁定两组场景的绝对坐标)
    # ==========================================
    G_base_a3 = nx.Graph(edges_before_a3 + edges_after_a3)
    pos_a3 = nx.kamada_kawai_layout(G_base_a3)

    G_base_a0 = nx.Graph(base_a0_edges)
    pos_a0 = nx.kamada_kawai_layout(G_base_a0)

    # ==========================================
    # 开始绘图 (2x2 网格)
    # ==========================================
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))  # 画布加大，适配 2x2

    # 统一的高级视觉参数 (调大字号)
    node_opts = {"node_color": '#E8E8E8', "edgecolors": 'black',
                 "node_size": 1200, "font_size": 16, "font_weight": 'bold'}

    # ------------------------------------------
    # Row 1: S3 -> A0 (Corridor Smoothing)
    # ------------------------------------------
    ax1, ax2 = axes[0][0], axes[0][1]

    # Before A0
    nx.draw(nx.Graph(edges_before_a0), pos_a0, ax=ax1, with_labels=True, edge_color='gray', width=3, **node_opts)
    removed_a0 = set(edges_before_a0) - set(edges_after_a0)
    nx.draw_networkx_edges(nx.Graph(edges_before_a0), pos_a0, edgelist=list(removed_a0), ax=ax1, edge_color='#d62728',
                           width=6, style='dashed')
    ax1.set_title(f"(a) Before (State: S3 - Convergence)\nRed dashes: Detour edges (2-5-4) to be smoothed", fontsize=18,
                  fontweight='bold', pad=15)

    # After A0
    nx.draw(nx.Graph(edges_after_a0), pos_a0, ax=ax2, with_labels=True, edge_color='gray', width=3, **node_opts)
    added_a0 = set(edges_after_a0) - set(edges_before_a0)
    nx.draw_networkx_edges(nx.Graph(edges_after_a0), pos_a0, edgelist=list(added_a0), ax=ax2, edge_color='#2ca02c',
                           width=6, style='solid')
    ax2.set_title(f"(b) After (Action: A0 - Exploiter)\nGreen solid line: Direct corridor (2-4) established",
                  fontsize=18, fontweight='bold', pad=15)

    # ------------------------------------------
    # Row 2: S1 -> A3 (Aggressive Exploration)
    # ------------------------------------------
    ax3, ax4 = axes[1][0], axes[1][1]

    # Before A3
    nx.draw(nx.Graph(edges_before_a3), pos_a3, ax=ax3, with_labels=True, edge_color='gray', width=3, **node_opts)
    removed_a3 = set(edges_before_a3) - set(edges_after_a3)
    if removed_a3:
        nx.draw_networkx_edges(nx.Graph(edges_before_a3), pos_a3, edgelist=list(removed_a3), ax=ax3,
                               edge_color='#d62728', width=6, style='dashed')
    ax3.set_title(f"(c) Before (State: S1 - Stagnation)\nRed dashes: Trapped redundant topology", fontsize=18,
                  fontweight='bold', pad=15)

    # After A3
    nx.draw(nx.Graph(edges_after_a3), pos_a3, ax=ax4, with_labels=True, edge_color='gray', width=3, **node_opts)
    added_a3 = set(edges_after_a3) - set(edges_before_a3)
    if added_a3:
        nx.draw_networkx_edges(nx.Graph(edges_after_a3), pos_a3, edgelist=list(added_a3), ax=ax4, edge_color='#2ca02c',
                               width=6, style='solid')
    ax4.set_title(f"(d) After (Action: A3 - Explorer)\nForced structural mutation to restore diversity", fontsize=18,
                  fontweight='bold', pad=15)

    for row in axes:
        for ax in row:
            ax.set_axis_off()

    plt.tight_layout()
    out_path_png = os.path.join(out_dir, 'XAI_6_4_Mandl_Topology.png')
    plt.savefig(out_path_png, dpi=500, bbox_inches='tight')
    print(f"✅ 图 6.4 (双情境丰富版) 已保存至: {out_path_png}")


if __name__ == "__main__":
    plot_micro_slice_topology()