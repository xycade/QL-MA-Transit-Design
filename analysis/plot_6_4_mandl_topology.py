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
    # Scenario 1: Extract S1 -> A3 (Aggressive Exploration) from log files
    # ==========================================
    pattern = os.path.join(xai_dir, 'MicroSlice_mandl_Seed_*.json')
    slice_files = glob.glob(pattern)

    if not slice_files:
        print("⚠️ Warning: No MicroSlice data found for Mandl.")
        return

    with open(slice_files[0], 'r') as f:
        data_a3 = json.load(f)

    edges_before_a3 = [tuple(e) for e in data_a3['Before_Topology']]
    edges_after_a3 = [tuple(e) for e in data_a3['After_Topology']]

    # ==========================================
    # Scenario 2: S3 -> A0 (Micro-Exploitation) based on Mandl topology
    # Illustrates corridor smoothing: detour 2->5->4 smoothed into direct connection 2->4
    # ==========================================
    # Local physical edges for Mandl network
    base_a0_edges = [(1, 2), (2, 3), (2, 4), (2, 5), (4, 5), (4, 12), (3, 6), (4, 6)]

    # Before A0 (detour): 1 -> 2 -> 5 -> 4 -> 12
    edges_before_a0 = [(1, 2), (2, 5), (5, 4), (4, 12), (2, 3), (3, 6)]

    # After A0 (smoothed): 1 -> 2 -> 4 -> 12 (bypasses node 5, activates direct link 2-4)
    edges_after_a0 = [(1, 2), (2, 4), (4, 12), (2, 3), (3, 6)]

    # ==========================================
    # Layout computation (lock coordinates for both scenarios)
    # ==========================================
    G_base_a3 = nx.Graph(edges_before_a3 + edges_after_a3)
    pos_a3 = nx.kamada_kawai_layout(G_base_a3)

    G_base_a0 = nx.Graph(base_a0_edges)
    pos_a0 = nx.kamada_kawai_layout(G_base_a0)

    # ==========================================
    # Plotting (2x2 grid layout)
    # ==========================================
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))  # Expanded canvas for 2x2 grid

    # Visual style parameters
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
    print(f"✅ Fig 6.4 (Dual Scenario Topology Analysis) saved to: {out_path_png}")


if __name__ == "__main__":
    plot_micro_slice_topology()