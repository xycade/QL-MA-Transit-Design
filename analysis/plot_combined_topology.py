import os
import json
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# ==========================================
# Global Typography & Style Configuration for Publication
# ==========================================
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "axes.titlesize": 18,  # 【修改】：调大全局默认标题字号 (原 14)
    "legend.fontsize": 12,  # 【修改】：调大全局默认图例字号 (原 9)
    "pdf.fonttype": 42,
    "ps.fonttype": 42
})

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..'))


def generate_combined_topology_map():
    perf_csv = os.path.join(PROJECT_ROOT, "output", "exp1_2_data", "Table_Performance_Summary_Raw.csv")
    json_dir = os.path.join(PROJECT_ROOT, "output", "exp1_2_data")
    networks = ['mandl', 'mumford0', 'mumford1']
    titles = ['(a) Mandl (15 Nodes)', '(b) Mumford0 (30 Nodes)', '(c) Mumford1 (70 Nodes)']

    if not os.path.exists(perf_csv):
        print(f"❌ Performance summary file not found: {perf_csv}")
        return

    df = pd.read_csv(perf_csv)

    # Create a 1x3 panoramic figure canvas for publication
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), dpi=600)
    fig.patch.set_facecolor('white')

    # Use a high-contrast professional color palette
    colors = list(mcolors.TABLEAU_COLORS.values())

    for i, net in enumerate(networks):
        ax = axes[i]
        ax.set_facecolor('#f8f9fa')  # Ultra-light background to accentuate transit lines

        # Locate the best seed
        df_isa = df[(df['Network'] == net) & (df['Mode'] == 'QL-MA') & (df['Unmet_Demand'] == 0)]
        if df_isa.empty:
            df_isa = df[(df['Network'] == net) & (df['Mode'] == 'QL-MA')]

        best_seed = df_isa.loc[df_isa['Final_Fitness'].idxmin(), 'Seed']
        json_filepath = os.path.join(json_dir, f"Routes_{net}_QL-MA_Seed_{best_seed}.json")

        # Load underlying GIS/Node network data
        base_path = os.path.join(PROJECT_ROOT, "data", net)
        node_df = pd.read_csv(os.path.join(base_path, 'node.csv'))
        edge_df = pd.read_csv(os.path.join(base_path, 'edge.csv'))

        with open(json_filepath, 'r') as f:
            data = json.load(f)
            routes = data.get('routes', data) if isinstance(data, dict) else data

        pos = {row['nodeID']: (row['X'], row['Y']) for _, row in node_df.iterrows()}

        G = nx.Graph()
        for _, row in edge_df.iterrows():
            G.add_edge(row['FromNode'], row['ToNode'])

        # Draw base physical road network with light dashed lines
        nx.draw_networkx_edges(G, pos, ax=ax, edge_color='#cccccc', style='dashed', width=1.5, alpha=0.6)

        # Draw optimized transit trunk routes
        for idx, route in enumerate(routes):
            route_edges = [(route[i], route[i + 1]) for i in range(len(route) - 1)]
            color = colors[idx % len(colors)]

            # Use thicker lines and solid markers to enhance network distinguishability
            nx.draw_networkx_edges(G, pos, edgelist=route_edges, ax=ax,
                                   edge_color=color, width=3.5, alpha=0.85, label=f'R{idx + 1}')
            nx.draw_networkx_nodes(G, pos, nodelist=route, ax=ax,
                                   node_color=color, node_size=70, edgecolors='white', linewidths=1.0)

        # Highlight uncovered blind zones (if any)
        covered_nodes = set([node for route in routes for node in route])
        uncovered_nodes = set(G.nodes()) - covered_nodes
        if uncovered_nodes:
            nx.draw_networkx_nodes(G, pos, nodelist=list(uncovered_nodes), ax=ax,
                                   node_color='white', node_size=40, edgecolors='#ff4c4c', linewidths=1.2,
                                   label='Uncovered')

        # Set subplot title and legend
        # 【修改】：明确放大子图标题
        ax.set_title(titles[i], fontweight='bold', fontsize=18, pad=15)

        # 【修改】：将硬编码的 fontsize=7 修改为 12，放大图例
        ax.legend(loc='best', fontsize=12, framealpha=0.9)

        # 【核心修改】：替换原来的 ax.axis('off')
        # 隐藏坐标轴刻度数字和刻度线
        ax.set_xticks([])
        ax.set_yticks([])
        # 显式保留四个方向的边框（Spines），并将其设置为灰色的细线
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color('gray')
            spine.set_linewidth(1.0)  # 1.0 即为细线，可以根据喜好调整为 0.5

    plt.tight_layout()
    out_dir = os.path.join(PROJECT_ROOT, "output", "analysis_results")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "Fig_Combined_Topology_ISAMA.png")  # Supports JPG or PNG format
    plt.savefig(out_path, bbox_inches='tight', format='png', dpi=500)
    plt.close()
    print(f"✅ Combined 3-in-1 physical network topology map generated successfully: {out_path}")


if __name__ == "__main__":
    generate_combined_topology_map()