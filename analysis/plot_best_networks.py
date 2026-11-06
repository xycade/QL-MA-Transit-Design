import os
import json
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# Get the absolute path of the script directory to ensure safe execution from any working directory
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..'))


def plot_single_network(network_name, json_filepath, output_filename):
    """Draw a single network topology map based on route sequences from a JSON file."""
    base_path = os.path.join(PROJECT_ROOT, "data", network_name)

    if not os.path.exists(json_filepath):
        print(f"  ❌ Route file not found: {json_filepath}")
        return

    try:
        node_df = pd.read_csv(os.path.join(base_path, 'node.csv'))
        edge_df = pd.read_csv(os.path.join(base_path, 'edge.csv'))
    except FileNotFoundError:
        print(f"  ❌ Underlying road network files not found for network: {network_name}.")
        return

    with open(json_filepath, 'r') as f:
        data = json.load(f)
        routes = data.get('routes', data) if isinstance(data, dict) else data

    pos = {row['nodeID']: (row['X'], row['Y']) for _, row in node_df.iterrows()}

    G = nx.Graph()
    for _, row in edge_df.iterrows():
        G.add_edge(row['FromNode'], row['ToNode'])

    plt.figure(figsize=(12, 10), dpi=600)
    ax = plt.gca()
    ax.set_facecolor('#f8f9fa')

    nx.draw_networkx_edges(G, pos, ax=ax, edge_color='#cccccc', style='dashed', width=1.5, alpha=0.7)

    # Use a high-contrast color palette
    colors = list(mcolors.TABLEAU_COLORS.values())

    for idx, route in enumerate(routes):
        route_edges = [(route[i], route[i + 1]) for i in range(len(route) - 1)]
        color = colors[idx % len(colors)]

        # Route edges
        nx.draw_networkx_edges(G, pos, edgelist=route_edges, ax=ax,
                               edge_color=color, width=4.5, alpha=0.85,
                               label=f'Route {idx + 1}')

        # Route nodes
        nx.draw_networkx_nodes(G, pos, nodelist=route, ax=ax,
                               node_color=color, node_size=120, edgecolors='white', linewidths=1.5)

    covered_nodes = set([node for route in routes for node in route])
    uncovered_nodes = set(G.nodes()) - covered_nodes
    if uncovered_nodes:
        nx.draw_networkx_nodes(G, pos, nodelist=list(uncovered_nodes), ax=ax,
                               node_color='white', node_size=60, edgecolors='#ff4c4c', linewidths=1.5,
                               label='Uncovered Nodes')

    plt.title(f"QL-MA Optimized Topology: {network_name.capitalize()}", fontsize=18, fontweight='bold', pad=15)

    # Adjust legend position to avoid overlapping with the main figure
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=10, frameon=True, shadow=True)
    plt.axis('off')
    plt.tight_layout()

    out_dir = os.path.join(PROJECT_ROOT, "output", "analysis_results")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, output_filename)
    plt.savefig(out_path, bbox_inches='tight')
    plt.close()
    print(f"  ✅ Successfully generated: {out_path}")


def generate_three_best_networks():
    perf_csv = os.path.join(PROJECT_ROOT, "output", "exp1_2_data", "Table_Performance_Summary_Raw.csv")
    json_dir = os.path.join(PROJECT_ROOT, "output", "exp1_2_data")
    networks = ['mandl', 'mumford0', 'mumford1']

    print("\n" + "=" * 80)
    print(" 🗺️ GENERATING TOPOLOGY MAPS FOR BEST QL-MA NETWORKS ")
    print("=" * 80)

    if not os.path.exists(perf_csv):
        print(
            f"❌ Performance summary file not found: {perf_csv}\n   Please verify that the file exists under the output/exp1_2_data/ directory.")
        return

    df = pd.read_csv(perf_csv)

    for net in networks:
        print(f"\nProcessing {net.upper()}...")
        # Filter QL-MA runs that satisfy all passenger demand (Unmet_Demand == 0)
        df_isa = df[(df['Network'] == net) & (df['Mode'] == 'QL-MA') & (df['Unmet_Demand'] == 0)]

        if df_isa.empty:
            print(f"  ⚠️ No QL-MA run with Unmet_Demand == 0 found for {net}. Falling back to lowest fitness...")
            df_isa = df[(df['Network'] == net) & (df['Mode'] == 'QL-MA')]
            if df_isa.empty:
                print(f"  ❌ No QL-MA data found for {net}.")
                continue

        # Find the seed with the minimum (best) fitness
        best_seed = df_isa.loc[df_isa['Final_Fitness'].idxmin(), 'Seed']
        best_fitness = df_isa.loc[df_isa['Final_Fitness'].idxmin(), 'Final_Fitness']

        print(f"  🏆 Best seed identified: {best_seed} (Fitness: {best_fitness:,.2f})")

        # Construct JSON route filename
        json_filename = f"Routes_{net}_QL-MA_Seed_{best_seed}.json"
        json_filepath = os.path.join(json_dir, json_filename)
        output_filename = f"Fig_Best_Topology_{net.capitalize()}.png"

        # Plot topology
        plot_single_network(net, json_filepath, output_filename)


if __name__ == "__main__":
    generate_three_best_networks()