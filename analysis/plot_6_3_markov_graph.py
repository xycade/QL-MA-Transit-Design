import os
import glob
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np


def plot_markov_transition():
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    xai_dir = os.path.join(base_dir, 'output', 'exp_benchmark_xai')
    out_dir = os.path.join(base_dir, 'output/analysis_results')
    os.makedirs(out_dir, exist_ok=True)

    networks = ['mandl', 'mumford0']
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))

    # 1. 绝对坐标锁定 (呈完美的正方形排列)
    fixed_pos = {
        'S0': (-1, 1),  # 左上 (停滞+高多样)
        'S1': (-1, -1),  # 左下 (停滞+低多样)
        'S2': (1, 1),  # 右上 (健康+高多样)
        'S3': (1, -1)  # 右下 (健康+低多样)
    }

    # 节点语义着色 (左侧停滞用浅红，右侧健康用浅绿)
    node_colors = {'S0': '#FFCCCC', 'S1': '#FFCCCC', 'S2': '#CCFFCC', 'S3': '#CCFFCC'}

    # 自环文字的偏移位置 (上下浮动防止遮挡)
    offset_dict = {'S0': (0, 0.35), 'S1': (0, -0.35), 'S2': (0, 0.35), 'S3': (0, -0.35)}

    for idx, net in enumerate(networks):
        pattern = os.path.join(xai_dir, f'OpHistory_{net}_Seed_*.csv')
        csv_files = glob.glob(pattern)

        if not csv_files:
            continue

        # 汇总 20 个 Seed
        df_all = pd.concat([pd.read_csv(f) for f in csv_files], ignore_index=True)

        transitions = df_all.groupby(['Current_State', 'Next_State']).agg(
            Count=('Action_Selected', 'count'),
            Top_Action=('Action_Selected', lambda x: x.mode()[0]),
            Avg_Reward=('Reward', 'mean')
        ).reset_index()

        ax = axes[idx]
        G = nx.DiGraph()

        # 分离: 跨状态转移 (Cross edges) 和 自我循环 (Self loops)
        cross_edges = transitions[transitions['Current_State'] != transitions['Next_State']]
        self_loops = transitions[transitions['Current_State'] == transitions['Next_State']]

        # 强制添加所有节点
        for state in ['S0', 'S1', 'S2', 'S3']:
            G.add_node(state)

        # 2. 处理跨状态转移 (跨域连线)
        for _, row in cross_edges.iterrows():
            if row['Count'] >= 5:  # 降低门槛，保留哪怕是罕见的“关键破局动作”
                G.add_edge(row['Current_State'], row['Next_State'],
                           weight=np.log1p(row['Count']) * 1.2,  # 使用对数平滑粗细，防止极值失衡
                           label=f"{row['Top_Action']} (R:{row['Avg_Reward']:.2f})")

        # 绘制节点
        colors = [node_colors[node] for node in G.nodes()]
        nx.draw_networkx_nodes(G, fixed_pos, node_size=3500, node_color=colors, edgecolors='black', ax=ax)
        nx.draw_networkx_labels(G, fixed_pos, font_size=14, font_weight='bold', ax=ax)

        # 绘制转移连线 (带有明显的弧度避免双向覆盖)
        edges = G.edges(data=True)
        nx.draw_networkx_edges(G, fixed_pos, ax=ax, arrowstyle='->', arrowsize=25,
                               width=[d['weight'] for u, v, d in edges],
                               connectionstyle='arc3, rad=0.2', edge_color='#666666', alpha=0.8)

        # 绘制转移连线标签 (加白色背景框防遮挡)
        edge_labels = {(u, v): d['label'] for u, v, d in edges}
        nx.draw_networkx_edge_labels(G, fixed_pos, edge_labels=edge_labels, font_size=12,
                                     label_pos=0.25, ax=ax, font_color='darkred',
                                     bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=1))

        # 3. 处理自我循环 (化作精美的文字浮浮在节点旁边)
        for _, row in self_loops.iterrows():
            state = row['Current_State']
            if row['Count'] >= 5:
                x, y = fixed_pos[state]
                dx, dy = offset_dict[state]
                ax.text(x + dx, y + dy,
                        f"Stay: {row['Top_Action']}\n(R:{row['Avg_Reward']:.2f})",
                        ha='center', va='center', fontsize=10, color='darkblue',
                        bbox=dict(boxstyle="round,pad=0.3", facecolor='white', edgecolor='lightblue', alpha=0.9))

        ax.set_title(f'({chr(97 + idx)}) {net.capitalize()} - Causal Flow of Meta-Policy', fontweight='bold',
                     fontsize=16)
        ax.set_xlim(-1.6, 1.6)
        ax.set_ylim(-1.6, 1.6)
        ax.axis('off')

    plt.tight_layout()
    out_path = os.path.join(out_dir, 'XAI_6_3_Markov_Graph.png')
    plt.savefig(out_path, dpi=500, bbox_inches='tight')
    print(f"✅ 高清无遮挡版图 6.3 已保存至: {out_path}")


if __name__ == "__main__":
    plot_markov_transition()