import os
import sys
import glob
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

# ---------------------------------------------------------
# Global Academic Plotting Configuration
# ---------------------------------------------------------
sns.set_theme(style="whitegrid", context="paper")

# 保持你原本设定的大字号，同时适配热力图和堆叠图
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "axes.labelsize": 18,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
    "legend.fontsize": 16,
    "legend.title_fontsize": 18,
    "axes.titlesize": 18
})

# 【仅用于面积图的高级学术配色 (冷 -> 暖)】
MODE_COLORS = {
    'A0_Exploiter': '#2B4C7E',  # 深蓝 (稳定/保守)
    'A1_Mutator': '#5681B9',  # 灰蓝 (中度)
    'A2_Reformer': '#DDA77B',  # 哑光金 (过渡/重构)
    'A3_Explorer': '#993333'  # 深红 (激进/探索)
}

STATE_LABELS = [
    'S0: Converged & High-Div',
    'S1: Stagnant & Low-Div',
    'S2: Converged & Low-Div',
    'S3: Stagnant & High-Div'
]

ACTION_LABELS = ['A0: Exploiter', 'A1: Mutator', 'A2: Reformer', 'A3: Explorer']


def min_max_normalize_row(x):
    x_min = np.min(x, axis=1, keepdims=True)
    x_max = np.max(x, axis=1, keepdims=True)
    denominator = np.where(x_max - x_min == 0, 1e-9, x_max - x_min)
    return (x - x_min) / denominator


def analyze_and_plot_xai():
    data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'output', 'exp4_6_xai_data'))
    perf_csv = os.path.abspath(
        os.path.join(os.path.dirname(__file__), '..', 'output', 'exp1_2_data', 'Table_Performance_Summary_Raw.csv'))
    out_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'output', 'analysis_results'))
    os.makedirs(out_dir, exist_ok=True)

    networks = ['mandl', 'mumford0', 'mumford1']

    print("\n" + "=" * 90)
    print(" 🧠 EXP 4: XAI GLOBAL ATTRIBUTION & RULE EXTRACTION (REFINED) ")
    print("=" * 90)

    median_seeds = {}
    if os.path.exists(perf_csv):
        df_perf = pd.read_csv(perf_csv)
        for net in networks:
            df_net = df_perf[(df_perf['Network'] == net) & (df_perf['Mode'] == 'QL-MA')]
            if not df_net.empty:
                median_fit = df_net['Final_Fitness'].median()
                idx_closest = (df_net['Final_Fitness'] - median_fit).abs().idxmin()
                median_seeds[net] = df_net.loc[idx_closest, 'Seed']

    # ---------------------------------------------------------
    # PART 1: 2x2 Q-Table Heatmaps (100% 保持原样)
    # ---------------------------------------------------------
    fig_q, axes_q = plt.subplots(2, 2, figsize=(14, 12))
    titles = ['(a) Mandl (15 Nodes)', '(b) Mumford0 (30 Nodes)', '(c) Mumford1 (70 Nodes)']

    axes_q[1, 1].axis('off')
    cbar_ax = inset_axes(axes_q[1, 1], width="15%", height="80%", loc='center')

    q_tables_found = False

    for idx, net in enumerate(networks):
        row, col = divmod(idx, 2)
        ax = axes_q[row, col]

        target_seed = median_seeds.get(net)
        if target_seed:
            q_files = glob.glob(os.path.join(data_dir, f"QTable_{net}_Seed_{target_seed}.npy"))
        else:
            q_files = glob.glob(os.path.join(data_dir, f"QTable_{net}_Seed_*.npy"))

        if not q_files:
            ax.set_title(f"{titles[idx]} - Data Not Found")
            ax.axis('off')
            continue

        q_tables_found = True
        q_table_norm = min_max_normalize_row(np.load(q_files[0]))

        sns.heatmap(
            q_table_norm,
            annot=True,
            annot_kws={"size": 16},
            fmt=".2f",
            cmap='YlGnBu',
            cbar=True if idx == 2 else False,
            cbar_ax=cbar_ax if idx == 2 else None,
            linewidths=0.5,
            linecolor='lightgray',
            ax=ax,
            xticklabels=['A0', 'A1', 'A2', 'A3'],
            yticklabels=['S0', 'S1', 'S2', 'S3'] if col == 0 else []
        )

        ax.set_title(titles[idx], fontweight='bold')
        ax.set_xlabel('Heuristic Operators')
        if col == 0:
            ax.set_ylabel('Semantic States')

    if q_tables_found:
        plt.tight_layout()
        plot_path_heatmap = os.path.join(out_dir, "Fig_QTable_Attribution.png")
        plt.savefig(plot_path_heatmap, dpi=500, bbox_inches='tight', format='png')
        plt.close(fig_q)
        print(f"\n  ✅ 1. Generated 2x2 Q-Table Heatmap: {plot_path_heatmap}")

    # ---------------------------------------------------------
    # PART 2: 1x3 Stacked Area Charts (升级排版与配色)
    # ---------------------------------------------------------
    fig_a, axes_a = plt.subplots(1, 3, figsize=(18, 5))
    area_charts_found = False
    legend_handles = None
    legend_labels = None

    for idx, net in enumerate(networks):
        ax = axes_a[idx]

        target_seed = median_seeds.get(net)
        op_files = glob.glob(os.path.join(data_dir, f"OpHistory_{net}_Seed_{target_seed if target_seed else '*'}.csv"))

        if not op_files:
            ax.set_title(f"{titles[idx]} - Data Not Found")
            ax.axis('off')
            continue

        area_charts_found = True
        df_op = pd.read_csv(op_files[0])

        if 'Generation' in df_op.columns and 'Action' in df_op.columns:
            bin_size = 50
            df_op['Gen_Bin'] = (df_op['Generation'] // bin_size) * bin_size

            action_counts = df_op.groupby(['Gen_Bin', 'Action']).size().unstack(fill_value=0)
            action_pct = action_counts.div(action_counts.sum(axis=1), axis=0) * 100

            for a in range(4):
                if a not in action_pct.columns:
                    action_pct[a] = 0

            action_labels_mapped = ['A0_Exploiter', 'A1_Mutator', 'A2_Reformer', 'A3_Explorer']
            action_pct = action_pct[[0, 1, 2, 3]]
            action_pct.columns = action_labels_mapped

            last_gen = action_pct.index.max()
            if last_gen < 1000:
                for mb in range(int(last_gen) + bin_size, 1001, bin_size):
                    action_pct.loc[mb] = action_pct.loc[last_gen]

            x = action_pct.index.values
            y = [action_pct[c].values for c in action_labels_mapped]
            colors = [MODE_COLORS[c] for c in action_labels_mapped]

            # 堆叠图，增加白边提升质感
            stacks = ax.stackplot(x, y, labels=[lbl.replace('_', ': ') for lbl in action_labels_mapped],
                                  colors=colors, alpha=0.9, edgecolor='white', linewidth=0.5)

            if idx == 0:
                legend_handles = stacks
                legend_labels = [lbl.replace('_', ': ') for lbl in action_labels_mapped]

            ax.set_title(titles[idx], fontweight='bold')
            ax.set_xlabel('Generations')
            ax.set_xlim(0, 1000)
            ax.set_ylim(0, 100)

            # 去除顶部和右侧的边框线
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

            if idx == 0:
                ax.set_ylabel('Selection Probability (%)')
            else:
                ax.set_ylabel('')
                ax.tick_params(labelleft=False)

    if area_charts_found:
        plt.tight_layout()
        # 将图例统一放置在 1x3 底部
        fig_a.legend(legend_handles, legend_labels, loc='lower center', ncol=4,
                     bbox_to_anchor=(0.5, -0.15), title="Heuristic Operators", frameon=False)

        plot_path_area = os.path.join(out_dir, "Fig_Policy_Trajectory.png")
        plt.savefig(plot_path_area, dpi=500, bbox_inches='tight', format='png')
        plt.close(fig_a)
        print(f"  ✅ 2. Generated 1x3 Stacked Area Chart: {plot_path_area}")


if __name__ == '__main__':
    analyze_and_plot_xai()