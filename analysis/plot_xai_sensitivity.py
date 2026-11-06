import os
import sys
import glob
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# ---------------------------------------------------------
# Global Academic Plotting Configuration
# ---------------------------------------------------------
sns.set_theme(style="whitegrid", context="paper")

# 同样大幅调高字号以适应 2x2 布局
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


def analyze_phase_transition():
    data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'output', 'exp4_6_xai_data'))
    perf_csv = os.path.abspath(
        os.path.join(os.path.dirname(__file__), '..', 'output', 'exp1_2_data', 'Table_Performance_Summary_Raw.csv'))
    out_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'output', 'analysis_results'))
    os.makedirs(out_dir, exist_ok=True)

    networks = ['mandl', 'mumford0', 'mumford1']

    print("\n" + "=" * 90)
    print(" 🔬 EXP 6: MICRO-LEVEL PHASE TRANSITION ANALYSIS (2x2 CROSS-SCALE) ")
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

    # 【改动1】：构建 2x2 画板
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    titles = ['(a) Mandl (15 Nodes)', '(b) Mumford0 (30 Nodes)', '(c) Mumford1 (70 Nodes)']

    # 关闭右下角格子，作为图例专属位
    axes[1, 1].axis('off')

    for idx, net in enumerate(networks):
        row, col = divmod(idx, 2)
        ax1 = axes[row, col]

        target_seed = median_seeds.get(net)
        op_files = glob.glob(os.path.join(data_dir, f"OpHistory_{net}_Seed_{target_seed if target_seed else '*'}.csv"))

        if not op_files:
            ax1.set_title(f"{titles[idx]} - Data Not Found")
            ax1.axis('off')
            continue

        df = pd.read_csv(op_files[0])

        if not all(c in df.columns for c in ['Generation', 'Action', 'CV']):
            ax1.set_title(f"{titles[idx]} - Missing Columns")
            continue

        df = df[df['Generation'] <= 500].copy()
        df['Is_Explorer'] = (df['Action'] == 3).astype(float) * 100

        df_gen = df.groupby('Generation').agg({'CV': 'mean', 'Is_Explorer': 'mean'}).reset_index()

        window_size = 15
        df_gen['Smooth_CV'] = df_gen['CV'].rolling(window=window_size, min_periods=1).mean()
        df_gen['Smooth_A3'] = df_gen['Is_Explorer'].rolling(window=window_size, min_periods=1).mean()

        peak_idx = df_gen['Smooth_A3'].idxmax()
        peak_gen = df_gen.loc[peak_idx, 'Generation'] if not pd.isna(peak_idx) else 0
        peak_a3_val = df_gen.loc[peak_idx, 'Smooth_A3'] if not pd.isna(peak_idx) else 0

        # 防报错转换
        x_arr = df_gen['Generation'].to_numpy()
        y_cv_arr = df_gen['Smooth_CV'].to_numpy()
        y_a3_arr = df_gen['Smooth_A3'].to_numpy()

        # --- 绘图：左侧 Y 轴 ---
        color_div = '#009E73'
        ax1.plot(x_arr, y_cv_arr, color=color_div, linewidth=3, label='Population Diversity (CV)')
        ax1.set_xlabel('Generations', fontweight='bold')
        ax1.tick_params(axis='y', labelcolor=color_div)
        ax1.set_xlim(0, 500)

        # 只在第一列(图a, c)显示左侧的 Y 轴标签
        if col == 0:
            ax1.set_ylabel('Diversity (CV)', color=color_div, fontweight='bold')

        # --- 绘图：右侧 Y 轴 ---
        ax2 = ax1.twinx()
        color_act = '#D55E00'
        ax2.plot(x_arr, y_a3_arr, color=color_act, linewidth=3, linestyle='--', label='Explorer (A3) Selection')
        ax2.tick_params(axis='y', labelcolor=color_act)
        ax2.set_ylim(0, max(40, peak_a3_val * 1.2) if peak_a3_val > 0 else 100)

        # 针对 2x2 布局，图b (col=1) 和 图c (idx=2 且没被阻挡) 都能安全展示右侧 Y 轴标签
        if col == 1 or idx == 2:
            ax2.set_ylabel('Selection Probability (%)', color=color_act, fontweight='bold')

        # 标注相变点
        if peak_gen > 0 and peak_a3_val > 5:
            ax1.axvline(x=peak_gen, color='grey', linestyle=':', linewidth=2, alpha=0.7)
            if idx >= 1:
                ax2.annotate('Phase Transition\n(Response Point)',
                             xy=(peak_gen, peak_a3_val),
                             xytext=(peak_gen + 15, peak_a3_val * 0.75),
                             arrowprops=dict(facecolor='grey', shrink=0.05, width=2, headwidth=8),
                             fontsize=14, fontweight='bold', color='#333333')

        ax1.set_title(titles[idx], fontweight='bold', fontsize=16)

        # 【改动2】：抓取所有的线段和标签，统一扔进右下角的专属格子里居中展示
        if idx == 2:
            lines_1, labels_1 = ax1.get_legend_handles_labels()
            lines_2, labels_2 = ax2.get_legend_handles_labels()
            axes[1, 1].legend(lines_1 + lines_2, labels_1 + labels_2, loc='center', frameon=True)

    plt.tight_layout()
    plot_path = os.path.join(out_dir, "Fig_Phase_Transition_All.png")
    plt.savefig(plot_path, dpi=500, bbox_inches='tight', format='png')
    plt.close(fig)

    print(f"\n  ✅ Generated 2x2 Cross-Scale Phase Transition plot: {plot_path}")


if __name__ == '__main__':
    analyze_phase_transition()