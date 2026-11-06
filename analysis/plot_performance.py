import os
import sys
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon
import warnings

warnings.filterwarnings('ignore')

# ---------------------------------------------------------
# 1. Global Academic Plotting Configuration
# ---------------------------------------------------------
# Enforce Colorblind-friendly palette and sans-serif fonts for SCI journals
sns.set_theme(style="whitegrid", context="paper")

# 【改动1】：单纯调大全局字号，确保图片在Word/LaTeX中缩放时清晰可读
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "axes.labelsize": 16,        # 坐标轴标签调大 (原12)
    "xtick.labelsize": 14,       # X轴刻度调大 (原10)
    "ytick.labelsize": 14,       # Y轴刻度调大 (原10)
    "legend.fontsize": 14,       # 图例字体调大 (原10)
    "legend.title_fontsize": 15, # 图例标题调大 (原11)
    "axes.titlesize": 16,        # 子图标题调大
    "pdf.fonttype": 42,          # Ensure fonts are embedded in PDF
    "ps.fonttype": 42
})

# Standardized color palette for algorithms
MODE_COLORS = {
    'GA': '#E69F00',  # Orange
    'MA-Random': '#56B4E9',  # Light Blue
    'MA-Fixed': '#009E73',  # Green
    'QL-MA': '#D55E00'  # Red/Vermillion (Highlights our method)
}


def analyze_and_plot():
    data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'output', 'exp1_2_data'))
    out_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'output', 'analysis_results'))
    os.makedirs(out_dir, exist_ok=True)

    perf_csv = os.path.join(data_dir, "Table_Performance_Summary_Raw.csv")
    time_csv = os.path.join(data_dir, "Table_Computational_Overhead_Raw.csv")
    traj_csv = os.path.join(data_dir, "Fitness_Trajectory_Raw.csv")

    if not all(os.path.exists(p) for p in [perf_csv, time_csv, traj_csv]):
        print("❌ Error: Raw CSV data files not found.")
        return

    # ---------------------------------------------------------
    # Output 1: Statistical Performance Summary (Feasible Runs)
    # ---------------------------------------------------------
    print("\n" + "=" * 130)
    print(" 📊 OUTPUT 1: STATISTICAL PERFORMANCE SUMMARY (ONLY FEASIBLE RUNS)")
    print("=" * 130)

    df_perf = pd.read_csv(perf_csv)
    networks = ['mandl', 'mumford0', 'mumford1']
    modes = ['GA', 'MA-Random', 'MA-Fixed', 'QL-MA']

    for net in networks:
        print(f"\n[{net.upper()}] Performance across 30 seeds:")
        df_net = df_perf[df_perf['Network'] == net]

        unmet_cols = [c for c in df_net.columns if 'Unmet' in c]
        unmet_col = unmet_cols[0] if unmet_cols else None
        direct_cols = [c for c in df_net.columns if 'Direct' in c]
        direct_col = direct_cols[0] if direct_cols else None

        isa_data_full = df_net[df_net['Mode'] == 'QL-MA']
        if unmet_col:
            isa_data_feasible = isa_data_full[isa_data_full[unmet_col] == 0]
        else:
            isa_data_feasible = isa_data_full

        for mode in modes:
            df_mode = df_net[df_net['Mode'] == mode]
            if df_mode.empty:
                continue

            total_runs = len(df_mode)
            if unmet_col:
                feasible_runs = (df_mode[unmet_col] == 0).sum()
                feasibility_rate = (feasible_runs / total_runs) * 100 if total_runs > 0 else 0
                df_mode_feasible = df_mode[df_mode[unmet_col] == 0]
            else:
                feasibility_rate = 100.0
                df_mode_feasible = df_mode
            feasibility_str = f"{feasibility_rate:.1f}%"

            if len(df_mode_feasible) == 0:
                print(f"  {mode:<10} | Feasible: {feasibility_str} | (No feasible runs to compute stats)")
                continue

            costs = df_mode_feasible['Final_Fitness'].values
            best = np.min(costs)
            mean = np.mean(costs)
            std = np.std(costs)
            median = np.median(costs)

            if direct_col:
                d_vals = df_mode_feasible[direct_col].values
                direct_str = f"{np.mean(d_vals):.4f}±{np.std(d_vals):.4f}"
            else:
                direct_str = "N/A"

            p_val_str = "-"
            if mode != 'QL-MA' and len(isa_data_feasible) > 0:
                try:
                    merged = pd.merge(df_mode_feasible[['Seed', 'Final_Fitness']],
                                      isa_data_feasible[['Seed', 'Final_Fitness']],
                                      on='Seed', suffixes=('_mode', '_isa'))
                    if len(merged) >= 10:
                        stat, p_val = wilcoxon(merged['Final_Fitness_mode'], merged['Final_Fitness_isa'])
                        p_val_str = f"{p_val:.2e}" if p_val < 0.05 else f"{p_val:.4f}"
                    else:
                        p_val_str = "N/A (n<10)"
                except Exception as e:
                    p_val_str = "N/A"

            print(
                f"  {mode:<10} | Best: {best:,.0f} | Mean: {mean:,.0f}±{std:,.0f} | Med: {median:,.0f} | "
                f"Feasible: {feasibility_str} | DR: {direct_str} | p-val: {p_val_str}"
            )

    # ---------------------------------------------------------
    # Output 2: Computational Overhead Analysis
    # ---------------------------------------------------------
    print("\n" + "=" * 130)
    print(" ⏱️ OUTPUT 2: COMPUTATIONAL OVERHEAD ANALYSIS (First 15 Seeds)")
    print("=" * 130)

    df_time = pd.read_csv(time_csv)
    df_time_filtered = df_time[(df_time['Seed'] >= 43) & (df_time['Seed'] <= 57)]

    for net in networks:
        print(f"\n[{net.upper()}] Average execution times (Sec) over first 15 seeds:")
        df_net_t = df_time_filtered[df_time_filtered['Network'] == net]

        for mode in modes:
            df_mode_t = df_net_t[df_net_t['Mode'] == mode]
            if df_mode_t.empty:
                continue

            avg_total = df_mode_t['Total_Time_Sec'].mean()
            avg_rl = df_mode_t['RL_Time_Sec'].mean()
            rl_pct = (avg_rl / avg_total) * 100 if avg_total > 0 else 0

            if mode == 'QL-MA':
                print(f"  {mode:<10} | Total Time: {avg_total:.2f}s | RL Overhead: {avg_rl:.4f}s ({rl_pct:.4f}%)")
            else:
                print(f"  {mode:<10} | Total Time: {avg_total:.2f}s")

    # ---------------------------------------------------------
    # Output 3: Convergence Plot (Robust Median & Percentiles)
    # ---------------------------------------------------------
    print("\n" + "=" * 120)
    print(" 📈 OUTPUT 3: GENERATING HIGH-RES PLOTS...")
    print("=" * 120)

    try:
        print("  Molding trajectory data (this may take a moment due to 30x1000 generations)...")
        df_traj = pd.read_csv(traj_csv)

        gen_cols = [c for c in df_traj.columns if c.startswith('Gen_')]
        df_melt = pd.melt(df_traj, id_vars=['Network', 'Mode', 'Seed'], value_vars=gen_cols,
                          var_name='Generation_Str', value_name='Fitness')

        df_melt['Generation'] = df_melt['Generation_Str'].str.replace('Gen_', '').astype(int)

        df_melt['Fitness'] = pd.to_numeric(df_melt['Fitness'], errors='coerce').astype(float)

        # 【改动2】：图幅微调为 (16, 5)，这个比例在A4纸排版时不会过于扁平，配合大字号效果最佳
        fig, axes = plt.subplots(1, 3, figsize=(16, 5))
        titles = ['(a) Mandl (15 Nodes)', '(b) Mumford0 (30 Nodes)', '(c) Mumford1 (70 Nodes)']

        for idx, net in enumerate(networks):
            df_plot = df_melt[df_melt['Network'] == net]
            if df_plot.empty:
                continue

            ax = axes[idx]

            plot_data = df_plot.reset_index(drop=True)

            x_arr = plot_data['Generation'].to_numpy()
            y_arr = plot_data['Fitness'].to_numpy()
            hue_arr = plot_data['Mode'].to_numpy()

            # 完全保留你喜欢的 Seaborn 绘图逻辑
            sns.lineplot(
                x=x_arr,
                y=y_arr,
                hue=hue_arr,
                palette=MODE_COLORS,
                errorbar=('ci', 95),  # 95% Confidence Interval band
                linewidth=2,
                ax=ax,
                legend=(idx == 1)
            )

            ax.set_title(titles[idx], fontweight='bold')
            ax.set_xlabel('Generations')
            ax.set_ylabel('Objective Value (Log Scale)' if idx == 0 else '')
            ax.set_yscale('log')

            if idx == 1:
                handles, labels = ax.get_legend_handles_labels()
                ax.legend(handles=handles, labels=labels, title="Algorithm", frameon=True, loc='upper right')

        plt.tight_layout()
        plot_path = os.path.join(out_dir, "Fig_Convergence_CI.png")
        plt.savefig(plot_path, dpi=500, bbox_inches='tight', format='png')
        plt.close(fig)
        print(f"  ✅ Successfully generated 500 DPI plot: {plot_path}")

    except Exception as e:
        print(f"  ❌ Error generating plot: {e}")


if __name__ == '__main__':
    analyze_and_plot()