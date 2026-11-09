import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import ranksums


def analyze_and_plot_benchmark():
    # ==========================================
    # 1. 路径与数据加载
    # ==========================================
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    exp_dir = os.path.join(base_dir, 'output', 'exp_benchmark')
    fig_dir = os.path.join(base_dir, 'output', 'analysis_results')
    os.makedirs(fig_dir, exist_ok=True)

    perf_csv = os.path.join(exp_dir, 'Table_Performance_Summary.csv')
    traj_csv = os.path.join(exp_dir, 'Fitness_Trajectory.csv')

    if not os.path.exists(perf_csv) or not os.path.exists(traj_csv):
        print("❌ 找不到 CSV 数据文件，请确认 Benchmark 是否已产生输出。")
        return

    df_perf = pd.read_csv(perf_csv)
    df_traj = pd.read_csv(traj_csv)

    # 规范化算法名称 (为了图表美观)
    mode_mapping = {'GA': 'GA', 'RMA': 'RMA', 'SMA': 'SMA', 'QL-MA': 'QL-MA'}
    df_perf['Mode'] = df_perf['Mode'].map(mode_mapping)
    df_traj['Mode'] = df_traj['Mode'].map(mode_mapping)

    networks = ['mandl', 'mumford0']
    modes = ['GA', 'RMA', 'SMA', 'QL-MA']

    # ==========================================
    # 2. 计算 Table 2: 性能统计与 Wilcoxon 秩和检验
    # ==========================================
    print("\n" + "=" * 80)
    print("📊 TABLE 2: OPTIMIZATION PERFORMANCE & STATISTICAL ANALYSIS")
    print("=" * 80)

    for net in networks:
        print(f"\n🌍 Network: {net.upper()}")
        print(
            f"{'Algorithm':<10} | {'Best Cost':<12} | {'Mean ± Std Dev':<22} | {'Median Cost':<12} | {'Unmet Demand':<12} | {'Direct Rate':<12} | {'p-value'}")
        print("-" * 105)

        # 提取 QL-MA 的 fitness 数组作为假设检验的 baseline
        ql_fitness = df_perf[(df_perf['Network'] == net) & (df_perf['Mode'] == 'QL-MA')]['Final_Fitness'].values

        for mode in modes:
            sub_df = df_perf[(df_perf['Network'] == net) & (df_perf['Mode'] == mode)]
            if len(sub_df) == 0:
                continue

            fit_vals = sub_df['Final_Fitness'].values
            best_cost = np.min(fit_vals)
            mean_cost = np.mean(fit_vals)
            std_cost = np.std(fit_vals)
            median_cost = np.median(fit_vals)

            # Unmet Demand 取均值 (理想情况下全是 0)
            unmet_demand = np.mean(sub_df['Unmet_Demand'].values)
            direct_rate = np.mean(sub_df['Direct_Rate'].values) * 100

            # Wilcoxon rank-sum test
            if mode == 'QL-MA':
                p_val_str = "-"
            else:
                stat, p_val = ranksums(ql_fitness, fit_vals)
                p_val_str = f"{p_val:.2e}"

            print(
                f"{mode:<10} | {best_cost:<12,.0f} | {mean_cost:,.0f} ± {std_cost:<7.0f} | {median_cost:<12,.0f} | {unmet_demand:<12.1f} | {direct_rate:.2f}%{'':<5} | {p_val_str}")

    print("=" * 80 + "\n")

    # ==========================================
    # 3. 绘制 Figure 5: 收敛曲线图 (带 95% 置信区间)
    # ==========================================
    print("🎨 正在生成高清收敛曲线图 (带 95% 置信区间)...")

    # 将宽表 (Wide format) 转为长表 (Long format) 以供 Seaborn 画图
    id_vars = ['Network', 'Mode', 'Seed']
    value_vars = [col for col in df_traj.columns if col.startswith('Gen_')]
    df_melted = df_traj.melt(id_vars=id_vars, value_vars=value_vars, var_name='Generation', value_name='Fitness')
    df_melted['Generation'] = df_melted['Generation'].str.replace('Gen_', '').astype(int)

    # ⚠️ 修改1：context 从 "paper" 改为 "notebook"，font_scale 从 1.2 增大到 1.5
    # 这样能让全局字体显著变大，更加清晰
    sns.set_theme(style="whitegrid", context="notebook", font_scale=1.5)

    # ⚠️ 修改2：适当增加画布尺寸 (12x5 -> 14x6)，为大字体留出空间
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # 统一调色板 (QL-MA用显眼的红色，其它用冷色调)
    palette = {'GA': '#8c564b', 'RMA': '#2ca02c', 'SMA': '#1f77b4', 'QL-MA': '#d62728'}

    # 子图 1: Mandl
    ax1 = axes[0]
    sns.lineplot(data=df_melted[df_melted['Network'] == 'mandl'], x='Generation', y='Fitness',
                 hue='Mode', errorbar=('ci', 95), palette=palette, linewidth=2.5, ax=ax1)  # 线条加粗一点
    ax1.set_title('(a) Mandl (15 Nodes)', fontweight='bold', fontsize=18)  # ⚠️ 修改3：标题显式加字号
    ax1.set_xlabel('Generations', fontsize=16)
    ax1.set_ylabel('Objective Value (Log Scale)', fontsize=16)
    ax1.set_yscale('log')  # 对数坐标系，更好地展示收敛细节

    # 子图 2: Mumford0
    ax2 = axes[1]
    sns.lineplot(data=df_melted[df_melted['Network'] == 'mumford0'], x='Generation', y='Fitness',
                 hue='Mode', errorbar=('ci', 95), palette=palette, linewidth=2.5, ax=ax2)
    ax2.set_title('(b) Mumford0 (30 Nodes)', fontweight='bold', fontsize=18)
    ax2.set_xlabel('Generations', fontsize=16)
    ax2.set_ylabel('')  # 右侧不需要 Y 轴标签
    ax2.set_yscale('log')

    # ⚠️ 修改4：优化图例，调大图例内的文字
    for ax in axes:
        ax.tick_params(axis='both', labelsize=14)  # 刻度标签大小
        ax.legend(title='Algorithm', frameon=True, loc='upper right',
                  fontsize=14, title_fontsize=15)  # 图例字体大小

    plt.tight_layout()

    # 保存高分辨率图片
    png_path = os.path.join(fig_dir, 'Convergence_Trajectories.png')
    plt.savefig(png_path, dpi=500, bbox_inches='tight')

    print(f"✅ 图表已保存至: {png_path}")


if __name__ == "__main__":
    analyze_and_plot_benchmark()