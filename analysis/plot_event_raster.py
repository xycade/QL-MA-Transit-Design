import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# 极简/顶刊风格配置
sns.set_theme(style="ticks", context="paper")
plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica"],
    "axes.labelsize": 12, "xtick.labelsize": 11, "ytick.labelsize": 11,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": 1.2, "xtick.major.width": 1.2, "ytick.major.width": 1.2
})

# 高级配色 (Tableau 风格)
COLOR_CV = '#117A65'  # 深青色 (Teal)
ACTION_COLORS = {
    0: '#2CA02C',  # Green (Exploiter)
    1: '#1F77B4',  # Blue (Mutator)
    2: '#FF7F0E',  # Orange (Reformer)
    3: '#D62728'  # Red (Explorer)
}
ACTION_LABELS = ['$A_0$ (Exploiter)', '$A_1$ (Mutator)', '$A_2$ (Reformer)', '$A_3$ (Explorer)']


def plot_raster_event():
    data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'output', 'exp4_6_xai_data'))
    out_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'output', 'analysis_results'))

    base_file = os.path.join(data_dir, "SingleBranch_mumford0_Base.csv")
    branch_a_file = os.path.join(data_dir, "SingleBranch_mumford0_Branch_A.csv")

    if not (os.path.exists(base_file) and os.path.exists(branch_a_file)):
        print("❌ 错误: 找不到上一阶段的实验数据。")
        return

    df_base = pd.read_csv(base_file)
    df_a = pd.read_csv(branch_a_file)

    # 截取 200 代以后的基线数据
    df_base_cut = df_base[(df_base['Generation'] >= 200) & (df_base['Generation'] < 250)].copy()

    # 拼接数据 (200 - 300代)
    df = pd.concat([df_base_cut, df_a], ignore_index=True)
    df = df[df['Generation'] <= 300]

    # 根据 Greedy 概率反推被选中的绝对动作
    # （因为之前的脚本里，被选中的最优动作概率被设为了 1.0，其余为 0）
    def get_chosen_action(row):
        probs = [row['Prob_A0'], row['Prob_A1'], row['Prob_A2'], row['Prob_A3']]
        return np.argmax(probs)

    df['Chosen_Action'] = df.apply(get_chosen_action, axis=1)

    # ==========================================
    # 创建共享 X 轴的非等高子图 (GridSpec)
    # ==========================================
    fig = plt.figure(figsize=(10, 6.5))
    gs = gridspec.GridSpec(2, 1, height_ratios=[1, 2.5], hspace=0.15)

    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1], sharex=ax1)

    # ------------------------------------------
    # 子图 1: Environmental Stimulus (CV)
    # ------------------------------------------
    ax1.plot(df['Generation'], df['CV'], color=COLOR_CV, linewidth=2)

    # 填充 250 代右侧的底部阴影以增强环境崩塌的视觉氛围
    ax1.fill_between(df['Generation'], df['CV'], where=(df['Generation'] >= 250), color=COLOR_CV, alpha=0.1)

    ax1.set_ylabel('Diversity (CV)', fontweight='bold')
    ax1.set_title('Counterfactual Intervention: Neuromorphic Event Raster Plot', fontweight='bold', pad=15, fontsize=14)
    ax1.set_ylim(-0.1, df['CV'].max() * 1.2)
    ax1.tick_params(labelbottom=False)  # 隐藏顶部子图的 X 轴刻度标签，让画面更干净

    # ------------------------------------------
    # 子图 2: Agent Strategic Response (Raster Plot)
    # ------------------------------------------
    # 使用密集的竖线 (marker='|') 模拟神经脉冲
    for action_idx in range(4):
        action_data = df[df['Chosen_Action'] == action_idx]
        ax2.scatter(action_data['Generation'], np.repeat(action_idx, len(action_data)),
                    color=ACTION_COLORS[action_idx], s=120, marker='|', linewidths=2.5, alpha=0.9)

    ax2.set_yticks([0, 1, 2, 3])
    ax2.set_yticklabels(ACTION_LABELS, fontweight='bold')
    ax2.set_ylabel('Selected Operator', fontweight='bold')
    ax2.set_xlabel('Generations', fontweight='bold')
    ax2.set_ylim(-0.5, 3.5)
    ax2.set_xlim(200, 300)

    # ------------------------------------------
    # 全局标注: Intervention Line
    # ------------------------------------------
    for ax in [ax1, ax2]:
        ax.axvline(x=250, color='#8B0000', linestyle='--', linewidth=1.5, alpha=0.8)

    # 仅在顶图加上箭头标注
    ax1.annotate('Intervention Point\n(Clone Injection)', xy=(250, df['CV'].max() * 0.8),
                 xytext=(255, df['CV'].max() * 0.8),
                 arrowprops=dict(facecolor='#8B0000', shrink=0.05, width=1.2, headwidth=6),
                 fontsize=10, fontweight='bold', color='#8B0000', va='center')

    plt.tight_layout()
    plot_path = os.path.join(out_dir, "Fig_Event_Raster_Intervention.png")
    plt.savefig(plot_path, dpi=600, bbox_inches='tight')
    plt.close(fig)
    print(f"✅ Event Raster Plot: {plot_path}")


if __name__ == '__main__':
    plot_raster_event()