import os
import glob
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def plot_state_space():
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    xai_dir = os.path.join(base_dir, 'output', 'exp_benchmark_xai')
    out_dir = os.path.join(base_dir, 'output', 'analysis_results')
    os.makedirs(out_dir, exist_ok=True)

    networks = ['mandl', 'mumford0']

    # 沿用您的优秀排版：2行1列
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))

    # 配色：A0(绿色), A1(蓝色), A2(橙色), A3(红色)
    palette = {'A0': '#2ca02c', 'A1': '#1f77b4', 'A2': '#ff7f0e', 'A3': '#d62728'}

    for idx, net in enumerate(networks):
        pattern = os.path.join(xai_dir, f'OpHistory_{net}_Seed_*.csv')
        csv_files = glob.glob(pattern)

        if not csv_files:
            print(f"⚠️ 警告: 找不到 {net} 的 OpHistory 数据。")
            continue

        # 1. 拼接 20 个 Seed 的所有数据
        df_all = pd.concat([pd.read_csv(f) for f in csv_files], ignore_index=True)

        # 2. 核心修正：除了算众数(Mode)，还要计算“重合度(Count)”！
        agg_df = df_all.groupby(['Generation', 'Current_State']).agg(
            Action_Selected=('Action_Selected', lambda x: x.mode()[0]),
            Seed_Count=('Action_Selected', 'count')  # 统计有多少个 Seed 处于这个状态
        ).reset_index()

        # 3. 强制剥离出整数用于安全绘图，防止 Seaborn 乱序
        agg_df['State_Idx'] = agg_df['Current_State'].str.replace('S', '').astype(int)

        ax = axes[idx]

        # 💡 高级制图：加入 size='Seed_Count'，让点的大小代表“共识的强烈程度”
        sns.scatterplot(data=agg_df, x='Generation', y='State_Idx', hue='Action_Selected',
                        size='Seed_Count', sizes=(20, 150),  # 最小的孤立点 20，大部队点 150
                        palette=palette, alpha=0.7, edgecolor='white', linewidth=0.5,
                        ax=ax, hue_order=['A0', 'A1', 'A2', 'A3'])

        ax.set_title(f'({chr(97 + idx)}) {net.capitalize()} - Consensus Action Firing & State Density',
                     fontweight='bold', fontsize=16)
        ax.set_xlabel('Generations', fontsize=14)
        ax.set_ylabel('Semantic States', fontsize=16)

        # 锁定 Y 轴
        ax.set_yticks([0, 1, 2, 3])
        ax.set_yticklabels(['S0 (Stag+HighDiv)', 'S1 (Stag+LowDiv)', 'S2 (Conv+HighDiv)', 'S3 (Conv+LowDiv)'])
        ax.set_ylim(-0.5, 3.5)  # 给点留出上下空间，防止被边框切掉

    # 4. 沿用您的全局图例逻辑，并额外剔除 Size 的干扰图例
    handles, labels = [], []
    for ax in axes:
        h, l = ax.get_legend_handles_labels()
        # 过滤掉 Seaborn 自动生成的 'Seed_Count' 大小图例，只保留颜色图例
        clean_h, clean_l = [], []
        for handle, label in zip(h, l):
            if label in palette.keys():
                clean_h.append(handle)
                clean_l.append(label)

        if clean_h:
            handles, labels = clean_h, clean_l

        if ax.get_legend() is not None:
            ax.get_legend().remove()

    if handles:
        fig.legend(handles, labels, title='Majority Action Executed', loc='lower center',
                   bbox_to_anchor=(0.5, -0.03), ncol=4, frameon=True, fontsize=14, title_fontsize=14)

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.1)  # 留出底部空间

    out_path_png = os.path.join(out_dir, 'XAI_6_1_State_Space.png')
    plt.savefig(out_path_png, dpi=500, bbox_inches='tight')
    print(f"✅ 图 6.1 (共识密度映射版) 已保存至: {out_path_png}")


if __name__ == "__main__":
    plot_state_space()