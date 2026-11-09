import os
import glob
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


def plot_empirical_q_table():
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    xai_dir = os.path.join(base_dir, 'output', 'exp_benchmark_xai')

    networks = ['mandl', 'mumford0']
    # 修改1：稍微增大画布高度，给大字体留出空间
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for idx, net in enumerate(networks):
        pattern = os.path.join(xai_dir, f'OpHistory_{net}_Seed_*.csv')
        csv_files = glob.glob(pattern)

        if not csv_files:
            print(f"⚠️ 警告: 找不到 {net} 的 OpHistory 数据。")
            continue

        df_all = pd.concat([pd.read_csv(f) for f in csv_files], ignore_index=True)

        df_all['State_Idx'] = df_all['Current_State'].str.replace('S', '').astype(int)
        df_all['Action_Idx'] = df_all['Action_Selected'].str.replace('A', '').astype(int)

        empirical_q_table = np.zeros((4, 4))
        empirical_q_table.fill(-1.0)

        agg_df = df_all.groupby(['State_Idx', 'Action_Idx'])['Reward'].mean().reset_index()
        for _, row in agg_df.iterrows():
            empirical_q_table[int(row['State_Idx']), int(row['Action_Idx'])] = row['Reward']

        temp = 0.2
        exp_q = np.exp(empirical_q_table / temp)
        policy_probs = exp_q / np.sum(exp_q, axis=1, keepdims=True)

        ax = axes[idx]

        # 修改2：使用 annot_kws 调大热力图内部的数字
        sns.heatmap(policy_probs, annot=True, fmt=".2f", cmap="YlGnBu", cbar=True, ax=ax,
                    annot_kws={"size": 14},  # 内部数字大小
                    xticklabels=['A0', 'A1', 'A2', 'A3'],
                    yticklabels=['S0', 'S1', 'S2', 'S3'])

        # 修改3：调大标题和坐标轴标签
        ax.set_title(f'({chr(97 + idx)}) {net.capitalize()} - Empirical Policy Preference',
                     fontweight='bold', fontsize=16)
        ax.set_xlabel('Semantic Actions', fontsize=14)
        ax.set_ylabel('Semantic States' if idx == 0 else '', fontsize=14)

        # 修改4：调大坐标轴刻度标签（A0-A3, S0-S3）
        ax.tick_params(axis='both', labelsize=13)

        # 修改5：调大颜色条（Colorbar）的刻度文字
        cbar = ax.collections[0].colorbar
        cbar.ax.tick_params(labelsize=12)

    # 如果想让总图更紧凑，可以适当调整 hspace/wspace
    plt.tight_layout(pad=2.0)

    out_path = os.path.join(base_dir, 'output/analysis_results', 'XAI_6_2_QTable.png')
    plt.savefig(out_path, dpi=500, bbox_inches='tight')
    print(f"✅ 图 6.2 (OpHistory 直接提取版) 已保存至: {out_path}")


if __name__ == "__main__":
    plot_empirical_q_table()