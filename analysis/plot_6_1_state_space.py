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

    # Layout: 2 rows by 1 column
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))

    # Color palette: A0 (Green), A1 (Blue), A2 (Orange), A3 (Red)
    palette = {'A0': '#2ca02c', 'A1': '#1f77b4', 'A2': '#ff7f0e', 'A3': '#d62728'}

    for idx, net in enumerate(networks):
        pattern = os.path.join(xai_dir, f'OpHistory_{net}_Seed_*.csv')
        csv_files = glob.glob(pattern)

        if not csv_files:
            print(f"⚠️ Warning: No OpHistory data found for {net}.")
            continue

        # 1. Concatenate all data across 20 seeds
        df_all = pd.concat([pd.read_csv(f) for f in csv_files], ignore_index=True)

        # 2. Core aggregation: compute both mode (majority action) and sample count (consensus density)
        agg_df = df_all.groupby(['Generation', 'Current_State']).agg(
            Action_Selected=('Action_Selected', lambda x: x.mode()[0]),
            Seed_Count=('Action_Selected', 'count')  # Count the number of seeds in this state
        ).reset_index()

        # 3. Extract integer index for safe plotting to prevent categorical reordering by Seaborn
        agg_df['State_Idx'] = agg_df['Current_State'].str.replace('S', '').astype(int)

        ax = axes[idx]

        # Advanced plotting: set size='Seed_Count' so marker size reflects consensus density
        sns.scatterplot(data=agg_df, x='Generation', y='State_Idx', hue='Action_Selected',
                        size='Seed_Count', sizes=(20, 150),  # Scale from isolated points (20) to high-density clusters (150)
                        palette=palette, alpha=0.7, edgecolor='white', linewidth=0.5,
                        ax=ax, hue_order=['A0', 'A1', 'A2', 'A3'])

        ax.set_title(f'({chr(97 + idx)}) {net.capitalize()} - Consensus Action Firing & State Density',
                     fontweight='bold', fontsize=16)
        ax.set_xlabel('Generations', fontsize=14)
        ax.set_ylabel('Semantic States', fontsize=16)

        # Lock Y-axis ticks and labels
        ax.set_yticks([0, 1, 2, 3])
        ax.set_yticklabels(['S0 (Stag+HighDiv)', 'S1 (Stag+LowDiv)', 'S2 (Conv+HighDiv)', 'S3 (Conv+LowDiv)'])
        ax.set_ylim(-0.5, 3.5)  # Provide vertical padding to prevent markers from clipping

    # 4. Global legend configuration: retain color mapping while filtering out size handles
    handles, labels = [], []
    for ax in axes:
        h, l = ax.get_legend_handles_labels()
        # Filter out Seaborn's automatically generated 'Seed_Count' size handles, retaining only color handles
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
    plt.subplots_adjust(bottom=0.1)  # Reserve space at the bottom for the global legend

    out_path_png = os.path.join(out_dir, 'XAI_6_1_State_Space.png')
    plt.savefig(out_path_png, dpi=500, bbox_inches='tight')
    print(f"✅ Fig 6.1 (Consensus Density Mapping) saved to: {out_path_png}")


if __name__ == "__main__":
    plot_state_space()