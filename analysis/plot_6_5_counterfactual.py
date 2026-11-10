import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def plot_counterfactual_raster():
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    csv_path = os.path.join(base_dir, 'output', 'exp_counterfactual', 'OpHistory_Counterfactual.csv')

    if not os.path.exists(csv_path):
        print("❌ Counterfactual data not found. Please run run_counterfactual.py first.")
        return

    df = pd.read_csv(csv_path)

    # Slice data window around the perturbation generation for detailed localized view
    zoom_df = df[(df['Generation'] >= 200) & (df['Generation'] <= 300)].copy()

    sns.set_theme(style="ticks", context="paper", font_scale=1.2)

    # Create top and bottom subplots sharing the X-axis
    fig, (ax_top, ax_bottom) = plt.subplots(2, 1, figsize=(12, 7), sharex=True, gridspec_kw={'height_ratios': [1, 2]})

    # ==========================================
    # Top Panel: Sharp Drop in Population Diversity (CV)
    # ==========================================
    sns.lineplot(data=zoom_df, x='Generation', y='Diversity_CV', ax=ax_top, color='black', linewidth=2)
    ax_top.axvline(x=250, color='#d62728', linestyle='--', linewidth=2, alpha=0.8)
    ax_top.set_ylabel('Population\nDiversity (CV)', fontweight='bold')
    ax_top.set_title('Causal Verification via Counterfactual Clone Injection', fontweight='bold', fontsize=16)
    ax_top.grid(True, linestyle=':', alpha=0.6)

    # Text annotation
    ax_top.text(251, zoom_df['Diversity_CV'].max() * 0.8,
                "Generation 250:\nArtificial Clone Injection\n(Diversity forced to 0)",
                color='#d62728', fontsize=11, fontweight='bold',
                bbox=dict(facecolor='white', edgecolor='none', alpha=0.7))

    # ==========================================
    # Bottom Panel: Heuristic Action Raster Plot (Neural Spike Train Analogy)
    # ==========================================
    markers = {'A0': '|', 'A1': 'x', 'A2': 'D', 'A3': '*'}
    colors = {'A0': '#2ca02c', 'A1': '#1f77b4', 'A2': '#ff7f0e', 'A3': '#d62728'}
    sizes = {'A0': 300, 'A1': 150, 'A2': 150, 'A3': 500}  # Strongly highlight A3

    for action in ['A0', 'A1', 'A2', 'A3']:
        sub = zoom_df[zoom_df['Action_Selected'] == action]
        if not sub.empty:
            ax_bottom.scatter(sub['Generation'], sub['Action_Selected'],
                              marker=markers[action], color=colors[action],
                              s=sizes[action], label=action, alpha=0.9)

    ax_bottom.axvline(x=250, color='#d62728', linestyle='--', linewidth=2, alpha=0.8)
    ax_bottom.set_ylabel('Selected Heuristic Action', fontweight='bold')
    ax_bottom.set_xlabel('Generations', fontweight='bold')
    ax_bottom.set_yticks(['A0', 'A1', 'A2', 'A3'])
    ax_bottom.set_yticklabels(['A0 (Exploiter)', 'A1 (Mutator)', 'A2 (Reformer)', 'A3 (Explorer)'])
    ax_bottom.grid(axis='y', linestyle=':', alpha=0.6)

    # Arrow annotation at the causal reflex trigger point
    ax_bottom.annotate("Immediate Causal Reflex\n(Burst of A3)",
                       xy=(251, 'A3'), xytext=(265, 'A2'),
                       arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=8),
                       fontsize=12, fontweight='bold', color='#d62728')

    plt.tight_layout()
    out_dir = os.path.join(base_dir, 'output', 'analysis_results')
    os.makedirs(out_dir, exist_ok=True)
    out_path_png = os.path.join(out_dir, 'XAI_6_5_Counterfactual_Raster.png')
    plt.savefig(out_path_png, dpi=500, bbox_inches='tight')
    print(f"✅ Fig 6.5 (Counterfactual Raster Plot) saved to: {out_path_png}")


if __name__ == "__main__":
    plot_counterfactual_raster()