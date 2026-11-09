import pandas as pd
import numpy as np
import os


def clean_edges():
    # 1. Set paths (ensure the working directory is the project root)
    input_path = os.path.join('data', 'HXUT', 'raw_edges.csv')
    output_path = os.path.join('data', 'HXUT', 'edge.csv')

    if not os.path.exists(input_path):
        print(f"❌ File not found: {input_path}")
        return

    print("Loading raw data...")
    # 2. Read CSV and extract only the required columns (bypassing messy 'Name' columns from Excel)
    df = pd.read_csv(input_path, encoding='gb18030')

    # Extract Origin ID, Destination ID, and Total Travel Time
    # Standard column names: 'OriginID', 'DestinationID', 'Total_TravelTime'
    df = df[['OriginID', 'DestinationID', 'Total_TravelTime']]
    df.columns = ['FromNode', 'ToNode', 'TT']  # Rename to target schema

    # 3. Preliminary filtering: remove self-loops (e.g., 1 -> 1)
    df = df[df['FromNode'] != df['ToNode']]

    print(f"Initial edge count (after removing self-loops): {len(df)}")

    # 4. Build distance dictionary to accelerate lookups
    dist_dict = {}
    for _, row in df.iterrows():
        u = int(row['FromNode'])
        v = int(row['ToNode'])
        t = float(row['TT'])
        dist_dict[(u, v)] = t

    # 5. Core filtering logic: eliminate transitive / pass-through edges
    true_edges = []
    # Tolerance setting: 0.01 minutes (~0.6 seconds) to prevent floating-point precision issues
    # where A->B + B->C may not strictly equal A->C
    tolerance = 0.01

    all_nodes = pd.unique(df[['FromNode', 'ToNode']].values.ravel('K'))

    print("Pruning transitive edges that bypass intermediate stops...")
    for (u, v), t_uv in dist_dict.items():
        is_adjacent = True

        # Iterate over all possible intermediate nodes m
        for m in all_nodes:
            if m == u or m == v:
                continue

            # If both paths u -> m and m -> v exist
            if (u, m) in dist_dict and (m, v) in dist_dict:
                t_um = dist_dict[(u, m)]
                t_mv = dist_dict[(m, v)]

                # If the direct travel time equals the sum of segmented transit times, it is a pass-through edge
                if abs((t_um + t_mv) - t_uv) < tolerance:
                    is_adjacent = False
                    break  # Break early once an intermediate node is detected

        # Retain if it is a truly adjacent edge
        if is_adjacent:
            true_edges.append({'TT': t_uv, 'FromNode': u, 'ToNode': v})

    # 6. Generate final DataFrame and export
    final_df = pd.DataFrame(true_edges)

    # Reorder columns as required: TT, FromNode, ToNode
    final_df = final_df[['TT', 'FromNode', 'ToNode']]

    final_df.to_csv(output_path, index=False)
    print(f"✅ Cleaning complete! True adjacent physical edge count: {len(final_df)}")
    print(f"📂 File saved to: {output_path}")


if __name__ == '__main__':
    clean_edges()