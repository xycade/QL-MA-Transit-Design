import os
import pandas as pd
import json


def generate_arcgis_csv():
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    hxut_out_dir = os.path.join(base_dir, 'output', 'exp_casestudy_hxut')
    perf_csv = os.path.join(hxut_out_dir, 'HXUT_Performance_Summary.csv')

    # Read node coordinates (assumes node.csv or coords.txt exists in data/HXUT)
    # Modify columns if your coordinate file headers differ
    coords_path = os.path.join(base_dir, 'data', 'HXUT', 'node.csv')
    nodes_df = pd.read_csv(coords_path)
    # Build dictionary mapping: NodeID -> (X, Y)
    coords_dict = {row.iloc[0]: (row.iloc[1], row.iloc[2]) for _, row in nodes_df.iterrows()}

    # Identify the best seed
    df = pd.read_csv(perf_csv)
    best_seed = int(df.loc[df['Final_Fitness'].idxmin()]['Seed'])

    # Read JSON route file
    json_path = os.path.join(hxut_out_dir, f"HXUT_Routes_Seed_{best_seed}.json")
    with open(json_path, 'r') as f:
        best_ind = json.load(f)

    routes = best_ind.get('routes', [])
    frequencies = best_ind.get('frequencies', [])

    # Prepare flattened tabular data for ArcGIS ingestion
    arcgis_data = []
    for route_idx, route in enumerate(routes):
        route_id = f"Route_{route_idx + 1}"
        freq = frequencies[route_idx] if route_idx < len(frequencies) else 0

        # Classify route as Trunk or Feeder line (threshold set at 8 departures/hour; adjustable)
        line_type = "Trunk" if freq >= 8 else "Feeder"

        for stop_order, node_id in enumerate(route):
            x, y = coords_dict.get(node_id, (0, 0))
            arcgis_data.append({
                "Route_ID": route_id,
                "Line_Type": line_type,
                "Frequency": freq,
                "Stop_Order": stop_order + 1,
                "Node_ID": node_id,
                "X": x,
                "Y": y
            })

    # Export to CSV
    out_csv = os.path.join(hxut_out_dir, 'ArcGIS_Routes_Format.csv')
    pd.DataFrame(arcgis_data).to_csv(out_csv, index=False)
    print(f"✅ ArcGIS-compatible format exported to: {out_csv}")
    print("👉 In ArcGIS Pro, use the [XY Table To Point] tool followed by [Points To Line].")


if __name__ == "__main__":
    generate_arcgis_csv()