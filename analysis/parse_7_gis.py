import os
import pandas as pd
import json


def generate_arcgis_csv():
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    hxut_out_dir = os.path.join(base_dir, 'output', 'exp_casestudy_hxut')
    perf_csv = os.path.join(hxut_out_dir, 'HXUT_Performance_Summary.csv')

    # 读取节点坐标 (假设您 data/HXUT 目录下有 node.csv 或 coords.txt)
    # 如果您的坐标列名不同，请修改此处 'X', 'Y'
    coords_path = os.path.join(base_dir, 'data', 'HXUT', 'node.csv')
    nodes_df = pd.read_csv(coords_path)
    # 建立字典映射 NodeID -> (X, Y)
    coords_dict = {row.iloc[0]: (row.iloc[1], row.iloc[2]) for _, row in nodes_df.iterrows()}

    # 找最佳 Seed
    df = pd.read_csv(perf_csv)
    best_seed = int(df.loc[df['Final_Fitness'].idxmin()]['Seed'])

    # 读 JSON
    json_path = os.path.join(hxut_out_dir, f"HXUT_Routes_Seed_{best_seed}.json")
    with open(json_path, 'r') as f:
        best_ind = json.load(f)

    routes = best_ind.get('routes', [])
    frequencies = best_ind.get('frequencies', [])

    # 准备扁平化数据供 ArcGIS 使用
    arcgis_data = []
    for route_idx, route in enumerate(routes):
        route_id = f"Route_{route_idx + 1}"
        freq = frequencies[route_idx] if route_idx < len(frequencies) else 0

        # 判断是主干线还是微循环线 (这里以发车频率 8 辆/小时为界，可微调)
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

    # 导出 CSV
    out_csv = os.path.join(hxut_out_dir, 'ArcGIS_Routes_Format.csv')
    pd.DataFrame(arcgis_data).to_csv(out_csv, index=False)
    print(f"✅ ArcGIS 专用格式已导出至: {out_csv}")
    print("👉 请在 ArcGIS Pro 中使用 [XY Table To Point] -> [Points To Line] 工具。")


if __name__ == "__main__":
    generate_arcgis_csv()