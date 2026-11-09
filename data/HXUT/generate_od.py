import pandas as pd
import numpy as np
import os


def generate_bus_od():
    # ==========================================
    # 1. Setup paths and parameters
    # ==========================================
    data_dir = os.path.join('data', 'HXUT')

    # Ensure filenames match the provided files
    mapping_file = os.path.join(data_dir, 'Stops_Join_TAZ.csv')
    time_matrix_file = os.path.join(data_dir, 'full_time_matrix.csv')
    output_od_file = os.path.join(data_dir, 'ODmatrix.csv')

    # Impedance coefficient Beta (BETA): reflects spatial decay effect on travel
    BETA = 1.5

    print("Loading data...")
    if not os.path.exists(mapping_file) or not os.path.exists(time_matrix_file):
        print("❌ Input files not found. Please verify the path and file names under data/HXUT!")
        return

    # ==========================================
    # 2. Allocate TAZ Production/Attraction (P/A) to specific transit stops
    # ==========================================
    print("Allocating P/A to transit stops based on spatial joins...")
    df_map = pd.read_csv(mapping_file)

    # Count associated transit stops per TAZ (for equal flow splitting)
    taz_stop_counts = df_map.groupby('TAZ_ID')['nodeID'].count().reset_index()
    taz_stop_counts.rename(columns={'nodeID': 'stop_count'}, inplace=True)

    # Merge counts and distribute total production and attraction equally among all boundary stops
    df_map = pd.merge(df_map, taz_stop_counts, on='TAZ_ID')
    df_map['allocated_P'] = df_map['Production'] / df_map['stop_count']
    df_map['allocated_A'] = df_map['Attraction'] / df_map['stop_count']

    # Aggregate final P and A for each stop (summing allocations from adjacent TAZs)
    stop_pa = df_map.groupby('nodeID')[['allocated_P', 'allocated_A']].sum().reset_index()
    stop_pa.columns = ['StopID', 'P', 'A']

    print(f"✅ Stop P/A allocation complete. A total of {len(stop_pa)} stops involved in computation.")

    # ==========================================
    # 3. Read impedance matrix (Total_TravelTime)
    # ==========================================
    print("Constructing fully connected travel-time impedance matrix...")
    df_time = pd.read_csv(time_matrix_file, encoding='gb18030')

    # Build lookup dictionary with an epsilon offset to prevent division by zero
    time_dict = {}
    for _, row in df_time.iterrows():
        o = int(row['OriginID'])
        d = int(row['DestinationID'])
        t = float(row['Total_TravelTime'])
        time_dict[(o, d)] = max(t, 0.1)

    # ==========================================
    # 4. Execute single-constrained gravity model
    # ==========================================
    print("Estimating gravity model OD matrix (production-constrained)...")
    stops = stop_pa['StopID'].tolist()
    od_records = []

    for i in stops:
        P_i = stop_pa.loc[stop_pa['StopID'] == i, 'P'].values[0]

        # Skip if stop production is zero
        if P_i <= 0:
            continue

        # 4.1 Compute denominator: Sum (A_k * f(d_ik))
        denominator = 0
        for k in stops:
            if i == k: continue  # Exclude intra-stop self-trips
            A_k = stop_pa.loc[stop_pa['StopID'] == k, 'A'].values[0]
            t_ik = time_dict.get((i, k), 999)  # Assign large impedance for missing values
            f_ik = t_ik ** (-BETA)
            denominator += A_k * f_ik

        # 4.2 Compute specific T_ij (Passenger Demand / PD)
        for j in stops:
            if i == j: continue
            A_j = stop_pa.loc[stop_pa['StopID'] == j, 'A'].values[0]
            t_ij = time_dict.get((i, j), 999)
            f_ij = t_ij ** (-BETA)

            if denominator > 0:
                T_ij = P_i * ((A_j * f_ij) / denominator)
            else:
                T_ij = 0

            # Filter out negligible passenger flows to minimize redundancy
            if T_ij > 0.05:
                od_records.append({
                    'OriginID': int(i),
                    'DestinationID': int(j),
                    'PD': round(T_ij, 2)
                })

    # ==========================================
    # 5. Output in flattened table format
    # ==========================================
    df_od = pd.DataFrame(od_records)

    # Ensure column ordering strictly matches: OriginID, DestinationID, PD
    df_od = df_od[['OriginID', 'DestinationID', 'PD']]

    df_od.to_csv(output_od_file, index=False)
    print(f"✅ OD data generated successfully! Total transit passenger demand: {df_od['PD'].sum():.0f} trips.")
    print(f"📂 Flattened matrix file saved to: {output_od_file}")


if __name__ == '__main__':
    generate_bus_od()