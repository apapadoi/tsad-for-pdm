import pandas as pd
import numpy as np

from datetime import datetime, timedelta
import os

folder_to_read_from = './CMAPSSData'

scenario_files = ['FD001.txt', 'FD002.txt', 'FD003.txt', 'FD004.txt']
healthy_files = [file_name for file_name in scenario_files]

scenario_files = [folder_to_read_from + '/train_' + word for word in scenario_files]
healthy_files = [folder_to_read_from + '/test_' + word for word in healthy_files]

scenario_output_folder = '../../DataFolder/c_mapss/scenarios'
healthy_output_folder = '../../DataFolder/c_mapss/healthy'

os.makedirs(scenario_output_folder, exist_ok=True)
os.makedirs(healthy_output_folder, exist_ok=True)


dataframes = []
for file_index, file_path in enumerate(scenario_files):
    file_name = file_path.split('/')[-1][-9:-4]

    df = pd.read_csv(file_path, header=None, delim_whitespace=True)

    df[df.columns[0]] = file_name + '_' + df[df.columns[0]].astype(str)

    dataframes.append(df)

scenario_dataframes = [dataframe for dataframe in dataframes]


dataframes = []
for file_index, file_path in enumerate(healthy_files):
    file_name = file_path.split('/')[-1][-9:-4]

    df = pd.read_csv(file_path, header=None, delim_whitespace=True)

    df[df.columns[0]] = file_name + '_' + df[df.columns[0]].astype(str)

    dataframes.append(df)

healthy_dataframes = [dataframe for dataframe in dataframes]


def generate_timestamps(start_timestamp, num_rows):
    timestamps = [start_timestamp + timedelta(days=i) for i in range(num_rows)]
    return timestamps


std_tolerance = 0.000001
len_of_columns_to_keep_per_scenario = []
columns_kept_per_source = {}
for idx, df in enumerate(healthy_dataframes):
    grouped = df.groupby(0)

    start_timestamp = datetime.strptime('2000-01-01 00:01:40', '%Y-%m-%d %H:%M:%S')

    for group_name, group_data in grouped:
        group_data = group_data.drop(columns=[i for i in range(5)])
        columns_to_keep = group_data.columns[~np.isclose(group_data.std(), 0, atol=std_tolerance)]
        columns_to_keep = columns_to_keep.tolist()
        columns_kept_per_source[group_name] = columns_to_keep
        len_of_columns_to_keep_per_scenario.append(len(columns_to_keep))

        group_data = group_data[columns_to_keep]

        num_rows = group_data.shape[0]
        timestamps = generate_timestamps(start_timestamp, num_rows)
        group_data.insert(0, 'Artificial_timestamp', timestamps)

        output_file_path = os.path.join(healthy_output_folder, f"{group_name}.csv")
        group_data.to_csv(output_file_path, index=False)


for idx, df in enumerate(scenario_dataframes):
    grouped = df.groupby(0)

    start_timestamp = datetime.strptime('2000-01-01 00:01:40', '%Y-%m-%d %H:%M:%S')

    for group_name, group_data in grouped:
        group_data = group_data.drop(columns=[i for i in range(5)])
        if group_name in columns_kept_per_source:
            columns_to_keep = columns_kept_per_source[group_name]
        else:
            columns_to_keep = group_data.columns[~np.isclose(group_data.std(), 0, atol=std_tolerance)]
            columns_to_keep = columns_to_keep.tolist()

        len_of_columns_to_keep_per_scenario.append(len(columns_to_keep))

        group_data = group_data[columns_to_keep]

        if group_name in columns_kept_per_source:
            if group_data.shape[1] != len(columns_kept_per_source[group_name]):
                raise RuntimeError(f"Different number of columns for source {group_name}")

        num_rows = group_data.shape[0]
        timestamps = generate_timestamps(start_timestamp, num_rows)
        group_data.insert(0, 'Artificial_timestamp', timestamps)

        output_file_path = os.path.join(scenario_output_folder, f"{group_name}.csv")
        group_data.to_csv(output_file_path, index=False)
