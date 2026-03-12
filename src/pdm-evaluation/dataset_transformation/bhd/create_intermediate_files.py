import pandas as pd
import os
from datetime import datetime, timedelta

import numpy as np


def generate_timestamps(start_timestamp, num_rows):
    timestamps = [start_timestamp + timedelta(days=i) for i in range(num_rows)]
    return timestamps


# columns_to_count = ['model', 'datacenter', 'cluster_id', 'vault_id', 'pod_id', 'pod_slot_num', 'failure', 'serial_number']
columns_to_count = ['model', 'failure', 'serial_number']
unique_value_counts = {}
for column in columns_to_count:
    unique_value_counts[column] = set()

number_of_sources_with_failure = 0
sources_with_failure = []
models_with_failure = []

def read_csv_files_from_folder(input_folder):
    global number_of_sources_with_failure
    dfs = []
    total_read = 0
    
    for file_name in sorted(list(filter(lambda filename: filename.endswith(".csv"), os.listdir(input_folder))), key=lambda filename: datetime.strptime(filename.split('.')[0], "%Y-%m-%d")):
        if file_name.endswith(".csv"):
            file_path = os.path.join(input_folder, file_name)
            print(file_name)
            df = pd.read_csv(file_path)
            for column in columns_to_count:
                for unique_value in df[column].unique():
                    unique_value_counts[column].add(unique_value)

            df_with_failures_only = df[df.failure == 1]
            number_of_sources_with_failure += len(df_with_failures_only.model.tolist())
            sources_with_failure.extend(df_with_failures_only.serial_number.tolist())
            models_with_failure.extend(df_with_failures_only.model.tolist())
            # dfs.append(df)
            total_read += 1

    # return dfs

input_folder = './data_2023'

read_csv_files_from_folder(input_folder)
print(f'Number of sources with failure: {number_of_sources_with_failure}')
sources_with_failure_df = pd.DataFrame(sources_with_failure, columns=['serial_number'])
models_with_failure_df = pd.DataFrame(models_with_failure, columns=['model'])
# final_dfs = []

# concatenated_df = pd.concat(list_of_dfs, ignore_index=True)



# 

# for column in columns_to_count:
#     unique_value_counts[column] = concatenated_df[column].nunique()

sources_with_failure_df.to_csv('sources_with_failure.csv')
models_with_failure_df.to_csv('models_with_failure.csv')
for column, count in unique_value_counts.items():
    print(f"Column '{column}' has {len(count)} unique values.")