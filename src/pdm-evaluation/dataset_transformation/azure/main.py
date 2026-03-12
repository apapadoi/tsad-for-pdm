import pandas as pd
import os
import sys

signals_df = pd.read_csv('./PdM_telemetry.csv')
failures_df = pd.read_csv('./PdM_failures.csv')

print(signals_df.shape)
print(failures_df.shape)

train_output_folder = '../../DataFolder/azure/healthy'
os.makedirs(train_output_folder, exist_ok=True)

test_output_folder = '../../DataFolder/azure/scenarios'
os.makedirs(test_output_folder, exist_ok=True)

for group_name, group_data in signals_df.groupby('machineID'):
        print(f"Machine: {group_name}")
        sorted_group = group_data.sort_values(by='datetime')

        output_file_path = os.path.join(test_output_folder, f"{group_name}.csv")
        print(output_file_path)
        sorted_group = sorted_group.drop('machineID', axis=1)
        sorted_group.to_csv(output_file_path, index=False)