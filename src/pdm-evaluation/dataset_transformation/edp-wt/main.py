import pandas as pd
import os
import sys

signals_df = pd.read_csv('./Wind-Turbine-SCADA-signals-2017_0.csv')
failures_df = pd.read_csv('./opendata-wind-failures-2017.csv')

print(signals_df.shape)
print(failures_df.shape)

train_output_folder = '../../DataFolder/edp-wt/healthy'
os.makedirs(train_output_folder, exist_ok=True)

test_output_folder = '../../DataFolder/edp-wt/scenarios'
os.makedirs(test_output_folder, exist_ok=True)

if sys.argv[1] == 'statistics': # TODO this needs to be removed 
    for group_name, group_data in signals_df.groupby('Turbine_ID'):
        print(f"Group: {group_name}")
        sorted_group = group_data.sort_values(by='Timestamp')
        sorted_group['Timestamp'] = pd.to_datetime(sorted_group['Timestamp'])

        failures_for_current_group = failures_df[failures_df['Turbine_ID'] == group_name]
        failures_for_current_group = failures_for_current_group.sort_values(by='Timestamp').reset_index(drop=True)

        for failure_index, failure in failures_for_current_group.iterrows():
            current_df = sorted_group[sorted_group['Timestamp'] <= failure.Timestamp]
            sorted_group = sorted_group[sorted_group['Timestamp'] > failure.Timestamp]

            if current_df.shape[0] > 0:            
                output_file_path = os.path.join(test_output_folder, f"{group_name}_{failure_index}.csv")
                print(output_file_path)
                current_df = current_df.drop('Turbine_ID', axis=1)
                current_df.to_csv(output_file_path, index=False)

        if sorted_group.shape[0] != 0:            
            output_file_path = os.path.join(train_output_folder, f"{group_name}.csv")
            print(output_file_path)
            sorted_group = sorted_group.drop('Turbine_ID', axis=1)
            sorted_group.to_csv(output_file_path, index=False)
else:
    for group_name, group_data in signals_df.groupby('Turbine_ID'):
            print(f"Group: {group_name}")
            sorted_group = group_data.sort_values(by='Timestamp')

            output_file_path = os.path.join(test_output_folder, f"{group_name}.csv")
            print(output_file_path)
            sorted_group = sorted_group.drop('Turbine_ID', axis=1)
            sorted_group.to_csv(output_file_path, index=False)