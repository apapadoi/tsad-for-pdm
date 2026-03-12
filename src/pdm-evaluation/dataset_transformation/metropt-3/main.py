import pandas as pd
import os
import sys


signals_df = pd.read_csv('./MetroPT3(AirCompressor).csv').drop(columns=['Unnamed: 0'])
failures_df = pd.read_csv('./metropt3-failures.csv')

print(signals_df.shape)
print(failures_df.shape)

train_output_folder = '../../DataFolder/metropt-3/healthy'
os.makedirs(train_output_folder, exist_ok=True)

test_output_folder = '../../DataFolder/metropt-3/scenarios'
os.makedirs(test_output_folder, exist_ok=True)

if sys.argv[1] == 'statistics':
    sorted_signals_df = signals_df.sort_values(by='timestamp')
    sorted_signals_df['timestamp'] = pd.to_datetime(sorted_signals_df['timestamp'])

    sorted_failures_df = failures_df.sort_values(by='timestamp').reset_index(drop=True)

    for failure_index, failure in sorted_failures_df.iterrows():
        current_df = sorted_signals_df[sorted_signals_df['timestamp'] <= failure.timestamp]
        sorted_signals_df = sorted_signals_df[sorted_signals_df['timestamp'] > failure.timestamp]

        if current_df.shape[0] > 0:
            output_file_path = os.path.join(test_output_folder, f"{failure_index}.csv")
            print(output_file_path)
            current_df.to_csv(output_file_path, index=False)

    if sorted_signals_df.shape[0] != 0:
        output_file_path = os.path.join(train_output_folder, f"healthy.csv")
        print(output_file_path)
        sorted_signals_df.to_csv(output_file_path, index=False)
else:
    sorted_signals_df = signals_df.sort_values(by='timestamp')
    output_file_path = os.path.join(test_output_folder, f"1.csv")
    print(output_file_path)
    sorted_signals_df.to_csv(output_file_path, index=False)
