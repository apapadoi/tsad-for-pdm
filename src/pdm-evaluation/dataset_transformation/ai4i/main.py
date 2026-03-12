import pandas as pd
import os
import sys
from datetime import datetime, timedelta


def generate_timestamps(start_timestamp, num_rows):
    timestamps = [start_timestamp + timedelta(days=i) for i in range(num_rows)]
    return timestamps


data_df = pd.read_csv('./ai4i2020.csv')

data_df = data_df.drop(columns=['UDI', 'Product ID', 'Type', 'TWF', 'HDF', 'PWF', 'OSF', 'RNF'])

start_timestamp = datetime.strptime('2000-01-01 00:01:40', '%Y-%m-%d %H:%M:%S')
num_rows = data_df.shape[0]
timestamps = generate_timestamps(start_timestamp, num_rows)

data_df.insert(0, 'Artificial_timestamp', timestamps)

test_output_folder = '../../DataFolder/ai4i/scenarios'
os.makedirs(test_output_folder, exist_ok=True)

failure_rows_df = data_df[data_df['Machine failure'] == 1].drop(columns=[column for column in data_df.columns if column != 'Artificial_timestamp'])

data_df['Previous_Machine_failure'] = data_df['Machine failure'].shift(1)
mask = (data_df['Tool wear [min]'] == 0) & (data_df['Previous_Machine_failure'] == 0)
reset_rows_df = data_df[mask].drop(columns=[column for column in data_df.columns if column != 'Artificial_timestamp'])
data_df = data_df.drop(columns=['Previous_Machine_failure'])


print(reset_rows_df.shape)


event_df = pd.DataFrame([], columns=['date', 'type', 'source', 'description'])

for index, row in failure_rows_df.iterrows():
    row_to_append = [
        row['Artificial_timestamp'],
        'failure',
        '1',
        'One failure from the 5 available failure types'
    ]

    event_df.loc[len(event_df)] = row_to_append

for index, row in reset_rows_df.iterrows():
    row_to_append = [
        row['Artificial_timestamp'],
        'reset',
        '1',
        'Tool replacement'
    ]

    event_df.loc[len(event_df)] = row_to_append

print(failure_rows_df.shape)
print(event_df.shape)
print(data_df.shape)

data_df = data_df.drop(columns=['Machine failure'])
final_df = pd.DataFrame([], columns=data_df.columns)

total_rows_less_than_12_gap = 0
total_episodes_less_than_12_gap = 0
failure_dates_to_remove = []
for failure_index, failure in failure_rows_df.iterrows():
    current_df = data_df[data_df['Artificial_timestamp'] <= failure.Artificial_timestamp]
    data_df = data_df[data_df['Artificial_timestamp'] > failure.Artificial_timestamp]
    print('current_df_shape', current_df.shape)

    if current_df.shape[0] <= 12:
        total_rows_less_than_12_gap += current_df.shape[0]
        total_episodes_less_than_12_gap += 1

        final_df = pd.concat([
            final_df,
            current_df
        ])

        failure_dates_to_remove.append(current_df.iloc[-1].Artificial_timestamp)
    else:
        if final_df.shape[0] == 0:
            final_df = current_df.copy()
        else:
            final_df = pd.concat([
                final_df,
                current_df
            ])

if data_df.shape[0] != 0:
    final_df = pd.concat([
        final_df,
        data_df
    ])

event_df = event_df[(event_df['type'] == 'reset') | ((~event_df['date'].isin(failure_dates_to_remove)) & (event_df['type'] == 'failure'))]

print('total_rows_less_than_12_gap', total_rows_less_than_12_gap)
print('total_episodes_less_than_12_gap', total_episodes_less_than_12_gap)

output_file_path = os.path.join(test_output_folder, '1.csv')
print(output_file_path)
print('final_df.shape', final_df.shape)
final_df.to_csv(output_file_path, index=False)


event_output_file_path = os.path.join(test_output_folder, 'events.csv')
print(event_output_file_path)
print('event_df.shape', event_df.shape)
event_df.to_csv(event_output_file_path, index=False)
