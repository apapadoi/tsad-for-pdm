import os
import pandas as pd
import math
import statistics
import sys 

directory = '../../DataFolder/ai4i/scenarios'

all_dataframes = []
sum_of_lengths = 0
list_of_dimensions = []
lengths = []
for root, dirs, files in os.walk(directory):
    for file in files:
        if file.endswith('.csv') and 'event' not in file:
            file_path = os.path.join(root, file)

            df = pd.read_csv(file_path)
            sum_of_lengths += df.shape[0]
            lengths.append(df.shape[0])
            list_of_dimensions.append(df.shape[1] - 1) # do not consider the Artificial_timestamp column
            all_dataframes.append(df)

# TODO violin plots or box plots
combined_df = pd.concat(all_dataframes, ignore_index=True)
print(f'#records: {combined_df.shape[0]}')
print(f'sum_of_lengths: {sum_of_lengths}')
print(f'average scenario length: {sum_of_lengths / len(all_dataframes)}')
print(f'min scenario length: {min(lengths)}')
print(f'scenario length standard deviation: {statistics.stdev(lengths) if len(lengths) >= 2 else 0}')
print(f'list_of_dimensions')
print(pd.Series(list_of_dimensions).describe())
print(f'scenario length')
print(pd.Series(lengths).describe())

events_df = pd.read_csv(f'{directory}/events.csv')
failures_df = events_df[events_df['type'] == 'failure']
sorted_group = combined_df.copy()

episode_lengths = []
for failure_index, failure in failures_df.iterrows():
    current_df = sorted_group[sorted_group['Artificial_timestamp'] <= failure.date]
    sorted_group = sorted_group[sorted_group['Artificial_timestamp'] > failure.date]

    if current_df.shape[0] > 12:
        episode_lengths.append(current_df.shape[0])


if sorted_group.shape[0] != 0:
    episode_lengths.append(sorted_group.shape[0])

print(pd.Series(episode_lengths).describe())
print(f'PH: {math.ceil(0.1 * min(episode_lengths))}')