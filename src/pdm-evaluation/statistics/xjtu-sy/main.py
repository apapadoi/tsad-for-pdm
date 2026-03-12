import os
import pandas as pd
import math
import statistics
import sys 

if sys.argv[1] == 'statistics':
    directory = '../../DataFolder/xjtu-sy/'
else:
    directory = '../../DataFolder/xjtu-sy/scenarios'

all_dataframes = []
sum_of_lengths = 0
list_of_dimensions = []
lengths = []
for root, dirs, files in os.walk(directory):
    for file in files:
        if file.endswith('.csv'):
            file_path = os.path.join(root, file)

            df = pd.read_csv(file_path)
            sum_of_lengths += df.shape[0]
            lengths.append(df.shape[0])
            list_of_dimensions.append(df.shape[1] - 1) # do not consider the timestamp column
            all_dataframes.append(df)

# TODO violin plots or box plots
combined_df = pd.concat(all_dataframes, ignore_index=True)
print(f'#records: {combined_df.shape[0]}')
print(f'sum_of_lengths: {sum_of_lengths}')
print(f'average scenario length: {sum_of_lengths / len(all_dataframes)}')
print(f'min scenario length: {min(lengths)}')
print(f'scenario length standard deviation: {statistics.stdev(lengths)}')
print(f'list_of_dimensions')
print(pd.Series(list_of_dimensions).describe())
print(f'scenario length')
print(pd.Series(lengths).describe())
print(f'PH: {math.ceil(0.1 * min(lengths))}')