# Copyright 2026 Anastasios Papadopoulos, Apostolos Giannoulidis
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import pandas as pd
import math
import statistics
import sys 

if sys.argv[1] == 'statistics':
    directory = '../../DataFolder/formula1/'
else:
    directory = '../../DataFolder/formula1/out/'

all_dataframes = []
sum_of_lengths = 0
list_of_dimensions = []
lengths = []

for dirpath, dirnames, filenames in os.walk(directory):
    dirnames.sort()
    filenames.sort()  

    for file_name in filenames:
            file_path = os.path.join(dirpath, file_name)
            print(file_path)

            reason_for_failure = file_path.split('/')[-1].split('.')[0].split('_')

            if reason_for_failure[-1].lower() == 'collision' \
                or reason_for_failure[-1].lower() == 'accident' \
                or reason_for_failure[-1].lower() == 'illness' \
                or reason_for_failure[-1].lower() == 'debris' \
                or (reason_for_failure[-1].lower() == 'damage' and reason_for_failure[-2].lower() == 'collision') \
                or (reason_for_failure[-1].lower() == 'off' and reason_for_failure[-2].lower() == 'spun'):
                    continue
                

            df = pd.read_csv(file_path)

            if df.shape[0] <= 4800: # do not account time series with length less than 20 minutes
                print(f'Skipping {file_path} because of {df.shape[0]} points')
                continue

            df.drop(columns=['X', 'Y', 'Z'], inplace=True)

            sum_of_lengths += df.shape[0]
            lengths.append(df.shape[0])
            list_of_dimensions.append(df.shape[1] - 1) # do not consider the Date column
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