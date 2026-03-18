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

from datetime import datetime
import statistics
from utils import loadDataset
import pandas as pd

dataset_names_list = [
    'cmapss',
    'navarchos',
    'femto',
    'ims',
    'edp-wt',
    'metropt-3',
    'xjtu',
    'bhd',
    'azure',
    'ai4i'
]

number_of_operational_cycles_until_failure = []

for dataset_name in dataset_names_list:
    dataset = loadDataset.get_dataset(dataset_name)

    if len(dataset['event_preferences']['failure']) == 0: # run to failure case
        for scenario in dataset['target_data']:
            number_of_operational_cycles_until_failure.append(scenario.shape[0])
    else:
        failure_data_df = dataset['event_data'][dataset['event_data']['type'] == 'failure']
        timestamp_column_name = dataset['dates']

        for scenario, current_source in zip(dataset['target_data'], dataset['target_sources']):
            splits = []
            failures_dates_for_current_source = failure_data_df[failure_data_df['source'] == current_source]['date'].sort_values().reset_index(drop=True)

            for index, current_failure_date in failures_dates_for_current_source.items():
                rows_before_current_failure = scenario[scenario[timestamp_column_name] < current_failure_date]

                number_of_operational_cycles_until_failure.append(rows_before_current_failure.shape[0])

                scenario = scenario[scenario[timestamp_column_name] >= current_failure_date]

            if scenario.shape[0] != 0:
                number_of_operational_cycles_until_failure.append(scenario.shape[0])


    print(dataset_name)


print(statistics.median(number_of_operational_cycles_until_failure)) # 339
print(min(number_of_operational_cycles_until_failure)) # 0
# 1420 cases are in the range of 0 - 999, overall cases are 1897