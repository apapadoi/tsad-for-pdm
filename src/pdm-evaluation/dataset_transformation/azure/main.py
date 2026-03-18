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