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

scenarios_folder="../../DataFolder/edp-wt/scenarios"
data_list=[]
for file in os.listdir(scenarios_folder):
    if file.endswith('.csv') and "failures" not in file:
        file_path = os.path.join(scenarios_folder, file)
        df = pd.read_csv(file_path)
        df["Timestamp"] = pd.to_datetime(df["Timestamp"])
        df["source"] = [file.split(".csv")[0] for _i in range(df.shape[0])]
        data_list.append(df)


columns=data_list[0].columns.tolist()
for df in data_list:
    # Example preprocessing: Fill missing values with the mean of each column
    print(len(df.columns.tolist()))
    columns= set(columns).intersection(set(df.columns.tolist()))

# print("-------------------")
# print(len(columns))
# print("-------------------")
# for df in data_list:
#     # Example preprocessing: Fill missing values with the mean of each column
#     print(set(df.columns.tolist()).difference(columns))

# combine all dataframes into a single dataframe
combined_df = pd.concat(data_list, ignore_index=True)

combined_df.to_csv("EDP_combined.csv", index=False)

