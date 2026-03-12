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

