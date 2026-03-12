import pandas as pd
import os
from datetime import datetime, timedelta

import numpy as np

input_folders = ['./data_2013', './data_2014', './data_2015', './data_2016', './data_2017', './data_2018', './data_2019', './data_2020', './data_2021', './data_2022', './data_2023']
sources_with_failure_df = pd.read_csv('./sources_with_failure.csv', index_col=0)
models_with_failure_df = pd.read_csv('./models_with_failure.csv', index_col=0)

initial_number_of_sources = len(sources_with_failure_df.serial_number.tolist())
test_output_folder = '../../DataFolder/bhd/scenarios'
os.makedirs(test_output_folder, exist_ok=True)
output_df_per_source = {}
sources_failed = []
# total_read = 0

print(f'Initial number of sources: {initial_number_of_sources}')

for input_folder in input_folders:
    if sources_with_failure_df.shape[0] == 0:
        break

    for file_name in sorted(
            list(filter(
                lambda filename: filename.endswith(".csv"),
                os.listdir(input_folder))
            ),
            key=lambda filename: datetime.strptime(filename.split('.')[0], "%Y-%m-%d")):
        if sources_with_failure_df.shape[0] == 0:
            break

        if file_name.endswith(".csv"):
            file_path = os.path.join(input_folder, file_name)
            print(file_name)
            df = pd.read_csv(file_path)
            # total_read += 1
            print(f'Number of sources with failure left {sources_with_failure_df.shape[0]}')
            for _, current_source_row in sources_with_failure_df.iterrows():
                current_df_for_current_source = df[(df.serial_number == current_source_row.serial_number) & (df.capacity_bytes > 0)]

                if len(current_df_for_current_source.serial_number.tolist()) == 0:
                    continue

                if current_source_row.serial_number in output_df_per_source:
                    pass
                else:
                    not_null_columns = current_df_for_current_source.iloc[0].notnull()
                    columns_for_output_df_of_current_source = not_null_columns[not_null_columns].index.tolist()
                    output_df_per_source[current_source_row.serial_number] = pd.DataFrame([], columns=list(filter(lambda column_name: column_name != 'serial_number', columns_for_output_df_of_current_source)))

                row_to_append = []
                for column in output_df_per_source[current_source_row.serial_number]:
                    if column == 'failure':
                        if current_df_for_current_source.iloc[0][column] == 1:
                            sources_failed.append(current_source_row.serial_number)

                    if column == 'date':
                        row_to_append.append(current_df_for_current_source.iloc[0][column] + ' 00:00:00')
                    else:
                        row_to_append.append(current_df_for_current_source.iloc[0][column])

                output_df_per_source[current_source_row.serial_number].loc[len(output_df_per_source[current_source_row.serial_number])] = row_to_append

            for source in sources_failed:
                sources_with_failure_df = sources_with_failure_df[sources_with_failure_df['serial_number'] != source]

            sources_failed = []


for current_serial_number, current_df in output_df_per_source.items():
    output_file_path = os.path.join(test_output_folder, f"{current_serial_number}.csv")
    print(output_file_path)

    current_df.to_csv(output_file_path, index=False)
