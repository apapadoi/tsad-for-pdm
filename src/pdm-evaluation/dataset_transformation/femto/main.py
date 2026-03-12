import pandas as pd
import os
from datetime import datetime, timedelta

import numpy as np

def generate_timestamps(start_timestamp, num_rows):
    timestamps = [start_timestamp + timedelta(days=i) for i in range(num_rows)]
    return timestamps

# Train data
initial_train_dataframes_per_bearing = {}

def read_csv_files_in_directory(directory):
    for root, dirs, files in os.walk(directory):
        print(f'root: {root}')
        for dir in dirs:
            print(f'dir: {dir}')
        print(f'len(files): {len(files)}')

        if 'Bearing' not in root:
            continue

        initial_train_dataframes_per_bearing[root[13:]] = []

        for file in sorted(files, key=lambda file_name: int(file_name.split('.')[0].split('_')[1])):
            if file.endswith('.csv') and 'acc' in file:
                file_path = os.path.join(root, file)
                print(file_path)
                df = pd.read_csv(file_path, header=None)
                initial_train_dataframes_per_bearing[root[13:]].append(df[[4, 5]])

read_csv_files_in_directory('training_set')

# print(len(initial_train_dataframes_per_bearing['Bearing1_1']))

# P2P

p2p_transformed_train_dfs_per_bearing = {}

for current_bearing in initial_train_dataframes_per_bearing.keys():
    p2p_transformed_train_dfs_per_bearing[current_bearing] = []

    for index, current_dataframe in enumerate(initial_train_dataframes_per_bearing[current_bearing]):
        p2p_transformed_train_dfs_per_bearing[current_bearing].append(pd.DataFrame([], columns=['p2p_' + str(column) for column in current_dataframe.columns]))
        row_to_append = []
        
        for column in current_dataframe.columns:
            row_to_append.append(np.ptp(current_dataframe[column]))

        p2p_transformed_train_dfs_per_bearing[current_bearing][index].loc[len(p2p_transformed_train_dfs_per_bearing[current_bearing][index])] = row_to_append
        
        for current_record_df in p2p_transformed_train_dfs_per_bearing[current_bearing]:
            assert current_record_df.shape[0] == 1
        
    assert len(p2p_transformed_train_dfs_per_bearing[current_bearing]) == len(initial_train_dataframes_per_bearing[current_bearing])

# print(len(p2p_transformed_train_dfs_per_bearing['Bearing1_1']))

# print(p2p_transformed_train_dfs_per_bearing['Bearing1_1'][0])

final_train_dataframes_per_bearing_p2p = {}

for current_bearing in p2p_transformed_train_dfs_per_bearing.keys():
    final_train_dataframes_per_bearing_p2p[current_bearing] = pd.DataFrame([], columns=p2p_transformed_train_dfs_per_bearing[current_bearing][0].columns)

    for current_dataframe in p2p_transformed_train_dfs_per_bearing[current_bearing]:
        final_train_dataframes_per_bearing_p2p[current_bearing].loc[len(final_train_dataframes_per_bearing_p2p[current_bearing])] = current_dataframe.iloc[0]
    
    assert final_train_dataframes_per_bearing_p2p[current_bearing].shape[1] == p2p_transformed_train_dfs_per_bearing[current_bearing][0].shape[1] and \
            final_train_dataframes_per_bearing_p2p[current_bearing].shape[0] == len(p2p_transformed_train_dfs_per_bearing[current_bearing])
    

# RMS
rms_transformed_train_dfs_per_bearing = {}

for current_bearing in initial_train_dataframes_per_bearing.keys():
    rms_transformed_train_dfs_per_bearing[current_bearing] = []

    for index, current_dataframe in enumerate(initial_train_dataframes_per_bearing[current_bearing]):
        rms_transformed_train_dfs_per_bearing[current_bearing].append(pd.DataFrame([], columns=['rms_' + str(column) for column in current_dataframe.columns]))
        row_to_append = []
        
        for column in current_dataframe.columns:
            row_to_append.append(
                np.sqrt(
                    current_dataframe[column].apply(lambda value: value ** 2).sum() / len(current_dataframe[column])
                )
            )

        rms_transformed_train_dfs_per_bearing[current_bearing][index].loc[len(rms_transformed_train_dfs_per_bearing[current_bearing][index])] = row_to_append
        
        for current_record_df in rms_transformed_train_dfs_per_bearing[current_bearing]:
            assert current_record_df.shape[0] == 1
        
    assert len(rms_transformed_train_dfs_per_bearing[current_bearing]) == len(initial_train_dataframes_per_bearing[current_bearing])

# print(len(rms_transformed_train_dfs_per_bearing['Bearing1_1']))

final_train_dataframes_per_bearing_rms = {}

for current_bearing in rms_transformed_train_dfs_per_bearing.keys():
    final_train_dataframes_per_bearing_rms[current_bearing] = pd.DataFrame([], columns=rms_transformed_train_dfs_per_bearing[current_bearing][0].columns)

    for current_dataframe in rms_transformed_train_dfs_per_bearing[current_bearing]:
        final_train_dataframes_per_bearing_rms[current_bearing].loc[len(final_train_dataframes_per_bearing_rms[current_bearing])] = current_dataframe.iloc[0]
    
    assert final_train_dataframes_per_bearing_rms[current_bearing].shape[1] == rms_transformed_train_dfs_per_bearing[current_bearing][0].shape[1] and \
            final_train_dataframes_per_bearing_rms[current_bearing].shape[0] == len(rms_transformed_train_dfs_per_bearing[current_bearing])
    

# FFT
    # https://docs.scipy.org/doc/scipy/tutorial/fft.html
from scipy.fft import fft, fftfreq

# sample spacing
T = 1.0 / 100
NUMBER_OF_BINS = 20

fft_transformed_train_dfs_per_bearing = {}

for current_bearing in initial_train_dataframes_per_bearing.keys():
    fft_transformed_train_dfs_per_bearing[current_bearing] = []

    for index, current_dataframe in enumerate(initial_train_dataframes_per_bearing[current_bearing]):
        fft_transformed_train_dfs_per_bearing[current_bearing].append(pd.DataFrame([], columns=['fft_' + str(column) + '_' + str(i) for column in current_dataframe.columns for i in range(NUMBER_OF_BINS)]))
        rows_to_append = []
        
        for column in current_dataframe.columns:
            N = len(current_dataframe[column])
            current_data = np.array(current_dataframe[column].tolist())

            current_fft_result = fft(current_data)

            hist, bin_edges = np.histogram(2.0/N * np.abs(current_fft_result[0:N//2]), bins=NUMBER_OF_BINS)

            rows_to_append.append([max(current_data[(current_data>=i)&(current_data<i+1)]) if current_data[(current_data>=i)&(current_data<i+1)].size else 0 for i in bin_edges[:-1]])

        fft_transformed_train_dfs_per_bearing[current_bearing][index].loc[len(fft_transformed_train_dfs_per_bearing[current_bearing][index])] = [item for sublist in rows_to_append for item in sublist]
        
        for current_record_df in fft_transformed_train_dfs_per_bearing[current_bearing]:
            assert current_record_df.shape[0] == 1
        
    assert len(fft_transformed_train_dfs_per_bearing[current_bearing]) == len(initial_train_dataframes_per_bearing[current_bearing])

# print(len(fft_transformed_train_dfs_per_bearing['Bearing1_1']))

final_train_dataframes_per_bearing_fft = {}

for current_bearing in fft_transformed_train_dfs_per_bearing.keys():
    final_train_dataframes_per_bearing_fft[current_bearing] = pd.DataFrame([], columns=fft_transformed_train_dfs_per_bearing[current_bearing][0].columns)

    for current_dataframe in fft_transformed_train_dfs_per_bearing[current_bearing]:
        final_train_dataframes_per_bearing_fft[current_bearing].loc[len(final_train_dataframes_per_bearing_fft[current_bearing])] = current_dataframe.iloc[0]
    
    assert final_train_dataframes_per_bearing_fft[current_bearing].shape[1] == fft_transformed_train_dfs_per_bearing[current_bearing][0].shape[1] and \
            final_train_dataframes_per_bearing_fft[current_bearing].shape[0] == len(fft_transformed_train_dfs_per_bearing[current_bearing])

train_output_folder = '../../DataFolder/femto/healthy'
os.makedirs(train_output_folder, exist_ok=True)

for key in final_train_dataframes_per_bearing_rms.keys():
    start_timestamp = datetime.strptime('2000-01-01 00:01:40', '%Y-%m-%d %H:%M:%S')

    current_train_df = pd.concat([
        final_train_dataframes_per_bearing_p2p[key], 
        final_train_dataframes_per_bearing_rms[key], 
        final_train_dataframes_per_bearing_fft[key]
    ], axis=1)

    num_rows = current_train_df.shape[0]
    timestamps = generate_timestamps(start_timestamp, num_rows)
    current_train_df.insert(0, 'Artificial_timestamp', timestamps)

    output_file_path = os.path.join(train_output_folder, f"{key}.csv")
    current_train_df.to_csv(output_file_path, index=False)

# Test
initial_test_dataframes_per_bearing = {}

def read_csv_files_in_directory(directory):
    for root, dirs, files in os.walk(directory):
        print(f'root: {root}')
        for dir in dirs:
            print(f'dir: {dir}')
        print(f'len(files): {len(files)}')

        if 'Bearing' not in root:
            continue

        initial_test_dataframes_per_bearing[root[12:]] = []

        for file in sorted(files, key=lambda file_name: int(file_name.split('.')[0].split('_')[1])):
            if file.endswith('.csv') and 'acc' in file:
                file_path = os.path.join(root, file)
                print(file_path)
                df = pd.read_csv(file_path, header=None)
                initial_test_dataframes_per_bearing[root[12:]].append(df[[4, 5]])

read_csv_files_in_directory('testing_set')

# P2P
p2p_transformed_test_dfs_per_bearing = {}

for current_bearing in initial_test_dataframes_per_bearing.keys():
    p2p_transformed_test_dfs_per_bearing[current_bearing] = []

    for index, current_dataframe in enumerate(initial_test_dataframes_per_bearing[current_bearing]):
        p2p_transformed_test_dfs_per_bearing[current_bearing].append(pd.DataFrame([], columns=['p2p_' + str(column) for column in current_dataframe.columns]))
        row_to_append = []
        
        for column in current_dataframe.columns:
            row_to_append.append(np.ptp(current_dataframe[column]))

        p2p_transformed_test_dfs_per_bearing[current_bearing][index].loc[len(p2p_transformed_test_dfs_per_bearing[current_bearing][index])] = row_to_append
        
        for current_record_df in p2p_transformed_test_dfs_per_bearing[current_bearing]:
            assert current_record_df.shape[0] == 1
        
    assert len(p2p_transformed_test_dfs_per_bearing[current_bearing]) == len(initial_test_dataframes_per_bearing[current_bearing])

final_test_dataframes_per_bearing_p2p = {}

for current_bearing in p2p_transformed_test_dfs_per_bearing.keys():
    final_test_dataframes_per_bearing_p2p[current_bearing] = pd.DataFrame([], columns=p2p_transformed_test_dfs_per_bearing[current_bearing][0].columns)

    for current_dataframe in p2p_transformed_test_dfs_per_bearing[current_bearing]:
        final_test_dataframes_per_bearing_p2p[current_bearing].loc[len(final_test_dataframes_per_bearing_p2p[current_bearing])] = current_dataframe.iloc[0]
    
    assert final_test_dataframes_per_bearing_p2p[current_bearing].shape[1] == p2p_transformed_test_dfs_per_bearing[current_bearing][0].shape[1] and \
            final_test_dataframes_per_bearing_p2p[current_bearing].shape[0] == len(p2p_transformed_test_dfs_per_bearing[current_bearing])
    
# RMS
rms_transformed_test_dfs_per_bearing = {}

for current_bearing in initial_test_dataframes_per_bearing.keys():
    rms_transformed_test_dfs_per_bearing[current_bearing] = []

    for index, current_dataframe in enumerate(initial_test_dataframes_per_bearing[current_bearing]):
        rms_transformed_test_dfs_per_bearing[current_bearing].append(pd.DataFrame([], columns=['rms_' + str(column) for column in current_dataframe.columns]))
        row_to_append = []
        
        for column in current_dataframe.columns:
            row_to_append.append(
                np.sqrt(
                    current_dataframe[column].apply(lambda value: value ** 2).sum() / len(current_dataframe[column])
                )
            )

        rms_transformed_test_dfs_per_bearing[current_bearing][index].loc[len(rms_transformed_test_dfs_per_bearing[current_bearing][index])] = row_to_append
        
        for current_record_df in rms_transformed_test_dfs_per_bearing[current_bearing]:
            assert current_record_df.shape[0] == 1
        
    assert len(rms_transformed_test_dfs_per_bearing[current_bearing]) == len(initial_test_dataframes_per_bearing[current_bearing])

final_test_dataframes_per_bearing_rms = {}

for current_bearing in rms_transformed_test_dfs_per_bearing.keys():
    final_test_dataframes_per_bearing_rms[current_bearing] = pd.DataFrame([], columns=rms_transformed_test_dfs_per_bearing[current_bearing][0].columns)

    for current_dataframe in rms_transformed_test_dfs_per_bearing[current_bearing]:
        final_test_dataframes_per_bearing_rms[current_bearing].loc[len(final_test_dataframes_per_bearing_rms[current_bearing])] = current_dataframe.iloc[0]
    
    assert final_test_dataframes_per_bearing_rms[current_bearing].shape[1] == rms_transformed_test_dfs_per_bearing[current_bearing][0].shape[1] and \
            final_test_dataframes_per_bearing_rms[current_bearing].shape[0] == len(rms_transformed_test_dfs_per_bearing[current_bearing])
    

# FFT
# https://docs.scipy.org/doc/scipy/tutorial/fft.html
from scipy.fft import fft, fftfreq

# sample spacing
T = 1.0 / 100
NUMBER_OF_BINS = 20

fft_transformed_test_dfs_per_bearing = {}

for current_bearing in initial_test_dataframes_per_bearing.keys():
    fft_transformed_test_dfs_per_bearing[current_bearing] = []

    for index, current_dataframe in enumerate(initial_test_dataframes_per_bearing[current_bearing]):
        fft_transformed_test_dfs_per_bearing[current_bearing].append(pd.DataFrame([], columns=['fft_' + str(column) + '_' + str(i) for column in current_dataframe.columns for i in range(NUMBER_OF_BINS)]))
        rows_to_append = []
        
        for column in current_dataframe.columns:
            N = len(current_dataframe[column])
            current_data = np.array(current_dataframe[column].tolist())

            current_fft_result = fft(current_data)

            hist, bin_edges = np.histogram(2.0/N * np.abs(current_fft_result[0:N//2]), bins=NUMBER_OF_BINS)

            rows_to_append.append([max(current_data[(current_data>=i)&(current_data<i+1)]) if current_data[(current_data>=i)&(current_data<i+1)].size else 0 for i in bin_edges[:-1]])

        fft_transformed_test_dfs_per_bearing[current_bearing][index].loc[len(fft_transformed_test_dfs_per_bearing[current_bearing][index])] = [item for sublist in rows_to_append for item in sublist]
        
        for current_record_df in fft_transformed_test_dfs_per_bearing[current_bearing]:
            assert current_record_df.shape[0] == 1
        
    assert len(fft_transformed_test_dfs_per_bearing[current_bearing]) == len(initial_test_dataframes_per_bearing[current_bearing])

final_test_dataframes_per_bearing_fft = {}

for current_bearing in fft_transformed_test_dfs_per_bearing.keys():
    final_test_dataframes_per_bearing_fft[current_bearing] = pd.DataFrame([], columns=fft_transformed_test_dfs_per_bearing[current_bearing][0].columns)

    for current_dataframe in fft_transformed_test_dfs_per_bearing[current_bearing]:
        final_test_dataframes_per_bearing_fft[current_bearing].loc[len(final_test_dataframes_per_bearing_fft[current_bearing])] = current_dataframe.iloc[0]
    
    assert final_test_dataframes_per_bearing_fft[current_bearing].shape[1] == fft_transformed_test_dfs_per_bearing[current_bearing][0].shape[1] and \
            final_test_dataframes_per_bearing_fft[current_bearing].shape[0] == len(fft_transformed_test_dfs_per_bearing[current_bearing])
    

test_output_folder = '../../DataFolder/femto/scenarios'
os.makedirs(test_output_folder, exist_ok=True)

for key in final_test_dataframes_per_bearing_rms.keys():
    start_timestamp = datetime.strptime('2000-01-01 00:01:40', '%Y-%m-%d %H:%M:%S')

    current_test_df = pd.concat([
        final_test_dataframes_per_bearing_p2p[key], 
        final_test_dataframes_per_bearing_rms[key], 
        final_test_dataframes_per_bearing_fft[key]
    ], axis=1)

    num_rows = current_test_df.shape[0]
    timestamps = generate_timestamps(start_timestamp, num_rows)
    current_test_df.insert(0, 'Artificial_timestamp', timestamps)

    output_file_path = os.path.join(test_output_folder, f"{key}.csv")
    current_test_df.to_csv(output_file_path, index=False)
