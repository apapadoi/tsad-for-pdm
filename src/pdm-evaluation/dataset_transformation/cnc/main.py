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
from pathlib import Path
import os

# Get the directory where this script is located
script_dir = Path(__file__).parent

# Read the metadata.csv file
metadata_path = script_dir / "metadata.csv"
metadata_df = pd.read_csv(metadata_path)

# Display basic information about the dataframe
print(f"Shape of the metadata dataframe: {metadata_df.shape}")
assert metadata_df.shape == (968, 13)

# Create output directory
output_folder = '../../DataFolder/cnc'
os.makedirs(output_folder, exist_ok=True)

# Define the raw_data folder path
raw_data_folder = script_dir / "raw_data"

# Mode selection for data processing:
# 'failure_only': Save only experiments with CycleToFailure == 0 (file suffix: '_failure')
# 'failure_and_first': Save both failure (CycleToFailure == 0) and first experiment (largest CycleToFailure) as separate files
# 'all': Save all experiments concatenated together (file suffix: '_all')
PROCESSING_MODE = 'failure_only'  # Options: 'failure_only', 'failure_and_first', 'all'

# Initialize total row accumulator
total_rows_accumulated = 0

# Group by ToolIndex
for tool_index, tool_group in metadata_df.groupby('ToolIndex'):
    print(f"\n{'='*60}")
    print(f"Processing ToolIndex: {tool_index}")
    print(f"{'='*60}")

    # if tool_index != 11:
    #     continue
    
    # Sort tool_group by CycleToFailure in descending order
    tool_group = tool_group.sort_values(by='CycleToFailure', ascending=False).reset_index(drop=True)
    
    # Assert that the rows are in descending order by CycleToFailure
    cycle_values = tool_group['CycleToFailure'].tolist()
    assert cycle_values == sorted(cycle_values, reverse=True), \
        f"ToolIndex {tool_index}: Rows are not sorted in descending order by CycleToFailure"
    print(f"✓ Verified: Rows are sorted in descending order by CycleToFailure")
    
    # Get the largest CycleToFailure value (first row after sorting)
    largest_cycle_to_failure = tool_group.iloc[0]['CycleToFailure'] if len(tool_group) > 0 else None
    
    # Lists to store processed dataframes for this tool
    failure_df_list = []  # For CycleToFailure == 0
    first_experiment_df_list = []  # For largest CycleToFailure
    all_experiments_df_list = []  # For all experiments
    
    # Iterate over each ExperimentIndex in this tool group
    for idx, row in tool_group.iterrows():
        experiment_index = row['ExperimentIndex']
        cycle_to_failure = row['CycleToFailure']
        csv_file_path = raw_data_folder / f"{experiment_index}.csv"
        
        # Determine if this should be processed
        is_failure = (cycle_to_failure == 0)
        is_first_experiment = (cycle_to_failure == largest_cycle_to_failure)
        
        # Determine processing based on mode
        should_process = False
        target_lists = []
        
        if PROCESSING_MODE == 'failure_only':
            if is_failure:
                should_process = True
                target_lists = [failure_df_list]
        elif PROCESSING_MODE == 'failure_and_first':
            if is_failure:
                should_process = True
                target_lists = [failure_df_list]
            elif is_first_experiment:
                should_process = True
                target_lists = [first_experiment_df_list]
        elif PROCESSING_MODE == 'all':
            should_process = True
            target_lists = [all_experiments_df_list]
        
        if not should_process:
            print(f"Skipping: {experiment_index} (CycleToFailure={cycle_to_failure})")
            continue
        
        print(f"Processing: {experiment_index} (CycleToFailure={cycle_to_failure})")
        
        # Check if file exists
        if not csv_file_path.exists():
            print(f"  WARNING: File not found - {csv_file_path}")
            continue
        
        # Read the CSV file
        experiment_df = pd.read_csv(csv_file_path)
        print(f"  Original shape: {experiment_df.shape}")
        
        # Split into two dataframes
        # First 9 columns
        df_acc = experiment_df.iloc[:, :9].copy()
        # Rest of the columns
        df_current = experiment_df.iloc[:, 9:].copy()
        
        print(f"  Acceleration dataframe shape (before dropping NaN): {df_acc.shape}")
        print(f"  Current dataframe shape (before dropping NaN): {df_current.shape}")
        
        # Drop rows with NaN or empty values from both dataframes
        df_acc = df_acc.dropna()
        df_current = df_current.dropna()
        
        print(f"  Acceleration dataframe shape (after dropping NaN): {df_acc.shape}")
        print(f"  Current dataframe shape (after dropping NaN): {df_current.shape}")
        
        # Filter rows where 'Timestamps - Acc' exists in 'Timestamps - Current'
        if 'Timestamps - Acc' in df_acc.columns and 'Timestamps - Current' in df_current.columns:
            # Get the timestamps that exist in both
            common_timestamps = df_acc['Timestamps - Acc'].isin(df_current['Timestamps - Current'])
            df_acc_filtered = df_acc[common_timestamps].copy()
            
            print(f"  Filtered acceleration dataframe shape: {df_acc_filtered.shape}")
            
            # Reset indices for proper concatenation
            df_acc_filtered = df_acc_filtered.reset_index(drop=True)
            df_current = df_current.reset_index(drop=True)
            
            # Concatenate the two dataframes horizontally
            combined_df = pd.concat([df_acc_filtered, df_current], axis=1)
            print(f"  Combined dataframe shape: {combined_df.shape}")
        else:
            print(f"  WARNING: Expected timestamp columns not found")
            print(f"  First 9 columns: {df_acc.columns.tolist()}")
            print(f"  Remaining columns: {df_current.columns.tolist()[:5]}...")
            combined_df = experiment_df
        
        # Append to the appropriate list(s)
        for target_list in target_lists:
            target_list.append(combined_df)
        
        if PROCESSING_MODE == 'all':
            print(f"  Added to all experiments dataframe list")
        else:
            experiment_type = "failure" if is_failure else "first_experiment"
            print(f"  Added to {experiment_type} dataframe list")
    
    # Function to process and save a dataframe list
    def process_and_save_df(df_list, suffix):
        if not df_list:
            return 0
        
        final_df = pd.concat(df_list, axis=0, ignore_index=True)
        print(f"\nFinal concatenated dataframe shape for ToolIndex {tool_index} ({suffix}): {final_df.shape}")
        
        # Drop 'Timestamps - Current' column and rename 'Timestamps - Acc' to 'timestamp'
        if 'Timestamps - Current' in final_df.columns:
            final_df = final_df.drop(columns=['Timestamps - Current'])
            print(f"  Dropped 'Timestamps - Current' column")
        
        if 'Timestamps - Acc' in final_df.columns:
            final_df = final_df.rename(columns={'Timestamps - Acc': 'timestamp'})
            print(f"  Renamed 'Timestamps - Acc' to 'timestamp'")
        
        # Reset timestamp column to start from 0 and increase by 2 for each row
        if 'timestamp' in final_df.columns:
            final_df['timestamp'] = range(0, len(final_df) * 2, 2)
            print(f"  Reset timestamp column: starts at 0, increments by 2")
        
        print(f"Final dataframe shape before downsampling: {final_df.shape}")
        
        # Downsample to 1 row per 0.2 second (200ms windows)
        if 'timestamp' in final_df.columns:
            # Create a grouping key based on 0.2-second windows
            final_df['time_window'] = (final_df['timestamp'] // 200).astype(int)
            
            # Get all columns except timestamp and time_window
            feature_columns = [col for col in final_df.columns if col not in ['timestamp', 'time_window']]
            
            # Define aggregation functions
            agg_dict = {}
            for col in feature_columns:
                agg_dict[col] = ['min', 'max', 'mean', 'std', 'skew', ('kurtosis', lambda x: x.kurtosis())]
            
            # Group by time_window and aggregate
            downsampled_df = final_df.groupby('time_window').agg(agg_dict)
            
            # Flatten column names
            downsampled_df.columns = ['_'.join(col).strip() if col[1] else col[0] 
                                      for col in downsampled_df.columns.values]
            
            # Reset index and create proper timestamp column
            downsampled_df = downsampled_df.reset_index()
            downsampled_df['timestamp'] = downsampled_df['time_window'] * 200
            downsampled_df = downsampled_df.drop(columns=['time_window'])
            
            # Move timestamp to first column
            cols = ['timestamp'] + [col for col in downsampled_df.columns if col != 'timestamp']
            downsampled_df = downsampled_df[cols]
            
            final_df = downsampled_df
            print(f"  Downsampled to 1 row per 0.2 second (200ms windows)")
            print(f"Final dataframe shape after downsampling: {final_df.shape}")
        
        # Save to CSV
        output_file_path = os.path.join(output_folder, f"tool_{tool_index}_{suffix}.csv")
        final_df.to_csv(output_file_path, index=False)
        print(f"Saved to: {output_file_path}")
        
        return len(final_df)
    
    # Process and save based on mode
    rows_added = 0
    
    if PROCESSING_MODE == 'failure_only':
        if failure_df_list:
            rows_added += process_and_save_df(failure_df_list, 'failure')
        else:
            print(f"\nNo failure data found for ToolIndex {tool_index}")
    
    elif PROCESSING_MODE == 'failure_and_first':
        if failure_df_list:
            rows_added += process_and_save_df(failure_df_list, 'failure')
        else:
            print(f"\nNo failure data found for ToolIndex {tool_index}")
        
        if first_experiment_df_list:
            rows_added += process_and_save_df(first_experiment_df_list, 'first_experiment')
        else:
            print(f"\nNo first experiment data found for ToolIndex {tool_index}")
    
    elif PROCESSING_MODE == 'all':
        if all_experiments_df_list:
            rows_added += process_and_save_df(all_experiments_df_list, 'all')
        else:
            print(f"\nNo data found for ToolIndex {tool_index}")
    
    # Update total row accumulator
    total_rows_accumulated += rows_added
    
    print("---")

print("\n" + "="*60)
print("All CNC data processed successfully!")
print(f"Total rows accumulated across all tools: {total_rows_accumulated}")
print("="*60)

