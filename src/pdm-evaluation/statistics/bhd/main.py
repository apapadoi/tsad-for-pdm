import os
import pandas as pd
import math
import statistics
import sys 

if sys.argv[1] == 'statistics':
    directory = '../../DataFolder/bhd/'
else:
    directory = '../../DataFolder/bhd/scenarios'

def transform_date(date):
    if '2/25/18' in date:
        return date.replace('2/25/18', '2018-02-25')
    
    if '6/17/19' in date:
        return date.replace('6/17/19', '2019-06-17')

    return date

all_dataframes = []
sum_of_lengths = 0
list_of_dimensions = []
lengths = []
total_failed_after_being_installed_in_2023 = 0
for root, dirs, files in os.walk(directory):
    for file in files:
        if file.endswith('.csv'):
            file_path = os.path.join(root, file)

            df = pd.read_csv(file_path)
            df['date'] = df['date'].apply(transform_date)
            df['date'] = pd.to_datetime(df['date'])
            first_date = df['date'].iloc[0]

            if first_date >= pd.Timestamp('2023-01-02 00:00:00'):
                total_failed_after_being_installed_in_2023 += 1
                # print(f'{file} failed after being installed after 1st of January')
            
            last_date = df['date'].iloc[-1]

            # Check if the last date is before or equal to March 31, 2023
            # if last_date <= pd.Timestamp('2023-03-31 00:00:00'):
                # print(f'skipping dataframe with first row date: {df.iloc[0].date}')
                # continue
            
            found_smart_measurements = False
            for column in df.columns:
                if 'smart' in column:
                    found_smart_measurements = True
                    break

            if not found_smart_measurements:
                print(f'{file} has no SMART measurements')
                continue

            if df.shape[0] <= 10:
                print(f'skipping {file} has less than 10 data points')
                continue
            
            # TODO another option is interpolating using previous and next value
            # nan_indices = df.index[df.isnull().any(axis=1)].tolist()
            # nan_columns = df.columns[df.isnull().any()].tolist()

            df = df.dropna() # 1597 total NaN values
            sum_of_lengths += df.shape[0]
            lengths.append(df.shape[0])

            columns_to_drop = []
            for column in df.columns:
                if column != 'date' and 'smart' not in column:
                    columns_to_drop.append(column)
            
            df = df.drop(columns=columns_to_drop, axis=1)
            list_of_dimensions.append(df.shape[1] - 1) # do not consider the date column
            all_dataframes.append(df)

# TODO violin plots or box plots
combined_df = pd.concat(all_dataframes, ignore_index=True)
print(f'total scenarios: {len(all_dataframes)}')
print(f'#records: {combined_df.shape[0]}')
print(f'average scenario length: {sum_of_lengths / len(all_dataframes)}')
print(f'min scenario length: {min(lengths)}')
print(f'scenario length standard deviation: {statistics.stdev(lengths)}')
print(f'list_of_dimensions')
print(pd.Series(list_of_dimensions).describe())
print(f'scenario length')
print(pd.Series(lengths).describe())
print(f'PH: {math.ceil(0.1 * min(lengths))}')
print(f'{total_failed_after_being_installed_in_2023} failed after being installed after 1st of January')