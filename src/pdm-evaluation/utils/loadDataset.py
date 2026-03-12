import os
import pickle
import statistics
import sys
import math
import subprocess
from collections import defaultdict
# import subprocess

import pandas as pd

from utils import navarchos_data, utils
from pdm_evaluation_types.types import EventPreferences, EventPreferencesTuple



def load_pickle(name):
    with open(f'./DataFolder/{name}', 'rb') as handle:
        dataset = pickle.load(handle)

        dataset["beta"] = 1
        if len(dataset["historic_sources"])==0:
            dataset["min_historic_scenario_len"] = sys.maxsize
        else:
            dataset["min_historic_scenario_len"] = min(df.shape[0] for df in dataset["historic_data"])
        dataset["min_target_scenario_len"] = min(df.shape[0] for df in dataset["target_data"])
        if "max_wait_time" not in dataset:
            dataset["max_wait_time"] = max(dataset["min_target_scenario_len"] // 10, 10)
        elif dataset["max_wait_time"] is None:
            dataset["max_wait_time"] = max(dataset["min_target_scenario_len"]//10,10)

        if dataset["predictive_horizon"] is None:
            dataset["predictive_horizon"] = max(dataset["min_target_scenario_len"]//10,2)

        return dataset



# todo check correct reading of files
# NOTE max_wait_time is calculated based on min episode and not min scenario
def cmapss(semi):
    historic_data = []
    historic_sources = []
    folder_path = './DataFolder/c_mapss/healthy/'
    file_list = os.listdir(folder_path)
    if semi:
        for index, file_name in enumerate(sorted(file_list, key=lambda file_name_sorting: file_name_sorting.split('.')[0])):
            file_path = os.path.join(folder_path, file_name)
            # print(file_path)

            df = pd.read_csv(file_path, header=0)
            # print(df.columns)
            historic_data.append(df)
            historic_sources.append(file_name.split('.')[0])

    target_data = []
    target_sources = []
    folder_path = './DataFolder/c_mapss/scenarios/'
    file_list = os.listdir(folder_path)
    for index, file_name in enumerate(sorted(file_list, key=lambda file_name_sorting: file_name_sorting.split('.')[0])):
        if file_name.split('.')[0] not in historic_sources and semi:
            continue


        file_path = os.path.join(folder_path, file_name)
        # print(file_path)

        df = pd.read_csv(file_path, header=0)
        # print(df.columns)
        target_data.append(df)
        target_sources.append(file_name.split('.')[0])

    if semi:
        assert all(x == y for x, y in zip(target_sources, historic_sources))

    event_data = pd.DataFrame(columns=["date", "type", "source", "description"])

    event_preferences: EventPreferences = {
        'failure': [],
        'reset': []
    }

    dataset={}
    dataset["dates"]="Artificial_timestamp"
    dataset["event_preferences"]=event_preferences
    dataset["event_data"]=event_data
    dataset["target_data"]=target_data
    dataset["target_sources"]=target_sources
    dataset["historic_data"]=historic_data
    dataset["historic_sources"]=historic_sources
    dataset["predictive_horizon"]='13 days'
    dataset["slide"]=29
    dataset["lead"]='2 days'
    dataset["beta"]=1
    dataset["min_historic_scenario_len"] = min(df.shape[0] for df in historic_data) if semi else sys.maxsize
    dataset["min_target_scenario_len"] = min(df.shape[0] for df in target_data)
    dataset["max_wait_time"] = math.ceil((1/3) * dataset["min_target_scenario_len"])

    return dataset


def navarchos(transformation="raw"):
    historic_data = []
    sourceori,dfs,dfevents=navarchos_data.localNavarchosSimulation(datasetname=transformation, oilInReset=False,ExcludeNoInformationVehicles=True)
    target_data=dfs

    event_data = dfevents

    event_preferences: EventPreferences = {
        'failure': [
            EventPreferencesTuple(description='*', type='fail', source='*', target_sources='=')
        ],
        'reset': [
            EventPreferencesTuple(description='*', type='reset', source='*', target_sources='='),
            EventPreferencesTuple(description='*', type='fail', source='*', target_sources='=')
        ]
    }

    dataset = {}
    dataset["dates"] = "dt"
    dataset["event_preferences"] = event_preferences
    dataset["event_data"] = event_data
    dataset["target_data"] = target_data
    dataset["target_sources"] = sourceori
    dataset["historic_data"] = historic_data
    dataset["historic_sources"] = []
    dataset["predictive_horizon"] = '15 days'
    dataset["slide"] = 15
    dataset["lead"] = '1 days'
    dataset["beta"] = 1
    dataset["min_historic_scenario_len"] = sys.maxsize
    dataset["min_target_scenario_len"] = min(df.shape[0] for df in target_data)
    dataset["max_wait_time"] = 3200

    return dataset


def femto(semi):
    historic_data = []
    historic_sources = []
    if semi:
        source_mapping = {
            'Bearing1_3': 'Bearing1_1',
            'Bearing1_4': 'Bearing1_2',
            'Bearing2_3': 'Bearing2_1',
            'Bearing2_4': 'Bearing2_2',
            'Bearing3_3': 'Bearing3_2',
        }

        folder_path = './DataFolder/femto/healthy/'
        file_list = os.listdir(folder_path)
        for index, file_name in enumerate(sorted(file_list, key=lambda file_name_sorting: file_name_sorting.split('.')[0])):
            if file_name.split('.')[0] not in source_mapping:
                continue

            file_path = os.path.join(folder_path, file_name)
            # print(file_path)

            df = pd.read_csv(file_path, header=0)
            # print(df.columns)

            if '3_3' in file_name.split('.')[0]:
                historic_data.append(df)
                historic_sources.append('Bearing3_1')

            historic_data.append(df)
            historic_sources.append(source_mapping[file_name.split('.')[0]])
            

    target_data = []
    target_sources = []
    folder_path = './DataFolder/femto/scenarios/'
    file_list = os.listdir(folder_path)
    for index, file_name in enumerate(sorted(file_list, key=lambda file_name_sorting: file_name_sorting.split('.')[0])):
        # if index == 10:
        #     break

        file_path = os.path.join(folder_path, file_name)
        # print(file_path)

        df = pd.read_csv(file_path, header=0)
        # print(df.columns)
        target_data.append(df)
        target_sources.append(file_name.split('.')[0])

    event_data = pd.DataFrame(columns=["date", "type", "source", "description"])

    event_preferences: EventPreferences = {
        'failure': [],
        'reset': []
    }

    dataset={}
    dataset["dates"]="Artificial_timestamp"
    dataset["event_preferences"]=event_preferences
    dataset["event_data"]=event_data
    dataset["target_data"]=target_data
    dataset["target_sources"]=target_sources
    dataset["historic_data"]=historic_data
    dataset["historic_sources"]=historic_sources
    dataset["predictive_horizon"]='52 days'
    dataset["slide"]=117
    dataset["lead"]='2 days'
    dataset["beta"]=1
    dataset["min_historic_scenario_len"] = min(df.shape[0] for df in historic_data) if semi else sys.maxsize
    dataset["min_target_scenario_len"] = min(df.shape[0] for df in target_data)
    dataset["max_wait_time"] = math.ceil((1/3) * dataset["min_target_scenario_len"])

    return dataset


def ims():
    historic_data = []
    historic_sources = []

    target_data = []
    target_sources = []
    folder_path = './DataFolder/ims/scenarios/'
    file_list = os.listdir(folder_path)
    for index, file_name in enumerate(file_list):
        file_path = os.path.join(folder_path, file_name)
        # print(file_path)

        df = pd.read_csv(file_path, header=0)
        # print(df.columns)
        target_data.append(df)
        target_sources.append(file_name.split('.')[0])

    event_data = pd.DataFrame(columns=["date", "type", "source", "description"])

    event_preferences: EventPreferences = {
        'failure': [],
        'reset': []
    }

    dataset={}
    dataset["dates"]="Artificial_timestamp"
    dataset["event_preferences"]=event_preferences
    dataset["event_data"]=event_data
    dataset["target_data"]=target_data
    dataset["target_sources"]=target_sources
    dataset["historic_data"]=historic_data
    dataset["historic_sources"]=historic_sources
    dataset["predictive_horizon"]='99 days'
    dataset["slide"]=225
    dataset["lead"]='2 days'
    dataset["beta"]=1
    dataset["min_historic_scenario_len"] = sys.maxsize
    dataset["min_target_scenario_len"] = min(df.shape[0] for df in target_data)
    dataset["max_wait_time"] = 328


    return dataset


def edp():
    historic_data = []
    historic_sources = []

    target_data = []
    target_sources = []
    folder_path = './DataFolder/edp-wt/scenarios/'
    file_list = os.listdir(folder_path)
    failures_df = pd.read_csv('./DataFolder/edp-wt/scenarios/failures.csv')
    failures_df['date'] = pd.to_datetime(failures_df['date'])
    
    episode_lengths = []
    for index, file_name in enumerate(file_list):
        if 'fail' in file_name:
            continue

        # if index == 10:
        #     break

        file_path = os.path.join(folder_path, file_name)
        # print(file_path)

        df = pd.read_csv(file_path, header=0)
        df = df.ffill()
        df['Timestamp'] = pd.to_datetime(df['Timestamp'])
        df = df.drop_duplicates(subset=['Timestamp'])

        constant_columns = [col for col in df.columns if df[col].nunique() == 1]

        df = df.drop(columns=constant_columns, axis=1)

        target_data.append(df.copy())
        target_sources.append(file_name.split('.')[0])

        failures_for_current_source = failures_df[failures_df['source'] == file_name.split('.')[0]]
        failures_for_current_source = failures_for_current_source.sort_values(by='date').reset_index(drop=True)

        # final_df = pd.DataFrame([], columns=df.columns)

        for failure_index, failure in failures_for_current_source.iterrows():
            current_df = df[df['Timestamp'] <= failure.date]
            episode_lengths.append(current_df.shape[0])
            # final_df = pd.concat([final_df, current_df])
            df = df[df['Timestamp'] > failure.date]

        episode_lengths.append(df.shape[0])

    event_data = pd.read_csv('./DataFolder/edp-wt/scenarios/failures.csv')
    event_data['date'] = pd.to_datetime(event_data['date'])

    event_preferences: EventPreferences = {
        'failure': [
            EventPreferencesTuple(description='*', type='failure', source='*', target_sources='=')
        ],
        'reset': [
            EventPreferencesTuple(description='*', type='reset', source='*', target_sources='='),
            EventPreferencesTuple(description='*', type='failure', source='*', target_sources='=')
        ]
    }

    dataset={}
    dataset["dates"]="Timestamp"
    dataset["event_preferences"]=event_preferences
    dataset["event_data"]=event_data
    dataset["target_data"]=target_data
    dataset["target_sources"]=target_sources
    dataset["historic_data"]=historic_data
    dataset["historic_sources"]=historic_sources
    dataset["predictive_horizon"]="86400 minutes"
    dataset["slide"]=0
    dataset["lead"]="2880 minutes"
    dataset["beta"]=1
    dataset["min_historic_scenario_len"] = sys.maxsize
    dataset["min_target_scenario_len"] = min(df.shape[0] for df in target_data)
    dataset["max_wait_time"] = math.ceil((1/3) * min(episode_lengths))
    
    return dataset


def metropt():
    historic_data = []
    historic_sources = []

    target_data = []
    target_sources = []
    folder_path = './DataFolder/metropt-3/scenarios/'
    file_list = os.listdir(folder_path)
    failures_df = pd.read_csv('./DataFolder/metropt-3/scenarios/failures.csv')
    failures_df['date'] = pd.to_datetime(failures_df['date'])

    episode_lengths = []
    for index, file_name in enumerate(file_list):
        if 'fail' in file_name:
            continue

        file_path = os.path.join(folder_path, file_name)
        # print(file_path)

        df = pd.read_csv(file_path, header=0)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.index = df['timestamp'].copy()
        df.drop(columns=['timestamp'], inplace=True)

        df = df.resample(f'10T').mean()

        df['timestamp'] = df.index
        df = df.reset_index(drop=True)
        df.dropna(inplace=True)

        target_data.append(df.copy())
        target_sources.append(file_name.split('.')[0])

        failures_for_current_source = failures_df.sort_values(by='date').reset_index(drop=True)
        for failure_index, failure in failures_df.iterrows():
            current_df = df[df['timestamp'] <= failure.date]
            episode_lengths.append(current_df.shape[0])
            # final_df = pd.concat([final_df, current_df])
            df = df[df['timestamp'] > failure.date]

        episode_lengths.append(df.shape[0])


    event_data = pd.read_csv('./DataFolder/metropt-3/scenarios/failures.csv')
    event_data['date'] = pd.to_datetime(event_data['date'])
    event_data['source'] = event_data['source'].astype(str)

    event_preferences: EventPreferences = {
        'failure': [
            EventPreferencesTuple(description='*', type='failure', source='*', target_sources='=')
        ],
        'reset': [
            EventPreferencesTuple(description='*', type='reset', source='*', target_sources='='),
            EventPreferencesTuple(description='*', type='failure', source='*', target_sources='=')
        ]
    }

    dataset={}
    dataset["dates"]="timestamp"
    dataset["event_preferences"]=event_preferences
    dataset["event_data"]=event_data
    dataset["target_data"]=target_data
    dataset["target_sources"]=target_sources
    dataset["historic_data"]=historic_data
    dataset["historic_sources"]=historic_sources
    dataset["predictive_horizon"]="48 hours"
    dataset["slide"]=720
    dataset["lead"]="2 hours"
    dataset["beta"]=1
    dataset["min_historic_scenario_len"] = sys.maxsize
    dataset["min_target_scenario_len"] = min(df.shape[0] for df in target_data)
    dataset["max_wait_time"] = math.ceil((1/3) * min(episode_lengths))

    return dataset


def azure():
    dftelemetry = pd.read_csv("./DataFolder/azure/PdM_telemetry.csv", header=0)
    dfmainentance = pd.read_csv("./DataFolder/azure/PdM_maint.csv", header=0)
    dferrors = pd.read_csv("./DataFolder/azure/PdM_errors.csv", header=0)
    dffailures = pd.read_csv("./DataFolder/azure/PdM_failures.csv", header=0)
    dftelemetry["machineID"]=[str(m_id) for m_id in dftelemetry["machineID"].values]
    dfmainentance["machineID"]=[str(m_id) for m_id in dfmainentance["machineID"].values]
    dferrors["machineID"]=[str(m_id) for m_id in dferrors["machineID"].values]
    dffailures["machineID"]=[str(m_id) for m_id in dffailures["machineID"].values]

    #print(dferrors.head())
    #print(dfmainentance.head())
    #print(dffailures.head())

    #print(dftelemetry.head())

    event_data = pd.DataFrame(columns=["date", "type", "source", "description"])

    dferrors=dferrors.rename({'datetime': 'date', 'machineID': 'source', 'errorID' : 'description'}, axis=1)
    dferrors['type']=["error" for i in dferrors.index]

    event_data=pd.concat([event_data,dferrors], ignore_index=True)

    dfmainentance = dfmainentance.rename({'datetime': 'date', 'machineID': 'source', 'comp': 'description'}, axis=1)
    dfmainentance['type'] = ["maintenance" for i in dfmainentance.index]

    event_data = pd.concat([event_data, dfmainentance], ignore_index=True)

    dffailures = dffailures.rename({'datetime': 'date', 'machineID': 'source', 'failure': 'description'}, axis=1)
    dffailures['type'] = ["failure" for i in dffailures.index]

    event_data = pd.concat([event_data, dffailures], ignore_index=True)

    event_data['date']=pd.to_datetime(event_data['date'])

    event_preferences: EventPreferences = {
        'failure': [
            EventPreferencesTuple(description='*', type='failure', source='*', target_sources='=')
        ],
        'reset': [
            #EventPreferencesTuple(description='*', type='reset', source='*', target_sources='='),
            EventPreferencesTuple(description='*', type='failure', source='*', target_sources='=')
        ]
    }

    historic_data = []
    historic_sources = []
    episode_lengths =[]
    target_data = []
    target_sources = []
    for machine_idi in range(1,100):
        machine_id=str(machine_idi)
        df = dftelemetry[dftelemetry["machineID"] == machine_id]
        df = df.drop(["machineID"],axis=1)
        df = df.rename({'datetime': 'timestamp'}, axis=1)
        df['timestamp']=pd.to_datetime(df['timestamp'])
        df=df.sort_values(by='timestamp')
        target_data.append(df)
        target_sources.append(machine_id)

        dftemp=df.copy()
        ailures_for_current_source = event_data[event_data["source"]==machine_id].sort_values(by='date')
        for failure_index, failure in ailures_for_current_source.iterrows():
            current_df = dftemp[dftemp['timestamp'] <= failure.date]
            if current_df.shape[0] > 0:
                episode_lengths.append(current_df.shape[0])
            # final_df = pd.concat([final_df, current_df])
            dftemp = dftemp[dftemp['timestamp'] > failure.date]
        #if dftemp.shape[0] > 0: # these are without failure
        #    episode_lengths.append(dftemp.shape[0])
    #print(episode_lengths)
            
    #print(f"Ph + slide: {min(episode_lengths)/3}")
    #print(f"Ph: {min(episode_lengths)/10}")
    #episode_lengths.sort()
    #print(episode_lengths)

    dataset = {}
    dataset["dates"] = "timestamp"
    dataset["event_preferences"] = event_preferences
    dataset["event_data"] = event_data
    dataset["target_data"] = target_data
    dataset["target_sources"] = target_sources
    dataset["historic_data"] = historic_data
    dataset["historic_sources"] = historic_sources
    dataset["predictive_horizon"] = "96 hours" # there are many episodes of length 1,2,3 ...
    dataset["slide"] = 96 
    dataset["lead"] = "2 hours"
    dataset["beta"] = 1
    dataset["min_historic_scenario_len"] = sys.maxsize
    dataset["min_target_scenario_len"] = min(df.shape[0] for df in target_data)
    dataset["max_wait_time"] = 150

    return dataset


def xjtu():
    historic_data = []
    historic_sources = []
    
    target_data = []
    target_sources = []
    folder_path = './DataFolder/xjtu-sy/scenarios/'
    file_list = os.listdir(folder_path)
    for index, file_name in enumerate(sorted(file_list, key=lambda file_name_sorting: file_name_sorting.split('.')[0])):
        # if index == 10:
        #     break

        file_path = os.path.join(folder_path, file_name)
        # print(file_path)

        df = pd.read_csv(file_path, header=0)
        # print(df.columns)
        target_data.append(df)
        target_sources.append(file_name.split('.')[0])

    event_data = pd.DataFrame(columns=["date", "type", "source", "description"])

    event_preferences: EventPreferences = {
        'failure': [],
        'reset': []
    }

    dataset={}
    dataset["dates"]="Artificial_timestamp"
    dataset["event_preferences"]=event_preferences
    dataset["event_data"]=event_data
    dataset["target_data"]=target_data
    dataset["target_sources"]=target_sources
    dataset["historic_data"]=historic_data
    dataset["historic_sources"]=historic_sources
    dataset["predictive_horizon"]='18 days'
    dataset["slide"]=8
    dataset["lead"]='2 days'
    dataset["beta"]=1
    dataset["min_historic_scenario_len"] = sys.maxsize
    dataset["min_target_scenario_len"] = min(df.shape[0] for df in target_data)
    dataset["max_wait_time"] = math.ceil((1/3) * dataset["min_target_scenario_len"])

    return dataset


def bhd(sample):
    def count_lines(file_path):
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                line_count = sum(1 for line in file)
            return line_count
        except IOError as e:
            print(f"Error reading file: {e}")
            return None
    def transform_date(date):
        if '2/25/18' in date:
            return date.replace('2/25/18', '2018-02-25')

        if '6/17/19' in date:
            return date.replace('6/17/19', '2019-06-17')

        return date

    historic_data = []
    historic_sources = []

    target_data = []
    target_sources = []
    folder_path = './DataFolder/bhd/scenarios/'
    file_list = os.listdir(folder_path)
    model_per_source = {}
    length_per_source = {}

    if sample:
        for index, file_name in enumerate(file_list):
            file_path = os.path.join(folder_path, file_name)

            current_target_source = ''
            with pd.read_csv(file_path, header=0, chunksize=1) as reader:

                for df in reader:
                    current_target_source = file_name.split('.')[0]
                    model_per_source[current_target_source] = df['model'].iloc[0]
                    break

            assert current_target_source != ''
            wc_result = count_lines(file_path)
            length_per_source[current_target_source] = int(wc_result)

    final_sources = set()
    model_counts = {}
    if sample:
        sorted_sources = sorted(length_per_source, key=length_per_source.get, reverse=True)

        for source in sorted_sources:
            model = model_per_source[source]
            if model not in model_counts:
                model_counts[model] = 0
            if model_counts[model] < 2:
                final_sources.add(source)
                model_counts[model] += 1

    for index, file_name in enumerate(file_list):
        file_path = os.path.join(folder_path, file_name)
        # print(file_path)

        if sample and file_name.split('.')[0] not in final_sources:
            continue

        df = pd.read_csv(file_path, header=0)

        found_smart_measurements = False
        for column in df.columns:
            if 'smart' in column:
                found_smart_measurements = True
                break

        if not found_smart_measurements:
            print(f'{file_name} has no SMART measurements')
            continue

        df = df.dropna()
        if df.shape[0] < 30:
            model_for_current_source = model_per_source[file_name.split('.')[0]]
            print(
                f'skipping {file_name} has less than 30 data points ({df.shape[0]}) (total sources with this model type: {model_counts[model_for_current_source]})')
            continue

        columns_to_drop = []
        for column in df.columns:
            if column != 'date' and 'smart' not in column:
                columns_to_drop.append(column)

        df = df.drop(columns=columns_to_drop, axis=1)

        df['date'] = df['date'].apply(transform_date)

        df = df.loc[:, (df != df.iloc[0]).any()]

        df = df.drop_duplicates(subset='date', keep='last')

        # print(df.columns)
        target_data.append(df)
        target_sources.append(file_name.split('.')[0])

    event_data = pd.DataFrame(columns=["date", "type", "source", "description"])

    event_preferences: EventPreferences = {
        'failure': [],
        'reset': []
    }

    dataset = {}
    dataset["dates"] = "date"
    dataset["event_preferences"] = event_preferences
    dataset["event_data"] = event_data
    dataset["target_data"] = target_data
    dataset["target_sources"] = target_sources
    dataset["historic_data"] = historic_data
    dataset["historic_sources"] = historic_sources
    dataset["predictive_horizon"] = '10 days'
    dataset["slide"] = 10
    dataset["lead"] = '2 days'
    dataset["beta"] = 1
    dataset["min_historic_scenario_len"] = sys.maxsize
    dataset["min_target_scenario_len"] = min(df.shape[0] for df in target_data)
    dataset["max_wait_time"] = math.ceil((1 / 3) * dataset["min_target_scenario_len"])

    return dataset


def formula1():
    COLUMNS_TO_TRANSFORM_FROM_BOOLEAN = ['Brake', 'Status_OffTrack', 'Status_OnTrack', 'Rainfall']
    historic_data = []
    historic_sources = []

    target_data = []
    target_sources = []
    root_dir = './DataFolder/formula1/out/'
    # reason_to_retire_list = []
    counter = 0
    for dirpath, dirnames, filenames in os.walk(root_dir):
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

                # reason_to_retire_list.append(reason_for_failure[-2] + ' ' + reason_for_failure[-1])

                df[COLUMNS_TO_TRANSFORM_FROM_BOOLEAN] = df[COLUMNS_TO_TRANSFORM_FROM_BOOLEAN].astype(int)
                df.drop(columns=['X', 'Y', 'Z'], inplace=True)
                df.sort_values(by=['Date'], inplace=True)
                target_data.append(df)
                target_sources.append(f'{counter}')
                counter += 1


    # with pd.option_context('display.max_rows', None, 'display.max_colwidth', None):
        # print(pd.Series(reason_to_retire_list).value_counts())
        # print(pd.Series([current_df.shape[0] for current_df in target_data]).value_counts())

    # print(len(target_data))

    event_data = pd.DataFrame(columns=["date", "type", "source", "description"])

    event_preferences: EventPreferences = {
        'failure': [],
        'reset': []
    }

    dataset={}
    dataset["dates"]="Date"
    dataset["event_preferences"]=event_preferences
    dataset["event_data"]=event_data
    dataset["target_data"]=target_data
    dataset["target_sources"]=target_sources
    dataset["historic_data"]=historic_data
    dataset["historic_sources"]=historic_sources
    dataset["predictive_horizon"]='8 minutes'
    dataset["slide"]=1920 # slide the ph up to 16 minutes before the failure for VUS metrics
    dataset["lead"]='4 minutes' # enough even under SC or VSC or red flag to do a full lap on any track in order for the team to know if to pit
    dataset["beta"]=1
    dataset["min_historic_scenario_len"] = sys.maxsize
    dataset["min_target_scenario_len"] = min(df.shape[0] for df in target_data)
    dataset["max_wait_time"] = 960 # assuming an average sample rate of 250 ms - 250ms x 960 = 4 minutes - here we want to collect data from the formation lap but the interval is enough even under SC or VSC or red flag to do a full lap on any track

    return dataset


def cnc():
    from datetime import datetime, timedelta
    
    historic_data = []
    historic_sources = []

    event_data = pd.DataFrame(columns=["date", "type", "source", "description"])

    target_data = []
    target_sources = []
    folder_path = './DataFolder/cnc/'
    file_list = os.listdir(folder_path)
    for index, file_name in enumerate(sorted(file_list, key=lambda file_name_sorting: file_name_sorting.split('.')[0])):
        # if index == 10:
        #     break
        
        file_path = os.path.join(folder_path, file_name)
        # print(file_path)

        df = pd.read_csv(file_path, header=0)
        
        # Convert timestamp column from milliseconds to datetime format
        if 'timestamp' in df.columns:
            # Check if timestamp is numeric (milliseconds) or already datetime
            if pd.api.types.is_numeric_dtype(df['timestamp']):
                start_datetime = datetime(2025, 12, 1, 0, 0, 0)
                df['timestamp'] = df['timestamp'].apply(
                    lambda x: start_datetime + timedelta(milliseconds=x)
                )
                # Format as string with milliseconds
                df['timestamp'] = df['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S.%f')
        
        # print(df.columns)
        
        # Check for NaN values
        if df.isnull().values.any():
            print(f"WARNING: NaN values found in {file_name.split('.')[0]}")
            nan_columns = df.columns[df.isnull().any()].tolist()
            print(f"  Columns with NaN: {nan_columns}")
            print(f"  Total NaN count: {df.isnull().sum().sum()}")
        
        df = df.dropna()

        # if 'failure' not in file_name.split('.')[0]:
        #     continue

        target_data.append(df)
        source_name = file_name.split('.')[0]
        target_sources.append(source_name)
        
        # Add failure event if filename contains 'failure'
        # if 'failure' in file_name.lower():
        #     # Get the last timestamp value from the dataframe
        #     last_timestamp = df['timestamp'].iloc[-1]
        #     # # Convert to datetime if it's a string
        #     # if isinstance(last_timestamp, str):
        #     #     last_timestamp = pd.to_datetime(last_timestamp)
            
        #     # Add event to event_data
        #     event_data.loc[len(event_data)] = {
        #         'date': last_timestamp,
        #         'type': 'failure',
        #         'source': source_name,
        #         'description': 'milling tool failure'
        #     }

    # event_data['date'] = pd.to_datetime(event_data['date'])

    event_preferences: EventPreferences = {
        'failure': [],
        'reset': []
        # 'failure': [
        #     EventPreferencesTuple(description='*', type='failure', source='*', target_sources='=')
        # ],
        # 'reset': [
        #     EventPreferencesTuple(description='*', type='reset', source='*', target_sources='=')
        # ]
    }

    dataset={}
    dataset["dates"]="timestamp"
    dataset["event_preferences"]=event_preferences
    dataset["event_data"]=event_data
    dataset["target_data"]=target_data
    dataset["target_sources"]=target_sources
    dataset["historic_data"]=historic_data
    dataset["historic_sources"]=historic_sources
    dataset["predictive_horizon"]='3 seconds' #'60 seconds'
    dataset["slide"]=33 #60
    dataset["lead"]='1 seconds'
    dataset["beta"]=1
    dataset["min_historic_scenario_len"] = sys.maxsize
    dataset["min_target_scenario_len"] = min(df.shape[0] for df in target_data)
    dataset["max_wait_time"] = math.ceil((1/3) * dataset["min_target_scenario_len"])

    return dataset


def fuhrlander():
    historic_data = []
    historic_sources = []

    target_data = []
    target_sources = []
    folder_path = './DataFolder/fuhrlander/'
    event_data = pd.DataFrame(columns=["date", "type", "source", "description"])
    episode_lengths = []

    for turbine_id in range(80, 85):
        sensor_data = pd.read_csv(f'{folder_path}turbine_{turbine_id}.csv')
        events_occurred = pd.read_csv(f'{folder_path}turbine_{turbine_id}_alarms.csv')

        sensor_data.drop(columns=['turbine_id'], inplace=True)
        sensor_data.rename(columns={'date_time': 'timestamp'}, inplace=True)

        target_data.append(sensor_data)
        target_sources.append(f'turbine_{turbine_id}')

        failure_event_ids = [#16, 31, 66,
                             100,
                             # 114, 155,
                             # 128,
                             # 206,
                             # 236,
                             # 237, 614,
                             # 663,
            # 779, 780, 904, 964,
                             # 1024,
                             # 1025, 1026, 1027,
                             # 1113, 1210, 1271, 1273, 1329,
                             # 1379, 1380, 1382,
                             1404,
            # 1405,
                             1406,
            # 1684,
            # 1689,
            # 5430
                             # 1407, 1408, 1409, 1410, 1411, 1412,
                             # 1743, 1744, 1813,
                             # 1919, 1920, 1921,
                             # 1934, 2029, 2047, 2154,
                             # 2240, 2241,
                             # 2243,
                             # 2251,
            # 2254,
                             # 2300, 2301, 2302, 2303, 2304, 2305, 2306,
                            #  2703,
            3108,
            # 3122,
            3123,
            # 5196,
            # 5197,
            # 5198,
            # 5199,
                            #  5494,
                             # 5496, 5497, 5498, 5499,
                             # 5930
                             ]

        # events_occurred = events_occurred[(events_occurred['alarm_desc'].str.contains('stop')) | (events_occurred['alarm_desc'].str.contains('stp'))]
        events_occurred = events_occurred[events_occurred['alarm_id'].isin(failure_event_ids)]
        events_occurred = events_occurred[events_occurred['date_time_ini'] == events_occurred['date_time_end']]
        events_occurred.drop_duplicates(subset=['date_time_ini'], inplace=True)

        # trigger reset for Resid.curr.guard stp to the whole wind farm and investigate this error
        # also check if it is better for justification in the text to use the other 5 errors gpt recommended
        # use SCANIA dataset for showcasing number of steps needed until optimized performance because it has also repairs for reset
        for _, row in events_occurred.iterrows():
            current_event_date = pd.to_datetime(row['date_time_ini'])
            
            # Check if there's a failure in the event_data for this turbine within the previous 2 months
            # two_months_ago = current_event_date - pd.DateOffset(months=2)
            # recent_failures = event_data[
            #     (event_data['source'] == f'turbine_{turbine_id}') &
            #     (event_data['type'] == 'failure') &
            #     (event_data['date'] >= two_months_ago) &
            #     (event_data['date'] < current_event_date)
            # ]
            #
            # # Only append if no failure occurred within the previous 2 months
            # if recent_failures.empty:
            event_data.loc[len(event_data)] = [
                current_event_date,
                'failure' if row['alarm_id'] in [100,1404,1406] else 'reset',
                f'turbine_{turbine_id}',
                row['alarm_desc']
            ]

        df = sensor_data.copy()
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        for failure_index, failure_row in event_data[(event_data['source'] == f'turbine_{turbine_id}') & (event_data['type'] == 'failure')].iterrows():
            current_df = df[df['timestamp'] <= failure_row['date']]

            if current_df.shape[0] > 0:
                episode_lengths.append(current_df.shape[0])

            df = df[df['timestamp'] > failure_row['date']]

    event_preferences: EventPreferences = {
        'failure': [
            EventPreferencesTuple(description='*', type='failure', source='*', target_sources='=')
        ],
        'reset': [
            EventPreferencesTuple(description='*', type='failure', source='*', target_sources='='),
            # EventPreferencesTuple(description='TSO Stop', type='reset', source='*', target_sources='='),
            # EventPreferencesTuple(description='GCA Stop', type='reset', source='*', target_sources='=')
        ]
    }

    dataset = {}
    dataset["dates"] = "timestamp"
    dataset["event_preferences"] = event_preferences
    dataset["event_data"] = event_data
    dataset["target_data"] = target_data
    dataset["target_sources"] = target_sources
    dataset["historic_data"] = historic_data
    dataset["historic_sources"] = historic_sources
    dataset["predictive_horizon"]="86400 minutes"
    dataset["slide"]=0
    dataset["lead"]="2880 minutes"
    dataset["beta"]=1
    dataset["min_historic_scenario_len"] = sys.maxsize
    dataset["min_target_scenario_len"] = min(df.shape[0] for df in target_data)
    dataset["max_wait_time"] = math.ceil((1/3) * min(episode_lengths))
    
    return dataset


def scania_train():
    historic_data = []
    historic_sources = []

    target_data = []
    target_sources = []
    folder_path = './DataFolder/scania/train/'
    file_list = os.listdir(folder_path)
    for index, file_name in enumerate(file_list):
        file_path = os.path.join(folder_path, file_name)
        # print(file_path)

        df = pd.read_csv(file_path, header=0)
        # print(df.columns)
        target_data.append(df)
        target_sources.append(file_name.split('_')[0])

    event_data = pd.DataFrame(columns=["date", "type", "source", "description"])

    event_preferences: EventPreferences = {
        'failure': [],
        'reset': []
    }

    dataset={}
    dataset["dates"]="timestamp"
    dataset["event_preferences"]=event_preferences
    dataset["event_data"]=event_data
    dataset["target_data"]=target_data
    dataset["target_sources"]=target_sources
    dataset["historic_data"]=historic_data
    dataset["historic_sources"]=historic_sources
    dataset["predictive_horizon"]='48 days'
    dataset["slide"]=225
    dataset["lead"]='24 hours'
    dataset["beta"]=1
    dataset["min_historic_scenario_len"] = sys.maxsize
    dataset["min_target_scenario_len"] = min(df.shape[0] for df in target_data)
    dataset["max_wait_time"] = math.ceil((1/3) * min([temp_df.shape[0] for temp_df in target_data]))


    return dataset


def scania_test():
    historic_data = []
    historic_sources = []

    target_data = []
    target_sources = []
    folder_path = './DataFolder/scania/test/'
    file_list = os.listdir(folder_path)
    for index, file_name in enumerate(file_list):
        file_path = os.path.join(folder_path, file_name)
        # print(file_path)

        df = pd.read_csv(file_path, header=0)
        # print(df.columns)
        target_data.append(df)
        target_sources.append(file_name.split('_')[0])

    event_data = pd.DataFrame(columns=["date", "type", "source", "description"])

    event_preferences: EventPreferences = {
        'failure': [],
        'reset': []
    }

    dataset={}
    dataset["dates"]="timestamp"
    dataset["event_preferences"]=event_preferences
    dataset["event_data"]=event_data
    dataset["target_data"]=target_data
    dataset["target_sources"]=target_sources
    dataset["historic_data"]=historic_data
    dataset["historic_sources"]=historic_sources
    dataset["predictive_horizon"]='48 days'
    dataset["slide"]=225
    dataset["lead"]='24 hours'
    dataset["beta"]=1
    dataset["min_historic_scenario_len"] = sys.maxsize
    dataset["min_target_scenario_len"] = min(df.shape[0] for df in target_data)
    dataset["max_wait_time"] = math.ceil((1/3) * min([temp_df.shape[0] for temp_df in target_data]))


    return dataset


def ai4i():
    historic_data = []
    historic_sources = []

    target_data = []
    target_sources = []
    folder_path = './DataFolder/ai4i/scenarios/'
    file_list = os.listdir(folder_path)

    event_data = pd.read_csv('./DataFolder/ai4i/scenarios/events.csv')
    event_data['date'] = pd.to_datetime(event_data['date'])
    event_data = event_data.sort_values(by='date').reset_index(drop=True)
    event_data['source'] = event_data['source'].astype(str)

    failures_df = event_data[event_data['type'] == 'failure']
    failures_df = failures_df.sort_values(by='date').reset_index(drop=True)

    episode_lengths = []
    for index, file_name in enumerate(file_list):
        if 'event' in file_name:
            continue

        file_path = os.path.join(folder_path, file_name)

        df = pd.read_csv(file_path, header=0)
        df = df.ffill()
        df['Artificial_timestamp'] = pd.to_datetime(df['Artificial_timestamp'])
        df = df.drop_duplicates(subset=['Artificial_timestamp'])

        constant_columns = [col for col in df.columns if df[col].nunique() == 1]

        df = df.drop(columns=constant_columns, axis=1)

        target_data.append(df.copy())
        target_sources.append(file_name.split('.')[0])

        failures_for_current_source = failures_df[failures_df['source'] == file_name.split('.')[0]]
        failures_for_current_source = failures_for_current_source.sort_values(by='date').reset_index(drop=True)

        for failure_index, failure in failures_for_current_source.iterrows():
            current_df = df[df['Artificial_timestamp'] <= failure.date]
            episode_lengths.append(current_df.shape[0])
            df = df[df['Artificial_timestamp'] > failure.date]

        if df.shape[0] != 0:
            episode_lengths.append(df.shape[0])

    event_preferences: EventPreferences = {
        'failure': [
            EventPreferencesTuple(description='*', type='failure', source='*', target_sources='=')
        ],
        'reset': [
            EventPreferencesTuple(description='*', type='reset', source='*', target_sources='='),
            EventPreferencesTuple(description='*', type='failure', source='*', target_sources='=')
        ]
    }

    dataset = {}
    dataset["dates"] = "Artificial_timestamp"
    dataset["event_preferences"] = event_preferences
    dataset["event_data"] = event_data
    dataset["target_data"] = target_data
    dataset["target_sources"] = target_sources
    dataset["historic_data"] = historic_data
    dataset["historic_sources"] = historic_sources
    dataset["predictive_horizon"] = "4 days"
    dataset["slide"] = 2
    dataset["lead"] = "2 days"
    dataset["beta"] = 1
    dataset["min_historic_scenario_len"] = sys.maxsize
    dataset["min_target_scenario_len"] = min(df.shape[0] for df in target_data)
    dataset["max_wait_time"] = math.ceil((1 / 3) * min(episode_lengths))

    return dataset


def combine_close_labels(labels,limit):
    combined_labels = [0 for i in range(len(labels))]
    pos = 0
    while pos < len(labels):
        if labels[pos] == 0:
            next_pos = pos + 1

            if next_pos < len(labels):
                while labels[next_pos] != 1:
                    next_pos += 1

                    if next_pos >= len(labels):
                        break
            else:
                break

            if next_pos - pos <= limit:
                for qi in range(pos, next_pos):
                    combined_labels[qi] = 1
            else:
                for qi in range(pos, next_pos):
                    combined_labels[qi] = 0

            pos = next_pos
        else:
            combined_labels[pos] = 1
            pos += 1

    return combined_labels


def generate_early_detection_labels_SETUP_1_constant(original_labels,PH_span, PH_span_early):
    """
    Setup 1 place the  middle of the predictive horizon in the start of the anomaly range and spans +- PH_span length
    :param original_labels:
    :param PH:
    :return:
    """
    early_detection_labels=[0 for i in range(len(original_labels))]
    for i in range(1, len(original_labels)):
        if original_labels[i] == 1 and original_labels[i - 1] == 0:
            center = i
            stop_labels = len(original_labels)
            for j in range(center, len(original_labels)):
                if original_labels[j] == 0:
                    stop_labels = j
                    break
            for qi in range(max(0, center - PH_span), min(stop_labels, center + PH_span_early)):
                early_detection_labels[qi] = 1
    return early_detection_labels


def plot_labels_TSB(early_detection_labels,leadtime,combined_labels,df):
    import matplotlib
    matplotlib.use('TkAgg')  # or any other backend
    import matplotlib.pyplot as plt
    #df = (df - df.min()) / (df.max() - df.min())

    plt.title(f"PH span {40} centered in start of failure (possible setup)")
    plt.plot(early_detection_labels, color="green", linewidth=2)
    plt.fill_between([i for i in range(len(early_detection_labels))], 0, 1, where=early_detection_labels, color="green",
                     alpha=0.3, label="PH")

    plt.plot(leadtime, color="orange", linewidth=2)
    plt.fill_between([i for i in range(len(early_detection_labels))], 0, 1, where=leadtime, color="orange",
                     alpha=0.3, label="lead")

    plt.plot(df[0].values, alpha=0.8, label="Data")
    plt.plot(combined_labels, color="magenta", linewidth=5, alpha=0.8, label="Original label")

    plt.legend()
    # plt.savefig("todeleteFig")
    plt.show()


def TSB_dataset(name, reset_after_fail, setup_1_predict, setup_1_early, max_sizes=None):
    dataset_paths = {
        "daphnet": './DataFolder/TSB-UAD-Public/Daphnet',
        "dodgers": './DataFolder/TSB-UAD-Public/Dodgers',
        "ecg": './DataFolder/TSB-UAD-Public/ECG',
        "genesis": './DataFolder/TSB-UAD-Public/Genesis',
        "ghl": './DataFolder/TSB-UAD-Public/GHL',
        "iops": './DataFolder/TSB-UAD-Public/IOPS',
        "kdd21": './DataFolder/TSB-UAD-Public/KDD21',
        "mgab": './DataFolder/TSB-UAD-Public/MGAB',
        "mitdb": './DataFolder/TSB-UAD-Public/MITDB',
        "nab": './DataFolder/TSB-UAD-Public/NAB',
        "nasa-msl": './DataFolder/TSB-UAD-Public/NASA-MSL',
        "nasa-smap": './DataFolder/TSB-UAD-Public/NASA-SMAP',
        "occupancy": './DataFolder/TSB-UAD-Public/Occupancy',
        "opportunity": './DataFolder/TSB-UAD-Public/OPPORTUNITY',
        "sensor-scope": './DataFolder/TSB-UAD-Public/SensorScope',
        "smd": './DataFolder/TSB-UAD-Public/SMD',
        "smd_1" : './DataFolder/TSB-UAD-Public/SMD1',
        "smd_multivariate": './DataFolder/TSB-UAD-Public/SMD',
        "smd_1_multivariate": './DataFolder/TSB-UAD-Public/SMD1',
        "svdb": './DataFolder/TSB-UAD-Public/SVDB',
        "yahoo": './DataFolder/TSB-UAD-Public/YAHOO'
    }

    assert name in dataset_paths, f"No dataset with name {name}"

    target_data = []
    target_sources = []
    historic_data=[]
    historic_sources=[]

    event_dict={
        "date":[],
        "type":[],
        "source":[],
        "description":[]
    }

    lead_time = []
    early_detection_labels = []


    sizesc=0
    label_size=[]
    folder_path = dataset_paths[name]
    file_list = os.listdir(folder_path)
    file_list.sort()
    for index, file_name in enumerate(file_list):
        file_path = os.path.join(folder_path, file_name)
        print(file_path)
        df = pd.read_csv(file_path, header=None)

        if "train" in file_name:
            historic_data.append(df.drop([1], axis=1))

            source = file_name
            if 'NASA-MSL' in file_path or 'NASA-SMAP' in file_path or 'IOPS' in file_path:
                source = file_name.split('.')[0]
            elif 'Occupancy' in file_path:
                source = file_name.split('@')[1]

            historic_sources.append(source)
            continue

        labels=df[1].values

        sizesc += len(labels)

        rangelabel = 0
        for l in labels:
            if l == 0 and rangelabel != 0:
                label_size.append(rangelabel)
                rangelabel = 0
            elif l == 1:
                rangelabel += 1

        if rangelabel != 0:
            label_size.append(rangelabel)

        combined_labels = combine_close_labels(labels, setup_1_predict)
        
        current_early_detection_labels = generate_early_detection_labels_SETUP_1_constant(combined_labels, setup_1_predict, setup_1_early)
        early_detection_labels.append(current_early_detection_labels)

        current_lead_time = [1 if el==0 and ol==1 else 0 for el,ol in zip(current_early_detection_labels,combined_labels)]
        lead_time.append(current_lead_time)

        start_datetime = pd.to_datetime('2024-01-01 00:00:00')
        num_timestamps = df.shape[0]
        timestamps = pd.date_range(start=start_datetime, periods=num_timestamps, freq='T')
        df['date'] = timestamps

        df = df.drop([1],axis=1)

        target_data.append(df)

        source = file_name
        if 'NASA-MSL' in file_path or 'NASA-SMAP' in file_path or 'IOPS' in file_path:
            source = file_name.split('.')[0]
        elif 'Occupancy' in file_path:
            source = file_name.split('@')[1]

        target_sources.append(source)

        for i in range(len(combined_labels)-1):
            if combined_labels[i] == 1 and combined_labels[i + 1] == 0:
                event_dict['date'].append(timestamps[i])
                event_dict['type'].append('failure')
                event_dict['source'].append(file_name)
                event_dict['description'].append('Annotated anomaly')

        if max_sizes is not None:
            if sizesc >= max_sizes:
                if sum([sum(lll) for lll in early_detection_labels])>1:
                    break


    if reset_after_fail:
        event_preferences: EventPreferences = {
            'failure': [
                EventPreferencesTuple(description='*', type='failure', source='*', target_sources='=')
            ],
            'reset': [
                EventPreferencesTuple(description='*', type='reset', source='*', target_sources='='),
                EventPreferencesTuple(description='*', type='failure', source='*', target_sources='=')
            ]
        }
    else:
        event_preferences: EventPreferences = {
            'failure': [
                EventPreferencesTuple(description='*', type='failure', source='*', target_sources='=')
            ],
            'reset': [
                EventPreferencesTuple(description='*', type='reset', source='*', target_sources='='),
            ]
        }

    event_data = pd.DataFrame(event_dict)

    if name == 'smd_multivariate' or name == 'smd_1_multivariate':
        dataframes_per_machine = defaultdict(list)

        for s, df in zip(target_sources, target_data):
            prefix = s.split('.')[0]
            if len(dataframes_per_machine[prefix]) == 0:
                dataframes_per_machine[prefix].append(df)
            else:
                dataframes_per_machine[prefix].append(df.drop(columns=['date'], axis=0).rename({0: len(dataframes_per_machine[prefix])}, axis=1))

        target_sources = []
        target_data = []
        for prefix, dfs in dataframes_per_machine.items():
            target_sources.append(prefix)
            target_data.append(pd.concat(dfs, axis=1))

    dataset = {}
    dataset["dates"] = "date"
    dataset["setup_1_period"] = setup_1_predict
    dataset["reset_after_fail"] = reset_after_fail
    dataset["anomaly_ranges"] = True
    dataset["lead"] = lead_time
    dataset["slide"] = setup_1_predict + setup_1_early
    dataset["predictive_horizon"] = early_detection_labels
    dataset["event_preferences"] = event_preferences
    dataset["event_data"] = event_data
    dataset["target_data"] = target_data
    dataset["target_sources"] = target_sources
    dataset["historic_data"] = historic_data
    dataset["historic_sources"] = historic_sources
    dataset["beta"] = 1
    dataset["min_historic_scenario_len"] = sys.maxsize if len(historic_data) == 0 else min(len(historic_data_df) for historic_data_df in historic_data)
    dataset["min_target_scenario_len"] = min(df.shape[0] for df in target_data)

    target_scenario_with_min_len = min(target_data, key=lambda df: len(df))

    sliding=utils.find_length(target_scenario_with_min_len[0])
    max_wait_time = 10 * sliding
    dataset["max_wait_time"] = max_wait_time # TODO discuss in a group meeting

    print(f" Original anomaly ranges mean {statistics.mean(label_size)}")
    print(f" Original anomaly ranges median {statistics.median(label_size)}")
    print(f" Original anomaly ranges number {len(label_size)}")
    print(f" Original anomaly ranges total length {sum(label_size)}")
    print(f" min Original anomaly ranges {min(label_size)}")
    print(f" max Original anomaly ranges {max(label_size)}")
    print(f" len min scenario {dataset['min_target_scenario_len']}")
    print(f" len median scenario {statistics.median([df.shape[0] for df in target_data])}")
    print(f" max_wait_time {dataset['max_wait_time']}")
    print(f" Suggested sub_sequence length {sliding}")

    return dataset
# return:
#    dictionary: event_preferences
#    event_data : pandas Dataframe
#    target_data : list[pd.Dataframe]
#    target_sources : list[string]
#    historic_data : list[pd.Dataframe]
#    historic_sources : list[string]


def get_dataset(name="cmapss", semi=False, sample=True, reset_after_fail=False, setup_1_predict=100, setup_1_early=10,
                max_sizes=None):
    if name == "cmapss":
        return cmapss(semi=semi)
    elif name == "navarchos":
        return navarchos()
    elif name == "femto":
        return femto(semi=semi)
    elif name == "ims":
        return ims()
    elif name == "edp-wt":
        return edp()
    elif name == "metropt-3":
        return metropt()
    elif name == "xjtu":
        return xjtu()
    elif name == "bhd":
        return bhd(sample=sample)
    elif name == "azure":
        return azure()
    elif name == "ai4i":
        return ai4i()
    elif name == "formula1":
        return formula1()
    elif name == "cnc":
        return cnc()
    elif name == "fuhrlander":
        return fuhrlander()
    elif name == "scania_train":
        return scania_train()
    elif name == "scania_test":
        return scania_test()
    elif "dictionaries" in name:
        return load_pickle(name)
    else:
        return TSB_dataset(name=name, reset_after_fail=reset_after_fail, setup_1_predict=setup_1_predict,
                           setup_1_early=setup_1_early, max_sizes=max_sizes)
