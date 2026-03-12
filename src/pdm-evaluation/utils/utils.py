import pandas as pd
from statsmodels.tsa.stattools import acf
from scipy.signal import argrelextrema
import numpy as np
import torch
from torch import nn
import os
import torch.utils.data
from pathlib import Path
from typing import List, Optional, Union

conda_env = os.environ.get('CONDA_DEFAULT_ENV')

if conda_env is not None:
    if 'Chronos' in conda_env:
        from gluonts.dataset.arrow import ArrowWriter

from pdm_evaluation_types.types import EventPreferences, EventPreferencesTuple


# determine sliding window (period) based on ACF
def find_length(data):
    if len(data.shape) > 1:
        return 0
    data = data[:min(20000, len(data))]

    base = 3
    auto_corr = acf(data, nlags=400, fft=True)[base:]

    local_max = argrelextrema(auto_corr, np.greater)[0]
    try:
        max_local_max = np.argmax([auto_corr[lcm] for lcm in local_max])
        if local_max[max_local_max] < 3 or local_max[max_local_max] > 300:
            return 125
        return local_max[max_local_max] + base
    except:
        return 125


class Window:
    """ The  class for rolling window feature mapping.
    The mapping converts the original timeseries X into a matrix.
    The matrix consists of rows of sliding windows of original X.
    """

    def __init__(self,  window = 100):
        self.window = window
        self.detector = None

    def convert(self, X):
        n = self.window
        X = pd.Series(X)
        L = []
        if n == 0:
            df = X
        else:
            for i in range(n):
                L.append(X.shift(i))
            df = pd.concat(L, axis = 1)
            df = df.iloc[n-1:]
        
        return df


def process_event_preference_with_one_dont_care_bit(event_preference: EventPreferencesTuple, event_data: pd.DataFrame, dont_care_bit_index: int) -> list[EventPreferencesTuple]:
    result = []

    if dont_care_bit_index == 0:
        filtered_event_data = event_data[(event_data['type'] == event_preference.type) & (event_data['source'] == event_preference.source)]
    elif dont_care_bit_index == 1:
        filtered_event_data = event_data[(event_data['description'] == event_preference.description) & (event_data['source'] == event_preference.source)]
    else: # dont_care_bit_index == 2
        filtered_event_data = event_data[(event_data['description'] == event_preference.description) & (event_data['type'] == event_preference.type)]

    for _, current_event_data_row in filtered_event_data.iterrows():
        result.append(EventPreferencesTuple(description=current_event_data_row.description, type=current_event_data_row.type, source=current_event_data_row.source, target_sources=event_preference.target_sources))


    return result


def process_event_preference_with_two_dont_care_bits(event_preference: EventPreferencesTuple, event_data: pd.DataFrame, dont_care_bit_1_index: int, dont_care_bit_2_index: int) -> list[EventPreferencesTuple]:
    result = []

    if dont_care_bit_1_index == 0 and dont_care_bit_2_index == 1:
        filtered_event_data = event_data[event_data['source'] == event_preference.source]
    elif dont_care_bit_1_index == 0 and dont_care_bit_2_index == 2:
        filtered_event_data = event_data[event_data['type'] == event_preference.type]
    else: # dont_care_bit_1_index == 1 and dont_care_bit_2_index == 2
        filtered_event_data = event_data[event_data['description'] == event_preference.description]

    for _, current_event_data_row in filtered_event_data.iterrows():
        result.append(EventPreferencesTuple(description=current_event_data_row.description, type=current_event_data_row.type, source=current_event_data_row.source, target_sources=event_preference.target_sources))


    return result


def process_event_preferences_key(event_data: pd.DataFrame, event_preferences: list[EventPreferencesTuple]) -> list[EventPreferencesTuple]:
    result_preferences = []
    for current_preference in event_preferences:
        if current_preference.description == '*' and current_preference.type != '*' and current_preference.source != '*':
            result_preferences = result_preferences + process_event_preference_with_one_dont_care_bit(current_preference, event_data, 0)

        elif current_preference.description != '*' and current_preference.type == '*' and current_preference.source != '*':
            result_preferences = result_preferences + process_event_preference_with_one_dont_care_bit(current_preference, event_data, 1)

        elif current_preference.description != '*' and current_preference.type != '*' and current_preference.source == '*':
            result_preferences = result_preferences + process_event_preference_with_one_dont_care_bit(current_preference, event_data, 2)

        # 2 dont care bits
        elif current_preference.description == '*' and current_preference.type == '*' and current_preference.source != '*':
            result_preferences = result_preferences + process_event_preference_with_two_dont_care_bits(current_preference, event_data, 0, 1)

        elif current_preference.description == '*' and current_preference.type != '*' and current_preference.source == '*':
            result_preferences = result_preferences + process_event_preference_with_two_dont_care_bits(current_preference, event_data, 0, 2)

        elif current_preference.description != '*' and current_preference.type == '*' and current_preference.source == '*':
            result_preferences = result_preferences + process_event_preference_with_two_dont_care_bits(current_preference, event_data, 1, 2)

        # 3 dont care bits
        elif current_preference.description == '*' and current_preference.type == '*' and current_preference.source == '*':
            for _, current_event_data_row in event_data.iterrows():
                result_preferences.append(EventPreferencesTuple(description=current_event_data_row.description, type=current_event_data_row.type, source=current_event_data_row.source, target_sources=current_preference.target_sources))

            break # we encountered a preference with 3 dont care bits so no need to continue looping through the rest of the preferences

        else: # 0 dont care bits
            result_preferences.append(current_preference)

    
    return list(set(result_preferences)) # remove duplicates


def expand_event_preferences(event_data: pd.DataFrame, event_preferences: EventPreferences) -> EventPreferences:
    result_event_preferences: EventPreferences = {
        'failure': [],
        'reset': [],
    }

    result_event_preferences['failure'] = process_event_preferences_key(event_data, event_preferences['failure'])
    result_event_preferences['reset'] = process_event_preferences_key(event_data, event_preferences['reset'])


    return result_event_preferences


def calculate_mango_parameters(current_param_space_dict, MAX_JOBS, INITIAL_RANDOM, MAX_RUNS):
    if MAX_RUNS <= MAX_JOBS:
        MAX_JOBS = MAX_RUNS
        return 1, MAX_JOBS, 1
    
    param_space_size = 1
    for _, item in current_param_space_dict.items():
        param_space_size *= len(item)

    if param_space_size<=MAX_JOBS:
        num=max(1, param_space_size-INITIAL_RANDOM)
        jobs=1
        initial_random=min(INITIAL_RANDOM, param_space_size)
    elif min(MAX_RUNS,param_space_size)%MAX_JOBS <INITIAL_RANDOM:
        initial_random=min(MAX_RUNS,param_space_size)%MAX_JOBS+MAX_JOBS
        num=max(1, min(MAX_RUNS,param_space_size)//MAX_JOBS -1)
        jobs=MAX_JOBS
    else:
        initial_random=min(MAX_RUNS,param_space_size)%MAX_JOBS
        num=max(1, min(MAX_RUNS,param_space_size)//MAX_JOBS)
        jobs=MAX_JOBS

    return num, jobs, initial_random


class EarlyStoppingTorch:
    def __init__(self, save_path=None, patience=7, verbose=False, delta=0.0001):
        self.save_path = save_path
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0
    
    def save_checkpoint(self, val_loss, model):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        if self.save_path:
            path = os.path.join(self.save_path, 'best_network.pth')
            torch.save(model.state_dict(), path)	
        self.val_loss_min = val_loss

if conda_env is not None:
    if 'Chronos' in conda_env:
        def convert_to_arrow(
            path: Union[str, Path],
            time_series: Union[List[np.ndarray], np.ndarray],
            start_times: Optional[Union[List[np.datetime64], np.ndarray]] = None,
            compression: str = "lz4",
        ):
            if start_times is None:
                # Set an arbitrary start time
                start_times = [np.datetime64("2000-01-01 00:00", "s")] * len(time_series)

            assert len(time_series) == len(start_times)

            dataset = [
                {"start": start, "target": ts} for ts, start in zip(time_series, start_times)
            ]

            ArrowWriter(compression=compression).write_to_file(
                dataset,
                path=path,
            )
