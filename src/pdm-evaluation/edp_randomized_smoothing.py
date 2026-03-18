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
import sys
import random
import argparse

import torch
import mlflow
import numpy as np
import pandas as pd

from experiment.batch.auto_profile_semi_supervised_experiment import AutoProfileSemiSupervisedPdMExperiment
from pipeline.pipeline import PdMPipeline
from method.ocsvm import OneClassSVM
from method.isolation_forest import IsolationForest
from method.TranADPdM import TranADPdM
from preprocessing.record_level.default import DefaultPreProcessor
from postprocessing.default import DefaultPostProcessor
from thresholding.constant import ConstantThresholder
from constraint_functions.constraint import (
    auto_profile_max_wait_time_constraint
)
from utils import loadDataset
from utils.utils import calculate_mango_parameters

seed = 42
random.seed(seed)
torch.manual_seed(seed)
np.random.seed(seed)

conda_env = os.environ.get('CONDA_DEFAULT_ENV')

def apply_gaussian_noise(df, sigma, seed):
    """
    Injects random Gaussian noise into all channels except 'Timestamp'.
    """
    if sigma == 0:
        return df.copy()
    
    # Store the original Timestamp column if it exists
    timestamp_col = None
    if 'Timestamp' in df.columns:
        timestamp_col = df['Timestamp'].copy()
        df_without_timestamp = df.drop(columns=['Timestamp'])
    else:
        df_without_timestamp = df
        
    # Apply noise only to non-Timestamp columns
    np.random.seed(seed)
    noise = np.random.normal(0, sigma, df_without_timestamp.shape)
    data_noisy = df_without_timestamp.values + noise
    
    # Create the noisy dataframe
    df_result = pd.DataFrame(data_noisy, index=df_without_timestamp.index, columns=df_without_timestamp.columns)
    
    # Put back the original Timestamp column if it existed
    if timestamp_col is not None:
        df_result['Timestamp'] = timestamp_col
        df_result = df_result[['Timestamp'] + [col for col in df_result.columns if col != 'Timestamp']]
    else:
        raise ValueError("DataFrame does not contain a 'Timestamp' column.")
    
    return df_result


def generate_noisy_dataset(
    dataset, 
    sigma,
    seed
):
    """
    Main Pipeline: Takes the list of clean DataFrames and returns 
    a corresponding list of Adversarial (Noisy) DataFrames.
    
    Args:
        dataset: The dataset dictionary
        sigma: Magnitude of value noise (Standard deviation)
        
    Returns:
        List[pd.DataFrame]: The attacked datasets.
    """
    attacked_dfs = []
    
    print(f"Generating Samples | Noise: {sigma}")

    for i, df in enumerate(dataset['target_data']):
        # We use specific seeds per DF to ensure reproducibility across experiments
        current_seed = seed + i 
        # 2. Apply Noise (Value Distortion)
        df_final = apply_gaussian_noise(
            df, 
            sigma=sigma, 
            seed=current_seed
        )
        
        attacked_dfs.append(df_final)
        
    return attacked_dfs


def execute(method_names_to_run, seed, sigma=None, moving_average=False, MAX_RUNS=1, MAX_JOBS=1, INITIAL_RANDOM=0):
    print("script: edp_randomized_smoothing.py")

    tracking_uri = mlflow.get_tracking_uri()
    print("MLflow Tracking URI:", tracking_uri)

    dataset = loadDataset.get_dataset("edp-wt")

    print(dataset['target_data'][0].head())

    if sigma is None:
        raise ValueError("For 'noise' mode, 'sigma' must be provided.")
    
    dataset['target_data'] = generate_noisy_dataset(dataset, sigma=sigma, seed=seed)

    if moving_average:
        print("Applying Causal Moving Average Filter (Window=100)")
        for i in range(len(dataset['target_data'])):
            df = dataset['target_data'][i]
            # Apply moving average to all columns except Timestamp
            # Using rolling mean with window 100, causal (right aligned)
            cols_to_smooth = [c for c in df.columns if c != 'Timestamp']
            
            original_values = df[cols_to_smooth].copy()
            
            # Rolling with window 100. Default center=False (causal).
            # Default min_periods = window size (return NaN if not enough points)
            rolled = df[cols_to_smooth].rolling(window=100).mean()
            
            # Fill NaNs (where we didn't have 100 points) with original values
            df[cols_to_smooth] = rolled.fillna(original_values)
            
            dataset['target_data'][i] = df

    print(dataset['target_data'][0].head())

    configs = []

    # Auto profile EDP IF
    configs.append({
        'method_name': 'IF',
        'experiment_type': 'Auto profile EDP',
        'method_class': IsolationForest,
        'params': {
            'n_estimators': [200],
            'max_samples': [400],
            'random_state': [42],
            'max_features': [0.6],
            'bootstrap': [False]
        },
        'common_params': {
            'profile_size': [900]
        }
    })

    # Auto profile EDP OCSVM
    configs.append({
        'method_name': 'OCSVM',
        'experiment_type': 'Auto profile EDP',
        'method_class': OneClassSVM,
        'params': {
            'kernel': ['linear'],
            'nu': [0.1],
            'gamma': ['auto'],
            'max_iter': [10000]
        },
        'common_params': {
            'profile_size': [800]
        }
    })

    # Auto profile EDP TRANAD
    configs.append({
        'method_name': 'TRANAD',
        'experiment_type': 'Auto profile EDP',
        'method_class': TranADPdM,
        'params': {
            'window_size': [75],
            'num_epochs': [5],
            'lr': [0.001]
        },
        'common_params': {
            'profile_size': [1500]
        }
    })
    
    experiment_class_map = {
        'Auto profile EDP': AutoProfileSemiSupervisedPdMExperiment,
    }

    for config in configs:
        if config['method_name'] not in method_names_to_run:
            continue

        experiment_type = config['experiment_type']
        method_name = config['method_name']
        ExperimentClass = experiment_class_map[experiment_type]
        
        postprocessor = DefaultPostProcessor
        
        my_pipeline = PdMPipeline(
            steps={
                'preprocessor': DefaultPreProcessor,
                'method': config['method_class'],
                'postprocessor': postprocessor,
                'thresholder': ConstantThresholder,
            },
            dataset=dataset,
            auc_resolution=100
        )

        current_param_space_dict = {
            'thresholder_threshold_value': [0.5],
        }
        
        # Add common params
        for key, value in config['common_params'].items():
            current_param_space_dict[key] = value

        # Add method params with prefix
        for key, value in config['params'].items():
            current_param_space_dict[f'method_{key}'] = value

        num, jobs, initial_random = calculate_mango_parameters(current_param_space_dict, MAX_JOBS, INITIAL_RANDOM, MAX_RUNS)

        # Determine constraint function
        constraint = None
        if experiment_type == 'Auto profile EDP':
            constraint = auto_profile_max_wait_time_constraint(my_pipeline)
            
        my_experiment = ExperimentClass(
            experiment_name=f'Median Randomized Smoothing (sigma={sigma}) (moving_average={moving_average}) EDP {method_name}',
            target_data=dataset['target_data'],
            target_sources=dataset['target_sources'],
            pipeline=my_pipeline,
            param_space=current_param_space_dict,
            num_iteration=num,
            n_jobs=jobs,
            initial_random=initial_random,
            artifacts='./artifacts/' + experiment_type + ' artifacts',
            constraint_function=constraint,
            debug=True,
            log_best_scores=True,
        )

        best_params = my_experiment.execute()
        print(experiment_type + ' ' + method_name)
        print(best_params)


parser = argparse.ArgumentParser(description='EDP Randomized Smoothing Experiment')

parser.add_argument('--method', type=str, required=True, help='Method name to run')
parser.add_argument('--seed', type=int, default=42, required=True, help='Random seed')
parser.add_argument('--sigma', type=float, required=True, help='Noise level')
parser.add_argument('--moving_average', action='store_true', help='Apply moving average filter after noise injection')

if __name__ == "__main__":
    args = parser.parse_args()
    
    method_names = [args.method]

    execute(method_names_to_run=method_names, sigma=args.sigma, seed=args.seed, moving_average=args.moving_average)