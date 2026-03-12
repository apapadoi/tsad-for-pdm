# Unsupervised EDP CHRONOS

# method_context_length	1500
# method_num_samples	1
# method_slide	15


# Auto profile EDP IF
# method_n_estimators	200
# method_max_samples	400
# method_random_state	42
# method_max_features	0.6
# method_bootstrap	False
# profile_size	900


# Auto profile EDP KNN
# method_k	2
# method_window_norm	True
# profile_size	800


# Auto profile EDP LOF
# n_neighbors	5
# profile_size	1000


# Auto profile EDP LTSF
# ltsf_type	DLinear
# features	M
# seq_len	50
# individual	True
# train_epochs	5
# learning_rate	0.01
# batch_size	2
# profile_size	800


# Incremental EDP NP
# n_nnballs   100
# max_sample  80
# sub_sequence_length  200
# aggregation_strategy  max
# initial_incremental_window_length 500
# incremental_window_length 500
# incremental_slide 300


# Auto profile EDP OCSVM
# kernel	linear
# nu	0.1
# gamma	auto
# max_iter	-1
# profile_size	800


# Auto profile EDP PB
# profile_size	1000


# Unsupervised EDP SAND
# pattern_length	50
# subsequence_length_multiplier	4
# alpha	0.25
# init_length	1500
# batch_size	1500
# k	9
# aggregation_strategy	avg


# Auto profile EDP TRANAD
# window_size	75
# num_epochs	5
# lr	0.001
# profile_size	1500


# Incremental EDP USAD
# window_size	100
# num_epochs	10
# lr	0.1
# BATCH_SIZE	4
# hidden_size	16
# initial_incremental_window_length	1750
# incremental_window_length	1750
# incremental_slide	1500

import os
import sys
import random
import argparse

import torch
import mlflow
import numpy as np
import pandas as pd

from experiment.batch.auto_profile_semi_supervised_experiment import AutoProfileSemiSupervisedPdMExperiment
from experiment.batch.unsupervised_experiment import UnsupervisedPdMExperiment
from experiment.batch.incremental_semi_supervised_experiment import IncrementalSemiSupervisedPdMExperiment
from pipeline.pipeline import PdMPipeline
from method.profile_based import ProfileBased
from method.ocsvm import OneClassSVM
from method.isolation_forest import IsolationForest
from method.usad import usad
from method.TranADPdM import TranADPdM
from method.NPsemi import NeighborProfileSemi
from method.dist_k_Semi import Distance_Based_Semi
from method.ltsf_linear.ltsf_linear import LTSFLinear
from method.lof_semi import LocalOutlierFactor
from method.sand import Sand
from preprocessing.record_level.default import DefaultPreProcessor
from postprocessing.default import DefaultPostProcessor
from thresholding.constant import ConstantThresholder
from constraint_functions.constraint import (
    combine_constraint_functions,
    auto_profile_max_wait_time_constraint,
    incremental_max_wait_time_constraint,
    incremental_constraint_function,
    sand_parameters_constraint_function,
    unsupervised_max_wait_time_constraint
)
from utils import loadDataset
from utils.utils import calculate_mango_parameters
# from postprocessing.min_max_scaler import MinMaxPostProcessor

seed = 42
random.seed(seed)
torch.manual_seed(seed)
np.random.seed(seed)

conda_env = os.environ.get('CONDA_DEFAULT_ENV')

def apply_out_of_order_attack(df, max_lag, prob, seed):
    """
    Simulates Out-of-Order arrival by swapping rows locally (except 'Timestamp' column).
    Constraints: 
    - Row t can only be swapped with row k where |t-k| <= max_lag.
    """
    if max_lag == 0 or prob == 0:
        return df.copy()
    
    # Store the original Timestamp column if it exists
    timestamp_col = None
    if 'Timestamp' in df.columns:
        timestamp_col = df['Timestamp'].copy()
        df_without_timestamp = df.drop(columns=['Timestamp'])
    else:
        df_without_timestamp = df
        
    # Set random seed for reproducibility
    np.random.seed(seed)
    
    # Perform swaps only on non-Timestamp columns
    data = df_without_timestamp.values.copy()
    n_rows, n_cols = data.shape
    
    # Iterate through time steps
    # We stop 'max_lag' steps early to ensure valid swap targets
    for t in range(n_rows - max_lag):
        
        # Probabilistic trigger
        if np.random.rand() < prob:
            
            # Select swap target within [t+1, t+max_lag]
            # This ensures we don't swap with ourselves (offset 0)
            offset = np.random.randint(1, max_lag + 1)
            swap_idx = t + offset
            
            # Perform the swap (All channels move together)
            # This simulates the entire timestamp's data arriving late
            data[[t, swap_idx]] = data[[swap_idx, t]]
    
    # Create the result dataframe
    df_result = pd.DataFrame(data, index=df_without_timestamp.index, columns=df_without_timestamp.columns)
    
    # Put back the original Timestamp column if it existed
    if timestamp_col is not None:
        df_result['Timestamp'] = timestamp_col
        df_result = df_result[['Timestamp'] + [col for col in df_result.columns if col != 'Timestamp']]
    else:
        raise ValueError("DataFrame does not contain a 'Timestamp' column.")
    
    return df_result


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


def generate_out_of_order_dataset(
    dataset, 
    max_lag,
    prob,
    seed
):
    """
    Main Pipeline: Takes the list of clean DataFrames and returns 
    a corresponding list of Adversarial (Out-of-Order) DataFrames.
    
    Args:
        dataset: The dataset dictionary
        max_lag: Maximum lag for out-of-order swapping
        prob: Probability of swapping at each time step
        
    Returns:
        List[pd.DataFrame]: The attacked datasets.
    """
    attacked_dfs = []
    
    print(f"Generating Samples | Max Lag: {max_lag} | Prob: {prob}")

    for i, df in enumerate(dataset['target_data']):
        # We use specific seeds per DF to ensure reproducibility across experiments
        current_seed = seed + i 
        # 1. Apply Out-of-Order Attack
        df_final = apply_out_of_order_attack(
            df, 
            max_lag=max_lag, 
            prob=prob, 
            seed=current_seed
        )
        
        attacked_dfs.append(df_final)
        
    return attacked_dfs


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


def execute(method_names_to_run, seed, mode='noise', sigma=None, max_lag=None, prob=None, MAX_RUNS=1, MAX_JOBS=1, INITIAL_RANDOM=1):
    print("script: edp_adversarial.py")

    tracking_uri = mlflow.get_tracking_uri()
    print("MLflow Tracking URI:", tracking_uri)

    dataset = loadDataset.get_dataset("edp-wt")

    print(dataset['target_data'][0].head())

    if mode == 'noise':
        if sigma is None:
            raise ValueError("For 'noise' mode, 'sigma' must be provided.")
        
        dataset['target_data'] = generate_noisy_dataset(dataset, sigma=sigma, seed=seed)
    else:
        if max_lag is None or prob is None:
            raise ValueError("For 'out_of_order' mode, 'max_lag' and 'prob' must be provided.")
        
        dataset['target_data'] = generate_out_of_order_dataset(dataset, max_lag=max_lag, prob=prob, seed=seed)

    print(dataset['target_data'][0].head())

    # Configuration list
    # Each entry contains: 'method_name', 'experiment_type', 'method_class', 'params', 'common_params'
    configs = []

    # Unsupervised EDP CHRONOS
    if 'Chronos' in conda_env:
        from method.chronos_uns import ChronosUns
        configs.append({
            'method_name': 'CHRONOS',
            'experiment_type': 'Unsupervised EDP',
            'method_class': ChronosUns,
            'params': {
                'context_length': [1500],
                'num_samples': [1],
                'slide': [15]
            },
            'common_params': {}
        })

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

    # Auto profile EDP KNN
    configs.append({
        'method_name': 'KNN',
        'experiment_type': 'Auto profile EDP',
        'method_class': Distance_Based_Semi,
        'params': {
            'k': [2],
            'window_norm': [True]
        },
        'common_params': {
            'profile_size': [800]
        }
    })

    # Auto profile EDP LOF
    configs.append({
        'method_name': 'LOF',
        'experiment_type': 'Auto profile EDP',
        'method_class': LocalOutlierFactor,
        'params': {
            'n_neighbors': [5]
        },
        'common_params': {
            'profile_size': [1000]
        }
    })

    # Auto profile EDP LTSF
    configs.append({
        'method_name': 'LTSF',
        'experiment_type': 'Auto profile EDP',
        'method_class': LTSFLinear,
        'params': {
            'ltsf_type': ['DLinear'],
            'features': ['M'],
            'target': ['p2p_0'],
            'seq_len': [50],
            'pred_len': [1],
            'individual': [True],
            'train_epochs': [5],
            'learning_rate': [0.01],
            'batch_size': [2]
        },
        'common_params': {
            'profile_size': [800]
        }
    })

    # Incremental EDP NP
    configs.append({
        'method_name': 'NP',
        'experiment_type': 'Incremental EDP',
        'method_class': NeighborProfileSemi,
        'params': {
            'n_nnballs': [100],
            'max_sample': [80],
            'sub_sequence_length': [200],
            'aggregation_strategy': ['max'],
            'random_state': [42]
        },
        'common_params': {
            'initial_incremental_window_length': [500],
            'incremental_window_length': [500],
            'incremental_slide': [300]
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

    # Auto profile EDP PB
    configs.append({
        'method_name': 'PB',
        'experiment_type': 'Auto profile EDP',
        'method_class': ProfileBased,
        'params': {},
        'common_params': {
            'profile_size': [1000]
        }
    })

    # Unsupervised EDP SAND
    configs.append({
        'method_name': 'SAND',
        'experiment_type': 'Unsupervised EDP',
        'method_class': Sand,
        'params': {
            'pattern_length': [50],
            'subsequence_length_multiplier': [4], #4*4 this is the sub size
            'alpha': [0.25],
            'init_length': [1500],
            'batch_size': [1500],
            'k': [9],
            'aggregation_strategy': ['avg']
        },
        'common_params': {}
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

    # Incremental EDP USAD
    configs.append({
        'method_name': 'USAD',
        'experiment_type': 'Incremental EDP',
        'method_class': usad,
        'params': {
            'window_size': [100],
            'num_epochs': [10],
            'lr': [0.1],
            'BATCH_SIZE': [4],
            'hidden_size': [16]
        },
        'common_params': {
            'initial_incremental_window_length': [1750],
            'incremental_window_length': [1750],
            'incremental_slide': [1500]
        }
    })
    
    experiment_class_map = {
        'Auto profile EDP': AutoProfileSemiSupervisedPdMExperiment,
        'Incremental EDP': IncrementalSemiSupervisedPdMExperiment,
        'Unsupervised EDP': UnsupervisedPdMExperiment,
    }

    for config in configs:
        if config['method_name'] not in method_names_to_run:
            continue

        experiment_name = config['experiment_type']
        method_name = config['method_name']
        ExperimentClass = experiment_class_map[experiment_name]
        
        postprocessor = DefaultPostProcessor
        # if len(sys.argv) > 1:
        #     if sys.argv[1] == 'minmax':
        #         postprocessor = MinMaxPostProcessor
        
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
        if method_name == 'SAND':
            constraint = sand_parameters_constraint_function(my_pipeline)
        elif experiment_name == 'Incremental EDP':
            constraint = combine_constraint_functions(incremental_max_wait_time_constraint(my_pipeline), incremental_constraint_function)
        elif experiment_name == 'Auto profile EDP':
            constraint = auto_profile_max_wait_time_constraint(my_pipeline)
        elif experiment_name == 'Unsupervised EDP':
            constraint = unsupervised_max_wait_time_constraint(my_pipeline)
            
        my_experiment = ExperimentClass(
            experiment_name=f'Adversarial Noise (sigma={sigma}) ' + experiment_name + ' ' + method_name if mode == 'noise' \
                else f'Adversarial Out-of-Order (max_lag={max_lag}, prob={prob}) ' + experiment_name + ' ' + method_name,
            target_data=dataset['target_data'],
            target_sources=dataset['target_sources'],
            pipeline=my_pipeline,
            param_space=current_param_space_dict,
            num_iteration=num,
            n_jobs=jobs,
            initial_random=initial_random,
            artifacts='./artifacts/' + experiment_name + ' artifacts',
            constraint_function=constraint,
            debug=True
        )

        best_params = my_experiment.execute()
        print(experiment_name + ' ' + method_name)
        print(best_params)


parser = argparse.ArgumentParser(description='EDP Adversarial Experiment')

parser.add_argument('--method', type=str, required=True, help='Method name to run')
parser.add_argument('--seed', type=int, default=42, required=True, help='Random seed')

parser.add_argument('--mode', type=str, choices=['noise', 'out_of_order'], default='noise', help='Type of adversarial attack')

parser.add_argument('--sigma', type=float, help='Noise level')

parser.add_argument('--max_lag', type=int, help='Maximum lag for out-of-order attack')
parser.add_argument('--prob', type=float, help='Probability for out-of-order attack')

if __name__ == "__main__":
    args = parser.parse_args()
    
    method_names = [args.method]

    if args.mode == 'noise':
        execute(method_names_to_run=method_names, mode=args.mode, sigma=args.sigma, seed=args.seed)
    else:
        execute(method_names_to_run=method_names, mode=args.mode, max_lag=args.max_lag, prob=args.prob, seed=args.seed)
