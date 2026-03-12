from experiment.batch.auto_profile_semi_supervised_experiment import AutoProfileSemiSupervisedPdMExperiment
from method.isolation_forest import IsolationForest
from method.ocsvm import OneClassSVM
from method.TranADPdM import TranADPdM
from median_randomized_smoothing import MedianRandomizedSmoothing
from utils import loadDataset
import pandas as pd
import numpy as np
import random
import os
import mlflow
import argparse
from pipeline.pipeline import PdMPipeline
from preprocessing.record_level.default import DefaultPreProcessor
from postprocessing.default import DefaultPostProcessor
from thresholding.constant import ConstantThresholder
from constraint_functions.constraint import (
    auto_profile_max_wait_time_constraint
)
from utils.utils import calculate_mango_parameters

seed = 42
random.seed(seed)
np.random.seed(seed)

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

def execute(method_names_to_run, seed, sigma_values, moving_average_values, MAX_RUNS=1, MAX_JOBS=1, INITIAL_RANDOM=0):
    print("script: edp_median_smoothing_experiment.py")

    tracking_uri = mlflow.get_tracking_uri()
    print("MLflow Tracking URI:", tracking_uri)

    # We need to iterate over combinations of sigma and moving_average
    for sigma in sigma_values:
        for moving_average in moving_average_values:
            print(f"Running for Sigma: {sigma}, Moving Average: {moving_average}")
            
            # Reload dataset for each iteration to apply fresh noise
            dataset = loadDataset.get_dataset("edp-wt")
            print(dataset['target_sources'])
            target_sources = dataset['target_sources']
            # Apply noise
            dataset['target_data'] = generate_noisy_dataset(dataset, sigma=sigma, seed=seed)
            
            configs = []

            # Auto profile EDP with MedianRandomizedSmoothing
            supported_techniques = ['OCSVM', 'IF', 'TRANAD']
            
            for technique in supported_techniques:
                 if technique in method_names_to_run or 'ALL' in method_names_to_run:
                    
                     # Define profile size based on the technique based on edp_adversarial.py
                    profile_size = 800
                    if technique == 'IF':
                        profile_size = 900
                    elif technique == 'TRANAD':
                         profile_size = 1500

                    configs.append({
                        'method_name': f'Median_{technique}',
                        'experiment_type': 'Auto profile EDP',
                        'method_class': MedianRandomizedSmoothing,
                        'params': {
                            'sources': [target_sources],
                            'sigma': [sigma],
                            'moving_average': [moving_average],
                            'technique': [technique]
                        },
                        'common_params': {
                            'profile_size': [profile_size] 
                        }
                    })

            experiment_class_map = {
                'Auto profile EDP': AutoProfileSemiSupervisedPdMExperiment,
            }

            for config in configs:
                method_name = config['method_name']
                experiment_name = config['experiment_type']
                # MethodClass = config['method_class']
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
                
                # Setup parameter space
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

                # Determine constraint function (only Auto profile EDP used here)
                constraint = auto_profile_max_wait_time_constraint(my_pipeline)

                # Experiment Name Construction
                # e.g., "Median Smoothing (sigma=100.0) (moving_average=True) Auto profile EDP Median_OCSVM"
                final_experiment_name = f"Median Smoothing (sigma={sigma}) (moving_average={moving_average}) {experiment_name} {method_name}"

                my_experiment = ExperimentClass(
                    experiment_name=final_experiment_name,
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
                print(final_experiment_name)
                print(best_params)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='EDP Median Randomized Smoothing Experiment')
    parser.add_argument('--method', type=str, default='ALL', help='Underlying technique name to run (OCSVM, IF, TRANAD or ALL)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')

    args = parser.parse_args()
    
    method_names = [args.method] if args.method != 'ALL' else ['OCSVM', 'IF', 'TRANAD']
    
    sigma_values = [100.0, 1000.0, 10000.0]
    moving_average_values = [True, False]

    execute(method_names_to_run=method_names, seed=args.seed, sigma_values=sigma_values, moving_average_values=moving_average_values)