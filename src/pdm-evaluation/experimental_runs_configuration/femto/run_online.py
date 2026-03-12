import pandas as pd
import os
import sys

import numpy as np
import mlflow

from experiment.batch.auto_profile_semi_supervised_experiment import AutoProfileSemiSupervisedPdMExperiment
from experiment.batch.unsupervised_experiment import UnsupervisedPdMExperiment
from experiment.batch.incremental_semi_supervised_experiment import IncrementalSemiSupervisedPdMExperiment
from experiment.batch.semi_supervised_experiment import SemiSupervisedPdMExperiment
from pipeline.pipeline import PdMPipeline
from method.profile_based import ProfileBased
from method.ocsvm import OneClassSVM
from method.isolation_forest import IsolationForest
from method.usad import usad
from method.TranADPdM import TranADPdM
from method.NPsemi import NeighborProfileSemi
from method.dist_k_Semi import Distance_Based_Semi
from method.timemixerpp.timemixerpp import TimeMixerPP
from pdm_evaluation_types.types import EventPreferences, EventPreferencesTuple
from preprocessing.record_level.default import DefaultPreProcessor
from postprocessing.default import DefaultPostProcessor
from thresholding.constant import ConstantThresholder
from preprocessing.record_level.min_max_scaler import MinMaxScaler
from constraint_functions.constraint import self_tuning_constraint_function, incremental_constraint_function, combine_constraint_functions, auto_profile_max_wait_time_constraint, incremental_max_wait_time_constraint
from method.ltsf_linear.ltsf_linear import LTSFLinear
from method.lof_semi import LocalOutlierFactor
from postprocessing.self_tuning import SelfTuningPostProcessor
from utils import loadDataset
from utils.utils import calculate_mango_parameters
from postprocessing.min_max_scaler import MinMaxPostProcessor

conda_env = os.environ.get('CONDA_DEFAULT_ENV')

def execute(method_names_to_run, MAX_RUNS=100, MAX_JOBS=16, INITIAL_RANDOM=4):
    print("script: femto/run_online.py")

    tracking_uri = mlflow.get_tracking_uri()
    print("MLflow Tracking URI:", tracking_uri)

    dataset = loadDataset.get_dataset("femto")

    experiments = [
        AutoProfileSemiSupervisedPdMExperiment,
    ]

    experiment_names = [
        'Auto profile FEMTO',
    ]

    param_space_dict_per_method = [
        {
            'n_estimators': [50, 100, 150, 200],
            'max_samples': [100, 200, 300, 400],
            'random_state': [42],
            'max_features': [0.5, 0.6, 0.7, 0.8],
            'bootstrap': [True, False]
        },
        {
            'kernel': ['linear', 'rbf', 'poly', 'sigmoid'],
            'nu': [0.01, 0.05, 0.1, 0.15, 0.2, 0.5],
            'gamma': ['scale', 'auto'],
            'degree': [2, 3, 4, 5],
        },
        {
            # Profile based
        },
        {
            'k': [2, 4, 5, 7, 9, 10, 15, 20, 30, 40, 50],
            'window_norm': [False, True],
        },
        {
            'n_nnballs': [10, 20, 50, 75, 100, 125, 150],
            'max_sample': [8, 10, 12, 14, 16, 18, 20],
            'sub_sequence_length': [5, 10, 15, 20],
            'aggregation_strategy': ['avg', 'max'],
            'random_state': [42]
        },
        {
            'n_neighbors': [2, 4, 5, 7, 9, 10, 15, 20, 30, 40, 50] 
        },
        {
            'ltsf_type': ['Linear', 'DLinear', 'NLinear'],
            'features': ['M', 'MS'],
            'target': ['p2p_0'],
            'seq_len': [5, 10, 15, 20],
            'pred_len': [1],
            'individual': [True, False],
            'train_epochs': [3, 5, 10, 15, 20, 25],
            'learning_rate': [0.001, 0.01, 0.1],
            'batch_size': [2, 4, 8, 16]
        },
        {
            'window_size': [5, 10, 15, 20],
            'num_epochs': [5, 10, 15, 20, 25],
            'lr': [0.001, 0.01, 0.1, 0.05, 0.005]
        },
        {
            'window_size': [5, 10, 15, 20],
            'num_epochs': [5, 10, 15, 20, 25],
            'lr': [0.001, 0.01, 0.1],
            'BATCH_SIZE': [2, 4, 8, 16],
            'hidden_size': [2, 4, 8, 16]
        },
        {
            'seq_len': [20],
            'moving_avg': [5],
            'train_epochs': [25],
        }
    ]

    methods = [
        IsolationForest,
        OneClassSVM,
        ProfileBased,
        Distance_Based_Semi,
        NeighborProfileSemi,
        LocalOutlierFactor,
        LTSFLinear,
        TranADPdM,
        usad,
        TimeMixerPP
    ]

    method_names = [
        'IF',
        'OCSVM',
        'PB',
        'KNN',
        'NP',
        'LOF',
        'LTSF',
        'TRANAD',
        'USAD',
        'TIMEMIXERPP'
    ]

    if 'Chronos' in conda_env:        
        if len(sys.argv) > 2:
            os.environ['CUDA_VISIBLE_DEVICES'] = sys.argv[2]
        else:
            os.environ['CUDA_VISIBLE_DEVICES'] = '0'
           
        from method.chronos_semi import ChronosSemi

        methods.append(ChronosSemi)

        method_names.append('CHRONOS')

        param_space_dict_per_method.append({
            'context_length': [5, 10, 15, 20],
            'num_samples': [1, 3, 5, 10],
            'slide': [15],
            'max_steps': [1, 3, 5],
            'learning_rate': [0.001, 0.01, 0.1]
        })

    if 'TimeLLM' in conda_env:
        from method.time_llm_pypots import TimeLLMPyPots

        methods.append(TimeLLMPyPots)

        method_names.append('TimeLLM')

        param_space_dict_per_method.append({
            'llm_model_type': ['GPT2'],
            'n_layers': [2], # 32
            'patch_size': [8], # 16
            'patch_stride': [4], # 8
            'd_llm': [768],
            'd_model': [2], # 32
            'd_ffn': [8], # 32
            'n_heads': [4], # 8
            'batch_size': [8],
            'n_steps': [20], # [5, 10, 15, 20],
            'domain_prompt_content': ['The dataset incorporates accelerated bearing experiments conducted on a small-scale bearing test rig, originates from the FEMTO-ST Institute and was employed in the 2012 IEEE Prognostics Challenge. Statistical and frequency domain features were extracted from each vibration measurement. Specifically, three features were computed: (1) the peak-to-peak amplitude; (2) the root mean square (RMS) value; and (3) spectral features derived from the discrete Fourier transform.'],
            'epochs': [25] # 10
        })

    for current_method, current_method_param_space, current_method_name in zip(methods, param_space_dict_per_method, method_names):
        if current_method_name not in method_names_to_run:
            continue

        postprocessor = DefaultPostProcessor
        if len(sys.argv) > 1:
            if sys.argv[1] == 'minmax':
                postprocessor = MinMaxPostProcessor
        
        my_pipeline = PdMPipeline(
            steps={
                'preprocessor': DefaultPreProcessor,
                'method': current_method,
                'postprocessor': postprocessor,
                'thresholder': ConstantThresholder,
            },
            dataset=dataset,
            auc_resolution=100
        )

        for experiment, experiment_name in zip(experiments, experiment_names):
            current_param_space_dict = {
                'thresholder_threshold_value': [0.5],
            }

            current_param_space_dict['profile_size'] = [50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200] if 'TimeLLM' not in conda_env else [170]
            
            if current_method_name == 'TIMEMIXERPP':
                current_param_space_dict['profile_size'] = [170]

            for key, value in current_method_param_space.items():
                current_param_space_dict[f'method_{key}'] = value

            num, jobs, initial_random = calculate_mango_parameters(current_param_space_dict, MAX_JOBS, INITIAL_RANDOM, MAX_RUNS)

            if 'Chronos' in conda_env and len(sys.argv) > 2:
                current_param_space_dict['method_device_type'] = ['cuda']
            elif 'Chronos' in conda_env:
                current_param_space_dict['method_device_type'] = ['cuda']

            my_experiment = experiment(
                experiment_name=experiment_name + ' ' + current_method_name,
                target_data=dataset['target_data'],
                target_sources=dataset['target_sources'],
                pipeline=my_pipeline,
                param_space=current_param_space_dict,
                num_iteration=num,
                n_jobs=jobs,
                initial_random=initial_random,
                artifacts='./artifacts/' + experiment_name + ' artifacts',
                constraint_function=auto_profile_max_wait_time_constraint(my_pipeline),
                debug=True
            )

            best_params = my_experiment.execute()
            print(experiment_name)
            print(best_params)
