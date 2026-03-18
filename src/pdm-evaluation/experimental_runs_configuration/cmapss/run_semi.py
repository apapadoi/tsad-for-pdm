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
    print("script: cmapss/run_semi.py")

    tracking_uri = mlflow.get_tracking_uri()
    print("MLflow Tracking URI:", tracking_uri)

    dataset = loadDataset.get_dataset("cmapss", True)

    experiments = [
        SemiSupervisedPdMExperiment,
    ]

    experiment_names = [
        'Semisupervised CMAPSS',
    ]

    param_space_dict_per_method = [
        {
            'n_estimators': [50, 100, 150, 200],
            'max_samples': [10, 20, 40],
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
            'k': [2, 4, 5, 7, 9, 10, 15, 20, 30, 40],
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
            'n_neighbors': [2, 4, 5, 7, 9, 10, 15, 20, 30, 40] 
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
            'batch_size': [16],
            'n_steps': [15],#[20], # [5, 10, 15, 20],
            'domain_prompt_content': [
"""Engine degradation simulation was carried out using the Commercial Modular Aero-Propulsion System Simulation (C-MAPSS).\
 The data include several sensor channels to characterize fault evolution.\
 The dataset was provided by the NASA Ames Prognostics Center of Excellence (PCoE)."""
            ],
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
            
            for key, value in current_method_param_space.items():
                current_param_space_dict[f'method_{key}'] = value

            num, jobs, initial_random = calculate_mango_parameters(current_param_space_dict, MAX_JOBS, INITIAL_RANDOM, MAX_RUNS)

            my_experiment = experiment(
                experiment_name=experiment_name + ' ' + current_method_name,
                historic_data=dataset['historic_data'],
                historic_sources=dataset['historic_sources'],
                target_data=dataset['target_data'],
                target_sources=dataset['target_sources'],
                pipeline=my_pipeline,
                param_space=current_param_space_dict,
                num_iteration=num,
                n_jobs=jobs,
                initial_random=initial_random,
                artifacts='./artifacts/' + experiment_name + ' artifacts',
                debug=True
            )

            best_params = my_experiment.execute()
            print(experiment_name)
            print(best_params)
