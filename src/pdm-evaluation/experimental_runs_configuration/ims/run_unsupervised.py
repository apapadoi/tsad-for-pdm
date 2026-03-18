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
from method.NPuns import NeighborProfileUns
from method.sand import Sand
from pdm_evaluation_types.types import EventPreferences, EventPreferencesTuple
from preprocessing.record_level.default import DefaultPreProcessor
from postprocessing.default import DefaultPostProcessor
from thresholding.constant import ConstantThresholder
from preprocessing.record_level.min_max_scaler import MinMaxScaler
from constraint_functions.constraint import sand_parameters_constraint_function, combine_constraint_functions, self_tuning_constraint_function, unsupervised_max_wait_time_constraint, unsupervised_distance_based
from method.ltsf_linear.ltsf_linear import LTSFLinear
from method.dist_k_uns import Distance_Based_Uns
from method.isolation_forest_uns import IsolationForestUnsupervised
from method.lof_uns import LocalOutlierFactorUnsupervised
from postprocessing.self_tuning import SelfTuningPostProcessor
from utils import loadDataset
from postprocessing.min_max_scaler import MinMaxPostProcessor
from utils.utils import calculate_mango_parameters

conda_env = os.environ.get('CONDA_DEFAULT_ENV')

def execute(method_names_to_run, MAX_RUNS=100, MAX_JOBS=16, INITIAL_RANDOM=4):
    print("script: ims/run_unsupervised.py")

    tracking_uri = mlflow.get_tracking_uri()
    print("MLflow Tracking URI:", tracking_uri)

    dataset = loadDataset.get_dataset("ims")

    experiments = [
        UnsupervisedPdMExperiment, 
    ]

    experiment_names = [
        'Unsupervised IMS',
    ]

    param_space_dict_per_method = [
        {
            'n_nnballs': [50, 75, 100, 125, 150],
            'max_sample': [8, 10, 12],
            'sub_sequence_length': [5, 10, 15, 20],
            'aggregation_strategy': ['avg', 'max'],
            'window': [50, 75, 100, 110, 120, 130, 140, 150, 200, 250, 300],
            'slide': [0.33, 0.5, 1.0],
            'overlap_aggregation_strategy': ['first', 'last', 'avg'],
            'random_state': [42]
        },
        {
            'window': [50, 75, 100, 110, 120, 130, 140, 150, 200, 250, 300],
            'slide': [0.33, 0.5, 1.0],
            'k': [2, 4, 5, 7, 9, 10, 15, 20, 30, 40, 50],
            'window_norm': [False, True],
            'policy': ['or', 'and', 'first', 'last']
        },
        {
            'window': [50, 75, 100, 110, 120, 130, 140, 150, 200, 250, 300],
            'slide': [0.33, 0.5, 1.0],
            'n_estimators': [50, 100, 150, 200],
            'max_samples': [100, 200, 300, 400],
            'max_features': [0.5, 0.6, 0.7, 0.8],
            'bootstrap': [True, False],
            'random_state': [42],
            'policy': ['or', 'and', 'first', 'last']
        },
        {
            'window': [50, 75, 100, 110, 120, 130, 140, 150, 200, 250, 300],
            'slide': [0.33, 0.5, 1.0],
            'n_neighbors': [2, 4, 5, 7, 9, 10, 15, 20, 30, 40, 50] 
        },
        {
            'pattern_length': [5, 10, 15, 20, 30],
            'subsequence_length_multiplier': [3, 4, 5], #4*4 this is the sub size
            'alpha': [0.5, 0.75, 0.25],
            'init_length': [80, 100, 150, 170, 200, 225, 250, 275, 300],
            'batch_size': [80, 100, 150, 170, 200, 225, 250, 275, 300],
            'k': [6, 7, 8, 9, 10],
            'aggregation_strategy': ['avg', 'max']
        }
    ]

    methods = [
        NeighborProfileUns,
        Distance_Based_Uns,
        IsolationForestUnsupervised,
        LocalOutlierFactorUnsupervised,
        Sand
    ]

    method_names = [
        'NP',
        'KNN',
        'IF',
        'LOF',
        'SAND',
    ]

    if 'Chronos' in conda_env:
        param_space_dict_per_method.append(
            {
                'context_length': [50, 75, 100, 110, 120, 130, 140, 150, 200, 250, 300],
                'num_samples': [1, 3, 5, 10],
                'slide': [1]
            }
        )

        from method.chronos_uns import ChronosUns

        methods.append(ChronosUns)

        method_names.append('CHRONOS')

        param_space_dict_per_method.append({
            'context_length': [300],
            'target': ['p2p_0']
        })

        from method.autogluon import AutogluonUns

        methods.append(AutogluonUns)

        method_names.append('AUTOGLUON')


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
                target_data=dataset['target_data'],
                target_sources=dataset['target_sources'],
                pipeline=my_pipeline,
                param_space=current_param_space_dict,
                num_iteration=num,
                n_jobs=jobs,
                initial_random=initial_random,
                artifacts='./artifacts/' + experiment_name + ' artifacts',
                constraint_function=
                    sand_parameters_constraint_function() if 'SAND' == current_method_name 
                    else combine_constraint_functions(unsupervised_distance_based, unsupervised_max_wait_time_constraint(my_pipeline)) if 'KNN' == current_method_name
                    else unsupervised_max_wait_time_constraint(my_pipeline),
                debug=True
            )

            best_params = my_experiment.execute()
            print(experiment_name)
            print(best_params)
