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
from utils.automatic_parameter_generation import online_technique, profile_values

conda_env = os.environ.get('CONDA_DEFAULT_ENV')

def execute(method_names_to_run, MAX_RUNS=100, MAX_JOBS=16, INITIAL_RANDOM=4):
    print("script: xjtu/run_online.py")

    tracking_uri = mlflow.get_tracking_uri()
    print("MLflow Tracking URI:", tracking_uri)

    dataset = loadDataset.get_dataset("xjtu")

    experiments = [
        AutoProfileSemiSupervisedPdMExperiment,
    ]

    experiment_names = [
        'Auto profile XJTU PH 18',
    ]

    param_space_dict_per_method = []

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

    if 'TimeLLM' in conda_env:
        from method.time_llm_pypots import TimeLLMPyPots

        methods.append(TimeLLMPyPots)

        method_names.append('TimeLLM')

    for method_name in method_names:
        param_space_dict_per_method.append(online_technique(method_name, dataset['max_wait_time']))

    if 'TimeLLM' in conda_env:
        param_space_dict_per_method[-1]['domain_prompt_content'] = [
            '15 rolling element bearings whose type is LDK UER204 are tested under 3 different operating conditions. Failure modes include inner race wear, outer race wear, outer race fracture. Two PCB 352C33 accelerometers are placed on the housing of the bearings and positioned at 90 degrees to each other.'
        ]

        param_space_dict_per_method[-1]['batch_size'] = [16]
        param_space_dict_per_method[-1]['n_steps'] = [30]
        dataset['max_wait_time'] = 32

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

            current_param_space_dict['profile_size'] = profile_values(dataset["max_wait_time"]) if 'TimeLLM' not in conda_env else [max(profile_values(dataset["max_wait_time"]))]
            
            if current_method_name == 'TIMEMIXERPP':
                current_param_space_dict['profile_size'] = [max(profile_values(dataset["max_wait_time"]))]
                current_method_param_space['down_sampling_layers'] = [1]

            for key, value in current_method_param_space.items():
                current_param_space_dict[f'method_{key}'] = value

            num, jobs, initial_random = calculate_mango_parameters(current_param_space_dict, MAX_JOBS, INITIAL_RANDOM, MAX_RUNS)

            if 'Chronos' in conda_env and len(sys.argv) > 2:
                current_param_space_dict['method_device_type'] = ['cuda:' + sys.argv[2]]
            elif 'Chronos' in conda_env:
                current_param_space_dict['method_device_type'] = ['cuda:0']

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
