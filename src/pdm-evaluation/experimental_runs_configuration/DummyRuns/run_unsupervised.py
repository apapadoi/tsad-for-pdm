import pandas as pd
import os
import sys

import numpy as np
import mlflow

from experiment.batch.auto_profile_semi_supervised_experiment import AutoProfileSemiSupervisedPdMExperiment
from experiment.batch.unsupervised_experiment import UnsupervisedPdMExperiment
from pipeline.pipeline import PdMPipeline
from pdm_evaluation_types.types import EventPreferences, EventPreferencesTuple
from preprocessing.record_level.default import DefaultPreProcessor
from postprocessing.default import DefaultPostProcessor
from thresholding.constant import ConstantThresholder
from preprocessing.record_level.min_max_scaler import MinMaxScaler
from constraint_functions.constraint import sand_parameters_constraint_function, combine_constraint_functions, self_tuning_constraint_function, unsupervised_max_wait_time_constraint, unsupervised_distance_based
from postprocessing.self_tuning import SelfTuningPostProcessor
from utils import loadDataset
from postprocessing.min_max_scaler import MinMaxPostProcessor
from utils.utils import calculate_mango_parameters
from utils.automatic_parameter_generation import unsupervised_technique
from method.dummy_all import DummyAll
from method.dummy_increase import DummyIncrease
from method.sand import Sand

def execute(method_names_to_run, MAX_RUNS=1, MAX_JOBS=1, INITIAL_RANDOM=1):
    dataset_names_to_run = [
        "femto"
        #"metropt-3"
        #"bhd"
    ]
    for dataset_name in dataset_names_to_run:

        print(f"script: {dataset_name}/run_unsupervised.py")

        tracking_uri = mlflow.get_tracking_uri()
        print("MLflow Tracking URI:", tracking_uri)

        dataset = loadDataset.get_dataset(dataset_name)

        experiments = [
            UnsupervisedPdMExperiment,
        ]

        experiment_names = [
            f"Unsupervised {dataset_name.upper()}",
        ]

        methods = [
            Sand
        ]

        method_names = [
            'SAND',
        ]

        param_space_dict_per_method = [
            # {
            #     'pattern_length': [3],
            #     'subsequence_length_multiplier': [3],  # 4*4 this is the sub size
            #     'alpha': [0.5],
            #     'init_length': [13],
            #     'batch_size': [12],
            #     'k': [3],
            #     'aggregation_strategy': ['max']
            # }
            {
                'pattern_length': [10],
                'subsequence_length_multiplier': [1],  # 4*4 this is the sub size
                'alpha': [0.5],
                'init_length': [100],
                'batch_size': [80],
                'k': [8],
                'aggregation_strategy': ['avg']
            }
            # {
            #     'pattern_length': [25],
            #     'subsequence_length_multiplier': [5],  # 4*4 this is the sub size
            #     'alpha': [0.75],
            #     'init_length': [206],
            #     'batch_size': [128],
            #     'k': [3],
            #     'aggregation_strategy': ['max']
            # }
        ]

        for current_method, current_method_param_space, current_method_name in zip(methods, param_space_dict_per_method,
                                                                                   method_names):
            if current_method_name not in method_names_to_run:
                continue

            postprocessor = DefaultPostProcessor
            if len(sys.argv) > 1:
                if sys.argv[1] == 'minmax':
                    postprocessor = MinMaxPostProcessor

            my_pipeline = PdMPipeline(
                steps={
                    # 'preprocessor': Windowing,
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
                    # 'preprocessor_slidingWindow': [1],
                }

                for key, value in current_method_param_space.items():
                    current_param_space_dict[f'method_{key}'] = value

                num, jobs, initial_random = calculate_mango_parameters(current_param_space_dict, MAX_JOBS,
                                                                       INITIAL_RANDOM, MAX_RUNS)

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
                    sand_parameters_constraint_function(my_pipeline) if 'SAND' == current_method_name
                    else combine_constraint_functions(unsupervised_distance_based,
                                                      unsupervised_max_wait_time_constraint(
                                                          my_pipeline)) if 'KNN' == current_method_name
                    else unsupervised_max_wait_time_constraint(my_pipeline),
                    debug=True
                )

                best_params = my_experiment.execute()
                print(experiment_name)
                print(best_params)
