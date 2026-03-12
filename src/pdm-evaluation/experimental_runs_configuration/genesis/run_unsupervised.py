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
from preprocessing.record_level.windowing import Windowing
from constraint_functions.constraint import sand_parameters_constraint_function, combine_constraint_functions, self_tuning_constraint_function, unsupervised_max_wait_time_constraint, unsupervised_distance_based
from method.ltsf_linear.ltsf_linear import LTSFLinear
from method.dist_k_uns import Distance_Based_Uns
from method.isolation_forest_uns import IsolationForestUnsupervised
from method.lof_uns import LocalOutlierFactorUnsupervised
from postprocessing.self_tuning import SelfTuningPostProcessor
from utils import loadDataset
from postprocessing.min_max_scaler import MinMaxPostProcessor
from utils.utils import calculate_mango_parameters
from utils import automatic_parameter_generation


def execute(method_names_to_run, MAX_RUNS=100, MAX_JOBS=16, INITIAL_RANDOM=4):
    print("script: genesis/run_unsupervised.py")

    tracking_uri = mlflow.get_tracking_uri()
    print("MLflow Tracking URI:", tracking_uri)

    dataset = loadDataset.get_dataset("genesis")

    experiments = [
        UnsupervisedPdMExperiment,
    ]

    experiment_names = [
        'Unsupervised Genesis',
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


    param_space_dict_per_method = [automatic_parameter_generation.unsupervised_technique(method_name,dataset["max_wait_time"]) for method_name in method_names]

    for current_method, current_method_param_space, current_method_name in zip(methods, param_space_dict_per_method, method_names):
        if current_method_name not in method_names_to_run:
            continue
        
        postprocessor = DefaultPostProcessor
        if len(sys.argv) > 1:
            if sys.argv[1] == 'minmax':
                postprocessor = MinMaxPostProcessor
                
        my_pipeline = PdMPipeline(
            steps={
                #'preprocessor': Windowing,
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
                #'preprocessor_slidingWindow': [1],
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
