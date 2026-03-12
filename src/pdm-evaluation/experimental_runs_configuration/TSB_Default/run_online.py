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
from pdm_evaluation_types.types import EventPreferences, EventPreferencesTuple
from preprocessing.record_level.default import DefaultPreProcessor
from postprocessing.default import DefaultPostProcessor
from thresholding.constant import ConstantThresholder
from preprocessing.record_level.min_max_scaler import MinMaxScaler
from constraint_functions.constraint import self_tuning_constraint_function, incremental_constraint_function, \
    combine_constraint_functions, auto_profile_max_wait_time_constraint, incremental_max_wait_time_constraint
from method.ltsf_linear.ltsf_linear import LTSFLinear
from method.lof_semi import LocalOutlierFactor
from method.HBOS import HBOS
from method.PCA import PCA_semi
from method.cnn import Cnn
from postprocessing.self_tuning import SelfTuningPostProcessor
from utils import loadDataset
from utils import automatic_parameter_generation
from utils.utils import calculate_mango_parameters
from postprocessing.min_max_scaler import MinMaxPostProcessor


def execute(method_names_to_run, MAX_RUNS=100, MAX_JOBS=16, INITIAL_RANDOM=4, setup_1_predict=100, setup_1_early=10, max_sizes=None):
    
    dataset_names_to_run=[
        "daphnet" ,
        "dodgers",
        "genesis",
        "ghl",
        "iops",
        "kdd21",
        "mgab",
        "mitdb",
        "nab",
        "nasa-msl",
        "nasa-smap",
        "occupancy",
        "opportunity",
        "sensor-scope",
        "smd",
        "svdb",
        "yahoo",
        "ecg",
    ]
    for dataset_name in dataset_names_to_run:
        print(f"script: {dataset_name}/run_online.py")

        tracking_uri = mlflow.get_tracking_uri()
        print("MLflow Tracking URI:", tracking_uri)

        dataset = loadDataset.get_dataset(f"{dataset_name}",reset_after_fail=False, setup_1_predict=setup_1_predict, setup_1_early=setup_1_early, max_sizes=max_sizes)

        experiments = [
            AutoProfileSemiSupervisedPdMExperiment,
        ]

        experiment_names = [
            f'TSB Default Auto profile {dataset_name}',
        ]

        methods = [
            IsolationForest,
            OneClassSVM,
            Distance_Based_Semi,
            NeighborProfileSemi,
            LocalOutlierFactor,
            HBOS,
            PCA_semi,
            Cnn
        ]

        method_names = [
            'IF',
            'OCSVM',
            'KNN',
            'NP',
            'LOF',
            'HBOS',
            'PCA',
            'CNN'
        ]

        param_space_dict_per_method = [
            automatic_parameter_generation.default_TSB_semi(method_name, dataset["max_wait_time"]) for method_name in
            method_names]

        for current_method, current_method_param_space, current_method_name in zip(methods, param_space_dict_per_method,
                                                                                method_names):
            print(f"Current method: {current_method}")
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

                current_param_space_dict['profile_size'] = [dataset['max_wait_time']] #automatic_parameter_generation.profile_values( dataset["max_wait_time"])

                for key, value in current_method_param_space.items():
                    current_param_space_dict[f'method_{key}'] = value

                num, jobs, initial_random = calculate_mango_parameters(current_param_space_dict, MAX_JOBS, INITIAL_RANDOM,
                                                                    MAX_RUNS)

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
