import pickle

import pandas as pd
import os

cwd = os.getcwd()
print(cwd)
import sys
sys.path.insert(0, cwd)

import numpy as np
import mlflow
from pipeline.pipeline import PdMPipeline
from pdm_evaluation_types.types import EventPreferences, EventPreferencesTuple
from utils.utils import calculate_mango_parameters

# Experiments
from experiment.batch.auto_profile_semi_supervised_experiment import AutoProfileSemiSupervisedPdMExperiment
from experiment.batch.unsupervised_experiment import UnsupervisedPdMExperiment
from experiment.batch.incremental_semi_supervised_experiment import IncrementalSemiSupervisedPdMExperiment
from experiment.batch.semi_supervised_experiment import SemiSupervisedPdMExperiment


from thresholding.constant import ConstantThresholder

# post processing
from postprocessing.default import DefaultPostProcessor
from postprocessing.dynamicth import DynamicThresholder
from postprocessing.Moving2T import Moving2Thresholder
from postprocessing.self_tuning import SelfTuningPostProcessor
from postprocessing.min_max_scaler import MinMaxPostProcessor

# pre_processing
from preprocessing.record_level.default import DefaultPreProcessor
from preprocessing.record_level.aggregator import MeanAggregator
from preprocessing.record_level.min_max_scaler import MinMaxScaler
from preprocessing.record_level.feature_selector import FeatureSelector
from preprocessing.record_level.windowing import Windowing



# automatic imports
from utils.automatic_parameter_generation import online_technique,incremental_technique,unsupervised_technique, semi_technique,profile_values

# constains
from constraint_functions.constraint import self_tuning_constraint_function, incremental_constraint_function, combine_constraint_functions, auto_profile_max_wait_time_constraint, incremental_max_wait_time_constraint
from constraint_functions.constraint import sand_parameters_constraint_function, combine_constraint_functions, self_tuning_constraint_function, unsupervised_max_wait_time_constraint, unsupervised_distance_based


# Methods SEMI
from method.lof_semi import LocalOutlierFactor
from method.dist_k_Semi import Distance_Based_Semi
from method.dummy_increase import DummyIncrease
from method.ltsf_linear.ltsf_linear import LTSFLinear
from method.profile_based import ProfileBased
from method.ocsvm import OneClassSVM
from method.isolation_forest import IsolationForest
from method.usad import usad
from method.TranADPdM import TranADPdM
from method.NPsemi import NeighborProfileSemi
from method.HBOS import HBOS
from method.PCA import PCA_semi
from method.cnn import Cnn

# Methods Unsupervised
from method.NPuns import NeighborProfileUns
from method.sand import Sand
from method.dist_k_uns import Distance_Based_Uns
from method.DAMP import Damp
from method.HBOS_uns import HBOSUns
from method.PCA_uns import PCA_uns
from method.isolation_forest_uns import IsolationForestUnsupervised
from method.lof_uns import LocalOutlierFactorUnsupervised


def execute(params):
    # print(params["dataset"]["event_data"])
    # print(params["dataset"]["event_preferences"])
    if params["methodology"]=="Unsupervised":
        execute_uns(params,MAX_RUNS=params["MAX_RUNS"],MAX_JOBS=params["MAX_JOBS"],INITIAL_RANDOM=params["INITIAL_RANDOM"])
    else:
        execute_semi(params,MAX_RUNS=params["MAX_RUNS"],MAX_JOBS=params["MAX_JOBS"],INITIAL_RANDOM=params["INITIAL_RANDOM"])
def execute_semi(params, MAX_RUNS=1, MAX_JOBS=1, INITIAL_RANDOM=1):
    # print(f"script: {dataset_name}/run_online.py")

    tracking_uri = mlflow.get_tracking_uri()
    print("MLflow Tracking URI:", tracking_uri)


    method_names_to_run = params['methods']
    dataset_name=params['dataset_name']


    dataset = params['dataset']

    methods = [
     LocalOutlierFactor,Distance_Based_Semi,ProfileBased,NeighborProfileSemi,
        OneClassSVM,IsolationForest,

     LTSFLinear,TranADPdM,usad,

     HBOS,PCA_semi,Cnn,

     DummyIncrease,
    ]

    method_names = [
        'LOF','KNN','PB','NP','OCSVM','IF',

        'LTSF','TRANAD','USAD',

        'HBOS','PCA','CNN',

        'DummyIncrease'
    ]

    experiments, experiment_names,exper_automatic=select_experiments(params['select_flavors'], dataset_name)

    preprocesor, param_space_dict_Pre, postprocesor, param_space_dict_Post = pre_post(params)

    param_space_dict_per_expetype_per_method = method_parameters(params, exper_automatic, method_names,
                                                                 method_names_to_run,dataset)

    experiment_parameters=[]
    for exp_name in params['select_flavors']:
        experiment_parameters.append(params[exp_name+"_parameters"])

    optimization_metric = params["optimization_metric"]

    run_experiments(dataset, method_names, param_space_dict_per_expetype_per_method, methods, method_names_to_run, preprocesor,
                    param_space_dict_Pre, postprocesor, param_space_dict_Post,
                    experiments, experiment_names, experiment_parameters,
                    MAX_JOBS, INITIAL_RANDOM, MAX_RUNS,optimization_metric)




def execute_uns(params, MAX_RUNS=1, MAX_JOBS=1, INITIAL_RANDOM=1):
    # print(f"script: {dataset_name}/run_online.py")

    tracking_uri = mlflow.get_tracking_uri()
    print("MLflow Tracking URI:", tracking_uri)

    method_names_to_run = params['methods']
    dataset_name = params['dataset_name']


    dataset = params['dataset']
    methods = [

        LocalOutlierFactorUnsupervised,
        Distance_Based_Uns,
        NeighborProfileUns,
        IsolationForestUnsupervised,
        Sand,

        Damp,
        HBOSUns,
        PCA_uns,

    ]

    method_names = [
        'LOF','KNN','NP','IF','SAND',
        'DAMP','HBOS','PCA',
    ]

    experiments=[UnsupervisedPdMExperiment]
    experiment_names=[f"Unsupervised {dataset_name.upper()}"]
    exper_automatic=[unsupervised_technique]

    preprocesor,param_space_dict_Pre,postprocesor,param_space_dict_Post = pre_post(params)

    param_space_dict_per_expetype_per_method=method_parameters(params, exper_automatic, method_names, method_names_to_run,dataset)

    experiment_parameters=[{}]

    optimization_metric=params["optimization_metric"]
    run_experiments(dataset, method_names,param_space_dict_per_expetype_per_method, methods, method_names_to_run, preprocesor, param_space_dict_Pre, postprocesor, param_space_dict_Post,
                    experiments, experiment_names, experiment_parameters,
                    MAX_JOBS, INITIAL_RANDOM, MAX_RUNS,optimization_metric)


def run_experiments(dataset,method_names,param_space_dict_per_expetype_per_method,methods,method_names_to_run,preprocesor, param_space_dict_Pre, postprocesor, param_space_dict_Post,experiments,experiment_names,experiment_parameters,MAX_JOBS,
                                                                   INITIAL_RANDOM, MAX_RUNS,optimization_metric):

    BEST_configuration=None
    metric_best=-1
    for experiment, experiment_name, param_space_dict_per_method,experiment_parameter_ in zip(experiments, experiment_names, param_space_dict_per_expetype_per_method,experiment_parameters):


        for current_method, current_method_param_space, current_method_name in zip(methods, param_space_dict_per_method,
                                                                                   method_names):
            if current_method_name not in method_names_to_run:
                continue

            my_pipeline = PdMPipeline(
                steps={
                    # 'preprocessor': Windowing,preprocesor,postprocesor
                    'preprocessor': preprocesor,
                    'method': current_method,
                    'postprocessor': postprocesor,
                    'thresholder': ConstantThresholder,
                },
                dataset=dataset,
                auc_resolution=100
            )

            current_param_space_dict = {
                'thresholder_threshold_value': [0.5],
                # 'preprocessor_slidingWindow': [1],
            }
            for key, value in param_space_dict_Pre.items():
                current_param_space_dict[f'preprocessor_{key}'] = value
            for key, value in current_method_param_space.items():
                current_param_space_dict[f'method_{key}'] = value
            for key, value in param_space_dict_Post.items():
                current_param_space_dict[f'postprocessor_{key}'] = value

            for key in experiment_parameter_.keys():
                current_param_space_dict['profile_size'] =experiment_parameter_[key]

            num, jobs, initial_random = calculate_mango_parameters(current_param_space_dict, MAX_JOBS,
                                                                   INITIAL_RANDOM, MAX_RUNS)


            exp_constrain = [auto_profile_max_wait_time_constraint(my_pipeline),
                             combine_constraint_functions(incremental_max_wait_time_constraint(my_pipeline),
                                                          incremental_constraint_function),
                             None,sand_parameters_constraint_function() if 'SAND' == current_method_name
                    else combine_constraint_functions(unsupervised_distance_based, unsupervised_max_wait_time_constraint(my_pipeline)) if 'KNN' == current_method_name
                    else unsupervised_max_wait_time_constraint(my_pipeline)]

            flavors = ["Online", "Sliding", "Historical","Unsupervised"]


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
                constraint_function=exp_constrain[flavors.index(experiment_name.split(" ")[0])],
                debug=True,
                log_best_scores=False,
                optimization_param=optimization_metric
            )

            best_params = my_experiment.execute()
            print(experiment_name)
            print(best_params)
            if BEST_configuration is None:
                BEST_configuration=best_params
                metric_best=best_params["best_objective"]
            elif metric_best<best_params["best_objective"]:
                BEST_configuration = best_params
                metric_best = best_params["best_objective"]
    return BEST_configuration

def method_parameters(params,exper_automatic,method_names,method_name_to_run,dataset):
    param_space_dict_per_expetype_per_method = []
    for exp_auto in exper_automatic:
        param_space_dict_per_method = []
        for method_name in method_names:
            if method_name not in method_name_to_run:
                param_space_dict_per_method.append({})
            elif params[f"automatic_{method_name}"]:
                param_space_dict_per_method.append(exp_auto(method_name, dataset['max_wait_time']))
            else:
                param_space_dict_per_method.append(params["method_"+method_name + "_parameters"])
        param_space_dict_per_expetype_per_method.append(param_space_dict_per_method)
    return param_space_dict_per_expetype_per_method
def pre_post(params):
    Allpreprocesors = [DefaultPreProcessor, FeatureSelector, MinMaxScaler, Windowing, MeanAggregator]
    pre_names = ["Default", "Keep Features", "MinMax Scaler (semi)", "Windowing (one column)", "Mean Aggregator"]

    Allpostprocesors = [DefaultPostProcessor, DynamicThresholder, Moving2Thresholder, SelfTuningPostProcessor]
    post_names = ["Default", "Dynamic Threshold", "Moving2T", "SelfTuning"]

    preprocesor = Allpreprocesors[pre_names.index(params["pre-processing"])]
    postprocesor = Allpostprocesors[post_names.index(params["post-processing"])]


    param_space_dict_Pre=params["pre-processing_parameters"]
    param_space_dict_Post = params["post-processing_parameters"]

    return preprocesor,param_space_dict_Pre,postprocesor,param_space_dict_Post
def select_experiments(select_flavors,dataset_name):
    exp_to_return=[]
    exp_names_to_return=[]
    exp_automatic=[]

    flavors = ["Online", "Sliding", "Historical"]

    flavormethods = [online_technique, incremental_technique, semi_technique]
    experiments = [
        AutoProfileSemiSupervisedPdMExperiment,
        IncrementalSemiSupervisedPdMExperiment,
        SemiSupervisedPdMExperiment,
    ]

    experiment_names = [
        f"Online {dataset_name.upper()}",
        f"Sliding {dataset_name.upper()}",
        f"Historical {dataset_name.upper()}",
    ]
    for select_flavor in select_flavors:
        if select_flavor in flavors:
            exp_to_return.append(experiments[flavors.index(select_flavor)])
            exp_names_to_return.append(experiment_names[flavors.index(select_flavor)])
            exp_automatic.append(flavormethods[flavors.index(select_flavor)])
        else:
            print(f"Flavor {select_flavor} not exist")
    if len(exp_to_return)==0:
        assert False, "No valid flavor."
    return exp_to_return,exp_names_to_return,exp_automatic


if __name__ == "__main__":

    file_path = sys.argv[1]

    # Read the bytes from the file
    with open(file_path, 'rb') as file:
        input_bytes = file.read()

    # Deserialize the bytes back into a dictionary
    my_dict = pickle.loads(input_bytes)

    # Process the dictionary
    execute(my_dict)
