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

import mlflow
import sys


def print_latex_table(times_dict,dataset_name):
    flavors = ["Auto profile", "Incremental", "Semisupervised", "Unsupervised"]
    technques = ["PB", "KNN", "IF", "LOF", "NP", "SAND", "OCSVM", "DISTANCE BASED", "LTSF", "TRANAD", "USAD","CHRONOS",]
    # technques = ["Dummy Increase", "Dummy All"]#, "IF", "LOF", "NP", "SAND", "OCSVM", "DISTANCE BASED", "LTSF", "TRANAD", "USAD","CHRONOS",]
    timetable = [["" for f in flavors] for tech in technques]

    for key in times_dict.keys():
        posi = -1
        for i in range(len(flavors)):
            if flavors[i] in key:
                posi = i
                break
        posj = -1
        for j in range(len(technques)):
            if technques[j] in key:
                posj = j
                break
        if posi == -1 or posj == -1:
            print(f"Unknown {key}")
        timetable[posj][posi] = times_dict[key]

    avg = [0 for f in flavors]
    avg_count = [0 for f in flavors]
    for i in range(len(technques)):
        str_temp = ""
        countj = -1
        for time_teck in timetable[i]:
            countj += 1
            try:
                str_temp += f"&{round(float(time_teck), 3)}"
                avg[countj] += round(float(time_teck), 3)
                avg_count[countj] += 1
            except:
                str_temp += f"&{time_teck}"
        print(f"{dataset_name}&{technques[i].lower()}{str_temp}")  # \\\\")
    str_temp = ""
    for avg_f, a_c in zip(avg, avg_count):
        if a_c != 0:
            str_temp += f"&{round(avg_f / a_c, 3)}"
        else:
            str_temp += f"&"
    print(f"Avg.{str_temp}")  # \\\\")
    # print(sum_duration*100*2/60)


def get_max_of_metric(metric="duration", dataset_name="CMAPSS", rules=[]):  # [("params.postprocessor","Default")]):
    # Connect to your MLflow tracking server
    experiments1 = get_experiments_from(dataset_name, urll="http://127.0.0.1:8080/")

    # Get a list of experiments
    times_dict = {}
    sum_duration = 0

        #print(f"Experiment '{experiment.name}' - Maximum {metric}: {max_value}")
    times_dict=search_runs(experiments1, "http://127.0.0.1:8080/", rules, metric, times_dict)


    print_latex_table(times_dict,dataset_name)

    if metric == "duration":
        print(f" total time: {sum_duration}")


def search_runs(experiments,urll,rules,metric,times_dict):
    mlflow.set_tracking_uri(urll)
    for experiment in experiments:
        if "correlated" in experiment.name:
            continue
        exp_id = experiment.experiment_id
        #print(exp_id)
        if len(sys.argv) == 1:
            runs = mlflow.search_runs(exp_id)
        else:
            runs = mlflow.search_runs(exp_id, filter_string=f'attributes.created > {sys.argv[1]}')

        max_value = None

        for index, run in runs.iterrows():
            skip = False
            for rule in rules:
                # print(experiment.name)
                # print(run.keys())
                if rule[0] not in run.keys():
                    skip = True
                    break
                if run[rule[0]] is None:
                    skip = True
                    break
                elif rule[1] in str(run[rule[0]]):
                    continue
                else:
                    skip = True
                    break
            if skip:
                continue
            if metric == "duration":
                duration = run["end_time"] - run[
                    'start_time']  # Replace 'your_field' with the field you want to analyze
                duration = duration.seconds
                value = duration
            else:
                if metric not in run.keys():
                    print(f"Not such metric, availiable: {run.keys()}")
                    continue
                value = run[metric]
            if max_value is None:
                max_value = value
            elif value > max_value:
                max_value = value
        times_dict[experiment.name] = max_value
        # if metric == "duration" and max_value is not None:
        #     sum_duration += max_value
    return times_dict


def get_experiments_from(dataset_name,urll):
    mlflow.set_tracking_uri(urll)

    # Get a list of experiments
    client = mlflow.tracking.MlflowClient()
    experiments = client.search_experiments(filter_string=f"name LIKE '%{dataset_name}%'")
    return experiments


def get_max_of_metric_multi_Dataset(metric="duration", dataset_name="TSB 500000 Auto profile ",
                                    rules=[],urll="http://127.0.0.1:8080/",urll2="http://127.0.0.1:8080/"):  # [("params.postprocessor","Default")]):

    experiments1 = get_experiments_from(dataset_name, urll=urll)
    print(len(experiments1))
    experiments=[exp for exp in experiments1]
    if urll2 !=urll:
        experiments2 = get_experiments_from(dataset_name, urll=urll2)
        print(len(experiments2))
        experiments.extend([exp for exp in experiments2])
    # Get a list of experiments
    times_dict = {}
    sum_duration = 0
    for experiment in experiments:
        exp_id = experiment.experiment_id
        if len(sys.argv) == 1:
            runs = mlflow.search_runs(exp_id)
        else:
            runs = mlflow.search_runs(exp_id, filter_string=f'attributes.created > {sys.argv[1]}')

        max_value = None
        show_message = True
        for index, run in runs.iterrows():
            skip = False
            for rule in rules:
                # print(experiment.name)
                # print(run.keys())
                if rule[0] not in run.keys():
                    skip = True
                    break
                if run[rule[0]] is None:
                    skip = True
                    break
                elif rule[1] in str(run[rule[0]]):
                    continue
                else:
                    skip = True
                    break
            if skip:
                continue
            if metric not in run.keys():
                if show_message:
                    print(f"{experiment.name} Not such metric, availiable: {run.keys()}")
                    show_message = False
                continue

            value = run[metric]
            if max_value is None:
                max_value = value
            elif value > max_value:
                max_value = value
        times_dict[experiment.name] = max_value
        if metric == "duration" and max_value is not None:
            sum_duration += max_value
        #print(f"Experiment '{experiment.name}' - Maximum {metric}: {max_value}")

get_max_of_metric(metric="metrics.AD2_f1",dataset_name="EDP",rules=[("params.postprocessor","Default")])
# get_max_of_metric(metric="metrics.AD2_f1",dataset_name="CMAPSS",rules=[("params.postprocessor","Default")])
# get_max_of_metric(metric="metrics.AD2_f1",dataset_name="METRO",rules=[("params.postprocessor","Default")])
# get_max_of_metric(metric="metrics.AD2_f1",dataset_name="NAVARCHOS",rules=[("params.postprocessor","Default")])
# get_max_of_metric(metric="metrics.AD2_f1",dataset_name="IMS",rules=[("params.postprocessor","Default")])
# get_max_of_metric(metric="metrics.AD2_f1",dataset_name="FEMTO",rules=[("params.postprocessor","Default")])
# get_max_of_metric(metric="metrics.AD2_f1",dataset_name="BHD",rules=[("params.postprocessor","Default")])
# get_max_of_metric(metric="metrics.AD2_f1",dataset_name="AZURE",rules=[("params.postprocessor","Default")])
# get_max_of_metric(metric="metrics.AD2_f1",dataset_name="XJTU",rules=[("params.postprocessor","Default")])
# get_max_of_metric(metric="metrics.AD2_f1",dataset_name="AI4I",rules=[("params.postprocessor","Default")])


#get_max_of_metric(metric="metrics.AD1_AUC",dataset_name="CMAPSS",rules=[("params.postprocessor","MinMax")])
# print("================================================")
#get_max_of_metric(metric="metrics.AD1_AUC",dataset_name="CMAPSS",rules=[("params.postprocessor","Default")])



# print("================================================")
# get_max_of_metric(metric="metrics.AD1_AUC",dataset_name="Unsupervised EDP",rules=[("params.postprocessor","MinMax")])