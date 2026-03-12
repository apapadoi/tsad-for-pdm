import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import wilcoxon


def load_data():
    df = pd.read_csv("data_analysis_runtime.csv", sep=",", header=0)
    df = df[~df['Technique'].isin(['ADD', 'ALL', 'XGBOOST', 'TimeLLMPyPots', 'TIMEMIXERPP', 'AutoGluon_chronos2', 'DummyAllAlarms'])]
    df = df[df["Flavor"] != "Semisupervised "]

    formal_flavor_name_map = {
        'Auto profile ': 'online',
        'Incremental ': 'sliding',
        'Semisupervised ': 'historical',
        'Unsupervised ': 'unsupervised'
    }

    df["Flavor"] = df["Flavor"].map(formal_flavor_name_map)
    df['Technique'] = df.apply(lambda row: 'KNN' if 'Distance' in row['Technique'] else row['Technique'], axis=1)

    df['Technique'] = df.apply(
        lambda row: row['Technique'].lower().replace('unsupervised', '').capitalize() if 'unsupervised' in row[
            'Technique'].lower() else row['Technique'], axis=1)

    df['Technique'] = df.apply(
        lambda row: row['Technique'].lower().replace('(uns)', '').capitalize() if '(uns)' in row['Technique'].lower() else
        row['Technique'], axis=1)

    df['Technique'] = df.apply(
        lambda row: row['Technique'].lower().replace('(semi)', '').capitalize() if '(semi)' in row['Technique'].lower() else
        row['Technique'], axis=1)

    df['Technique'] = df.apply(
        lambda row: row['Technique'].lower().replace('semi', '').capitalize() if 'semi' in row['Technique'].lower() else
        row['Technique'], axis=1)

    df['Technique'] = df.apply(lambda row: 'IsolationForest' if 'Isolation' in row['Technique'] else row['Technique'],
                            axis=1)

    df['Technique'] = df.apply(lambda row: 'LocalOutlierFactor' if 'Local' in row['Technique'] else row['Technique'],
                            axis=1)

    df = df.sort_values(by="Dataset", ascending=True)
    return df

def transform_name(name):
    if "Local" in name:
        name = "LOF"
    elif "Isolation" in name:
        name = "IF"
    elif "Distance" in name:
        name = "KNN"
    elif "NeighborProfile" in name:
        name = "NP"
    elif "OneClassSVM" in name:
        name = "OCSVM"
    elif "Chronos" in name:
        name = "Chronos"
    elif "Profile" in name:
        name = "PB"
    return name

df = load_data()
print(df.columns)

datasets = ['AZURE', 'BHD', 'CMAPSS', 'CNC', 'EDP', 'FEMTO', 'Formula 1', 'IMS', 'METRO', 'Navarchos', 'XJTU']
res={"ADD" : [0.070, 0.050, 0.160, 0.074, 0.670, 0.210, 0.248, 0.350, 0.290, 0.010, 0.110]}
resmedian={"ADD" : [0.070, 0.050, 0.160, 0.074, 0.670, 0.210, 0.248, 0.350, 0.290, 0.010, 0.110]}

for dataset in datasets:
    for method in df["Technique"].unique():
        subset = df[(df["Dataset"] == dataset) & (df["Technique"] == method)]
        if method == "ADD":
            continue
        else:
            maxperf=-1
            median_perf=-1
            name=""
            for flavor in subset["Flavor"].unique():
                subset_flavor = subset[subset["Flavor"] == flavor]
                perf = subset_flavor["AD1_AUC"].max()
                if perf > maxperf:
                    maxperf = perf
                    median_perf = subset_flavor["AD1_AUC"].median()

            if median_perf == -1 or maxperf == -1:
                raise ValueError('Max and median performance not found')
            name=method
            name=transform_name(name)
            if name not in res:
                res[name]=[]
                resmedian[name]=[]
            res[name].append(maxperf)
            resmedian[name].append(median_perf)

print(f"Technique & Median > ADD & Best > ADD & Median > ADD & Median < ADD & Best > ADD & Best < ADD \\\\")
for method in res:
    if method=="ADD":
        continue
    addacc=res["ADD"]
    methodaccM=resmedian[method]
    methodacc=res[method]
    resB = wilcoxon(methodacc, addacc, alternative='greater')
    resM = wilcoxon(methodaccM, addacc, alternative='greater')
    pv=resB[1]
    pvM=resM[1]

    wins=0
    for m,ad in zip(methodacc,addacc):
        if m>ad:
            wins+=1
    loses=len(methodacc)-wins

    winsM=0
    for m,ad in zip(methodaccM,addacc):
        if m>ad:
            winsM+=1
    losesM=len(methodaccM)-winsM

    singB="="
    if pv<0.05:
        singB="\\checkmark"
    elif wilcoxon(addacc,methodacc, alternative='greater')[1]<0.05:
        singB="\\cross"

    singM="="
    if pvM<0.05:
        singM="\\checkmark"
    elif wilcoxon(addacc, methodaccM, alternative='greater')[1] < 0.05:
        singM = "\\cross"

    print(f"{method} & {singM} & {singB} & {winsM} & {losesM} & {wins} & {loses} \\\\")