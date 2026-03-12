import pandas as pd
import itertools
import networkx as nx
import numpy as np
import choix
import matplotlib.pyplot as plt

def compare_rows(df):
    results = []
    # Iterate all unique row-pairs (i, j) with i < j
    for i, j in itertools.combinations(df.index, 2):
        # Subset to the two rows
        a, b = df.loc[i], df.loc[j]
        # Compare elementwise across all columns
        for col in df.columns:
            if col == 'Dataset':
                continue

            if a[col] > b[col]:
                results.append((i, j))
            elif a[col] < b[col]:
                results.append((j, i))
            # if equal, do nothing
    return results

import mlflow
import sys
import random
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib.patches import Patch
import matplotlib as mpl
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, KFold
from sklearn.metrics import make_scorer, f1_score
from mango.tuner import Tuner
from mango import scheduler
import os
from autorank import autorank, plot_stats, create_report


RANDOM_STATE = 42
METRIC = 'AD1_AUC'

random.seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)
mpl.rcParams['svg.hashsalt'] = str(RANDOM_STATE)

flavors = [
    "Auto profile ",
    "Incremental ",
    "Semisupervised ",
    "Unsupervised "

]
formal_flavor_name_map = {
    'Auto profile ': '_onl',
    'Incremental ': '_sli',
    # 'Semisupervised ': 'historical',
    'Unsupervised ': '_uns',
    'Classification ': ''
}
# FONT_SIZE = 65

print(sys.argv)
df = pd.read_csv(f'data_analysis_runtime.csv')


results_all_ad1_auc = [0.070, 0.010, 0.050, 0.023, 0.320, 0.040, 0.091, 0.030, 0.040, 0.040, 0.020]
results_all_vus_pr = [0.55, 0.50, 0.55, 0.526, 0.66, 0.54, 0.56, 0.53, 0.54, 0.52, 0.52]

results_add_ad1_auc = [0.070, 0.050, 0.160, 0.074, 0.670, 0.210, 0.248, 0.350, 0.290, 0.010, 0.110]
results_add_vus_pr =  [0.11, 0.01, 0.25, 0.123, 0.47, 0.11, 0.213, 0.16, 0.14, 0.03, 0.03]

# add dummy results
datasets = ['AZURE', 'BHD', 'CMAPSS', 'CNC', 'EDP', 'FEMTO', 'FORMULA 1', 'IMS', 'METROPT-3', 'Navarchos', 'XJTU-SY']
for dataset, result_ad1, result_vus in zip(datasets, results_all_ad1_auc, results_all_vus_pr):
    df.loc[len(df)] = ['Unsupervised ', 'ALL', dataset, 1000, result_ad1, result_vus]

for dataset, result_ad1, result_vus in zip(datasets, results_add_ad1_auc, results_add_vus_pr):
    df.loc[len(df)] = ['Auto profile ', 'ADD', dataset, 1000, result_ad1, result_vus]

df.drop_duplicates(inplace=True, ignore_index=True)

df['Technique'] = df.apply(lambda row: 'KNN' if 'Distance' in row['Technique'] else row['Technique'], axis=1)

df['Technique'] = df.apply(lambda row: row['Technique'].lower().replace('unsupervised', '').capitalize() if 'unsupervised' in row['Technique'].lower() else row['Technique'], axis=1)

df['Technique'] = df.apply(lambda row: row['Technique'].lower().replace('(uns)', '').capitalize() if '(uns)' in row['Technique'].lower() else row['Technique'], axis=1)

df['Technique'] = df.apply(lambda row: row['Technique'].lower().replace('(semi)', '').capitalize() if '(semi)' in row['Technique'].lower() else row['Technique'], axis=1)

df['Technique'] = df.apply(lambda row: row['Technique'].lower().replace('semi', '').capitalize() if 'semi' in row['Technique'].lower() else row['Technique'], axis=1)

df['Technique'] = df.apply(lambda row: 'IsolationForest' if 'Isolation' in row['Technique'] else row['Technique'], axis=1)

df['Technique'] = df.apply(lambda row: 'LocalOutlierFactor' if 'Local' in row['Technique'] else row['Technique'], axis=1)

# df.loc[len(df)] = ['Auto profile ', 'Chronos', 'CMAPSS', 259200, 0.200] # include chronos for online with cmapps ph 19

df = df[df['Flavor'] != 'Semisupervised ']
df['Flavor'] = df['Flavor'].map(formal_flavor_name_map)

technique_mapping_dict = {
    'ChronosUnsupervised': 'CHRONOS',
    'TranAD': 'TRANAD',
    'Sand': 'SAND',
    'LocalOutlierFactorUnsupervised': 'LOF',
    'IsolationForestUnsupervised': 'IF',
    'Distance Based k (uns)': 'KNN',
    'NeighborProfile': 'NP',
    'LocalOutlierFactor': 'LOF',
    'Distance Based k (semi)': 'KNN',
    'ProfileBased': 'PB',
    'OneClassSVM': 'OCSVM',
    'IsolationForest': 'IF',
    'ChronosSemi': 'CHRONOS',
    'XGBOOST': 'XGBOOST',
    'KNN': 'KNN',
    'LTSF': 'LTSF',
    'USAD': 'USAD',
    'Chronos': 'CHRONOS',
    'ADD': 'ADD',
    'ALL': 'ALL'
}

df['Technique'] = df['Technique'].map(technique_mapping_dict)



dataset_mapping_dict = {
    'AZURE': 'AZURE',
    'BHD': 'BHD 2023',
    'CMAPSS': 'CMAPSS',
    'CNC': 'CNC',
    'EDP': 'EDP-WT',
    'FEMTO': 'FEMTO',
    'Formula 1': 'FORMULA 1',
    'IMS': 'IMS',
    'METRO': 'METROPT-3',
    'Navarchos': 'NAVARCHOS',
    'XJTU': 'XJTU-SY'
}
df['Dataset'] = df['Dataset'].map(dataset_mapping_dict)

df = df[~df['Technique'].isin(['ADD', 'ALL', 'XGBOOST', 'TimeLLMPyPots', 'TIMEMIXERPP', 'AutoGluon_chronos2', 'DummyAllAlarms'])]

# Transform df to have techniques as rows and datasets as columns
# Get the best METRIC performance for each technique-dataset combination (regardless of flavor)
best_performance = df.groupby(['Technique', 'Dataset'])[METRIC].median().reset_index()

# Pivot to create the desired structure: techniques as rows, datasets as columns
performance_matrix = best_performance.pivot(index='Technique', columns='Dataset', values=METRIC)

print(performance_matrix)

# Create mapping from indices to dataset names before transposing
dataset_names = performance_matrix.columns.tolist()
index_to_dataset_mapping = {i: dataset for i, dataset in enumerate(dataset_names)}

performance_matrix = performance_matrix.T.reset_index(drop=True)

print("Performance matrix (Techniques x Datasets):")
print(performance_matrix)
print("\nMatrix shape:", performance_matrix.shape)
print("Index to dataset mapping:", index_to_dataset_mapping)

results = compare_rows(performance_matrix)

n_items = len(dataset_names)

print(results)

params = choix.ilsr_pairwise(n_items, results)
print(params)
ranking_indices = np.argsort(params)
print("ranking (worst to best):", ranking_indices)

# Map indices to dataset names
ranking_names = [index_to_dataset_mapping[idx] for idx in ranking_indices]
print("Dataset ranking (worst to best):", ranking_names)

# Create visualization
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

# Plot 1: Ranking indices
ax1.barh(range(len(ranking_indices)), params[ranking_indices])
ax1.set_yticks(range(len(ranking_indices)))
ax1.set_yticklabels([str(idx) for idx in ranking_indices])
ax1.set_xlabel('Ranking Score')
ax1.set_ylabel('Dataset Index')
ax1.set_title('Dataset Ranking by Index (worst to best)')
ax1.grid(axis='x', alpha=0.3)

# Plot 2: Dataset names
ax2.barh(range(len(ranking_names)), params[ranking_indices])
ax2.set_yticks(range(len(ranking_names)))
ax2.set_yticklabels(ranking_names)
ax2.set_xlabel('Ranking Score')
ax2.set_ylabel('Dataset Name')
ax2.set_title('Dataset Ranking by Name (worst to best)')
ax2.grid(axis='x', alpha=0.3)

plt.tight_layout()
# plt.savefig('dataset_ranking.pdf', format='pdf', bbox_inches='tight')
# plt.savefig('dataset_ranking.png', format='png', bbox_inches='tight', dpi=300)
plt.show()

print("Dataset ranking plot saved as 'dataset_ranking.pdf' and 'dataset_ranking.png'")
