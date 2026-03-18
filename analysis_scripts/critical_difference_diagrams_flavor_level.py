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
    'Semisupervised ': '_his',
    'Unsupervised ': '_uns',
    'Classification ': ''
}
FONT_SIZE = 65

print(sys.argv)
df = pd.read_csv(f'data_analysis_runtime.csv')

df.drop_duplicates(inplace=True, ignore_index=True)

df['Technique'] = df.apply(lambda row: 'KNN' if 'Distance' in row['Technique'] else row['Technique'], axis=1)

df['Technique'] = df.apply(lambda row: row['Technique'].lower().replace('unsupervised', '').capitalize() if 'unsupervised' in row['Technique'].lower() else row['Technique'], axis=1)

df['Technique'] = df.apply(lambda row: row['Technique'].lower().replace('(uns)', '').capitalize() if '(uns)' in row['Technique'].lower() else row['Technique'], axis=1)

df['Technique'] = df.apply(lambda row: row['Technique'].lower().replace('(semi)', '').capitalize() if '(semi)' in row['Technique'].lower() else row['Technique'], axis=1)

df['Technique'] = df.apply(lambda row: row['Technique'].lower().replace('semi', '').capitalize() if 'semi' in row['Technique'].lower() else row['Technique'], axis=1)

df['Technique'] = df.apply(lambda row: 'IsolationForest' if 'Isolation' in row['Technique'] else row['Technique'], axis=1)

df['Technique'] = df.apply(lambda row: 'LocalOutlierFactor' if 'Local' in row['Technique'] else row['Technique'], axis=1)

# df.loc[len(df)] = ['Auto profile ', 'Chronos', 'CMAPSS', 259200, 0.200] # include chronos for online with cmapps ph 19

df = df[df['Flavor'] != 'Classification ']
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
    
METRIC = 'AD1_AUC' # 'AD1_AUC' or 'VUS_PR'

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

# from utils import loadDataset

# datasets = ['azure', 'bhd', 'cmapss', 'edp-wt', 'femto', 'formula1', 'ims', 'metropt-3', 'navarchos', 'xjtu']


# dataset_characteristics_df = pd.DataFrame({
#     "Dataset": ["CMAPSS", "FEMTO", "IMS", "EDP-WT", "METROPT-3", "NAVARCHOS", "XJTU-SY", "BHD 2023", "AZURE", "FORMULA 1"],
#     # "type": ["s", "e", "e", "r", "r", "r", "e", "r", "s", "s", "r"],
#     "#records": [265256, 21493, 9464, 209236, 1516948, 854178, 9216, 6626869, 876100, 5563727],
#     "Min scenario length": [19, 172, 984, 52244, 1516948, 1482, 42, 11, 8761, 4809],
#     "Avg scenario length": [187, 1264, 3154, 52309, 1516948, 32853, 614, 1529, 8761, 22344],
#     "Std scenario length": [82, 772, 2806, 50, 0, 46824, 853, 776, 0, 10595],
#     "Min dimensions": [14, 44, 88, 79, 15, 6, 44, 9, 4, 17],
#     "Max dimensions": [21, 44, 176, 79, 15, 6, 44, 32, 4, 17],
#     "failures": [709, 6, 3, 8, 4, 21, 15, 4334, 761, 249],
#     "scenarios with failure": [709, 6, 3, 4, 1, 13, 15, 4334, 100, 249],
#     "scenarios without failure": [707, 11, 0, 0, 0, 13, 0, 0, 0, 0],
#     "PH": [13, 52, 99, 8640, 17280, 10800, 18, 10, 96, 1920],
#     # "maximum PH": [42, 169, 324, 8640, 60480, 21600, 19, 20, 192, 4800],
#     "lead": [2, 2, 2, 288, 720, 720, 2, 2, 2, 960]
# })

# # Calculate channel lengths, total cells, and total columns for each dataset
# channel_lengths = {}
# total_cells = {}
# total_columns = {}
# for dataset in datasets:
#     try:
#         data = loadDataset.get_dataset(dataset)
#         target_data = data['target_data']
#         total_length = sum(len(df) for df in target_data)
#         # Calculate total number of cells (rows × columns)
#         total_cells_count = sum(len(df) * len(df.columns) for df in target_data)
#         # Calculate total number of columns across all dataframes
#         total_columns_count = sum(len(df.columns) for df in target_data)
#         channel_lengths[dataset] = total_length
#         total_cells[dataset] = total_cells_count
#         total_columns[dataset] = total_columns_count
#         print(f"Dataset {dataset}: {total_length} records, {total_cells_count} total cells, {total_columns_count} total columns")
#     except Exception as e:
#         print(f"Error loading dataset {dataset}: {e}")

# Map dataset names to match the dataframe
# dataset_name_mapping = {
#     'azure': 'AZURE',
#     'bhd': 'BHD 2023', 
#     'cmapss': 'CMAPSS',
#     'edp-wt': 'EDP-WT',
#     'femto': 'FEMTO',
#     'formula1': 'FORMULA 1',
#     'ims': 'IMS',
#     'metropt-3': 'METROPT-3',
#     'navarchos': 'NAVARCHOS',
#     'xjtu': 'XJTU-SY'
# }

# Create mapping for total cells and total columns from calculated values
# total_cells_mapping = {}
# total_columns_mapping = {}
# for dataset_key, mapped_name in dataset_name_mapping.items():
#     if dataset_key in total_cells:
#         total_cells_mapping[mapped_name] = total_cells[dataset_key]
#     if dataset_key in total_columns:
#         total_columns_mapping[mapped_name] = total_columns[dataset_key]

# Calculate min, median, max duration for each technique-flavor-dataset combination
metric_stats_df = df.groupby(['Technique', 'Flavor', 'Dataset'])[METRIC].agg(['min', 'median', 'max']).reset_index()

# Create the 2x2 subplot figure for CD diagrams
fig, axes = plt.subplots(2, 2, figsize=(8, 6))
axes = axes.flatten()

flavors_list = ['_onl', '_sli', '_uns', '_his']  # Classification flavor is empty string
flavor_names = ['Online', 'Sliding', 'Unsupervised', 'Historical']

for i, (flavor, flavor_name) in enumerate(zip(flavors_list, flavor_names)):
    ax = axes[i]
    
    # Filter data for this flavor
    flavor_data = metric_stats_df[metric_stats_df['Flavor'] == flavor]
    
    if flavor_data.empty:
        ax.text(0.5, 0.5, f'No data for {flavor_name}', ha='center', va='center', transform=ax.transAxes)
        ax.set_title(f'{flavor_name}')
        continue
    
    # Create summary with min, median, max as separate "datasets"
    summary_data = []
    
    for _, row in flavor_data.iterrows():
        technique = row['Technique']
        dataset = row['Dataset']
        
        # Add min, median, max as separate rows with dataset suffixes
        summary_data.append({
            'Dataset': f"{dataset}_min",
            'Technique': technique,
            METRIC: row['min']
        })
        summary_data.append({
            'Dataset': f"{dataset}_median", 
            'Technique': technique,
            METRIC: row['median']
        })
        summary_data.append({
            'Dataset': f"{dataset}_max",
            'Technique': technique,
            METRIC: row['max']
        })
    
    summary_df = pd.DataFrame(summary_data)
    
    # Create pivot table for autorank
    table = summary_df.pivot(index='Dataset', columns='Technique', values=METRIC)
    
    # Remove techniques with all NaN values
    table = table.dropna(axis=1, how='all')
    
    if table.empty or table.shape[1] < 2:
        ax.text(0.5, 0.5, f'Insufficient data for {flavor_name}', ha='center', va='center', transform=ax.transAxes)
        ax.set_title(f'{flavor_name}')
        continue
    
    try:
        # Perform autorank analysis (descending order since higher metric is better, assumming AUC/VUS)
        result = autorank(
            table, 
            alpha=0.05,
            verbose=False,
            order='descending',  # Higher metric is better for AUC/VUS
            force_mode="nonparametric",
            random_state=RANDOM_STATE
        )
        
        # Plot CD diagram
        plot_stats(result, ax=ax, allow_insignificant=True if METRIC == 'VUS_PR' else False)
        ax.set_title(f'{flavor_name}', fontsize=FONT_SIZE // 3.5)
        
    except Exception as e:
        ax.text(0.5, 0.5, f'Error in {flavor_name}: {str(e)}', ha='center', va='center', transform=ax.transAxes)
        ax.set_title(f'{flavor_name}')

plt.tight_layout()
plt.savefig(f'cd_diagrams_by_flavor_{METRIC}.pdf', format='pdf', bbox_inches='tight')
plt.savefig(f'cd_diagrams_by_flavor_{METRIC}.png', format='png', bbox_inches='tight', dpi=300)
plt.show()

print(f"Critical difference diagrams saved as 'cd_diagrams_by_flavor_{METRIC}.pdf' and 'cd_diagrams_by_flavor_{METRIC}.png'")

