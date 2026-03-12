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
FONT_SIZE = 120
METRIC = 'Duration' # 'Duration'

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
df = df[df['Technique'] != 'XGBOOST']
df = df[df['Technique'] != 'ALL']
df = df[df['Technique'] != 'ADD']


technique_to_best_flavor = {
    'CHRONOS': '_uns',     
    'IF': '_onl',          
    'KNN': '_uns',         
    'LOF': '_onl',         
    'LTSF': '_onl',        
    'NP': '_onl',          
    'OCSVM': '_onl',       
    'PB': '_onl',          
    'TRANAD': '_onl',       
    'USAD': '_onl',          
    'SAND': '_uns',
    'ALL': '_uns',
    'ADD': '_onl',
    'XGBOOST': ''
}

# Filter dataframe to keep only the best flavor for each technique
def get_best_flavor_for_technique(row):
    technique = row['Technique']
    flavor = row['Flavor']
    best_flavor = technique_to_best_flavor.get(technique, '')
    return flavor == best_flavor

# # Keep only rows with the best flavor for each technique
df = df[df.apply(get_best_flavor_for_technique, axis=1)].copy()


df['Technique'] = df['Technique'].astype(str) + df['Flavor'].astype(str)
df.drop(columns=['Flavor'], inplace=True)


# Create max, median, min, and negative std summaries
summary_max = df.groupby(['Dataset', 'Technique'])[METRIC].max().reset_index()
summary_median = df.groupby(['Dataset', 'Technique'])[METRIC].median().reset_index()
summary_min = df.groupby(['Dataset', 'Technique'])[METRIC].min().reset_index()
# summary_neg_std = df.groupby(['Dataset', 'Technique'])[METRIC].std().reset_index()

# Make std negative for summary_neg_std
# summary_neg_std[METRIC] = -summary_neg_std[METRIC]

# Add suffix to distinguish different summary types
summary_max['Dataset'] = summary_max['Dataset'] + '_max'
summary_median['Dataset'] = summary_median['Dataset'] + '_median'
summary_min['Dataset'] = summary_min['Dataset'] + '_min'
# summary_neg_std['Dataset'] = summary_neg_std['Dataset'] + '_neg_std'

# Combine all summaries
summary = pd.concat([summary_max, summary_median, summary_min], ignore_index=True)
print(summary['Dataset'].unique().tolist())

table = summary.pivot(index='Dataset', columns='Technique', values=METRIC)

result = autorank(
    table, 
    alpha=0.05,
    verbose=False,
    order='ascending',
    force_mode="nonparametric",
    random_state=RANDOM_STATE
)

create_report(result)

plt.rcParams.update({'font.size': FONT_SIZE // 4})
fig, ax = plt.subplots(figsize=(10.5, 6.5))
plot_stats(result, ax=ax)
# ax.tick_params(labelsize=2 * FONT_SIZE)
# for label in ax.get_xticklabels() + ax.get_yticklabels():
#     label.set_fontsize(2 * FONT_SIZE)
plt.tight_layout()
plt.savefig(f'cd_technique_flavor_combinations_{METRIC}.pdf', format='pdf', bbox_inches='tight')
# plt.savefig(f'cd_technique_flavor_combinations_{METRIC}.jpeg', format='jpeg', bbox_inches='tight')
