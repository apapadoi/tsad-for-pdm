# Performance distribution across flavor

import mlflow
import sys
import random
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib.patches import Patch
import matplotlib as mpl

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
    'Auto profile ': 'online',
    'Incremental ': 'sliding',
    'Semisupervised ': 'historical',
    'Unsupervised ': 'unsupervised'
}

FONT_SIZE = 65

plt.rcParams.update({
    'font.size': 1.5*FONT_SIZE,
    'axes.titlesize': 2.5*FONT_SIZE,
    'axes.labelsize': 1.5*FONT_SIZE,
    'xtick.labelsize': 1.5*FONT_SIZE,
    'ytick.labelsize': 1.5*FONT_SIZE,
    'legend.fontsize': 1.5*FONT_SIZE
})

print(sys.argv)
df = pd.read_csv(f'data_analysis_runtime.csv')

df.drop_duplicates(inplace=True, ignore_index=True)

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

df = df[~df['Technique'].isin(['ADD', 'ALL', 'XGBOOST', 'TimeLLMPyPots', 'TIMEMIXERPP', 'AutoGluon_chronos2', 'DummyAllAlarms'])]
df = df[df['Flavor'] != 'Classification ']
df = df[~df['Flavor'].str.contains('dversarial')]

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

# df.loc[len(df)] = ['Auto profile ', 'Chronos', 'CMAPSS', 259200, 0.200]

df['Technique'] = df['Technique'].map({
    'NeighborProfile': 'NP',
    'IsolationForest': 'IF',
    'KNN': 'KNN',
    'LocalOutlierFactor': 'LOF',
    'OneClassSVM': 'OCSVM',
    'ProfileBased': 'PB',
    'TranAD': 'TRANAD',
    'USAD': 'USAD',
    'LTSF': 'LTSF',
    'Sand': 'SAND',
    'Chronos': 'CHRONOS'
}).astype(str)

all_techniques = sorted(df['Technique'].unique())
all_datasets = sorted(df['Dataset'].unique())

fig, axes = plt.subplots(2, 2, figsize=(120, 120))
axes = axes.flatten()

for ax, flavor in zip(axes, flavors):
    sub = df[df['Flavor'] == flavor]

    metric = sub.pivot_table(index='Technique', columns='Dataset', values='AD1_AUC', aggfunc='max')

    metric = metric.reindex(index=all_techniques, columns=all_datasets)

    sns.heatmap(metric, ax=ax, linewidths=20, cmap='Blues', cbar=True, annot=True, fmt=".2f", square=True, cbar_kws={'label': 'AD1 AUC PR'})
    ax.set_title(f"{formal_flavor_name_map[flavor].capitalize()} flavor", fontsize=2.5*FONT_SIZE, pad=100)


plt.tight_layout()
plt.show()

fig.savefig("grouped_heatmap.pdf", format='pdf', bbox_inches='tight')
