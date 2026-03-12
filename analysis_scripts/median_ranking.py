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

df = df[~df['Technique'].isin(['ADD', 'ALL', 'XGBOOST', 'TimeLLMPyPots', 'TIMEMIXERPP', 'AutoGluon_chronos2', 'DummyAllAlarms'])]
df = df[df['Flavor'] != 'Semisupervised ']
df = df[~df['Flavor'].str.contains('dversarial')]

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

# First, get the median performance per dataset for each technique-flavor combination
median_per_dataset = df.groupby(['Technique', 'Flavor', 'Dataset'])['AD1_AUC'].median().reset_index()
median_per_dataset.columns = ['Technique', 'Flavor', 'Dataset', 'Median_AD1_AUC']

# Then calculate median of medians across datasets for each technique-flavor combination
median_results = median_per_dataset.groupby(['Technique', 'Flavor'])['Median_AD1_AUC'].median().reset_index()
median_results.columns = ['Technique', 'Flavor', 'Median_of_Medians_AD1_AUC']

# Sort by flavor and then by median performance (descending)
median_results = median_results.sort_values(['Flavor', 'Median_of_Medians_AD1_AUC'], ascending=[True, False])

print("Median of median AD1_AUC values for each technique-flavor combination across datasets:")
print("=" * 80)

# Group by flavor and generate LaTeX tables
print("\\documentclass{article}")
print("\\usepackage{booktabs}")
print("\\usepackage{array}")
print("\\begin{document}")

for flavor in median_results['Flavor'].unique():
    flavor_data = median_results[median_results['Flavor'] == flavor]
    formal_name = formal_flavor_name_map.get(flavor, flavor.strip())
    
    print(f"\n% Table for {flavor.strip()} ({formal_name})")
    print("\\begin{table}[H]")
    print("\\centering")
    print(f"\\caption{{Median of median AD1 AUC-PR for the {formal_name} flavor}}")
    print(f"\\label{{tab:{formal_name}_ranking}}")
    print("\\begin{tabular}{lr}")
    print("\\toprule")
    print("Technique & Median AD1 AUC-PR \\\\")
    print("\\midrule")
    
    for idx, row in flavor_data.iterrows():
        technique_name = row['Technique'].replace('_', '\\_')  # Escape underscores for LaTeX
        print(f"{technique_name} & {row['Median_of_Medians_AD1_AUC']:.6f} \\\\")
    
    print("\\bottomrule")
    print("\\end{tabular}")
    print("\\end{table}")

# Generate overall ranking table
print(f"\n% Overall ranking table")
print("\\begin{table}[H]")
print("\\centering")
print("\\caption{Overall ranking by median of median AD1 AUC-PR (excluding the historical flavor)}")
print("\\label{tab:overall_ranking}")
print("\\begin{tabular}{llr}")
print("\\toprule")
print("Technique & Flavor & Median AD1 AUC-PR \\\\")
print("\\midrule")

overall_sorted = median_results.sort_values('Median_of_Medians_AD1_AUC', ascending=False)
for idx, row in overall_sorted.iterrows():
    formal_name = formal_flavor_name_map.get(row['Flavor'], row['Flavor'].strip())
    technique_name = row['Technique'].replace('_', '\\_')  # Escape underscores for LaTeX
    print(f"{technique_name} & {formal_name} & {row['Median_of_Medians_AD1_AUC']:.6f} \\\\")

print("\\bottomrule")
print("\\end{tabular}")
print("\\end{table}")
print("\\end{document}")

# Also print a simple text version for reference
print(f"\n\n% Text version for reference:")
print("=" * 80)
for flavor in median_results['Flavor'].unique():
    flavor_data = median_results[median_results['Flavor'] == flavor]
    formal_name = formal_flavor_name_map.get(flavor, flavor.strip())
    
    print(f"\n{flavor.strip()} ({formal_name}):")
    print("-" * 50)
    for idx, row in flavor_data.iterrows():
        print(f"{row['Technique']:25s}: {row['Median_of_Medians_AD1_AUC']:.6f}")

