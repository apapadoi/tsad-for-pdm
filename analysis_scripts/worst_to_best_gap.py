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
METRIC = 'VUS_PR'  # 'AD1_AUC' or 'VUS_PR'

random.seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)
mpl.rcParams['svg.hashsalt'] = str(RANDOM_STATE)

flavors = [
    "Auto profile ",
    "Incremental ",
    "Semisupervised ",
    "Unsupervised ",
    "Classification "
]
formal_flavor_name_map = {
    'Auto profile ': 'online',
    'Incremental ': 'sliding',
    'Semisupervised ': 'historical',
    'Unsupervised ': 'unsupervised',
    'Classification ': 'classification'
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

df = df[~df['Technique'].isin(['ADD', 'ALL', 'TimeLLMPyPots', 'TIMEMIXERPP', 'AutoGluon_chronos2', 'DummyAllAlarms'])]

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
    'Chronos': 'CHRONOS',
    'XGBOOST': 'XGBOOST',
}).astype(str)

# df.loc[len(df)] = ['Auto profile ', 'Chronos', 'CMAPSS', 259200, 0.200] # include chronos for online with cmapps ph 19


# Calculate oracle performance (maximum) for each dataset
oracle_performance = df.groupby('Dataset')[METRIC].max().reset_index()
oracle_performance.columns = ['Dataset', 'Oracle_AUC']
print("Oracle performance per dataset:")
print(oracle_performance)
print()

# Merge oracle performance back to the main dataframe
df = df.merge(oracle_performance, on='Dataset', how='left')

# Calculate worst performance for each technique-flavor-dataset combination
worst_performance = df.groupby(['Technique', 'Flavor', 'Dataset'])[METRIC].min().reset_index()
worst_performance.columns = ['Technique', 'Flavor', 'Dataset', 'Worst_AUC']

# Merge oracle performance with worst performance
worst_with_oracle = worst_performance.merge(oracle_performance, on='Dataset', how='left')

# Calculate gap to oracle (oracle - worst performance)
worst_with_oracle['Gap_to_Oracle'] = worst_with_oracle['Oracle_AUC'] - worst_with_oracle['Worst_AUC']

print("Gap analysis for each technique-flavor-dataset combination:")
print(worst_with_oracle.sort_values(['Flavor', 'Gap_to_Oracle']))
print()

# For each flavor, calculate average gap per technique and sort by increasing gap
flavor_technique_gaps = worst_with_oracle.groupby(['Flavor', 'Technique'])['Gap_to_Oracle'].mean().reset_index()

print("Techniques ranked by average gap to oracle (best to worst) for each flavor:")
print("=" * 80)

# Generate LaTeX tables
print("\\documentclass{article}")
print("\\usepackage{booktabs}")
print("\\usepackage{array}")
print("\\begin{document}")

for flavor in flavors:
    # Skip Classification flavor if it exists
    # if 'Classification' in flavor:
    #     continue
        
    flavor_data = flavor_technique_gaps[flavor_technique_gaps['Flavor'] == flavor]
    flavor_data_sorted = flavor_data.sort_values('Gap_to_Oracle')
    
    if len(flavor_data_sorted) > 0:  # Only create table if there's data
        formal_name = formal_flavor_name_map.get(flavor, flavor.strip())
        
        print(f"\n% Table for {flavor.strip()} ({formal_name})")
        print("\\begin{table}[htbp]")
        print("\\centering")
        print(f"\\caption{{Average worst-case gap to ORACLE across all datasets for the {formal_name.title()} flavor under {METRIC}.}}")
        print(f"\\label{{tab:{formal_name}_gap_oracle}}")
        print("\\begin{tabular}{lr}")
        print("\\toprule")
        print("Technique & Average worst-case gap to ORACLE \\\\")
        print("\\midrule")
        
        for idx, row in flavor_data_sorted.iterrows():
            technique_name = row['Technique'].replace('_', '\\_')  # Escape underscores for LaTeX
            print(f"{technique_name} & {row['Gap_to_Oracle']:.6f} \\\\")
        
        print("\\bottomrule")
        print("\\end{tabular}")
        print("\\end{table}")

print("\\end{document}")

# Also print text version for reference
print(f"\n\n% Text version for reference:")
for flavor in flavors:
    # Skip Classification flavor if it exists
    if 'Classification' in flavor:
        continue
        
    flavor_data = flavor_technique_gaps[flavor_technique_gaps['Flavor'] == flavor]
    flavor_data_sorted = flavor_data.sort_values('Gap_to_Oracle')
    
    print(f"\n{flavor.strip()} ({formal_flavor_name_map.get(flavor, flavor.strip())}):")
    print("-" * 50)
    for idx, row in flavor_data_sorted.iterrows():
        print(f"{row['Technique']:20s}: {row['Gap_to_Oracle']:.6f}")

# Also create a summary table showing worst gaps per flavor
# print("\n" + "=" * 80)
# print("Summary: Average gap to oracle by flavor:")
# print("=" * 80)
# flavor_avg_gaps = worst_with_oracle.groupby('Flavor')['Gap_to_Oracle'].mean().sort_values()
# for flavor, avg_gap in flavor_avg_gaps.items():
#     formal_name = formal_flavor_name_map.get(flavor, flavor)
#     print(f"{flavor.strip():15s} ({formal_name:12s}): {avg_gap:.6f}")

