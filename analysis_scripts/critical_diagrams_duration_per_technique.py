import sys
import pandas as pd
import numpy as np
from autorank import autorank

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
METRIC = 'Duration' 

formal_flavor_name_map = {
    'Auto profile ': '_onl',
    'Incremental ': '_sli',
    'Unsupervised ': '_uns',
    'Classification ': ''
}

print(sys.argv)
df = pd.read_csv(f'data_analysis_runtime.csv')

# Add dummy results
results_all_ad1_auc = [0.070, 0.010, 0.050, 0.023, 0.320, 0.040, 0.091, 0.030, 0.040, 0.040, 0.020]
results_all_vus_pr = [0.55, 0.50, 0.55, 0.526, 0.66, 0.54, 0.56, 0.53, 0.54, 0.52, 0.52]

results_add_ad1_auc = [0.070, 0.050, 0.160, 0.074, 0.670, 0.210, 0.248, 0.350, 0.290, 0.010, 0.110]
results_add_vus_pr =  [0.11, 0.01, 0.25, 0.123, 0.47, 0.11, 0.213, 0.16, 0.14, 0.03, 0.03]

datasets = ['AZURE', 'BHD', 'CMAPSS', 'CNC', 'EDP', 'FEMTO', 'FORMULA 1', 'IMS', 'METROPT-3', 'Navarchos', 'XJTU-SY']
for dataset, result_ad1, result_vus in zip(datasets, results_all_ad1_auc, results_all_vus_pr):
    df.loc[len(df)] = ['Unsupervised ', 'ALL', dataset, 1000, result_ad1, result_vus]

for dataset, result_ad1, result_vus in zip(datasets, results_add_ad1_auc, results_add_vus_pr):
    df.loc[len(df)] = ['Auto profile ', 'ADD', dataset, 1000, result_ad1, result_vus]

df.drop_duplicates(inplace=True, ignore_index=True)

# Clean technique names
df['Technique'] = df.apply(lambda row: 'KNN' if 'Distance' in row['Technique'] else row['Technique'], axis=1)

df['Technique'] = df.apply(lambda row: row['Technique'].lower().replace('unsupervised', '').capitalize() if 'unsupervised' in row['Technique'].lower() else row['Technique'], axis=1)

df['Technique'] = df.apply(lambda row: row['Technique'].lower().replace('(uns)', '').capitalize() if '(uns)' in row['Technique'].lower() else row['Technique'], axis=1)

df['Technique'] = df.apply(lambda row: row['Technique'].lower().replace('(semi)', '').capitalize() if '(semi)' in row['Technique'].lower() else row['Technique'], axis=1)

df['Technique'] = df.apply(lambda row: row['Technique'].lower().replace('semi', '').capitalize() if 'semi' in row['Technique'].lower() else row['Technique'], axis=1)

df['Technique'] = df.apply(lambda row: 'IsolationForest' if 'Isolation' in row['Technique'] else row['Technique'], axis=1)

df['Technique'] = df.apply(lambda row: 'LocalOutlierFactor' if 'Local' in row['Technique'] else row['Technique'], axis=1)

# Remove semisupervised
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

# Select the row with the highest AD1_AUC for each (Dataset, Technique, Flavor) combination
idx = df.groupby(['Dataset', 'Technique', 'Flavor'])['AD1_AUC'].idxmax()
summary = df.loc[idx, ['Dataset', 'Technique', 'Flavor', 'Duration']].reset_index(drop=True)

# Get all unique techniques
all_techniques = summary['Technique'].unique()

print(f"All techniques found: {all_techniques}")
print(f"Number of techniques to analyze: {len(all_techniques)}")

# Analyze each technique
for technique in all_techniques:
    technique_data = summary[summary['Technique'] == technique]
    
    if len(technique_data) == 0:
        continue
        
    # Check if we have multiple flavors for this technique
    unique_flavors = technique_data['Flavor'].nunique()
    if unique_flavors < 2:
        print(f"Skipping {technique}: only {unique_flavors} flavor(s) available")
        continue
        
    # Create pivot table with flavors as columns
    pivot_table = technique_data.pivot(index='Dataset', columns='Flavor', values=METRIC)
    
    # Remove columns with all NaN values
    pivot_table = pivot_table.dropna(axis=1, how='all')
    
    if pivot_table.shape[1] < 2:
        print(f"Skipping {technique}: not enough flavors with data")
        continue
        
    # Skip datasets with missing values
    pivot_table = pivot_table.dropna()
    
    if pivot_table.shape[0] < 3:
        print(f"Skipping {technique}: not enough datasets with complete data")
        continue
        
    print(f"\nProcessing {technique}:")
    print(f"  Datasets: {list(pivot_table.index)}")
    print(f"  Flavors: {list(pivot_table.columns)}")
    
    try:
        # Perform autorank analysis
        result = autorank(
            pivot_table, 
            alpha=0.05,
            verbose=False,
            order='ascending',
            force_mode="nonparametric",
            random_state=RANDOM_STATE
        )
        
        # Print the row with the lowest mean rank (best flavor)
        best_flavor = result.rankdf['meanrank'].idxmin()
        best_row = result.rankdf.loc[best_flavor]
        
        print(f"  Best flavor: {best_flavor}")
        print(f"  Mean rank: {best_row['meanrank']:.4f}")
        
    except Exception as e:
        print(f"Error processing {technique}: {str(e)}")
        continue

print("\nAnalysis complete!")
