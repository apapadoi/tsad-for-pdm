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
import scipy.stats as stats

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

# grouped = df.groupby(['Flavor', 'Technique', 'Dataset'])
METRIC = 'AD1_AUC'

df['rank_desc'] = (
    df.groupby(['Dataset'])[METRIC]
      .rank(method='dense', ascending=False)
)

df['is_top3'] = (df['rank_desc'] <= 3).astype(int)

# Drop duplicates for rows with same technique, flavor, dataset, and rank in top 3
# Keep only one row per combination of Technique, Flavor, Dataset when they have the same rank in top 3
df_deduplicated = df.copy()

# For rows in top 3, remove duplicates based on Technique, Flavor, Dataset, and rank
top3_mask = df_deduplicated['is_top3'] == 1
top3_df = df_deduplicated[top3_mask]
not_top3_df = df_deduplicated[~top3_mask]

# Remove duplicates in top 3 data based on Technique, Flavor, Dataset, and rank
top3_df_deduplicated = top3_df.drop_duplicates(subset=['Technique', 'Flavor', 'Dataset', 'rank_desc'], keep='first')

# Combine back the deduplicated top 3 with non-top 3 data
df_deduplicated = pd.concat([top3_df_deduplicated, not_top3_df], ignore_index=True)

df_deduplicated.drop(columns='rank_desc', inplace=True)

print(f"Original shape: {df.shape}")
print(f"After deduplication shape: {df_deduplicated.shape}")
print(f"Top 3 entries after deduplication: {df_deduplicated[df_deduplicated['is_top3'] == 1].shape}")

# Use the deduplicated dataframe for further analysis
df = df_deduplicated

# Keep only top 3 entries
df_top3 = df[df['is_top3'] == 1].copy()

print(f"Keeping only top 3 entries: {df_top3.shape}")

# Calculate percentage of online flavor in top 3 with 95% confidence interval

def calculate_online_top3_percentage_with_ci(df, confidence_level=0.95):
    """
    Calculate the percentage of times online flavor is in top 3
    and compute confidence interval using bootstrap method
    """
    # Filter for online flavor only
    online_df = df[df['Flavor'] == 'Auto profile ']
    
    if len(online_df) == 0:
        print("No online flavor data found!")
        return None
    
    # Since we're only looking at top 3, all entries are in top 3
    # Calculate the proportion of top 3 entries that are online flavor
    total_top3_entries = len(df)
    online_top3_count = len(online_df)
    proportion = online_top3_count / total_top3_entries
    percentage = proportion * 100
    
    # Bootstrap confidence interval calculation
    n_bootstrap = 10000
    bootstrap_proportions = []
    
    np.random.seed(RANDOM_STATE)  # For reproducibility
    
    # Create array of 1s for online entries and 0s for non-online entries
    is_online_array = (df['Flavor'] == 'Auto profile ').astype(int).values
    
    for _ in range(n_bootstrap):
        # Resample with replacement
        bootstrap_sample = np.random.choice(is_online_array, 
                                          size=total_top3_entries, 
                                          replace=True)
        bootstrap_proportion = np.mean(bootstrap_sample)
        bootstrap_proportions.append(bootstrap_proportion)
    
    # Calculate confidence interval
    alpha = 1 - confidence_level
    lower_percentile = (alpha/2) * 100
    upper_percentile = (1 - alpha/2) * 100
    
    ci_lower = np.percentile(bootstrap_proportions, lower_percentile) * 100
    ci_upper = np.percentile(bootstrap_proportions, upper_percentile) * 100
    
    # Alternative: Normal approximation confidence interval
    std_error = np.sqrt(proportion * (1 - proportion) / total_top3_entries)
    z_score = stats.norm.ppf(1 - alpha/2)
    ci_lower_normal = max(0, (proportion - z_score * std_error) * 100)
    ci_upper_normal = min(100, (proportion + z_score * std_error) * 100)
    
    return {
        'percentage': percentage,
        'ci_lower_bootstrap': ci_lower,
        'ci_upper_bootstrap': ci_upper,
        'ci_lower_normal': ci_lower_normal,
        'ci_upper_normal': ci_upper_normal,
        'online_top3_count': online_top3_count,
        'total_top3_entries': total_top3_entries,
        'confidence_level': confidence_level,
        'std_error': std_error
    }

# Calculate the results
results = calculate_online_top3_percentage_with_ci(df_top3, confidence_level=0.95)

if results:
    # Print results
    print("\n" + "="*80)
    print("ONLINE FLAVOR PERCENTAGE IN TOP-3 ENTRIES ANALYSIS")
    print("="*80)
    print(f"Metric used: {METRIC}")
    print(f"Total top-3 entries: {results['total_top3_entries']}")
    print(f"Online flavor entries in top-3: {results['online_top3_count']}")
    print(f"Percentage of top-3 that are online: {results['percentage']:.2f}%")
    print(f"95% Confidence Interval (Bootstrap): [{results['ci_lower_bootstrap']:.2f}%, {results['ci_upper_bootstrap']:.2f}%]")
    print(f"95% Confidence Interval (Normal): [{results['ci_lower_normal']:.2f}%, {results['ci_upper_normal']:.2f}%]")
    print(f"Standard Error: {results['std_error']:.4f}")

    # Detailed breakdown by dataset
    print("\n" + "="*80)
    print("DETAILED BREAKDOWN BY DATASET (TOP-3 ONLY)")
    print("="*80)
    
    dataset_breakdown = df_top3.groupby('Dataset').agg({
        'Flavor': lambda x: (x == 'Auto profile ').sum(),  # Count online entries
        'Dataset': 'count'  # Total entries per dataset
    }).rename(columns={'Flavor': 'Online_Count', 'Dataset': 'Total_Top3_Count'})
    
    dataset_breakdown['Online_Percentage'] = (dataset_breakdown['Online_Count'] / dataset_breakdown['Total_Top3_Count'] * 100).round(2)
    
    print(dataset_breakdown)
    
    # Flavor distribution in top 3
    print("\n" + "="*80)
    print("FLAVOR DISTRIBUTION IN TOP-3 ENTRIES")
    print("="*80)
    
    flavor_counts = df_top3['Flavor'].value_counts()
    flavor_percentages = (flavor_counts / len(df_top3) * 100).round(2)
    
    for flavor, count in flavor_counts.items():
        percentage = flavor_percentages[flavor]
        print(f"{flavor}: {count} entries ({percentage}%)")
    
    print(f"\nTotal top-3 entries analyzed: {len(df_top3)}")