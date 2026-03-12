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

grouped = df.groupby(['Flavor', 'Technique', 'Dataset'])

# Get all unique groups
group_keys = list(grouped.groups.keys())

print(f"Creating {len(group_keys) * 2} individual PDF files...")

# Create individual plots for each group and metric
for group_key in group_keys:
    group_data = grouped.get_group(group_key)
    flavor, technique, dataset = group_key

    # Clean names for filenames (remove spaces, special characters)
    clean_flavor = str(flavor).replace(' ', '_').replace('/', '_')
    clean_technique = str(technique).replace(' ', '_').replace('/', '_')
    clean_dataset = str(dataset).replace(' ', '_').replace('/', '_')

    # Create AD1_AUC plot
    fig, ax = plt.subplots(figsize=(3, 0.75))  # Compact size

    # Get data range for limiting green line
    data_min = group_data['AD1_AUC'].min()
    data_max = group_data['AD1_AUC'].max()

    # Create box plot
    box = ax.boxplot(
                        group_data['AD1_AUC'], vert=False, patch_artist=True, widths=0.65,
                        boxprops=dict(facecolor='white', edgecolor='black'),
                        medianprops=dict(color='orange', linewidth=1.5),
                        whiskerprops=dict(color='black'),
                        capprops=dict(color='black'),
                        flierprops=dict(marker='o', markersize=10)# markerfacecolor='red', markersize=0.5),
                     )

    mean_val = group_data['AD1_AUC'].mean()
    ax.axvline(mean_val, color='green', linewidth=1.5, ymin=0.0585, ymax=0.94)

    # Add mean line - limited to data range
    # mean_val = group_data['AD1_AUC'].mean()
    # ax.axvline(mean_val, color='green', linewidth=3,
    #            ymin=0.25, ymax=0.75)  # Only draw line within box area

    ax.spines['bottom'].set_visible(False)

    # Minimize whitespace
    ax.set_yticks([])
    ax.set_xlabel('')
    # Set x-axis ticks, show x-axis, and increase font size for XGBOOST technique
    if technique == 'XGBOOST':
        ax.set_xticks([0.0, 0.5, 1.0])
        ax.set_xticklabels(['0.0', '0.5', '1.0'], fontsize=FONT_SIZE // 3)
        ax.spines['bottom'].set_visible(True)
    else:
        ax.set_xticks([])
        ax.spines['bottom'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)

    # Set tight limits
    # data_range = data_max - data_min
    # margin = data_range * 0.05  # 5% margin
    # ax.set_xlim(data_min - margin, data_max + margin)
    ax.set_xlim(0, 1)
    ax.set_ylim(0.65, 1.35)

    # Remove all padding
    plt.subplots_adjust(left=0.1, right=0.95, top=0.85, bottom=0.25)

    # Save AD1_AUC plot
    filename_ad1 = f"{clean_flavor}_{clean_technique}_{clean_dataset}_AD1_AUC.pdf"
    filename_ad1 = filename_ad1.lower()
    filepath_ad1 = os.path.join('boxplots', filename_ad1)
    plt.savefig(filepath_ad1, format='pdf', bbox_inches='tight', pad_inches=0.05)
    plt.close()

    # Create VUS_PR plot
    fig, ax = plt.subplots(figsize=(3, 0.75))  # Compact size

    # Get data range for limiting green line
    data_min_vus = group_data['VUS_PR'].min()
    data_max_vus = group_data['VUS_PR'].max()

    # Create box plot with increased width
    box = ax.boxplot(
        group_data['VUS_PR'], 
        vert=False, 
        patch_artist=True,
        widths=0.65,  # Increase from default 0.5 to make box taller
        boxprops=dict(facecolor='white', edgecolor='black', linewidth=1.5),  # Thicker edges
        medianprops=dict(color='orange', linewidth=1.5),  # Thicker median line
        whiskerprops=dict(color='black', linewidth=1.5),  # Thicker whiskers
        capprops=dict(color='black', linewidth=1.5),  # Thicker caps
        flierprops=dict(marker='o', markersize=10)#markerfacecolor='red', markersize=3),  # Larger outliers
    )

    mean_val_vus = group_data['VUS_PR'].mean()
    ax.axvline(mean_val_vus, color='green', linewidth=1.5, ymin=0.0585, ymax=0.94)

    ax.spines['bottom'].set_visible(False)
    ax.set_yticks([])
    ax.set_xlabel('')
    # Set x-axis ticks, show x-axis, and increase font size for XGBOOST technique
    if technique == 'XGBOOST':
        ax.set_xticks([0.0, 0.5, 1.0])
        ax.set_xticklabels(['0.0', '0.5', '1.0'], fontsize=FONT_SIZE // 3)
        ax.spines['bottom'].set_visible(True)
    else:
        ax.set_xticks([])
        ax.spines['bottom'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)

    # Tighter y-limits to reduce whitespace
    # data_range_vus = data_max_vus - data_min_vus
    # margin_vus = data_range_vus * 0.05
    # ax.set_xlim(data_min_vus - margin_vus, data_max_vus + margin_vus)
    ax.set_xlim(0, 1)
    ax.set_ylim(0.65, 1.35)  # Tighter limits around the box

    # Remove all padding
    plt.subplots_adjust(left=0.1, right=0.95, top=0.85, bottom=0.25)

    # Save VUS_PR plot
    filename_vus = f"{clean_flavor}_{clean_technique}_{clean_dataset}_VUS_PR.pdf"
    filename_vus = filename_vus.lower()
    filepath_vus = os.path.join('boxplots', filename_vus)
    plt.savefig(filepath_vus, format='pdf', bbox_inches='tight', pad_inches=0.05)
    plt.close()

    print(f"✓ Saved: {filename_ad1}")
    print(f"✓ Saved: {filename_vus}")

# Create summary figures with max lines per dataset
unique_datasets = df['Dataset'].unique()
for dataset in unique_datasets:
    dataset_data = df[df['Dataset'] == dataset]
    # AD1_AUC max
    max_ad1 = dataset_data['AD1_AUC'].max()
    fig, ax = plt.subplots(figsize=(3, 0.75))
    ax.set_xlim(0, 1)
    ax.set_ylim(0.65, 1.35)
    ax.set_yticks([])
    ax.set_xticks([])
    ax.set_xlabel('')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    plt.subplots_adjust(left=0.1, right=0.95, top=0.85, bottom=0.25)
    # Draw the vertical red line at the max value
    ax.axvline(max_ad1, color='red', linewidth=2, ymin=0.0585, ymax=0.94)
    # Add text with the max value
    ax.text(max_ad1 - 0.4, 1.0, f'{max_ad1:.3f}', fontsize=FONT_SIZE // 3, fontweight='bold', color='black', va='center')
    filename = f"oracle_ad1_auc_{str(dataset).replace(' ', '_').replace('/', '_').lower()}.pdf"
    filepath = os.path.join('boxplots', filename)
    plt.savefig(filepath, format='pdf', bbox_inches='tight', pad_inches=0.05)
    plt.close()
    print(f"✓ Saved: {filename}")
    # VUS_PR max
    max_vus = dataset_data['VUS_PR'].max()
    fig, ax = plt.subplots(figsize=(3, 0.75))
    ax.set_xlim(0, 1)
    ax.set_ylim(0.65, 1.35)
    ax.set_yticks([])
    ax.set_xticks([])
    ax.set_xlabel('')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    plt.subplots_adjust(left=0.1, right=0.95, top=0.85, bottom=0.25)
    ax.axvline(max_vus, color='red', linewidth=2, ymin=0.0585, ymax=0.94)
    # Add text with the max value
    ax.text(max_vus - 0.4, 1.0, f'{max_vus:.3f}', fontsize=FONT_SIZE // 3, fontweight='bold', color='black', va='center')
    filename = f"oracle_vus_pr_{str(dataset).replace(' ', '_').replace('/', '_').lower()}.pdf"
    filepath = os.path.join('boxplots', filename)
    plt.savefig(filepath, format='pdf', bbox_inches='tight', pad_inches=0.05)
    plt.close()
    print(f"✓ Saved: {filename}")

datasets = ['AZURE', 'BHD', 'CMAPSS', 'CNC', 'EDP', 'FEMTO', 'FORMULA 1', 'IMS', 'METROPT-3', 'Navarchos', 'XJTU-SY']
results_all_ad1_auc = [0.070, 0.010, 0.050, 0.023, 0.320, 0.040, 0.091, 0.030, 0.040, 0.040, 0.020]
results_all_vus_pr = [0.55, 0.50, 0.55, 0.526, 0.66, 0.54, 0.56, 0.53, 0.54, 0.52, 0.52]

results_add_ad1_auc = [0.070, 0.050, 0.160, 0.074, 0.670, 0.210, 0.248, 0.350, 0.290, 0.010, 0.110]
results_add_vus_pr =  [0.11, 0.01, 0.25, 0.123, 0.47, 0.11, 0.213, 0.16, 0.14, 0.03, 0.03]

# === Create ALL method figures (blue) ===
for i, dataset in enumerate(datasets):
    # AD1_AUC ALL
    all_ad1_value = results_all_ad1_auc[i]
    fig, ax = plt.subplots(figsize=(3, 0.75))
    ax.set_xlim(0, 1)
    ax.set_ylim(0.65, 1.35)
    ax.set_yticks([])
    ax.set_xticks([])
    ax.set_xlabel('')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    plt.subplots_adjust(left=0.1, right=0.95, top=0.85, bottom=0.25)
    # Draw the vertical blue line at the ALL value
    ax.axvline(all_ad1_value, color='blue', linewidth=2, ymin=0.0585, ymax=0.94)
    # Add text with the ALL value
    ax.text(all_ad1_value + 0.075, 1.0, f'{all_ad1_value:.3f}', fontsize=FONT_SIZE // 3, fontweight='bold', color='black', va='center')
    filename = f"all_ad1_auc_{dataset.replace(' ', '_').replace('/', '_').lower()}.pdf"
    filepath = os.path.join('boxplots', filename)
    plt.savefig(filepath, format='pdf', bbox_inches='tight', pad_inches=0.05)
    plt.close()
    print(f"✓ Saved: {filename}")
    
    # VUS_PR ALL
    all_vus_value = results_all_vus_pr[i]
    fig, ax = plt.subplots(figsize=(3, 0.75))
    ax.set_xlim(0, 1)
    ax.set_ylim(0.65, 1.35)
    ax.set_yticks([])
    ax.set_xticks([])
    ax.set_xlabel('')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    plt.subplots_adjust(left=0.1, right=0.95, top=0.85, bottom=0.25)
    ax.axvline(all_vus_value, color='blue', linewidth=2, ymin=0.0585, ymax=0.94)
    # Add text with the ALL value
    ax.text(all_vus_value + 0.075, 1.0, f'{all_vus_value:.3f}', fontsize=FONT_SIZE // 3, fontweight='bold', color='black', va='center')
    filename = f"all_vus_pr_{dataset.replace(' ', '_').replace('/', '_').lower()}.pdf"
    filepath = os.path.join('boxplots', filename)
    plt.savefig(filepath, format='pdf', bbox_inches='tight', pad_inches=0.05)
    plt.close()
    print(f"✓ Saved: {filename}")

# === Create ADD method figures (purple) ===
for i, dataset in enumerate(datasets):
    # AD1_AUC ADD
    add_ad1_value = results_add_ad1_auc[i]
    fig, ax = plt.subplots(figsize=(3, 0.75))
    ax.set_xlim(0, 1)
    ax.set_ylim(0.65, 1.35)
    ax.set_yticks([])
    ax.set_xticks([])
    ax.set_xlabel('')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    plt.subplots_adjust(left=0.1, right=0.95, top=0.85, bottom=0.25)
    # Draw the vertical purple line at the ADD value
    ax.axvline(add_ad1_value, color='purple', linewidth=2, ymin=0.0585, ymax=0.94)
    # Add text with the ADD value
    ax.text(add_ad1_value + 0.075, 1.0, f'{add_ad1_value:.3f}', fontsize=FONT_SIZE // 3, fontweight='bold', color='black', va='center')
    filename = f"add_ad1_auc_{dataset.replace(' ', '_').replace('/', '_').lower()}.pdf"
    filepath = os.path.join('boxplots', filename)
    plt.savefig(filepath, format='pdf', bbox_inches='tight', pad_inches=0.05)
    plt.close()
    print(f"✓ Saved: {filename}")
    
    # VUS_PR ADD
    add_vus_value = results_add_vus_pr[i]
    fig, ax = plt.subplots(figsize=(3, 0.75))
    ax.set_xlim(0, 1)
    ax.set_ylim(0.65, 1.35)
    ax.set_yticks([])
    ax.set_xticks([])
    ax.set_xlabel('')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    plt.subplots_adjust(left=0.1, right=0.95, top=0.85, bottom=0.25)
    ax.axvline(add_vus_value, color='purple', linewidth=2, ymin=0.0585, ymax=0.94)
    # Add text with the ADD value
    ax.text(add_vus_value + 0.075, 1.0, f'{add_vus_value:.3f}', fontsize=FONT_SIZE // 3, fontweight='bold', color='black', va='center')
    filename = f"add_vus_pr_{dataset.replace(' ', '_').replace('/', '_').lower()}.pdf"
    filepath = os.path.join('boxplots', filename)
    plt.savefig(filepath, format='pdf', bbox_inches='tight', pad_inches=0.05)
    plt.close()
    print(f"✓ Saved: {filename}")
    