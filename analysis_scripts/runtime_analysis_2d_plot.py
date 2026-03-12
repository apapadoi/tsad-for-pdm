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
    # 'Semisupervised ': 'historical',
    'Unsupervised ': '_uns',
    'Classification ': ''
}
# FONT_SIZE = 65

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
FONT_SIZE = 65
METRIC = 'AD1_AUC'

# Get the best row for each technique-flavor-dataset combination
# First, filter to only rows with maximum METRIC per group
max_metric_mask = df.groupby(['Technique', 'Flavor', 'Dataset'])[METRIC].transform('max') == df[METRIC]
max_metric_df = df[max_metric_mask]

# Then, from those rows, select the one with minimum Duration per group (fastest among best)
best_results = max_metric_df.loc[max_metric_df.groupby(['Technique', 'Flavor', 'Dataset'])['Duration'].idxmin()]

# Calculate average duration and average METRIC for each technique-flavor combination
combo_stats = best_results.groupby(['Technique', 'Flavor']).agg({
    'Duration': 'mean',
    METRIC: 'mean'
}).reset_index()

# Create the 2D scatter plot
fig, ax = plt.subplots(figsize=(8, 4))

# Set background color
ax.set_facecolor((0.149, 0.149, 0.149, 0.15))  # #262626 with alpha=0.15

# Get unique combinations for color and marker mapping
unique_flavors = combo_stats['Flavor'].unique()
colors = plt.cm.Set1(np.linspace(0, 1, len(unique_flavors)))
flavor_colors = dict(zip(unique_flavors, colors))

# Define different markers for each flavor
markers = ['o', 's', '^', 'D']  # circle, square, triangle, diamond
flavor_markers = dict(zip(unique_flavors, markers[:len(unique_flavors)]))

# Plot points for each technique-flavor combination
for _, row in combo_stats.iterrows():
    technique = row['Technique']
    flavor = row['Flavor']
    avg_duration = row['Duration']
    avg_metric = row[METRIC]
    
    # Check if this is LOF with online flavor
    # if (technique == 'LOF' and flavor == '_onl' and METRIC == 'AD1_AUC') or \
    #     (technique == 'OCSVM' and flavor == '_onl' and METRIC == 'VUS_PR'):
    #     # Use star marker for LOF online
    #     plt.scatter(avg_duration, avg_metric, 
    #                color='red', 
    #                s=100, alpha=1, marker='*',
    #                label=f'{technique}{flavor}')
    # else:
    # Regular marker for points based on flavor
    ax.scatter(avg_duration, avg_metric, 
                color=flavor_colors[flavor], 
                s=50, alpha=1, 
                marker=flavor_markers[flavor],
                label=f'{technique}{flavor}')
    
    # Add text annotation for the technique
    ax.annotate(f'{technique}', 
                (avg_duration, avg_metric),
                xytext=(2, 2), textcoords='offset points',
                fontsize=FONT_SIZE // 4, alpha=1)

ax.set_xlabel('Average processing time (seconds)', fontsize=FONT_SIZE // 4)
ax.set_ylabel(f'Average {METRIC.replace("_", " ") + ("-PR" if "AUC" in METRIC else "")}', fontsize=FONT_SIZE // 4)
# ax.set_title('Technique-Flavor Combinations: Performance vs Speed Trade-off')
ax.set_xscale('log')
ax.tick_params(axis='x', labelsize=FONT_SIZE // 4)
ax.tick_params(axis='y', labelsize=FONT_SIZE // 4)
ax.grid(True, alpha=0.5)

# Add pink arrow pointing to upper left corner with "better" text
from matplotlib.patches import FancyArrowPatch

# Get current axis limits
x_min, x_max = ax.get_xlim()
y_min, y_max = ax.get_ylim()

# Calculate arrow positions
# Start: between 10^3 and 10^4 (around 5000), at a high performance value
arrow_y_start = 0.4  # Between 10^3 and 10^4
arrow_x_start = 2500  # High performance

# End: little bit right of 10^2 (around 200), at an even higher performance value
arrow_y_end =0.55     # Little bit right of 10^2
arrow_x_end = 650    # Even higher performance

# Add pink arrow
arrow = FancyArrowPatch((arrow_x_start, arrow_y_start), (arrow_x_end, arrow_y_end),
                       arrowstyle='-|>', 
                       color='hotpink', 
                       linewidth=3,
                       mutation_scale=30,
                       zorder=10)
ax.add_patch(arrow)

# Add "better" text near the arrow
text_x = (arrow_x_start + arrow_x_end) / 2 + 500
text_y = (arrow_y_start + arrow_y_end) / 1.95
ax.text(text_x, text_y, 'better', 
        fontsize=FONT_SIZE // 3, 
        color='hotpink', 
        weight='bold',
        ha='center', 
        va='center', 
        zorder=10)

# Create custom legend for flavors
flavor_names = {'_onl': 'Online', '_sli': 'Sliding', '_uns': 'Unsupervised', '': 'Supervised'}
legend_elements = [plt.Line2D([0], [0], marker=flavor_markers[flavor], color='w', 
                             markerfacecolor=flavor_colors[flavor], 
                             markersize=8, label=flavor_names.get(flavor, flavor))
                  for flavor in unique_flavors]

# Add Global best star to legend
# legend_elements.append(plt.Line2D([0], [0], marker='*', color='w', 
#                                  markerfacecolor='red', markersize=12, 
#                                  label='Global best'))

ax.legend(handles=legend_elements, title='Flavor', loc='lower left')

plt.tight_layout()
plt.savefig(f'technique_flavor_performance_vs_speed_{METRIC}.pdf', format='pdf', bbox_inches='tight')
plt.savefig(f'technique_flavor_performance_vs_speed_{METRIC}.png', format='png', bbox_inches='tight', dpi=300)
plt.show()

print(f"2D plot saved as 'technique_flavor_performance_vs_speed_{METRIC}.pdf' and 'technique_flavor_performance_vs_speed_{METRIC}.png'")
print("\nSummary statistics:")
print(combo_stats.sort_values(METRIC, ascending=False))

