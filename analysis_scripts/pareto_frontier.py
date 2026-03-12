import random
import sys

import matplotlib as mpl
import numpy as np
import pandas as pd
import paretoset
from matplotlib import pyplot as plt
import matplotlib.patches as mpatches

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

flavor_color_map = {
    "Auto profile ": "#D9534F",
    "Unsupervised ": "#5DA5DA",
    "Incremental ": "#60BD68",
    "Semisupervised ": "#F28E2B",
}

flavor_marker_map = {
    "Auto profile ": "o",     # circle
    "Unsupervised ": "s",     # square
    "Incremental ": "^",      # triangle up
    "Semisupervised ": "D",   # diamond
}

flavor_marker_map = {
    "Auto profile ": "o",     # circle
    "Unsupervised ": "s",     # square
    "Incremental ": "^",      # triangle up
    "Semisupervised ": "H",   # hexagon
}

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

METRIC = 'AD1_AUC'

df['is_max'] = (
    df[METRIC]
      .eq(df.groupby(['Dataset','Technique','Flavor'])[METRIC].transform('max'))
      .astype(int)
)

df = df[df['is_max'] > 0]

df = df[~df['Dataset'].isin(['AI4I'])]

df = df[~df['Technique'].isin(['ADD', 'ALL', 'XGBOOST', 'TimeLLMPyPots', 'TIMEMIXERPP', 'AutoGluon_chronos2', 'DummyAllAlarms'])]
df = df[df['Flavor'] != 'Semisupervised ']
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

df['Duration'] = np.log10(df['Duration'])

FONT_SIZE = 40

plt.rcParams.update({
    'font.size': FONT_SIZE,
    'axes.titlesize': FONT_SIZE,
    'axes.labelsize': FONT_SIZE,
    'xtick.labelsize': FONT_SIZE,
    'ytick.labelsize': FONT_SIZE,
    'legend.fontsize': FONT_SIZE
})


datasets = list(df['Dataset'].unique())
n_datasets = len(datasets)


fig, axes = plt.subplots(2, 6, figsize=(40, 15), sharex=True, sharey=True)
axes = axes.flatten()

pareto_rows = []

for idx, (current_dataset, grouped_df) in enumerate(df.groupby('Dataset')):
    ax = axes[idx]

    tick_dict = {
        0: '10⁰',
        1: '10¹',
        2: '10²',
        3: '10³',
        4: '10⁴',
        5: '10⁵',
        6: '10⁶',
    }

    ax.set_xticks(list(tick_dict.keys()))
    ax.set_xticklabels([tick_dict[i] for i in tick_dict.keys()])

    mask = paretoset.paretoset(
        grouped_df[['Duration', METRIC]].values,
        sense=['min', 'max']
    )
    pareto_df = grouped_df[mask]
    pareto_rows.append(pareto_df)

    ax.scatter(grouped_df['Duration'], grouped_df[METRIC], c='grey', alpha=0.25, s=350, marker='D')
    
    # Plot each flavor with different shapes and colors
    for flavor in pareto_df['Flavor'].unique():
        flavor_data = pareto_df[pareto_df['Flavor'] == flavor]
        ax.scatter(flavor_data['Duration'], flavor_data[METRIC], 
                  c=flavor_color_map[flavor], marker=flavor_marker_map[flavor], 
                  s=700, edgecolors='black', linewidth=1.5)
    frontier = pareto_df.sort_values('Duration')
    ax.plot(frontier['Duration'], frontier[METRIC], color='darkblue', linewidth=4, linestyle='--')
    
    # Apply Seaborn dark theme background color with 15% opacity (same as data_analysis_runtime)
    ax.set_facecolor((0.149, 0.149, 0.149, 0.15))  # #262626 with alpha=0.15
    
    ax.set_xlabel('Duration (s)')
    ax.set_ylabel('AD1 AUC-PR')
    ax.set_title(f'{current_dataset}', pad=30)
    ax.grid(True, alpha=0.3, linewidth=1)
    # ax.legend()

# Hide the last subplot in the second row
axes[11].axis('off')

handles, labels = axes[0].get_legend_handles_labels()

patches = []
for name in flavor_color_map.keys():
    # Create a custom legend entry with both color and marker shape
    from matplotlib.lines import Line2D
    patch = Line2D([0], [0], marker=flavor_marker_map[name], color='w', 
                   markerfacecolor=flavor_color_map[name], markersize=30, 
                   markeredgecolor='black', markeredgewidth=1.5,
                   label=formal_flavor_name_map[name].title() + ' flavor')
    patches.append(patch)

# Add dominated configuration with its marker shape
dominated_patch = Line2D([0], [0], marker='D', color='w', 
                        markerfacecolor='grey', markersize=30, alpha=0.5,
                        markeredgecolor='black', markeredgewidth=1.5,
                        label='Dominated configuration')
patches.append(dominated_patch)
patches.append(Line2D([0], [0], color='darkblue', linewidth=4, linestyle='--', label='Pareto frontier'))

fig.legend(
    handles=patches,
    # labels,
    loc='center',
    ncol=3,
    fontsize=FONT_SIZE,
)

# plt.tight_layout()
plt.tight_layout(h_pad=5)
plt.show()

fig.savefig("pareto_frontiers.pdf", format='pdf', bbox_inches='tight')

pareto_all = pd.concat(pareto_rows)

print("\nFlavor distribution on Pareto frontier:")
print(pareto_all['Flavor'].value_counts(normalize=True) * 100)

print("\nTechnique distribution on Pareto frontier:")
print(pareto_all['Technique'].value_counts(normalize=True) * 100)
