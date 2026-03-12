import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
from mpl_toolkits.mplot3d import Axes3D
import matplotlib as mpl

# Set style
FONT_SIZE = 44
plt.rcParams.update({
    'font.size': FONT_SIZE,
    'axes.titlesize': FONT_SIZE,
    'axes.labelsize': FONT_SIZE,
    'xtick.labelsize': 0.8*FONT_SIZE,
    'ytick.labelsize': 0.8*FONT_SIZE,
    'legend.fontsize': 0.8*FONT_SIZE,
    'axes.facecolor': '#EAEAF2',
    'grid.color': 'white',
    'grid.linestyle': '-',
    'grid.linewidth': 1.5,
})

# Read data
df = pd.read_csv('data_analysis_runtime.csv')

# Preprocessing
df.drop_duplicates(inplace=True, ignore_index=True)

df['Technique'] = df.apply(lambda row: 'KNN' if 'Distance' in row['Technique'] else row['Technique'], axis=1)

df['Technique'] = df.apply(lambda row: row['Technique'].lower().replace('unsupervised', '').capitalize() if 'unsupervised' in row['Technique'].lower() else row['Technique'], axis=1)

df['Technique'] = df.apply(lambda row: row['Technique'].lower().replace('(uns)', '').capitalize() if '(uns)' in row['Technique'].lower() else row['Technique'], axis=1)

df['Technique'] = df.apply(lambda row: row['Technique'].lower().replace('(semi)', '').capitalize() if '(semi)' in row['Technique'].lower() else row['Technique'], axis=1)

df['Technique'] = df.apply(lambda row: row['Technique'].lower().replace('semi', '').capitalize() if 'semi' in row['Technique'].lower() else row['Technique'], axis=1)

df['Technique'] = df.apply(lambda row: 'IsolationForest' if 'Isolation' in row['Technique'] else row['Technique'], axis=1)

df['Technique'] = df.apply(lambda row: 'LocalOutlierFactor' if 'Local' in row['Technique'] else row['Technique'], axis=1)

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
    'KNN': 'KNN',
    'LTSF': 'LTSF',
    'USAD': 'USAD',
    'Chronos': 'CHRONOS',
}

df['Technique'] = df['Technique'].map(technique_mapping_dict).fillna(df['Technique'])
df = df[~df['Technique'].isin(['ADD', 'ALL', 'XGBOOST', 'TimeLLMPyPots', 'TIMEMIXERPP', 'AutoGluon_chronos2', 'DummyAllAlarms'])]

# Calculate Best AD1_AUC per technique
best_auc_dict = df[df['Dataset'] == 'EDP'].groupby('Technique')['AD1_AUC'].max().to_dict()

# Filter for Adversarial
df_adv = df[df['Flavor'].str.contains('dversarial', na=False)].copy()

# Parse Rows
def parse_flavor(flavor):
    if 'Noise' in flavor:
        m = re.search(r'sigma=([0-9.]+)', flavor)
        if m:
            return 'Noise', float(m.group(1)), None, None
    elif 'Out-of-Order' in flavor:
        m = re.search(r'max_lag=(\d+), prob=([0-9.]+)', flavor)
        if m:
            return 'OOO', None, int(m.group(1)), float(m.group(2))
    return None, None, None, None

df_adv[['Type', 'Sigma', 'MaxLag', 'Prob']] = df_adv['Flavor'].apply(
    lambda x: pd.Series(parse_flavor(x))
)


# Validation: Check for 9 rows per technique for OOO and 3 for Noise
df_ooo_check = df_adv[df_adv['Type'] == 'OOO']
ooo_counts = df_ooo_check['Technique'].value_counts()
for technique, count in ooo_counts.items():
    if count != 9:
        raise RuntimeError(f"Technique {technique} has {count} OOO rows in df_adv (expected 9).")

df_noise_check = df_adv[df_adv['Type'] == 'Noise']
noise_counts = df_noise_check['Technique'].value_counts()
for technique, count in noise_counts.items():
    if count != 3:
        raise RuntimeError(f"Technique {technique} has {count} Noise rows in df_adv (expected 3).")


# Use all techniques from the full dataset to ensure consistent colors/markers with other plots
all_techniques_full = sorted(df['Technique'].dropna().unique())
plot_techniques = sorted(df_adv['Technique'].dropna().unique())

# Create colors and shapes based on full list
n_techniques = len(all_techniques_full)
tab10_colors = plt.cm.tab10(np.linspace(0, 1, 10))
color_list = []
for i in range(n_techniques):
    color_list.append(tab10_colors[i % 10])

color_map = dict(zip(all_techniques_full, color_list))
if 'CHRONOS' in color_map:
    color_map['CHRONOS'] = 'black'

available_markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h', 'H', '+', 'x', '8', 'd', '|', '_']
marker_map = {}
for i, technique in enumerate(all_techniques_full):
    marker_map[technique] = available_markers[i % len(available_markers)]

all_techniques = plot_techniques # Alias for compatibility with rest of script

# --- COMBINED PLOT ---
print("Generating Combined Robustness Plot (OOO + Noise)...")

fig = plt.figure(figsize=(25,18))
rows, cols = 2, 3

# 1. Out-of-Order Plots
df_ooo = df_adv[df_adv['Type'] == 'OOO']

if not df_ooo.empty and not df_ooo['MaxLag'].isnull().all():
    ooo_grouped = df_ooo.groupby(['Technique', 'MaxLag', 'Prob'])['AD1_AUC'].mean().reset_index()

    # Define plot groups
    # 1. CHRONOS, OCSVM
    # 2. TRANAD, KNN, NP
    # 3. USAD, PB
    # 4. IF, SAND
    # 5. LOF, LTSF
    
    ooo_groups = [
        ['CHRONOS', 'OCSVM'],
        ['TRANAD', 'KNN', 'NP'],
        ['USAD', 'PB'],
        ['IF', 'SAND'],
        ['LOF', 'LTSF']
    ]
    
    for i, tech_group in enumerate(ooo_groups):
        ax = fig.add_subplot(rows, cols, i+1, projection='3d')
        ax.view_init(elev=15, azim=-40, roll=0)
        ax.xaxis.set_pane_color((0.92, 0.92, 0.95, 1.0))
        ax.yaxis.set_pane_color((0.92, 0.92, 0.95, 1.0))
        ax.zaxis.set_pane_color((0.92, 0.92, 0.95, 1.0))
        
        # Add vertical dashed line at (0,0)
        ax.plot([0, 0], [0, 0], [0, 1], linestyle='--', color='black', linewidth=1.5, alpha=0.5)

        # Iterate through each technique in the current group
        for technique in tech_group:
            tech_data = ooo_grouped[ooo_grouped['Technique'] == technique]
            
            if not tech_data.empty:
                xs = tech_data['MaxLag']
                ys = tech_data['Prob']
                zs = tech_data['AD1_AUC']
                
                # Apply consistent coloring: Blues colormap mapped 0 to 1
                ax.plot_trisurf(xs, ys, zs, cmap='Blues', vmin=0, vmax=1, edgecolor='none', alpha=0.8)
                
                # Use markers to correspond to technique
                scatter_color = color_map.get(technique, 'black')
                if isinstance(scatter_color, np.ndarray):
                    scatter_color = tuple(scatter_color)
                
                # Use c=[color] * len to be explicit that all points get the same color
                ax.scatter(xs, ys, zs, c=[scatter_color]*len(xs), marker=marker_map.get(technique, 'o'), s=400, label=technique) 

                best_val = best_auc_dict.get(technique)
                if best_val is not None:
                    ax.scatter([0], [0], [best_val], facecolor='white', edgecolor=scatter_color, marker=marker_map.get(technique, 'o'), s=400, hatch='//', linewidth=2)
        
        ax.set_xlabel('Max Lag', labelpad=40)
        ax.set_xticks([36, 72, 144])
        
        # Set Y label 'Prob' for all OOO subplots
        ax.set_ylabel('Prob', labelpad=40)
            
        ax.set_yticks([0.1, 0.5, 0.75])
        
        # Remove Z label from all OOO subplots (label only on Noise plot)
        ax.set_zlabel('')

        ax.set_zlim(0, 1.0) 
        ax.set_zticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
        
        if i == 0 or i == 3:
            ax.tick_params(axis='z', labelsize=0.6*FONT_SIZE) # Decrease Z axis tick font size
        else:
            ax.set_zticklabels([])
        
        # Adjust camera/aspect to match 2D plot height better
        # Default dist is around 10. Lowering it zooms in.
        ax.dist = 7.5 
        # Make the Z axis usage more efficient within the box
        ax.set_box_aspect((1, 1, 0.9)) 
        
        ax.tick_params(axis='both', which='major', pad=15)

# 2. Noise Plot (Last position)
df_noise = df_adv[df_adv['Type'] == 'Noise']
if not df_noise.empty:
    ax_noise = fig.add_subplot(rows, cols, 6)
    # ax_noise.set_facecolor('#EAEAF2') # Already set by rcParams, but good to ensure if needed
    ax_noise.grid(color='white', linestyle='-', linewidth=2)
    
    # Group by Technique and Sigma, take mean of AD1_AUC
    noise_grouped = df_noise.groupby(['Technique', 'Sigma'])['AD1_AUC'].mean().reset_index()
    
    min_sigma = noise_grouped[noise_grouped['Sigma'] > 0]['Sigma'].min() if not noise_grouped.empty else 1e-3
    if pd.isna(min_sigma): min_sigma = 1e-3

    for technique in all_techniques:
        tech_data = noise_grouped[noise_grouped['Technique'] == technique]
        best_val = best_auc_dict.get(technique, None)

        if not tech_data.empty:
            tech_data = tech_data.sort_values('Sigma')
            xs = tech_data['Sigma'].tolist()
            ys = tech_data['AD1_AUC'].tolist()

            if best_val is not None:
                xs = [0] + xs
                ys = [best_val] + ys

            ax_noise.plot(xs, ys, 
                          marker=marker_map[technique], 
                          color=color_map[technique], 
                          markersize=25, linewidth=4, label=technique)

            if best_val is not None:
                 ax_noise.scatter([0], [best_val], 
                                 marker=marker_map[technique], 
                                 facecolor='white', 
                                 edgecolor=color_map[technique], 
                                 hatch='//', 
                                 s=625, 
                                 linewidth=2, 
                                 zorder=10)

    ax_noise.set_xscale('symlog', linthresh=min_sigma)
    ax_noise.set_xlabel('Sigma', labelpad=20)
    # Set y-label only for the last subplot in the row (Noise plot is last in Row 2)
    # Move the y-label to the right side
    ax_noise.set_ylabel('AD1 AUC-PR', labelpad=40, rotation=270)
    ax_noise.yaxis.set_label_position("right")
    ax_noise.yaxis.tick_right()
    
    ax_noise.set_ylim(0, 1.0)
    ax_noise.set_title('Noise injection robustness', pad=20)

    # Use ncol=2 for legend to fit it better in subplot
    # ax_noise.legend(fontsize=FONT_SIZE*0.7, markerscale=1.5, ncol=2) 
    ax_noise.tick_params(axis='both', which='major', pad=15)

# Create unified legend handles
legend_handles = []
# Calculate number of columns for 2 rows
n_cols_legend = int(np.ceil(len(all_techniques) / 2))

for tech in all_techniques:
    color = color_map.get(tech, 'black')
    if isinstance(color, np.ndarray):
        color = tuple(color)
    marker = marker_map.get(tech, 'o')
    
    label_text = tech

    h = mpl.lines.Line2D([0], [0], marker=marker, color='w', markerfacecolor=color, 
                         markersize=25, markeredgecolor='black', markeredgewidth=3, label=label_text)
    legend_handles.append(h)

plt.tight_layout()
# Increase hspace to make room for center legend/colorbar. Reduce wspace to make OOO plots closer.
# Added left margin to prevent clipping of first subplot's Prob label
plt.subplots_adjust(left=0.08, bottom=0.05, top=0.95, wspace=0.05, hspace=0.6)

# Center Legend (2 rows) - positioned relative to figure
fig.legend(handles=legend_handles, loc='center', bbox_to_anchor=(0.5, 0.52), 
           ncol=n_cols_legend, fontsize=FONT_SIZE*0.8, frameon=True, edgecolor='black', fancybox=False, shadow=False, facecolor='white', markerscale=1.5, labelspacing=0.4)

# Add shared colorbar for OOO surfaces in the center
cbar_ax = fig.add_axes([0.35, 0.46, 0.3, 0.02])
norm = mpl.colors.Normalize(vmin=0, vmax=1)
sm = plt.cm.ScalarMappable(cmap='Blues', norm=norm)
sm.set_array([])
cbar = fig.colorbar(sm, cax=cbar_ax, orientation='horizontal', label='AD1 AUC-PR (Surface)')
cbar.ax.tick_params(labelsize=0.75*FONT_SIZE)
cbar.set_label('AD1 AUC-PR', fontsize=0.9*FONT_SIZE)

fig.savefig("adversarial_robustness_combined.pdf", format='pdf', bbox_inches='tight', dpi=300)
fig.savefig("adversarial_robustness_combined.png", format='png', bbox_inches='tight', dpi=300)
print("Saved adversarial_robustness_combined.pdf/png")
