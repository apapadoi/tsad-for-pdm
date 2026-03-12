import mlflow
import sys
import random
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib.patches import Patch, Rectangle
import matplotlib as mpl
from scipy.stats import gaussian_kde
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

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
FONT_SIZE = 60

plt.rcParams.update({
    'font.size': FONT_SIZE,  # Change this number to adjust font size
    'axes.titlesize': FONT_SIZE,
    'axes.labelsize': 2*FONT_SIZE,
    'xtick.labelsize': 1.5*FONT_SIZE,
    'ytick.labelsize': 2*FONT_SIZE,
    'legend.fontsize': FONT_SIZE
})

if len(sys.argv) <= 1:
    mlflow_server_url_list = ["http://127.0.0.1:5000/", "http://127.0.0.1:8080/", "http://127.0.0.1:5001/", "http://localhost:43669/"]

    current_df = pd.DataFrame([], columns=['Flavor', 'Technique', 'Dataset', 'Duration', 'AD1_AUC', 'VUS_PR'])
    for current_url in mlflow_server_url_list:
        mlflow.set_tracking_uri(current_url)
        client = mlflow.tracking.MlflowClient()

        rules=[("params.postprocessor","Default")]
        ad1_auc_metric="metrics.AD1_AUC"
        vus_pr_metric="metrics.VUS_VUS_PR"

        # techniques = ["PB", "KNN", "IF", "LOF", "NP", "SAND", "OCSVM", "DISTANCE BASED", "LTSF", "TRANAD", "USAD","CHRONOS"]
        datasets = [
            'CMAPSS',
            'Navarchos',
            'FEMTO',
            'IMS',
            'EDP',
            'METRO',
            'XJTU',
            'BHD',
            'AZURE',
            'Formula 1',
            'CNC'
        ]

        for dataset in datasets:
            print(dataset)
            experiments = client.search_experiments(filter_string=f"name LIKE '%{dataset}%'")

            for experiment in experiments:
                if "correlated" in experiment.name:
                    continue

                print(experiment.name)

                if ('XJTU' in experiment.name and 'XJTU PH' not in experiment.name) \
                    or 'TSB' in experiment.name or 'Philips' in experiment.name or 'AI4I' in experiment.name \
                or 'My first experiment' in experiment.name or 'BHD PH 20' in experiment.name or 'CHRONOS LARGE' in experiment.name:
                        print(f'Skipping {experiment.name}')
                        continue

                exp_id = experiment.experiment_id

                if len(sys.argv) == 1:
                    runs = mlflow.search_runs(exp_id)
                else:
                    runs = mlflow.search_runs(exp_id, filter_string=f'attributes.created > {sys.argv[1]}')

                # Sort based on start_time values
                if len(runs) > 0 and 'start_time' in runs.columns:
                    runs = runs.sort_values('start_time', ascending=True)
                    print(f"Sorted {len(runs)} runs for experiment {experiment.name}")

                max_value = None
                runs_appended_count = 0

                for index, run in runs.iterrows():
                    skip = False
                    for rule in rules:
                        if rule[0] not in run.keys():
                            skip = True
                            break

                        if run[rule[0]] is None:
                            skip = True
                            break

                        elif rule[1] in str(run[rule[0]]):
                            continue

                        else:
                            skip = True
                            break

                    if skip:
                        continue


                    duration = run["end_time"] - run['start_time']
                    value_duration_seconds = duration.seconds

                    if ad1_auc_metric not in run.keys() or vus_pr_metric not in run.keys():
                        print(f"Not such metric, available: {run.keys()}")
                        continue
                    else:
                        ad1_auc_value_metric = run[ad1_auc_metric]
                        vus_pr_value_metric = run[vus_pr_metric]

                    if ad1_auc_value_metric == 0 or ad1_auc_value_metric == 0.0:
                        vus_pr_value_metric = 0

                    new_df = pd.DataFrame([[experiment.name.split(dataset)[0], run.loc['params.method'], dataset, value_duration_seconds, ad1_auc_value_metric, vus_pr_value_metric]], columns=current_df.columns)
                    current_df = pd.concat([current_df, new_df], ignore_index=True)
                    runs_appended_count += 1
                    
                    # Break when 50 runs have been appended
                    if runs_appended_count >= 50:
                        break

                print(f"Appended {runs_appended_count} out of {len(runs)} runs for experiment {experiment.name}")


    current_df.to_csv(f'data_analysis_runtime.csv', index=False)
else:
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
    
    # Add log duration for all data
    df['log_Duration'] = np.log10(df['Duration'])
    
    total = 0

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
        # 'XGBOOST': 'XGBOOST',
        'KNN': 'KNN',
        'LTSF': 'LTSF',
        'USAD': 'USAD',
        'Chronos': 'CHRONOS',
        # 'ADD': 'ADD',
        # 'ALL': 'ALL'
    }

    df = df[~df['Technique'].isin(['ADD', 'ALL', 'XGBOOST', 'TimeLLMPyPots', 'TIMEMIXERPP', 'AutoGluon_chronos2', 'DummyAllAlarms'])]
    df['Technique'] = df['Technique'].map(technique_mapping_dict)

    all_techniques = sorted(df['Technique'].unique())
    all_datasets = [
        'AZURE',
        'BHD 2023',
        'CMAPSS',
        'CNC',
        'EDP-WT',
        'FEMTO',
        'FORMULA 1',
        'IMS',
        'METROPT-3',
        'NAVARCHOS',
        'XJTU-SY',
    ]

    # Create colors and shapes for techniques using tab10 color palette
    n_techniques = len(all_techniques)
    
    # Use tab10 color palette and cycle through it if we have more than 10 techniques
    tab10_colors = plt.cm.tab10(np.linspace(0, 1, 10))
    color_list = []
    for i in range(n_techniques):
        color_list.append(tab10_colors[i % 10])  # Cycle through tab10 colors
    
    color_map = dict(zip(all_techniques, color_list))
    
    # Explicitly set CHRONOS to black color
    if 'CHRONOS' in color_map:
        color_map['CHRONOS'] = 'black'
    
    # Use different shapes and colors for each technique
    available_markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h', 'H', '+', 'x', '8', 'd', '|', '_']
    marker_map = {}
    for i, technique in enumerate(all_techniques):
        marker_map[technique] = available_markers[i % len(available_markers)]

    # Determine number of datasets per flavor
    datasets_per_flavor = {
        'Semisupervised ': ['CMAPSS', 'FEMTO'],  # Only 2 datasets
        'Auto profile ': all_datasets,
        'Incremental ': all_datasets, 
        'Unsupervised ': all_datasets
    }
    
    # Calculate per-dataset y-axis (duration) range for main flavors
    dataset_duration_ranges = {}
    for dataset in all_datasets:
        dataset_df_main = df[(df['Dataset'] == dataset) & (df['Flavor'].isin(["Auto profile ", "Incremental ", "Unsupervised "]))]
        if len(dataset_df_main) > 0:
            min_dur = dataset_df_main['Duration'].min()
            max_dur = dataset_df_main['Duration'].max()
            dataset_duration_ranges[dataset] = (min_dur, max_dur)
            print(f"Dataset {dataset} Duration range: [{min_dur:.2f}, {max_dur:.2f}]")
    
    # Calculate per-dataset y-axis (duration) range for historical flavor
    hist_dataset_duration_ranges = {}
    for dataset in ['CMAPSS', 'FEMTO']:
        dataset_df_hist = df[(df['Dataset'] == dataset) & (df['Flavor'] == "Semisupervised ")]
        if len(dataset_df_hist) > 0:
            min_dur = dataset_df_hist['Duration'].min()
            max_dur = dataset_df_hist['Duration'].max()
            hist_dataset_duration_ranges[dataset] = (min_dur, max_dur)
            print(f"Historical {dataset} Duration range: [{min_dur:.2f}, {max_dur:.2f}]")
    
    # Create separate figures for historical and other flavors
    flavors_main = ["Auto profile ", "Incremental ", "Unsupervised "]
    flavors_historical = ["Semisupervised "]
    
    # Calculate per-dataset maximum AD1_AUC for normalization
    dataset_max_ad1_auc_dict = {}
    for dataset in all_datasets:
        dataset_data = df[df['Dataset'] == dataset]['AD1_AUC']
        if len(dataset_data) > 0:
            dataset_max = dataset_data.max()
            dataset_max_ad1_auc_dict[dataset] = dataset_max
            print(f"Dataset {dataset} max AD1_AUC: {dataset_max:.4f}")
    
    # Calculate per-flavor x-axis (AD1_AUC) range for main flavors to ensure same range per column
    flavor_ad1_auc_ranges = {}
    for flavor in flavors_main:
        flavor_df = df[df['Flavor'] == flavor]
        if len(flavor_df) > 0:
            # Calculate normalized ranges per flavor
            normalized_values = []
            for dataset in all_datasets:
                dataset_subset = flavor_df[flavor_df['Dataset'] == dataset]
                if len(dataset_subset) > 0 and dataset in dataset_max_ad1_auc_dict and dataset_max_ad1_auc_dict[dataset] > 0:
                    norm_vals = dataset_subset['AD1_AUC'] / dataset_max_ad1_auc_dict[dataset]
                    normalized_values.extend(norm_vals.tolist())
            
            if normalized_values:
                min_auc = min(normalized_values)
                max_auc = max(normalized_values)
                flavor_ad1_auc_ranges[flavor] = (min_auc, max_auc)
                print(f"Flavor {flavor} normalized AD1_AUC range: [{min_auc:.4f}, {max_auc:.4f}]")
    
    # Calculate per-flavor x-axis (AD1_AUC) range for historical flavor
    hist_flavor_ad1_auc_ranges = {}
    for flavor in flavors_historical:
        flavor_df = df[df['Flavor'] == flavor]
        if len(flavor_df) > 0:
            # Calculate normalized ranges per flavor
            normalized_values = []
            for dataset in ['CMAPSS', 'FEMTO']:  # Only historical datasets
                dataset_subset = flavor_df[flavor_df['Dataset'] == dataset]
                if len(dataset_subset) > 0 and dataset in dataset_max_ad1_auc_dict and dataset_max_ad1_auc_dict[dataset] > 0:
                    norm_vals = dataset_subset['AD1_AUC'] / dataset_max_ad1_auc_dict[dataset]
                    normalized_values.extend(norm_vals.tolist())
            
            if normalized_values:
                min_auc = min(normalized_values)
                max_auc = max(normalized_values)
                hist_flavor_ad1_auc_ranges[flavor] = (min_auc, max_auc)
                print(f"Historical {flavor} normalized AD1_AUC range: [{min_auc:.4f}, {max_auc:.4f}]")
    
    # Create the main figure with 10 rows (datasets) and 3 columns (flavors)
    # Make it taller to prevent histogram overlap
    fig_main = plt.figure(figsize=(100, 85))
    # (100, 75)
    # Track all techniques that appear for legend creation
    all_legend_data = {}
    
    # NOTE: Main figure uses KDE offset of 0.06 * x_range
    # Iterate over datasets (rows) and flavors (columns)
    for dataset_idx, dataset in enumerate(all_datasets):
        for flavor_idx, flavor in enumerate(flavors_main):
            current_df = df[df['Flavor'] == flavor].copy()
            
            # Calculate subplot position (10 rows, 3 columns)
            subplot_idx = dataset_idx * 3 + flavor_idx + 1
            ax = plt.subplot(11, 3, subplot_idx)
            
            dataset_df = current_df[current_df['Dataset'] == dataset].copy()
            
            if len(dataset_df) == 0:
                ax.set_title(dataset, fontsize=FONT_SIZE*0.2)
                ax.set_xticks([])
                ax.set_yticks([])
                continue
            
            # Sort by AD1_AUC for left-to-right ordering
            dataset_df = dataset_df.sort_values('AD1_AUC')
            
            # Normalize AD1_AUC values by dataset maximum
            if dataset in dataset_max_ad1_auc_dict and dataset_max_ad1_auc_dict[dataset] > 0:
                normalized_ad1_auc = dataset_df['AD1_AUC'] / dataset_max_ad1_auc_dict[dataset]
            else:
                normalized_ad1_auc = dataset_df['AD1_AUC']  # Fallback to raw values
            
            # Store full dataset for KDE calculations before point reduction
            full_dataset_df = dataset_df.copy()
            full_normalized_ad1_auc = normalized_ad1_auc.copy()
            
            # Data point reduction: for combinations with >30 rows, split into ~10 equal parts and keep median points
            if len(dataset_df) > 30:
                print(f"Reducing data points for {dataset} - {flavor}: {len(dataset_df)} -> ", end="")
                
                # Group by technique and apply median selection within each technique
                reduced_rows = []
                for technique in dataset_df['Technique'].unique():
                    technique_df = dataset_df[dataset_df['Technique'] == technique].copy()
                    technique_normalized = normalized_ad1_auc[technique_df.index]
                    
                    if len(technique_df) > 30:
                        selected_indices = set()
                        
                        # Always include max and min AD1_AUC points
                        max_auc_idx = technique_df['AD1_AUC'].idxmax()
                        selected_indices.add(max_auc_idx)
                        min_auc_idx = technique_df['AD1_AUC'].idxmin()
                        selected_indices.add(min_auc_idx)
                        
                        # Always include max and min log duration points
                        technique_df['log_Duration_temp'] = np.log10(technique_df['Duration'])
                        max_log_dur_idx = technique_df['log_Duration_temp'].idxmax()
                        selected_indices.add(max_log_dur_idx)
                        min_log_dur_idx = technique_df['log_Duration_temp'].idxmin()
                        selected_indices.add(min_log_dur_idx)
                        
                        print(f"Always adding max AD1_AUC point for {dataset}-{technique}: {technique_df.loc[max_auc_idx]['AD1_AUC']:.4f}")
                        print(f"Always adding min AD1_AUC point for {dataset}-{technique}: {technique_df.loc[min_auc_idx]['AD1_AUC']:.4f}")
                        print(f"Always adding max log duration point for {dataset}-{technique}: {technique_df.loc[max_log_dur_idx]['log_Duration_temp']:.4f}")
                        print(f"Always adding min log duration point for {dataset}-{technique}: {technique_df.loc[min_log_dur_idx]['log_Duration_temp']:.4f}")
                        
                        # FIRST: Sort by AD1_AUC and split into ~10 parts, pick median based on AD1_AUC
                        technique_df_sorted_auc = technique_df.sort_values('AD1_AUC').copy()
                        n_parts = 10
                        part_size = len(technique_df_sorted_auc) // n_parts
                        if part_size == 0:
                            part_size = 1
                            n_parts = len(technique_df_sorted_auc)
                        
                        for i in range(n_parts):
                            start_idx = i * part_size
                            if i == n_parts - 1:  # Last part takes remaining rows
                                end_idx = len(technique_df_sorted_auc)
                            else:
                                end_idx = (i + 1) * part_size
                            
                            part_df = technique_df_sorted_auc.iloc[start_idx:end_idx]
                            if len(part_df) > 0:
                                # Select median point based on AD1_AUC (same column used for sorting)
                                median_auc = part_df['AD1_AUC'].median()
                                closest_idx = (part_df['AD1_AUC'] - median_auc).abs().idxmin()
                                selected_indices.add(closest_idx)
                        
                        # SECOND: Sort by log-scaled Duration and split into ~10 parts, pick median based on log Duration
                        technique_df['log_Duration_temp'] = np.log10(technique_df['Duration'])
                        technique_df_sorted_dur = technique_df.sort_values('log_Duration_temp').copy()
                        
                        for i in range(n_parts):
                            start_idx = i * part_size
                            if i == n_parts - 1:  # Last part takes remaining rows
                                end_idx = len(technique_df_sorted_dur)
                            else:
                                end_idx = (i + 1) * part_size
                            
                            part_df = technique_df_sorted_dur.iloc[start_idx:end_idx]
                            if len(part_df) > 0:
                                # Select median point based on log Duration (same column used for sorting)
                                median_log_dur = part_df['log_Duration_temp'].median()
                                closest_idx = (part_df['log_Duration_temp'] - median_log_dur).abs().idxmin()
                                selected_indices.add(closest_idx)
                        
                        # Add all selected rows (union of both approaches)
                        for idx in selected_indices:
                            reduced_rows.append(technique_df.loc[idx])
                    else:
                        # Keep all points if ≤30
                        for _, row in technique_df.iterrows():
                            reduced_rows.append(row)
                
                # Create reduced dataframe
                if reduced_rows:
                    dataset_df = pd.DataFrame(reduced_rows)
                    # Recalculate normalized values for reduced dataset
                    if dataset in dataset_max_ad1_auc_dict and dataset_max_ad1_auc_dict[dataset] > 0:
                        normalized_ad1_auc = dataset_df['AD1_AUC'] / dataset_max_ad1_auc_dict[dataset]
                    else:
                        normalized_ad1_auc = dataset_df['AD1_AUC']  # Fallback to raw values
                
                print(f"{len(dataset_df)} points")
            else:
                print(f"No reduction needed for {dataset} - {flavor}: {len(dataset_df)} points")
            
            # Plot points with normalized AD1_AUC on x-axis and duration on y-axis
            for i, (_, row) in enumerate(dataset_df.iterrows()):
                technique = row['Technique']
                x_pos = normalized_ad1_auc.loc[row.name]  # Direct normalized position without jitter
                y_pos = row['Duration']
                
                # Store legend data for each technique we encounter
                if technique not in all_legend_data:
                    all_legend_data[technique] = (color_map[technique], marker_map[technique])
                
                # Single layer: Colored marker with black edge
                scatter_plot = ax.scatter(
                    x_pos, y_pos,
                    color=color_map[technique],
                    marker=marker_map[technique],
                    s=2500,  # Larger single marker
                    alpha=1,
                    edgecolors='black',
                    linewidth=2.5,
                    zorder=6
                )
            
            # Configure subplot
            # Set y-axis to log scale (duration is now on y-axis)
            ax.set_yscale('log')
            
            # Apply Seaborn dark theme background color with 15% opacity
            ax.set_facecolor((0.149, 0.149, 0.149, 0.15))  # #262626 with alpha=0.15
            
            # Set per-dataset y-axis limits (same across row) - DO THIS BEFORE KDE
            if dataset in dataset_duration_ranges:
                min_dur, max_dur = dataset_duration_ranges[dataset]
                ax.set_ylim(min_dur * 0.8, max_dur * 1.2)
            
            # Set per-flavor x-axis limits (same across column) - DO THIS BEFORE KDE
            if flavor in flavor_ad1_auc_ranges:
                min_auc, max_auc = flavor_ad1_auc_ranges[flavor]
                ax.set_xlim(min_auc * 0.95, max_auc * 1.05)
            
            # Add zoom-in inset for BHD dataset in online, sliding, and unsupervised flavors
            # IMPORTANT: This must be done AFTER setting the main subplot axis limits
            if dataset == 'BHD 2023' and flavor in ['Auto profile ', 'Incremental ', 'Unsupervised ']:
                # Calculate the overcrowded region bounds for zoom
                if len(dataset_df) > 0:
                    x_values = [normalized_ad1_auc.loc[row.name] for _, row in dataset_df.iterrows()]
                    y_values = [row['Duration'] for _, row in dataset_df.iterrows()]
                    
                    if flavor in ['Auto profile ', 'Incremental ']:
                        # For online and sliding: include all points with full y-axis alignment
                        zoom_x_min = min(x_values) * 0.95
                        zoom_x_max = max(x_values) * 1.05
                        
                        # Use the SAME y-axis limits as the main subplot to align them
                        subplot_y_min, subplot_y_max = ax.get_ylim()
                        zoom_y_min = subplot_y_min
                        zoom_y_max = subplot_y_max
                        
                        # Create inset axes for zoom - positioned with stretched height for y-axis alignment
                        inset_ax = inset_axes(ax, width="60%", height="85%", loc='lower left', 
                                            bbox_to_anchor=(0.2, 0.02, 1, 1), bbox_transform=ax.transAxes)
                        
                        # Plot all data points in the inset
                        for i, (_, row) in enumerate(dataset_df.iterrows()):
                            technique = row['Technique']
                            x_pos = normalized_ad1_auc.loc[row.name]
                            y_pos = row['Duration']
                            
                            inset_ax.scatter(
                                x_pos, y_pos,
                                color=color_map[technique],
                                marker=marker_map[technique],
                                s=800,  # Smaller markers for inset
                                alpha=0.9,
                                edgecolors='black',
                                linewidth=1.5,
                                zorder=6
                            )
                        
                    elif flavor == 'Unsupervised ':
                        # For unsupervised: focus on low-performance region (x ≤ 0.1)
                        zoom_x_min = 0.0
                        zoom_x_max = 0.1
                        
                        # Filter for points in the low-performance region
                        low_perf_points = [(x, y) for x, y in zip(x_values, y_values) if x <= 0.1]
                        
                        if low_perf_points:
                            # Use data-driven y-range (not aligned with main subplot)
                            low_perf_y_values = [y for x, y in low_perf_points]
                            zoom_y_min = min(low_perf_y_values) * 0.8
                            zoom_y_max = max(low_perf_y_values) * 1.2
                        else:
                            # Fallback if no points in range
                            zoom_y_min = min(y_values) * 0.8
                            zoom_y_max = max(y_values) * 1.2
                        
                        # Create smaller inset axes for unsupervised (increased height by one-fifth)
                        inset_ax = inset_axes(ax, width="50%", height="65%", loc='lower left', 
                                            bbox_to_anchor=(0.25, 0.05, 1, 1), bbox_transform=ax.transAxes)
                        
                        # Plot only points with x ≤ 0.1
                        for i, (_, row) in enumerate(dataset_df.iterrows()):
                            technique = row['Technique']
                            x_pos = normalized_ad1_auc.loc[row.name]
                            y_pos = row['Duration']
                            
                            if x_pos <= 0.1:  # Only plot low-performance points
                                inset_ax.scatter(
                                    x_pos, y_pos,
                                    color=color_map[technique],
                                    marker=marker_map[technique],
                                    s=800,  # Smaller markers for inset
                                    alpha=0.9,
                                    edgecolors='black',
                                    linewidth=1.5,
                                    zorder=6
                                )
                    
                    # Configure inset (common for all flavors)
                    inset_ax.set_xlim(zoom_x_min, zoom_x_max)
                    inset_ax.set_ylim(zoom_y_min, zoom_y_max)
                    inset_ax.set_yscale('log')
                    inset_ax.set_facecolor((0.149, 0.149, 0.149, 0.1))
                    inset_ax.grid(True, alpha=0.3, linewidth=0.5)
                    
                    # Adjust font sizes: smaller x-ticks, larger y-ticks
                    inset_ax.tick_params(axis='x', labelsize=FONT_SIZE*0.5)  # Smaller x-tick labels
                    inset_ax.tick_params(axis='y', labelsize=FONT_SIZE*1.2)  # Larger y-tick labels
                    
                    # Add a border to the inset
                    for spine in inset_ax.spines.values():
                        spine.set_linewidth(2)
                        spine.set_color('red')
            
            # Store current limits before KDE adjustments
            x_lim_original = ax.get_xlim()
            y_lim_original = ax.get_ylim()
            
            # Add KDE plot for log duration distribution inside the subplot on the right side
            if len(full_dataset_df) > 1:
                try:
                    # Calculate KDE for log duration using FULL dataset
                    log_dur_values = np.log10(full_dataset_df['Duration'].values)
                    kde_dur = gaussian_kde(log_dur_values)
                    
                    # Get current axis limits
                    x_min_curr, x_max_curr = ax.get_xlim()
                    y_min_curr, y_max_curr = ax.get_ylim()
                    
                    # Calculate KDE values
                    log_dur_range = np.linspace(np.log10(y_min_curr), np.log10(y_max_curr), 200)
                    kde_values = kde_dur(log_dur_range)
                    dur_range = 10**log_dur_range  # Convert back to linear scale for plotting
                    
                    # Normalize KDE values to fit within the plot (use 12% of x-axis range to avoid overlap)
                    kde_max = kde_values.max()
                    x_range = x_max_curr - x_min_curr
                    kde_width = 0.12 * x_range
                    kde_normalized = (kde_values / kde_max) * kde_width
                    
                    # Position KDE further to the right to avoid overlap with points (starting at 1.06 beyond axis)
                    kde_x_offset = x_max_curr + 0.06 * x_range
                    kde_x_values = kde_x_offset + kde_normalized
                    
                    # Add histogram
                    hist_counts, hist_edges = np.histogram(log_dur_values, bins=20, range=(np.log10(y_min_curr), np.log10(y_max_curr)))
                    hist_bin_centers = (hist_edges[:-1] + hist_edges[1:]) / 2
                    hist_bin_centers_linear = 10**hist_bin_centers  # Convert back to linear scale
                    hist_width = 0.12 * x_range  # Histogram width (same as KDE)
                    hist_normalized = (hist_counts / hist_counts.max()) * hist_width if hist_counts.max() > 0 else hist_counts
                    hist_x_offset = kde_x_offset  # Same position as KDE
                    hist_x_values = hist_x_offset + hist_normalized
                    
                    # Plot histogram bars (underneath KDE, same position)
                    bin_height = hist_edges[1] - hist_edges[0]  # Height of each bin in log space
                    bin_height_linear = 10**(hist_edges[1:]) - 10**(hist_edges[:-1])  # Height in linear space
                    for i, (count, center, height) in enumerate(zip(hist_counts, hist_bin_centers_linear, bin_height_linear)):
                        if count > 0:
                            bottom = 10**hist_edges[i]
                            top = 10**hist_edges[i+1]
                            width = hist_normalized[i]
                            ax.fill_betweenx([bottom, top], hist_x_offset, hist_x_offset + width, 
                                           color='#87CEEB', alpha=0.6, zorder=1, edgecolor='black', linewidth=0.5)
                    
                    # Plot KDE with fill and curve (on top of histogram, same position)
                    # Fill area under the curve with dark blue color
                    ax.fill_betweenx(dur_range, kde_x_offset, kde_x_values, 
                                   color='#00008B', alpha=0.25, zorder=3)
                    # Plot the KDE curve with dark blue outline
                    ax.plot(kde_x_values, dur_range, color='#00008B', alpha=1, linewidth=2, zorder=4)
                    
                    # Extend x-axis limits to accommodate both histogram and KDE
                    max_x = max(hist_x_values.max(), kde_x_values.max())
                    ax.set_xlim(x_min_curr, max_x * 1.05)
                    
                except Exception as e:
                    # Skip KDE if it fails
                    print(f"KDE failed for {dataset} - {flavor}: {e}")
            
            ax.grid(True, alpha=0.3, linewidth=1)
            
            # Add column headers (flavors) for bottom row only - below x-axis
            if dataset_idx == len(all_datasets) - 1:  # Last row
                # Add flavor name below the x-axis label
                ax.text(0.5, -0.35, f'{formal_flavor_name_map[flavor]}', 
                       fontsize=2*FONT_SIZE, ha='center', va='top', transform=ax.transAxes)
            
            # Add row labels (datasets) for first column only - rotated
            if flavor_idx == 0:
                ax.set_ylabel(f'{dataset}', 
                             fontsize=FONT_SIZE*1.5, rotation=90, ha='center', va='center', labelpad=30)
                ax.tick_params(axis='y', labelsize=FONT_SIZE*0.9)
            else:
                # Remove y-axis labels for other columns
                ax.set_yticklabels([])
                ax.tick_params(axis='y', which='both', left=False)
            
            # Add "Duration (s)" label on the right side for last column
            if flavor_idx == len(flavors_main) - 1:  # Last column
                ax2 = ax.twinx()
                ax2.set_ylabel('Duration (s)', fontsize=FONT_SIZE*0.9, rotation=270, ha='center', va='center', labelpad=50)
                ax2.set_yticks([])  # Remove ticks on the right y-axis
            
            # X-axis: show labels only for bottom row
            if dataset_idx == len(all_datasets) - 1:  # Last row
                ax.set_xlabel('Normalized AD1 AUC PR', fontsize=FONT_SIZE*1.3)
                ax.tick_params(axis='x', labelsize=FONT_SIZE*1.2)
                # Set custom x-axis ticks for the last row
                ax.set_xticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
            else:
                # Hide x-axis labels for other rows
                ax.set_xticklabels([])
                ax.tick_params(axis='x', which='both', bottom=True, labelbottom=False)
            
            # Add pink arrows pointing to bottom right corner with "better" text for first row
            if dataset_idx == 0:  # First row only
                # Get current axis limits
                x_min_curr, x_max_curr = ax.get_xlim()
                y_min_curr, y_max_curr = ax.get_ylim()
                
                # Position arrow from absolute x value 0.8 to 1.0, with y between 10^3 and 10^4, ending lower
                arrow_x_start = 0.7
                arrow_x_end = 1.0
                arrow_y_start = 10**4.5  # Start between 10^3 and 10^4
                arrow_y_end = 10**3    # End lower (around 316)
                
                # Add pink arrow pointing to bottom right with increased mutation scale and filled arrowhead
                from matplotlib.patches import FancyArrowPatch
                arrow = FancyArrowPatch((arrow_x_start, arrow_y_start), (arrow_x_end, arrow_y_end),
                                      transform=ax.transData,
                                      arrowstyle='-|>', 
                                      color='hotpink', 
                                      linewidth=15,
                                      mutation_scale=140,
                                      zorder=10,
                                      clip_on=False)
                ax.add_patch(arrow)
                
                # Add "better" text without rotation, positioned 0.1 to the right of midpoint with larger font
                text_x = (arrow_x_start + arrow_x_end) / 2 + 0.1
                text_y = (arrow_y_start + arrow_y_end) / 3
                ax.text(text_x, text_y,
                       'better', fontsize=FONT_SIZE*1.8, color='hotpink', weight='bold',
                       ha='center', va='bottom', zorder=10, transform=ax.transData)
            
            # Add red horizontal arrows from zoom boxes to main plot for BHD dataset
            if dataset == 'BHD 2023' and flavor in ['Auto profile ', 'Incremental ', 'Unsupervised ']:
                if len(dataset_df) > 0:
                    if flavor in ['Auto profile ', 'Incremental ']:
                        # For online and sliding: arrow from LEFT edge of zoom box pointing left to concentrated area
                        zoom_left_edge = 0.2  # Left edge of zoom box (bbox_to_anchor x position)
                        arrow_start = zoom_left_edge - 0.02  # Start just outside the left edge
                        main_plot_target = 0.04 # Point to very left edge where points are concentrated
                        arrow_y = 0.5  # Middle of the plot
                        
                        # Add horizontal red arrow with increased visibility pointing left and filled arrowhead
                        from matplotlib.patches import FancyArrowPatch
                        arrow = FancyArrowPatch((arrow_start, arrow_y), (main_plot_target, arrow_y),
                                              transform=ax.transAxes,
                                              arrowstyle='-|>', 
                                              color='red', 
                                              linewidth=15,
                                              mutation_scale=120,
                                              zorder=100,
                                              clip_on=False)
                        ax.add_patch(arrow)
                        
                    elif flavor == 'Unsupervised ':
                        # For unsupervised: arrow from LEFT edge of zoom box pointing left to concentrated area
                        zoom_left_edge = 0.2  # Left edge of zoom box (bbox_to_anchor x position)
                        arrow_start = zoom_left_edge - 0.02  # Start just outside the left edge
                        main_plot_target = 0.04  # Point to very left edge where points are concentrated
                        arrow_y = 0.4  # Slightly lower for unsupervised
                        
                        # Add horizontal red arrow with increased visibility pointing left and filled arrowhead
                        from matplotlib.patches import FancyArrowPatch
                        arrow = FancyArrowPatch((arrow_start, arrow_y), (main_plot_target, arrow_y),
                                              transform=ax.transAxes,
                                              arrowstyle='-|>', 
                                              color='red', 
                                              linewidth=15,
                                              mutation_scale=120,
                                              zorder=100,
                                              clip_on=False)
                        ax.add_patch(arrow)
    
    # Create comprehensive legend for main figure
    legend_handles = []
    legend_labels = []
    for technique in sorted(all_legend_data.keys()):
        color, marker = all_legend_data[technique]
        handle = plt.Line2D([0], [0], marker=marker, color='w', markerfacecolor=color, 
                          markersize=55, markeredgecolor='black', markeredgewidth=3, label=technique)
        legend_handles.append(handle)
        legend_labels.append(technique)
    
    # Place legend at the top above the subplots in a single horizontal line, centered
    # Using bbox_transform to place it relative to the figure, not the axes
    fig_main.legend(legend_handles, legend_labels, 
              bbox_to_anchor=(0.425, 0.995), loc='upper center', 
              fontsize=FONT_SIZE*1.1, ncol=len(legend_handles), frameon=True, 
              edgecolor='black', fancybox=False, shadow=False,
              bbox_transform=fig_main.transFigure)

    plt.tight_layout()
    plt.subplots_adjust(top=0.97, right=0.85, hspace=0.05, wspace=0.05)
    
    # Save the main figure
    fig_main.savefig("effectiveness_runtime_main_analysis.pdf", format='pdf', bbox_inches='tight', dpi=300)
    fig_main.savefig("effectiveness_runtime_main_analysis.jpeg", format='jpeg', bbox_inches='tight', dpi=300)
    print("Saved main effectiveness-runtime analysis: effectiveness_runtime_main_analysis.pdf")
    print("Saved main effectiveness-runtime analysis: effectiveness_runtime_main_analysis.jpeg")

    # plt.show()
    
    # Create separate figure for historical (semisupervised) data
    fig_historical = plt.figure(figsize=(60, 30))
    
    flavor = 'Semisupervised '
    current_df = df[df['Flavor'] == flavor].copy()
    print(f"Historical {flavor}: {current_df.shape}")
    
    # Get datasets for historical flavor
    flavor_datasets = datasets_per_flavor[flavor]
    
    if len(current_df) > 0:
        # Create subplots for historical datasets
        for dataset_idx, dataset in enumerate(flavor_datasets):
            # Calculate subplot position (2 rows, 1 column)
            ax = plt.subplot(2, 1, dataset_idx + 1)
            
            dataset_df = current_df[current_df['Dataset'] == dataset].copy()
            
            if len(dataset_df) == 0:
                ax.text(0.5, 0.5, 'not\navailable', ha='center', va='center', 
                       transform=ax.transAxes, fontsize=FONT_SIZE*0.8, color='red', rotation=90,
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
                ax.set_title(dataset, fontsize=FONT_SIZE*0.2)
                ax.set_xticks([])
                ax.set_yticks([])
                continue
            
            # Sort by AD1_AUC for left-to-right ordering
            dataset_df = dataset_df.sort_values('AD1_AUC')
            
            # Normalize AD1_AUC values by dataset maximum
            if dataset in dataset_max_ad1_auc_dict and dataset_max_ad1_auc_dict[dataset] > 0:
                normalized_ad1_auc = dataset_df['AD1_AUC'] / dataset_max_ad1_auc_dict[dataset]
            else:
                normalized_ad1_auc = dataset_df['AD1_AUC']  # Fallback to raw values
            
            # Store full dataset for KDE calculations before point reduction
            full_dataset_df = dataset_df.copy()
            full_normalized_ad1_auc = normalized_ad1_auc.copy()
            
            # Data point reduction: for combinations with >30 rows, split into ~10 equal parts and keep median points
            if len(dataset_df) > 30:
                print(f"Reducing data points for {dataset} - historical: {len(dataset_df)} -> ", end="")
                
                # Group by technique and apply median selection within each technique
                reduced_rows = []
                for technique in dataset_df['Technique'].unique():
                    technique_df = dataset_df[dataset_df['Technique'] == technique].copy()
                    technique_normalized = normalized_ad1_auc[technique_df.index]
                    
                    if len(technique_df) > 30:
                        selected_indices = set()
                        
                        # Always include max and min AD1_AUC points
                        max_auc_idx = technique_df['AD1_AUC'].idxmax()
                        selected_indices.add(max_auc_idx)
                        min_auc_idx = technique_df['AD1_AUC'].idxmin()
                        selected_indices.add(min_auc_idx)
                        
                        # Always include max and min log duration points
                        technique_df['log_Duration_temp'] = np.log10(technique_df['Duration'])
                        max_log_dur_idx = technique_df['log_Duration_temp'].idxmax()
                        selected_indices.add(max_log_dur_idx)
                        min_log_dur_idx = technique_df['log_Duration_temp'].idxmin()
                        selected_indices.add(min_log_dur_idx)
                        
                        print(f"Always adding max AD1_AUC point for {dataset}-{technique}: {technique_df.loc[max_auc_idx]['AD1_AUC']:.4f}")
                        print(f"Always adding min AD1_AUC point for {dataset}-{technique}: {technique_df.loc[min_auc_idx]['AD1_AUC']:.4f}")
                        print(f"Always adding max log duration point for {dataset}-{technique}: {technique_df.loc[max_log_dur_idx]['log_Duration_temp']:.4f}")
                        print(f"Always adding min log duration point for {dataset}-{technique}: {technique_df.loc[min_log_dur_idx]['log_Duration_temp']:.4f}")
                        
                        # FIRST: Sort by AD1_AUC and split into ~10 parts, pick median based on AD1_AUC
                        technique_df_sorted_auc = technique_df.sort_values('AD1_AUC').copy()
                        n_parts = 10
                        part_size = len(technique_df_sorted_auc) // n_parts
                        if part_size == 0:
                            part_size = 1
                            n_parts = len(technique_df_sorted_auc)
                        
                        for i in range(n_parts):
                            start_idx = i * part_size
                            if i == n_parts - 1:  # Last part takes remaining rows
                                end_idx = len(technique_df_sorted_auc)
                            else:
                                end_idx = (i + 1) * part_size
                            
                            part_df = technique_df_sorted_auc.iloc[start_idx:end_idx]
                            if len(part_df) > 0:
                                # Select median point based on AD1_AUC (same column used for sorting)
                                median_auc = part_df['AD1_AUC'].median()
                                closest_idx = (part_df['AD1_AUC'] - median_auc).abs().idxmin()
                                selected_indices.add(closest_idx)
                        
                        # SECOND: Sort by log-scaled Duration and split into ~10 parts, pick median based on log Duration
                        technique_df['log_Duration_temp'] = np.log10(technique_df['Duration'])
                        technique_df_sorted_dur = technique_df.sort_values('log_Duration_temp').copy()
                        
                        for i in range(n_parts):
                            start_idx = i * part_size
                            if i == n_parts - 1:  # Last part takes remaining rows
                                end_idx = len(technique_df_sorted_dur)
                            else:
                                end_idx = (i + 1) * part_size
                            
                            part_df = technique_df_sorted_dur.iloc[start_idx:end_idx]
                            if len(part_df) > 0:
                                # Select median point based on log Duration (same column used for sorting)
                                median_log_dur = part_df['log_Duration_temp'].median()
                                closest_idx = (part_df['log_Duration_temp'] - median_log_dur).abs().idxmin()
                                selected_indices.add(closest_idx)
                        
                        # Add all selected rows (union of both approaches)
                        for idx in selected_indices:
                            reduced_rows.append(technique_df.loc[idx])
                    else:
                        # Keep all points if ≤30
                        for _, row in technique_df.iterrows():
                            reduced_rows.append(row)
                
                # Create reduced dataframe
                if reduced_rows:
                    dataset_df = pd.DataFrame(reduced_rows)
                    # Recalculate normalized values for reduced dataset
                    if dataset in dataset_max_ad1_auc_dict and dataset_max_ad1_auc_dict[dataset] > 0:
                        normalized_ad1_auc = dataset_df['AD1_AUC'] / dataset_max_ad1_auc_dict[dataset]
                    else:
                        normalized_ad1_auc = dataset_df['AD1_AUC']  # Fallback to raw values
                
                print(f"{len(dataset_df)} points")
            else:
                print(f"No reduction needed for {dataset} - historical: {len(dataset_df)} points")
            
            # Plot points with normalized AD1_AUC on x-axis and duration on y-axis
            
            for i, (_, row) in enumerate(dataset_df.iterrows()):
                technique = row['Technique']
                x_pos = normalized_ad1_auc.loc[row.name]  # Direct normalized position without jitter
                y_pos = row['Duration']
                
                # Single layer: Colored marker with black edge
                scatter_plot = ax.scatter(
                    x_pos, y_pos,
                    color=color_map[technique],
                    marker=marker_map[technique],
                    s=5000,  # Larger single marker
                    alpha=1,
                    edgecolors='black',
                    linewidth=3,
                    zorder=6
                )
            
            # Configure subplot
            # Set y-axis to log scale (duration is now on y-axis)
            ax.set_yscale('log')
            
            # Apply Seaborn dark theme background color with 15% opacity
            ax.set_facecolor((0.149, 0.149, 0.149, 0.15))  # #262626 with alpha=0.15
            
            # Set per-dataset y-axis limits for historical subplots - DO THIS BEFORE KDE
            if dataset in hist_dataset_duration_ranges:
                min_dur, max_dur = hist_dataset_duration_ranges[dataset]
                ax.set_ylim(min_dur * 0.8, max_dur * 1.2)
            
            # Set per-flavor x-axis limits for historical subplots - DO THIS BEFORE KDE
            if flavor in hist_flavor_ad1_auc_ranges:
                min_auc, max_auc = hist_flavor_ad1_auc_ranges[flavor]
                ax.set_xlim(min_auc * 0.95, max_auc * 1.05)
            
            # Store current limits before KDE adjustments
            x_lim_original = ax.get_xlim()
            y_lim_original = ax.get_ylim()
            
            # Add KDE plot for log duration distribution inside the subplot on the right side
            if len(full_dataset_df) > 1:
                try:
                    # Calculate KDE for log duration using FULL dataset
                    log_dur_values = np.log10(full_dataset_df['Duration'].values)
                    kde_dur = gaussian_kde(log_dur_values)
                    
                    # Get current axis limits
                    x_min_curr, x_max_curr = ax.get_xlim()
                    y_min_curr, y_max_curr = ax.get_ylim()
                    
                    # Calculate KDE values
                    log_dur_range = np.linspace(np.log10(y_min_curr), np.log10(y_max_curr), 200)
                    kde_values = kde_dur(log_dur_range)
                    dur_range = 10**log_dur_range  # Convert back to linear scale for plotting
                    
                    # Normalize KDE values to fit within the plot (use 12% of x-axis range to avoid overlap)
                    kde_max = kde_values.max()
                    x_range = x_max_curr - x_min_curr
                    kde_width = 0.12 * x_range
                    kde_normalized = (kde_values / kde_max) * kde_width
                    
                    # Position KDE further to the right to avoid overlap with points (starting at 1.06 beyond axis)
                    kde_x_offset = x_max_curr + 0.06 * x_range
                    kde_x_values = kde_x_offset + kde_normalized
                    
                    # Add histogram
                    hist_counts, hist_edges = np.histogram(log_dur_values, bins=20, range=(np.log10(y_min_curr), np.log10(y_max_curr)))
                    hist_bin_centers = (hist_edges[:-1] + hist_edges[1:]) / 2
                    hist_bin_centers_linear = 10**hist_bin_centers  # Convert back to linear scale
                    hist_width = 0.12 * x_range  # Histogram width (same as KDE)
                    hist_normalized = (hist_counts / hist_counts.max()) * hist_width if hist_counts.max() > 0 else hist_counts
                    hist_x_offset = kde_x_offset  # Same position as KDE
                    hist_x_values = hist_x_offset + hist_normalized
                    
                    # Plot histogram bars (underneath KDE, same position)
                    bin_height = hist_edges[1] - hist_edges[0]  # Height of each bin in log space
                    bin_height_linear = 10**(hist_edges[1:]) - 10**(hist_edges[:-1])  # Height in linear space
                    for i, (count, center, height) in enumerate(zip(hist_counts, hist_bin_centers_linear, bin_height_linear)):
                        if count > 0:
                            bottom = 10**hist_edges[i]
                            top = 10**hist_edges[i+1]
                            width = hist_normalized[i]
                            ax.fill_betweenx([bottom, top], hist_x_offset, hist_x_offset + width, 
                                           color='#87CEEB', alpha=0.6, zorder=1, edgecolor='black', linewidth=0.5)
                    
                    # Plot KDE with fill and curve (on top of histogram, same position)
                    # Fill area under the curve with dark blue color
                    ax.fill_betweenx(dur_range, kde_x_offset, kde_x_values, 
                                   color='#00008B', alpha=0.25, zorder=3)
                    # Plot the KDE curve with dark blue outline
                    ax.plot(kde_x_values, dur_range, color='#00008B', alpha=1, linewidth=3, zorder=4)
                    
                    # Extend x-axis limits to accommodate both histogram and KDE
                    max_x = max(hist_x_values.max(), kde_x_values.max())
                    ax.set_xlim(x_min_curr, max_x * 1.05)
                    
                except Exception as e:
                    # Skip KDE if it fails
                    print(f"KDE failed for {dataset} - historical: {e}")
            
            ax.grid(True, alpha=0.3, linewidth=2)
            
            # Add dataset name on the left side
            ax.set_ylabel(f'{dataset}', 
                         fontsize=FONT_SIZE*1.5, rotation=90, ha='center', va='center', labelpad=30)
            ax.tick_params(axis='y', labelsize=FONT_SIZE*1.5)
            
            # Add "Duration (s)" label on the right side
            ax2 = ax.twinx()
            ax2.set_ylabel('Duration (s)', fontsize=FONT_SIZE*1.5, rotation=270, ha='center', va='center', labelpad=50)
            ax2.set_yticks([])  # Remove ticks on the right y-axis
            
            # X-axis: show labels for bottom row only
            if dataset_idx == len(flavor_datasets) - 1:  # Last row
                ax.set_xlabel('Normalized AD1 AUC PR', fontsize=FONT_SIZE*1.3)
                ax.tick_params(axis='x', labelsize=FONT_SIZE*1.2)
                # Set custom x-axis ticks for the last row
                ax.set_xticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
            else:
                # Hide x-axis labels for other rows
                ax.set_xticklabels([])
                ax.tick_params(axis='x', which='both', bottom=True, labelbottom=False)
            
            # Add pink arrows pointing to bottom right corner with "better" text for first row
            if dataset_idx == 0:  # First row only
                # Get current axis limits
                x_min_curr, x_max_curr = ax.get_xlim()
                y_min_curr, y_max_curr = ax.get_ylim()
                
                # Position arrow from absolute x value 0.8 to 1.0, with y between 10^3 and 10^4, ending lower
                arrow_x_start = 0.7
                arrow_x_end = 0.95
                arrow_y_start = 10**3.5  # Start between 10^3 and 10^4
                arrow_y_end = 10**2    # End lower (around 316)
                
                # Add pink arrow pointing to bottom right with increased mutation scale and filled arrowhead
                from matplotlib.patches import FancyArrowPatch
                arrow = FancyArrowPatch((arrow_x_start, arrow_y_start), (arrow_x_end, arrow_y_end),
                                      transform=ax.transData,
                                      arrowstyle='-|>', 
                                      color='hotpink', 
                                      linewidth=15,
                                      mutation_scale=140,
                                      zorder=10,
                                      clip_on=False)
                ax.add_patch(arrow)
                
                # Add "better" text without rotation, positioned 0.1 to the right of midpoint with larger font
                text_x = (arrow_x_start + arrow_x_end) / 2 + 0.05
                text_y = (arrow_y_start + arrow_y_end) / 3
                ax.text(text_x, text_y,
                       'better', fontsize=FONT_SIZE*2.2, color='hotpink', weight='bold',
                       ha='center', va='bottom', zorder=10, transform=ax.transData)
    
    # Add legend to historical figure spanning two lines
    # Calculate how many techniques to put on each line for roughly equal distribution
    total_techniques = len(legend_handles)
    techniques_per_line = (total_techniques + 1) // 2  # Round up for first line
    
    # Place legend at the top above the subplots in two horizontal lines, centered
    legend = fig_historical.legend(legend_handles, legend_labels, 
                         bbox_to_anchor=(0.5, 0.99), loc='upper center', 
                         fontsize=FONT_SIZE*1.5, ncol=techniques_per_line, frameon=True, 
                         edgecolor='black', fancybox=False, shadow=False,
                         bbox_transform=fig_historical.transFigure)
    
    # Make the legend more transparent
    legend.get_frame().set_alpha(0.5)

    # Add flavor name as overall title
    # fig_historical.suptitle(f'{formal_flavor_name_map[flavor]}', fontsize=2*FONT_SIZE, y=0.98)

    plt.tight_layout()
    plt.subplots_adjust(right=0.85, hspace=0.05, wspace=0.05, top=0.94)
    
    # Save the historical figure
    fig_historical.savefig("effectiveness_runtime_historical_analysis.pdf", format='pdf', bbox_inches='tight', dpi=300)
    fig_historical.savefig("effectiveness_runtime_historical_analysis.jpeg", format='jpeg', bbox_inches='tight', dpi=300)
    print("Saved historical effectiveness-runtime analysis: effectiveness_runtime_historical_analysis.pdf")
    print("Saved historical effectiveness-runtime analysis: effectiveness_runtime_historical_analysis.jpeg")

    # plt.show()
    
    # total += df[df['Technique'] == 'XGBOOST'].shape[0] # xgboost is missing from above as it is only supervised

    # assert total == len(df)

    #  are the distributions using the full data