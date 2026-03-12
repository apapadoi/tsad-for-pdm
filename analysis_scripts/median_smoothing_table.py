import mlflow
import pandas as pd
import numpy as np
import sys

# Configuration
mlflow_server_url_list = ["http://127.0.0.1:5000/", "http://127.0.0.1:8080/", "http://127.0.0.1:5001/", "http://localhost:43669/"]

sigmas = [100.0, 1000.0, 10000.0]
techniques = ['IF', 'OCSVM', 'TRANAD']

# Load and process raw data for baseline comparison ---
try:
    raw_df = pd.read_csv('data_analysis_runtime.csv')
    
    # Filter for EDP dataset
    raw_df = raw_df[raw_df['Dataset'].isin(['EDP', 'EDP-WT'])]
    
    # Debug: Check techniques present
    print(f"Debug: Available techniques in EDP: {raw_df['Technique'].unique()}")

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

    best_raw = {}
    for tech in techniques:
        target_substring = ""
        if tech == 'IF': target_substring = "Isolation"
        elif tech == 'OCSVM': target_substring = "OneClassSVM"
        elif tech == 'TRANAD': target_substring = "TranAD"
        
        mask = raw_df['Technique'].str.contains(target_substring, case=False, na=False)
        tech_rows = raw_df[mask]
        
        if not tech_rows.empty:
            max_auc = tech_rows['AD1_AUC'].max()
            max_vus = tech_rows['VUS_PR'].max()
            best_raw[tech] = {'AD1_AUC': max_auc, 'VUS_PR': max_vus}
            print(f"Debug: Found raw baseline for {tech}: AUC={max_auc}, VUS={max_vus}")
        else:
            best_raw[tech] = {'AD1_AUC': 0, 'VUS_PR': 0}
            print(f"Warning: No raw data found for {tech} in EDP dataset using substring '{target_substring}'")

except Exception as e:
    print(f"Error reading raw data: {e}")
    best_raw = {t: {'AD1_AUC': 0, 'VUS_PR': 0} for t in techniques}

results = []

def get_experiment_data(client, exp_name):
    # Try exact match first which is safer
    exp = client.get_experiment_by_name(exp_name)
    experiments = [exp] if exp else []
            
    if not experiments:
        # Fallback to search if exact match fails
        filter_string = f"attribute.name = '{exp_name}'"
        experiments = client.search_experiments(filter_string=filter_string)

    if experiments:
        exp = experiments[0]
        runs = client.search_runs(experiment_ids=[exp.experiment_id])
        if runs:
            # Each experiment should have only one run
            assert len(runs) == 1, f"Expected 1 run for experiment '{exp_name}' (ID: {exp.experiment_id}), found {len(runs)}"
            
            run = runs[0]
            metrics = run.data.metrics
            return metrics.get('AD1_AUC', 0), metrics.get('VUS_VUS_PR', 0)
    return None, None

print("Starting extraction...")

# Collect data
for url in mlflow_server_url_list:
    try:
        print(f"Connecting to {url}...")
        mlflow.set_tracking_uri(url)
        client = mlflow.MlflowClient()
        
        # Check if server is alive by listing experiments (simple check)
        try:
            client.search_experiments(max_results=1)
        except Exception as e:
            print(f"  Skipping {url} - not responsive or error: {e}")
            continue

        for tech in techniques:
            for sigma in sigmas:
                # Base
                base_name = f"Adversarial Noise (sigma={sigma}) Auto profile EDP {tech}"
                
                # Median No MA
                med_noma_name = f"Median Smoothing (sigma={sigma}) (moving_average=False) Auto profile EDP Median_{tech}"
                
                # Median MA
                med_ma_name = f"Median Smoothing (sigma={sigma}) (moving_average=True) Auto profile EDP Median_{tech}"
                
                # Helper to add if not exists
                def add_result(variant, name):
                    # Check if we already have this data point collected from another server
                    already_collected = any((r['tech'] == tech and r['sigma'] == sigma and r['variant'] == variant) for r in results)
                    if not already_collected:
                        try:
                            auc, vus = get_experiment_data(client, name)
                            if auc is not None:
                                results.append({
                                    'tech': tech,
                                    'sigma': sigma,
                                    'variant': variant,
                                    'AD1_AUC': auc,
                                    'VUS_PR': vus
                                })
                                print(f"  Found {name}: AUC={auc:.3f}, VUS={vus:.3f}")
                        except AssertionError as e:
                             print(f"  Assertion failed: {e}")
                             raise e

                add_result('Base', base_name)
                add_result('Median_NoMA', med_noma_name)
                add_result('Median_MA', med_ma_name)
                
    except AssertionError as ae:
        print(f"Critical Error: {ae}")
        exit(1)
    except Exception as e:
        print(f"Error on {url}: {e}")

# Process into DataFrame
df = pd.DataFrame(results)

if df.empty:
    print("No data found!")
    sys.exit(1)

# Helper to format with diff
def fmt_val_diff(val, ref_val, is_vus=False):
    if val == -1: return "-"
    val = float(val)
    ref_val = float(ref_val)
    
    val_str = f"{val:.3f}"
    diff_str = ""
    
    if ref_val > 0:
        diff = val - ref_val
        if diff > 0:
            diff_str = f"(\\textcolor{{green!60!black}}{{+{diff:.3f}}})"
        elif diff < 0:
            diff_str = f"(\\textcolor{{red}}{{-{diff:.3f}}})"
        else:
            diff_str = f"(\\textcolor{{gray}}{{=}})"
            
    return val, diff_str

print("\n--- Generating LaTeX Table ---\n")

print("\\begin{table*}[ht]")
print("\\centering")
print("\\caption{Performance comparison under adversarial noise with and without median smoothing. The maximum value per noise-technique-metric combination is highlighted in bold. Parentheses indicate the difference compared to the best performance on raw (no noise) data for each technique.}")
print("\\resizebox{\\textwidth}{!}{")
print("\\begin{tabular}{|l|l|cc|cc|cc|}")
print("\\hline")
print("\\multirow{2}{*}{Noise ($\\sigma$)} & \\multirow{2}{*}{Technique} & \\multicolumn{2}{c|}{Raw detector on noisy data} & \\multicolumn{2}{c|}{\\makecell{Median Smoothing \\ (No moving average)}} & \\multicolumn{2}{c|}{\\makecell{Median Smoothing \\ (Moving average)}} \\\\")
print(" & & AD1 AUC-PR & \\textit{VUS-PR} & AD1 AUC-PR & \\textit{VUS-PR} & AD1 AUC-PR & \\textit{VUS-PR} \\\\")
print("\\hline")

added_footnotes = set()
footnote_marks = {'IF': 1, 'OCSVM': 2, 'TRANAD': 3}

for sigma in sigmas:
    first_row = True
    for tech in techniques:
        # Get values
        row_str = ""
        sigma_label = f"{sigma}"
        if sigma_label.endswith(".0"):
            sigma_label = sigma_label[:-2] # 10000.0 -> 10000
            
        if first_row:
             row_str += f"\\multirow{{3}}{{*}}{{{sigma_label}}} & "
             first_row = False
        else:
             row_str += " & "
             
        tech_cell = tech
        if tech not in added_footnotes:
             sym = "*" * footnote_marks.get(tech, 1)
             tech_cell += f"\\textsuperscript{{{sym}}}"
             added_footnotes.add(tech)
        
        row_str += tech_cell
        
        vals = {}
        for variant in ['Base', 'Median_NoMA', 'Median_MA']:
            row = df[(df['tech'] == tech) & (df['sigma'] == sigma) & (df['variant'] == variant)]
            if not row.empty:
                vals[variant] = {
                    'AUC': row.iloc[0]['AD1_AUC'],
                    'VUS': row.iloc[0]['VUS_PR']
                }
            else:
                vals[variant] = {'AUC': -1, 'VUS': -1} # Start with -1 for marking missing
        
        # Determine Max
        auc_values = [v['AUC'] for v in vals.values() if v['AUC'] != -1]
        vus_values = [v['VUS'] for v in vals.values() if v['VUS'] != -1]
        
        max_auc = max(auc_values) if auc_values else -1e9
        max_vus = max(vus_values) if vus_values else -1e9
        
        # Build cells per variant
        for variant in ['Base', 'Median_NoMA', 'Median_MA']:
            d = vals[variant]
            if d['AUC'] == -1:
                row_str += " & - & -"
            else:
                actual_auc = d['AUC']
                auc_val, auc_diff = fmt_val_diff(actual_auc, best_raw[tech]['AD1_AUC'])
                auc_str = f"{auc_val:.3f}"
                if actual_auc == max_auc:
                    auc_str = f"\\textbf{{{auc_str}}}"
                
                auc_final = f"{auc_str} {auc_diff}"
                
                actual_vus = d['VUS']
                vus_val, vus_diff = fmt_val_diff(actual_vus, best_raw[tech]['VUS_PR'])
                vus_str = f"{vus_val:.3f}"
                if actual_vus == max_vus:
                    formatted_vus = f"\\textbf{{\\textit{{{vus_str}}}}}"
                else:
                    formatted_vus = f"\\textit{{{vus_str}}}"
                    
                vus_final = f"{formatted_vus} {vus_diff}"
                
                row_str += f" & {auc_final} & {vus_final}"
                
        print(row_str + " \\\\")
    print("\\hline")

print("\\end{tabular}")
print("}")

print("\\par")
print("\\footnotesize")
for tech in techniques:
    sym = "*" * footnote_marks.get(tech, 1)
    raw = best_raw[tech]
    print(f"\\textsuperscript{{{sym}}} Best performance on raw data for {tech}: AD1 AUC-PR: {raw['AD1_AUC']:.3f}, VUS-PR: {raw['VUS_PR']:.3f} \\\\")

print("\\label{tab:median_smoothing}")
print("\\end{table*}")
