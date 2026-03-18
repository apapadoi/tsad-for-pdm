# Copyright 2026 Anastasios Papadopoulos, Apostolos Giannoulidis
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pandas as pd
import numpy as np

def generate_latex_tables():
    # Load data
    df = pd.read_csv('data_analysis_runtime.csv')
    df = df[~df['Technique'].isin(['ADD', 'ALL', 'DummyAllAlarms', 'DummyAddAlarms'])]

    # Mappings
    tech_map = {
        'TimeLLMPyPots': 'TimeLLM \\cite{jin2023time}',
        'TIMEMIXERPP': 'TimeMixer++ \\cite{wang2024timemixer++}',
        'AutoGluon_chronos2': 'CHRONOS-2 \\cite{ansari2025chronos}'
    }

    flavor_map = {
        'Auto profile': 'online',
        'Incremental': 'sliding',
        'Semisupervised': 'historical',
        'Unsupervised': 'unsupervised'
    }
    
    # Filter for relevant techniques
    # Create a separate df for the specific techniques, but keep original df for Oracle
    df_filtered = df[df['Technique'].isin(tech_map.keys())].copy()

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
    # Apply mapping to BOTH dataframes
    df['Dataset'] = df['Dataset'].map(dataset_mapping_dict)
    df_filtered['Dataset'] = df_filtered['Dataset'].map(dataset_mapping_dict)

    # Clean flavor names (remove trailing whitespace just in case)
    df['Flavor'] = df['Flavor'].str.strip()
    df_filtered['Flavor'] = df_filtered['Flavor'].str.strip()

    # Apply Log Scale to Duration
    # Handle 0 or negative durations if any (though unlikely for runtime)
    # Adding a small epsilon or just masking
    # Replace 0 or negative values with NaN or a small number before log
    df['Duration'] = df['Duration'].replace(0, np.nan)
    df_filtered['Duration'] = df_filtered['Duration'].replace(0, np.nan)

    df['Duration_Log'] = np.log10(df['Duration'])
    df_filtered['Duration_Log'] = np.log10(df_filtered['Duration'])

    # Determine Flavors for Main Table (Not Semisupervised)
    # For TimeLLM and TimeMixer, we want the non-semisupervised flavor which is 'Auto profile'
    # For Chronos2, it is 'Unsupervised'
    
    # selection mask for Table 1
    # We want rows where Flavor is NOT 'Semisupervised'
    mask_main = df_filtered['Flavor'] != 'Semisupervised'
    df_main = df_filtered[mask_main].copy()

    
    # Determine the Single Flavor per technique for the header
    # We expect one non-semi flavor per technique in this filtered view
    tech_flavors_main = {}
    for t in ['TimeLLMPyPots', 'TIMEMIXERPP', 'AutoGluon_chronos2']:
        flavors = df_main[df_main['Technique'] == t]['Flavor'].unique()
        if len(flavors) > 0:
            tech_flavors_main[t] = flavor_map.get(flavors[0], flavors[0])
        else:
            tech_flavors_main[t] = 'N/A'

    # Datasets - sort them alphabetically
    datasets = sorted(df_main['Dataset'].unique())
    
    # Techniques order for columns - Sort alphabetically by Mapped Name
    # Available techniques in this df
    available_techs = df_main['Technique'].unique()
    # Sort based on tech_map value
    tech_order_main = sorted(available_techs, key=lambda x: tech_map.get(x, x))

    # Generate Table 1
    # We need to pass the full dataframe to calculate ORACLE and Median
    # But Oracle should be calculated from techniques NOT in TimeLLM, TimeMixer++, Chronos2
    excluded_oracle_techs = ['TimeLLMPyPots', 'TIMEMIXERPP', 'AutoGluon_chronos2']
    generate_latex(df_main, datasets, tech_order_main, tech_map, tech_flavors_main, "Experimental results for two LLM-based detectors and TimeMixer++ (a general time series pattern machine) with single configurations. The best result per dataset and metric is highlighted in \\textbf{{bold}}, and the second-best is \\underline{{underlined}}.", full_df=df, excluded_oracle_techs=excluded_oracle_techs)

    # Table 2: Semisupervised
    mask_semi = df_filtered['Flavor'] == 'Semisupervised'
    df_semi = df_filtered[mask_semi].copy()
    
    # Available techniques in semi df
    available_semi_techs = df_semi['Technique'].unique()
    tech_order_semi = sorted(available_semi_techs, key=lambda x: tech_map.get(x, x))

    # Use only datasets present in this df
    datasets_semi = sorted(df_semi['Dataset'].unique())
    
    tech_flavors_semi = {t: flavor_map.get('Semisupervised', 'Historical') for t in tech_order_semi}

    print("\n\n% --- Semisupervised Table ---\n")
    generate_latex(df_semi, datasets_semi, tech_order_semi, tech_map, tech_flavors_semi, "Experimental results for TimeLLM and TimeMixer++ (a general time series pattern machine) with the historical flavor and single configurations. The best result per dataset and metric is highlighted in \\textbf{{bold}}, and the second-best is \\underline{{underlined}}.", full_df=df, excluded_oracle_techs=excluded_oracle_techs)


def generate_latex(df, datasets, tech_order, tech_map, tech_flavors, caption, full_df=None, excluded_oracle_techs=None):
    # Prepare header
    # \begin{tabular}{l | ccc | ccc ... | ccc | ccc}
    
    n_metrics = 3
    # Add 2 more groups for ORACLE and Median
    # The tech_order cols
    single_block = "|".join(["c"] * n_metrics)
    tech_cols = " | ".join([single_block for _ in tech_order])
    # Total col def
    col_def = f"|l | {tech_cols} | {single_block} | {single_block}|"
    
    print(f"\\begin{{table*}}[ht]")
    print(f"\\centering")
    print(f"\\caption{{{caption}}}")
    print(f"\\resizebox{{\\textwidth}}{{!}}{{")
    print(f"\\begin{{tabular}}{{{col_def}}}")
    print(f"\\hline")
    
    # Row 1: Technique Names
    header_1 = "Dataset"
    for tech in tech_order:
        tech_name = tech_map.get(tech, tech)
        header_1 += f" & \\multicolumn{{{n_metrics}}}{{c|}}{{\\textbf{{{tech_name}}}}}"
    
    # Add Oracle and Median headers
    header_1 += f" & \\multicolumn{{{n_metrics}}}{{c|}}{{\\textbf{{ORACLE}}}}"
    header_1 += f" & \\multicolumn{{{n_metrics}}}{{c|}}{{\\textbf{{\\makecell{{Median across the core \\\\ analysis TSAD techniques \\\\ (with their best configuration \\\\ w.r.t. AD1 AUC-PR)}}}}}}"
    
    print(header_1 + " \\\\")
    
    # Row 2: Flavor Names
    header_2 = ""
    for i, tech in enumerate(tech_order):
        flavor = tech_flavors.get(tech, "")
        if flavor:
            flavor = f"\\textit{{{flavor} flavor}}"
        header_2 += f" & \\multicolumn{{{n_metrics}}}{{c|}}{{{flavor}}}"
    
    # Add Oracle and Median flavor placeholders (empty or descriptive)
    header_2 += f" & \\multicolumn{{{n_metrics}}}{{c|}}{{}}"
    header_2 += f" & \\multicolumn{{{n_metrics}}}{{c|}}{{}}"

    print(f" {header_2} \\\\")

    print(f"\\hline")
    
    # Row 3: Metric Names (AD1_AUC, VUS_PR, Duration)
    # AD1_AUC, VUS_PR, and Duration columns
    # Duration is Log Scale
    metrics_header = " & AD1 AUC-PR & \\textit{VUS-PR} & \\makecell{{Log \\\\ duration (s)}}" * len(tech_order)
    # Add for Oracle and Median
    metrics_header += " & AD1 AUC-PR & \\textit{VUS-PR} & \\makecell{{Log \\\\ duration (s)}}" * 2

    print(f" {metrics_header} \\\\")
    print(f"\\hline")
    
    # Prep for ORACLE and Median calculation

    # Data Rows
    for ds in datasets:
        row_str = f"{ds}"
        
        # --- Collect Technique Data first to determine ranks ---
        tech_data = []
        auc_values = []
        vus_values = []
        
        # Calculate Oracle/Median first before ranking
        oracle_auc = None
        oracle_vus = None
        oracle_dur = None
        median_auc = None
        median_vus = None
        median_dur = None
        
        if full_df is not None:
             # Filter for Oracle calculation: Exclude specific techniques
            if excluded_oracle_techs:
                ds_data_oracle = full_df[(full_df['Dataset'] == ds) & (~full_df['Technique'].isin(excluded_oracle_techs))].copy()
            else:
                 ds_data_oracle = full_df[full_df['Dataset'] == ds].copy()
            
            ds_data_median = ds_data_oracle 

            if not ds_data_oracle.empty:
                # Oracle
                if 'Duration_Log' not in ds_data_oracle.columns:
                     ds_data_oracle['Duration_Log'] = np.log10(ds_data_oracle['Duration'])

                best_idx = ds_data_oracle['AD1_AUC'].idxmax()
                oracle_auc = ds_data_oracle.loc[best_idx, 'AD1_AUC']
                oracle_vus = ds_data_oracle.loc[best_idx, 'VUS_PR']
                oracle_dur = ds_data_oracle.loc[best_idx, 'Duration_Log']

                # --- Median Calculation ---
                # For each technique, find the row with the best AD1_AUC
                # Then take the metrics from that specific row
                # We need to group by Technique and pick the best index
                
                # Get the index of the max AD1_AUC for each technique
                idx_best_per_tech = ds_data_median.groupby('Technique')['AD1_AUC'].idxmax()
                
                # Select the rows corresponding to these best indices
                best_rows = ds_data_median.loc[idx_best_per_tech]
                
                median_auc = best_rows['AD1_AUC'].median()
                median_vus = best_rows['VUS_PR'].median()
                median_dur = best_rows['Duration_Log'].median()
        
        # Add to values list for ranking
        if oracle_auc is not None:
             auc_values.append(round(oracle_auc, 3))
        if median_auc is not None:
             auc_values.append(round(median_auc, 3))
             
        if oracle_vus is not None:
             vus_values.append(round(oracle_vus, 3))
        if median_vus is not None:
             vus_values.append(round(median_vus, 3))

        for tech in tech_order:
            entry = df[(df['Dataset'] == ds) & (df['Technique'] == tech)]
            
            if entry.empty:
                tech_data.append(None)
            else:
                # Get raw values
                raw_auc = entry['AD1_AUC'].values[0]
                raw_vus = entry['VUS_PR'].values[0]
                raw_dur = entry['Duration_Log'].values[0]
                
                # Round for comparison consistency with display (which uses .3f)
                val_auc = round(raw_auc, 3)
                val_vus = round(raw_vus, 3)
                
                tech_data.append({
                    'auc': val_auc,
                    'vus': val_vus, 
                    'dur': raw_dur
                })
                
                auc_values.append(val_auc)
                vus_values.append(val_vus)
        
        # Determine unique ranked values (descending)
        unique_auc = sorted(list(set(auc_values)), reverse=True)
        unique_vus = sorted(list(set(vus_values)), reverse=True)
        
        best_auc = unique_auc[0] if len(unique_auc) > 0 else None
        second_auc = unique_auc[1] if len(unique_auc) > 1 else None
        
        best_vus = unique_vus[0] if len(unique_vus) > 0 else None
        second_vus = unique_vus[1] if len(unique_vus) > 1 else None

        # --- Build Technique Columns String ---
        for data in tech_data:
            if data is None:
                row_str += " & - & - & -"
            else:
                # AD1_AUC Formatting
                auc_str = f"{data['auc']:.3f}"
                if best_auc is not None and data['auc'] == best_auc:
                    auc_str = f"\\textbf{{{auc_str}}}"
                elif second_auc is not None and data['auc'] == second_auc:
                    auc_str = f"\\underline{{{auc_str}}}"
                
                # VUS_PR Formatting (Base Italics + Highlight)
                vus_str = f"\\textit{{{data['vus']:.3f}}}"
                if best_vus is not None and data['vus'] == best_vus:
                    vus_str = f"\\textbf{{{vus_str}}}"
                elif second_vus is not None and data['vus'] == second_vus:
                    vus_str = f"\\underline{{{vus_str}}}"
                
                if pd.isna(data['dur']):
                    dur_str = "-"
                else:
                    dur_str = f"{data['dur']:.2f}"
                
                row_str += f" & {auc_str} & {vus_str} & {dur_str}"
        
        # --- ORACLE & Median Output ---
        if oracle_auc is not None:
            # Format Oracle
            o_auc_str = f"{oracle_auc:.3f}"
            if best_auc is not None and round(oracle_auc, 3) == best_auc:
                o_auc_str = f"\\textbf{{{o_auc_str}}}"
            elif second_auc is not None and round(oracle_auc, 3) == second_auc:
                o_auc_str = f"\\underline{{{o_auc_str}}}"
                
            o_vus_str = f"\\textit{{{oracle_vus:.3f}}}"
            if best_vus is not None and round(oracle_vus, 3) == best_vus:
                o_vus_str = f"\\textbf{{{o_vus_str}}}"
            elif second_vus is not None and round(oracle_vus, 3) == second_vus:
                o_vus_str = f"\\underline{{{o_vus_str}}}"
            
            if pd.isna(oracle_dur):
                o_dur_str = "-"
            else:
                o_dur_str = f"{oracle_dur:.2f}"
                
            row_str += f" & {o_auc_str} & {o_vus_str} & {o_dur_str}"
            
            # Format Median
            m_auc_str = f"{median_auc:.3f}"
            if best_auc is not None and round(median_auc, 3) == best_auc:
                m_auc_str = f"\\textbf{{{m_auc_str}}}"
            elif second_auc is not None and round(median_auc, 3) == second_auc:
                m_auc_str = f"\\underline{{{m_auc_str}}}"

            m_vus_str = f"\\textit{{{median_vus:.3f}}}"
            if best_vus is not None and round(median_vus, 3) == best_vus:
                m_vus_str = f"\\textbf{{{m_vus_str}}}"
            elif second_vus is not None and round(median_vus, 3) == second_vus:
                m_vus_str = f"\\underline{{{m_vus_str}}}"
            
            if pd.isna(median_dur):
                m_dur_str = "-"
            else:
                m_dur_str = f"{median_dur:.2f}"

            row_str += f" & {m_auc_str} & {m_vus_str} & {m_dur_str}"
                
        else:
             row_str += " & - & - & - & - & - & -"

        print(row_str + " \\\\")
        
    print(f"\\hline")
    print(f"\\end{{tabular}}")
    print(f"}}")
    print(f"\\end{{table*}}")

if __name__ == "__main__":
    generate_latex_tables()
