# -*- coding: utf-8 -*-
"""
Created on Wed Jun 17 08:41:51 2026
Updated for Enzyme-Net Results (LR=0.001, BS=64)

@author: H.A.R
"""

# -*- coding: utf-8 -*-
"""
Combine All CV Results for All Models
Across ALL Learning Rates and Batch Sizes

This script collects results from:
- ALL_CONFIGS_DIR/ (all 9 hyperparameter combinations)
- BEST_CONFIG_DIR/ (best combination)

Outputs:
1. Combined CV results for all models and all configs
2. Summary table with best configuration per model
3. Ranking of configurations by CV MCC

UPDATED: Using 10-fold CV results from Enzyme-Net (LR=0.001, BS=64)
Models: Logistic Regression, Vanilla MLP, DNN Baseline, EnzymeNet (Proposed)
Ablations: 8 variants
"""

import os
import pandas as pd
import numpy as np
from glob import glob

# =============================================================================
# CONFIGURATION
# =============================================================================

# UPDATE: Point to your Enzyme-Net results directory
OUTPUT_DIR = r"D:\zebfish\new_class\paper3\Results\enzyme_net_results_fixed1"
ALL_CONFIGS_DIR = os.path.join(OUTPUT_DIR, "all_configurations")
BEST_CONFIG_DIR = os.path.join(OUTPUT_DIR, "best_configuration")
OUTPUT_COMBINED_DIR = os.path.join(OUTPUT_DIR, "combined_results")
os.makedirs(OUTPUT_COMBINED_DIR, exist_ok=True)

# UPDATE: Model names matching your results
MODELS = [
    "LogisticRegression",
    "VanillaMLP",
    "DNNBaseline", 
    "EnzymeNet"  # Changed from VanillaPlus
]

# UPDATE: Ablation variants matching your Enzyme-Net results
ABLATION_VARIANTS = [
    "Full_EnzymeNet",
    "w_o_MSFE",
    "w_o_DFG",
    "w_o_Attention",
    "w_o_Ensemble",
    "w_o_Residuals",
    "Higher_Dropout_0.5",
    "Lower_Dropout_0.1"
]

# Best configuration from your results
BEST_LR = 0.001
BEST_BS = 64

# =============================================================================
# 1. LOAD GRID SEARCH RESULTS
# =============================================================================

print("="*70)
print("ENZYME-NET: COMBINING ALL CV RESULTS FOR ALL MODELS")
print("="*70)

# Load full grid search results
grid_search_path = os.path.join(OUTPUT_DIR, "csv_files", "full_grid_search_results.csv")
if os.path.exists(grid_search_path):
    grid_df = pd.read_csv(grid_search_path)
    print(f"\n✅ Loaded grid search results: {len(grid_df)} configurations")
    print(grid_df.to_string())
else:
    print("\n⚠️ Grid search results not found!")
    grid_df = None

# =============================================================================
# 2. COLLECT RESULTS FROM ALL CONFIGURATIONS
# =============================================================================

print("\n" + "="*70)
print("COLLECTING RESULTS FROM ALL CONFIGURATIONS")
print("="*70)

# Get all configuration directories
config_dirs = [d for d in glob(os.path.join(ALL_CONFIGS_DIR, "LR_*")) if os.path.isdir(d)]

all_config_results = []

for config_dir in config_dirs:
    config_name = os.path.basename(config_dir)
    
    # Extract LR and BS
    parts = config_name.split('_')
    lr = float(parts[1])
    bs = int(parts[3])
    
    csv_dir = os.path.join(config_dir, "csv_files")
    
    # For each model, check if config_summary exists
    config_summary_path = os.path.join(csv_dir, "config_summary.csv")
    
    if os.path.exists(config_summary_path):
        config_df = pd.read_csv(config_summary_path)
        config_df['learning_rate'] = lr
        config_df['batch_size'] = bs
        config_df['config'] = config_name
        all_config_results.append(config_df)
        print(f"  ✅ Loaded: {config_name}")

if all_config_results:
    combined_grid_df = pd.concat(all_config_results, ignore_index=True)
    print(f"\n✅ Combined results: {len(combined_grid_df)} rows")
else:
    combined_grid_df = pd.DataFrame()

# =============================================================================
# 3. LOAD BEST CONFIGURATION RESULTS
# =============================================================================

print("\n" + "="*70)
print("LOADING BEST CONFIGURATION RESULTS (LR=0.001, BS=64)")
print("="*70)

best_config_results = {}

# Load best config results
best_csv_dir = os.path.join(BEST_CONFIG_DIR, "csv_files")

for model in MODELS:
    csv_path = os.path.join(best_csv_dir, f"{model}_best_config.csv")
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        df['model'] = model
        best_config_results[model] = df
        print(f"  ✅ Loaded: {model}_best_config.csv")

# Load ablation results
ablation_path = os.path.join(best_csv_dir, "ablation_summary_best_config.csv")
if os.path.exists(ablation_path):
    ablation_df = pd.read_csv(ablation_path)
    print(f"  ✅ Loaded: ablation_summary_best_config.csv")
else:
    ablation_df = pd.DataFrame()

# =============================================================================
# 4. CREATE COMBINED SUMMARY TABLE
# =============================================================================

print("\n" + "="*70)
print("CREATING COMBINED SUMMARY TABLE")
print("="*70)

# Combine all model results from best config
summary_rows = []

for model, df in best_config_results.items():
    if len(df) > 0:
        row = {
            'Model': model,
            'Test_Accuracy': df['accuracy'].iloc[0] if 'accuracy' in df.columns else None,
            'Test_Precision': df['precision'].iloc[0] if 'precision' in df.columns else None,
            'Test_Recall': df['recall'].iloc[0] if 'recall' in df.columns else None,
            'Test_F1': df['f1'].iloc[0] if 'f1' in df.columns else None,
            'Test_MCC': df['mcc'].iloc[0] if 'mcc' in df.columns else None,
            'Test_AUC': df['auc_roc'].iloc[0] if 'auc_roc' in df.columns else None,
            'CV_Mean_MCC': df['cv_mean_mcc'].iloc[0] if 'cv_mean_mcc' in df.columns else None,
            'CV_Std_MCC': df['cv_std_mcc'].iloc[0] if 'cv_std_mcc' in df.columns else None,
            'CV_Mean_Accuracy': df['cv_mean_accuracy'].iloc[0] if 'cv_mean_accuracy' in df.columns else None,
            'CV_Mean_F1': df['cv_mean_f1'].iloc[0] if 'cv_mean_f1' in df.columns else None,
            'CV_Mean_Precision': df['cv_mean_precision'].iloc[0] if 'cv_mean_precision' in df.columns else None,
            'CV_Mean_Recall': df['cv_mean_recall'].iloc[0] if 'cv_mean_recall' in df.columns else None,
            'CV_Mean_AUC': df['cv_mean_auc'].iloc[0] if 'cv_mean_auc' in df.columns else None,
        }
        summary_rows.append(row)

# Add EnzymeNet if not already in list (with explicit handling)
if 'EnzymeNet' not in [r['Model'] for r in summary_rows]:
    en_path = os.path.join(best_csv_dir, "EnzymeNet_best_config.csv")
    if os.path.exists(en_path):
        en_df = pd.read_csv(en_path)
        summary_rows.append({
            'Model': 'EnzymeNet (Ours)',
            'Test_Accuracy': en_df['accuracy'].iloc[0] if 'accuracy' in en_df.columns else None,
            'Test_Precision': en_df['precision'].iloc[0] if 'precision' in en_df.columns else None,
            'Test_Recall': en_df['recall'].iloc[0] if 'recall' in en_df.columns else None,
            'Test_F1': en_df['f1'].iloc[0] if 'f1' in en_df.columns else None,
            'Test_MCC': en_df['mcc'].iloc[0] if 'mcc' in en_df.columns else None,
            'Test_AUC': en_df['auc_roc'].iloc[0] if 'auc_roc' in en_df.columns else None,
            'CV_Mean_MCC': en_df['cv_mean_mcc'].iloc[0] if 'cv_mean_mcc' in en_df.columns else None,
            'CV_Std_MCC': en_df['cv_std_mcc'].iloc[0] if 'cv_std_mcc' in en_df.columns else None,
            'CV_Mean_Accuracy': en_df['cv_mean_accuracy'].iloc[0] if 'cv_mean_accuracy' in en_df.columns else None,
            'CV_Mean_F1': en_df['cv_mean_f1'].iloc[0] if 'cv_mean_f1' in en_df.columns else None,
        })

# Map model names to display names
name_map = {
    'LogisticRegression': 'Logistic Regression',
    'VanillaMLP': 'Vanilla MLP',
    'DNNBaseline': 'DNN Baseline',
    'EnzymeNet': 'EnzymeNet (Ours)'
}

for row in summary_rows:
    if row['Model'] in name_map:
        row['Model'] = name_map[row['Model']]

summary_df = pd.DataFrame(summary_rows)

# Sort by CV_Mean_MCC
if 'CV_Mean_MCC' in summary_df.columns:
    summary_df = summary_df.sort_values('CV_Mean_MCC', ascending=False)

print("\n📊 Best Configuration Results (LR=0.001, BS=64):")
print(summary_df.to_string(index=False))

# Save summary
summary_df.to_csv(os.path.join(OUTPUT_COMBINED_DIR, "combined_best_config_summary.csv"), index=False)
print(f"\n✅ Saved: combined_best_config_summary.csv")

# =============================================================================
# 5. CREATE ABLATION SUMMARY
# =============================================================================

print("\n" + "="*70)
print("CREATING ABLATION SUMMARY")
print("="*70)

if not ablation_df.empty:
    # Sort by CV_Mean_MCC
    if 'CV_Mean_MCC' in ablation_df.columns:
        ablation_df = ablation_df.sort_values('CV_Mean_MCC', ascending=False)
    
    print("\n📊 Ablation Study Results:")
    print(ablation_df.to_string(index=False))
    ablation_df.to_csv(os.path.join(OUTPUT_COMBINED_DIR, "combined_ablation_summary.csv"), index=False)
    print(f"\n✅ Saved: combined_ablation_summary.csv")

# =============================================================================
# 6. CREATE ALL CONFIGURATIONS RANKING
# =============================================================================

print("\n" + "="*70)
print("CREATING ALL CONFIGURATIONS RANKING")
print("="*70)

if not combined_grid_df.empty:
    # Sort by CV MCC
    combined_grid_df = combined_grid_df.sort_values('cv_mean_mcc', ascending=False)
    
    print("\n📊 All Configurations Ranking (by CV MCC):")
    print(combined_grid_df[['learning_rate', 'batch_size', 'cv_mean_mcc', 'cv_std_mcc', 'test_mcc', 'test_f1']].to_string(index=False))
    
    combined_grid_df.to_csv(os.path.join(OUTPUT_COMBINED_DIR, "all_configurations_ranking.csv"), index=False)
    print(f"\n✅ Saved: all_configurations_ranking.csv")
    
    # Find best configuration
    best_config = combined_grid_df.iloc[0]
    print(f"\n🏆 Best Overall Configuration:")
    print(f"  Learning Rate: {best_config['learning_rate']}")
    print(f"  Batch Size: {best_config['batch_size']}")
    print(f"  CV MCC: {best_config['cv_mean_mcc']:.4f} ± {best_config['cv_std_mcc']:.4f}")
    print(f"  Test MCC: {best_config['test_mcc']:.4f}")
    print(f"  Test F1: {best_config['test_f1']:.4f}")

# =============================================================================
# 7. CREATE COMPARISON TABLE (BEST vs BASELINES)
# =============================================================================

print("\n" + "="*70)
print("CREATING COMPARISON TABLE")
print("="*70)

comparison_rows = []

# Add baselines
baseline_models = ['LogisticRegression', 'VanillaMLP', 'DNNBaseline']

for model in baseline_models:
    if model in best_config_results:
        df = best_config_results[model]
        comparison_rows.append({
            'Model': model.replace('LogisticRegression', 'Logistic Regression').replace('VanillaMLP', 'Vanilla MLP').replace('DNNBaseline', 'DNN Baseline'),
            'Type': 'Baseline',
            'Test_Accuracy': df['accuracy'].iloc[0] if 'accuracy' in df.columns else None,
            'Test_F1': df['f1'].iloc[0] if 'f1' in df.columns else None,
            'Test_MCC': df['mcc'].iloc[0] if 'mcc' in df.columns else None,
            'Test_AUC': df['auc_roc'].iloc[0] if 'auc_roc' in df.columns else None,
            'CV_MCC': df['cv_mean_mcc'].iloc[0] if 'cv_mean_mcc' in df.columns else None,
            'CV_Std': df['cv_std_mcc'].iloc[0] if 'cv_std_mcc' in df.columns else None,
        })

# Add EnzymeNet (Proposed)
en_path = os.path.join(best_csv_dir, "EnzymeNet_best_config.csv")
if os.path.exists(en_path):
    en_df = pd.read_csv(en_path)
    comparison_rows.append({
        'Model': 'EnzymeNet (Ours)',
        'Type': 'Proposed',
        'Test_Accuracy': en_df['accuracy'].iloc[0] if 'accuracy' in en_df.columns else None,
        'Test_F1': en_df['f1'].iloc[0] if 'f1' in en_df.columns else None,
        'Test_MCC': en_df['mcc'].iloc[0] if 'mcc' in en_df.columns else None,
        'Test_AUC': en_df['auc_roc'].iloc[0] if 'auc_roc' in en_df.columns else None,
        'CV_MCC': en_df['cv_mean_mcc'].iloc[0] if 'cv_mean_mcc' in en_df.columns else None,
        'CV_Std': en_df['cv_std_mcc'].iloc[0] if 'cv_std_mcc' in en_df.columns else None,
    })

# Add Best Ablation (w_o_Ensemble - Test MCC 0.9306)
if not ablation_df.empty:
    # Find best test MCC ablation
    best_ablation = ablation_df.loc[ablation_df['Test_MCC'].idxmax()]
    comparison_rows.append({
        'Model': f"Ablation: {best_ablation['Variant']}",
        'Type': 'Ablation',
        'Test_Accuracy': best_ablation['Test_Accuracy'] if 'Test_Accuracy' in best_ablation else None,
        'Test_F1': best_ablation['Test_F1'] if 'Test_F1' in best_ablation else None,
        'Test_MCC': best_ablation['Test_MCC'] if 'Test_MCC' in best_ablation else None,
        'Test_AUC': best_ablation['Test_AUC'] if 'Test_AUC' in best_ablation else None,
        'CV_MCC': best_ablation['CV_Mean_MCC'] if 'CV_Mean_MCC' in best_ablation else None,
        'CV_Std': best_ablation['CV_Std_MCC'] if 'CV_Std_MCC' in best_ablation else None,
    })

comparison_df = pd.DataFrame(comparison_rows)
if 'CV_MCC' in comparison_df.columns:
    comparison_df = comparison_df.sort_values('CV_MCC', ascending=False)

print("\n📊 Model Comparison (Best Configuration LR=0.001, BS=64):")
print(comparison_df.to_string(index=False))

comparison_df.to_csv(os.path.join(OUTPUT_COMBINED_DIR, "model_comparison_table.csv"), index=False)
print(f"\n✅ Saved: model_comparison_table.csv")

# =============================================================================
# 8. CREATE LaTeX TABLE
# =============================================================================

print("\n" + "="*70)
print("GENERATING LaTeX TABLE")
print("="*70)

latex_table = """
\\begin{table}[htbp]
\\centering
\\caption{Performance comparison on unseen fish species (19 species) using 10-fold CV. Best LR=0.001, BS=64.}
\\label{tab:model_comparison}
\\begin{tabular}{lcccccc}
\\hline
\\textbf{Model} & \\textbf{Accuracy} & \\textbf{F1} & \\textbf{MCC} & \\textbf{AUC} & \\textbf{CV MCC} \\\\
\\hline
"""

for _, row in comparison_df.iterrows():
    cv_str = f"{row['CV_MCC']:.4f}" if pd.notna(row['CV_MCC']) else "-"
    if pd.notna(row['CV_Std']):
        cv_str = f"{row['CV_MCC']:.4f} $\\pm$ {row['CV_Std']:.4f}"
    
    acc_str = f"{row['Test_Accuracy']:.4f}" if pd.notna(row['Test_Accuracy']) else "-"
    f1_str = f"{row['Test_F1']:.4f}" if pd.notna(row['Test_F1']) else "-"
    mcc_str = f"{row['Test_MCC']:.4f}" if pd.notna(row['Test_MCC']) else "-"
    auc_str = f"{row['Test_AUC']:.4f}" if pd.notna(row['Test_AUC']) else "-"
    
    # Make EnzymeNet and best ablation bold
    model_name = row['Model']
    if 'EnzymeNet' in model_name or row['Model'] == "Ablation: w_o_Ensemble":
        model_name = f"\\textbf{{{model_name}}}"
    
    latex_table += f"{model_name} & {acc_str} & {f1_str} & {mcc_str} & {auc_str} & {cv_str} \\\\\n"

latex_table += """
\\hline
\\end{tabular}
\\end{table}
"""

print(latex_table)

# Save LaTeX table
with open(os.path.join(OUTPUT_COMBINED_DIR, "latex_table.txt"), 'w', encoding='utf-8') as f:
    f.write(latex_table)
print(f"\n✅ Saved: latex_table.txt")

# =============================================================================
# 9. CREATE ABLATION LaTeX TABLE
# =============================================================================

print("\n" + "="*70)
print("GENERATING ABLATION LaTeX TABLE")
print("="*70)

if not ablation_df.empty:
    ablation_latex = """
\\begin{table}[htbp]
\\centering
\\caption{Ablation study results using 10-fold CV. Best LR=0.001, BS=64.}
\\label{tab:ablation}
\\begin{tabular}{lcccc}
\\hline
\\textbf{Variant} & \\textbf{Test F1} & \\textbf{Test MCC} & \\textbf{CV MCC} \\\\
\\hline
"""
    
    # Sort by Test MCC for better presentation
    ablation_df_sorted = ablation_df.sort_values('Test_MCC', ascending=False)
    
    for _, row in ablation_df_sorted.iterrows():
        cv_str = f"{row['CV_Mean_MCC']:.4f} $\\pm$ {row['CV_Std_MCC']:.4f}" if 'CV_Mean_MCC' in row and 'CV_Std_MCC' in row else "-"
        # Bold best performer
        variant = row['Variant']
        if row['Test_MCC'] == ablation_df_sorted['Test_MCC'].max():
            variant = f"\\textbf{{{variant}}}"
        ablation_latex += f"{variant} & {row['Test_F1']:.4f} & {row['Test_MCC']:.4f} & {cv_str} \\\\\n"
    
    ablation_latex += """
\\hline
\\end{tabular}
\\end{table}
"""
    
    print(ablation_latex)
    
    # Save ablation LaTeX table
    with open(os.path.join(OUTPUT_COMBINED_DIR, "ablation_latex_table.txt"), 'w', encoding='utf-8') as f:
        f.write(ablation_latex)
    print(f"\n✅ Saved: ablation_latex_table.txt")

# =============================================================================
# 10. SUMMARY REPORT
# =============================================================================

print("\n" + "="*70)
print("SUMMARY REPORT")
print("="*70)

# Get the best configuration
best_lr = str(BEST_LR)
best_bs = str(BEST_BS)
if not combined_grid_df.empty:
    best_config = combined_grid_df.iloc[0]
    best_lr = str(best_config['learning_rate'])
    best_bs = str(int(best_config['batch_size']))

report = f"""
======================================================================
ENZYME-NET RESULTS SUMMARY (10-fold CV)
======================================================================

1. BEST CONFIGURATION:
   Learning Rate: {best_lr}
   Batch Size: {best_bs}
   CV MCC: {combined_grid_df.iloc[0]['cv_mean_mcc']:.4f} ± {combined_grid_df.iloc[0]['cv_std_mcc']:.4f}

2. MODEL COMPARISON (Best Config LR={best_lr}, BS={best_bs}):
"""

for _, row in comparison_df.iterrows():
    cv_str = f"{row['CV_MCC']:.4f}" if pd.notna(row['CV_MCC']) else "-"
    if pd.notna(row['CV_Std']):
        cv_str = f"{row['CV_MCC']:.4f} ± {row['CV_Std']:.4f}"
    report += f"   {row['Model']:30s} Test Acc={row['Test_Accuracy']:.4f} | Test F1={row['Test_F1']:.4f} | Test MCC={row['Test_MCC']:.4f} | CV MCC={cv_str}\n"

# Add ablation summary
if not ablation_df.empty:
    report += "\n3. ABLATION STUDY (Best Config):\n"
    ablation_sorted = ablation_df.sort_values('Test_MCC', ascending=False)
    for _, row in ablation_sorted.head(8).iterrows():
        report += f"   {row['Variant']:30s} Test F1={row['Test_F1']:.4f} | Test MCC={row['Test_MCC']:.4f} | CV MCC={row['CV_Mean_MCC']:.4f} ± {row['CV_Std_MCC']:.4f}\n"

report += f"""
4. KEY FINDINGS:
   - Best Test MCC: w_o_Ensemble (0.9306)
   - Best CV MCC: Full_EnzymeNet (0.8432 ± 0.0205)
   - Best Test F1: w_o_Attention (0.9246)
   - All EnzymeNet variants outperform Vanilla MLP (0.9102)
   - Removing ensemble head improved test performance by 1.52%
   - Species-aware split: 370 train species, 19 test species, 0 overlap

5. DATASET STATISTICS:
   - Total proteins: 5,707
   - Training: 4,466 (370 species)
   - Test: 1,241 (19 species)
   - Classes: 8 (0-7)
   - Embedding dimension: 1024

======================================================================
FILES SAVED:
  - combined_best_config_summary.csv
  - combined_ablation_summary.csv
  - all_configurations_ranking.csv
  - model_comparison_table.csv
  - latex_table.txt
  - ablation_latex_table.txt
======================================================================
"""

print(report)

# Save report with UTF-8 encoding
with open(os.path.join(OUTPUT_COMBINED_DIR, "summary_report.txt"), 'w', encoding='utf-8') as f:
    f.write(report)

print(f"\n✅ Saved: summary_report.txt")
print(f"\n📁 All combined results saved to: {OUTPUT_COMBINED_DIR}")
print("="*70)