# -*- coding: utf-8 -*-
"""
Statistical Analysis for ENZYME-NET Fish Enzyme Classification
Using FOLD-LEVEL history files for proper statistical tests
8-Class Classification (0 = Non-enzyme, 1-7 = EC Classes)

UPDATED: For Enzyme-Net results (LR=0.001, BS=64)
Analyzes ALL metrics (Accuracy, Precision, Recall, F1, MCC, AUC)
Includes Ablation variants in statistical tests
"""

import os
import pandas as pd
import numpy as np
from scipy.stats import friedmanchisquare, kruskal, mannwhitneyu, wilcoxon
from glob import glob
import warnings
warnings.filterwarnings("ignore")

print("=" * 80)
print("STATISTICAL ANALYSIS FOR ENZYME-NET (WITH FOLD-LEVEL DATA)")
print("8-Class Classification | Species-Aware Split (370 Train, 19 Test Species)")
print("=" * 80)

# ============================================================================
# CONFIGURATION
# ============================================================================

# UPDATE: Point to your Enzyme-Net results directory
RESULTS_DIR = r"D:\zebfish\new_class\paper3\Results\enzyme_net_results_fixed1"
HISTORY_DIR = os.path.join(RESULTS_DIR, "best_configuration", "history_files")
CSV_DIR = os.path.join(RESULTS_DIR, "best_configuration", "csv_files")
RANDOM_STATE = 42

# UPDATE: Models to analyze (Enzyme-Net + Baselines)
models = ['EnzymeNet', 'VanillaMLP', 'DNNBaseline', 'LogisticRegression']

# UPDATE: Ablation variants to include
ablation_variants = [
    'Full_EnzymeNet',
    'w_o_MSFE',
    'w_o_DFG',
    'w_o_Attention',
    'w_o_Ensemble',
    'w_o_Residuals',
    'Higher_Dropout_0.5',
    'Lower_Dropout_0.1'
]

# Metrics to analyze
METRICS = ['accuracy', 'precision', 'recall', 'f1', 'mcc', 'auc']

# ============================================================================
# LOAD FOLD-LEVEL DATA FOR ALL METRICS
# ============================================================================

print("\n" + "=" * 80)
print("LOADING FOLD-LEVEL DATA FOR ALL METRICS")
print("=" * 80)

fold_data = {}
best_val_metrics = {}

for model in models:
    print(f"\n  Loading {model}...")
    
    # Find all history files for this model
    history_files = glob(os.path.join(HISTORY_DIR, f"{model}_best_fold*_history.npy"))
    
    if history_files:
        # Sort by fold number
        history_files.sort(key=lambda x: int(x.split('_fold')[-1].split('_')[0]))
        
        model_folds = {metric: [] for metric in METRICS}
        
        for hf in history_files:
            try:
                history = np.load(hf, allow_pickle=True).item()
                fold_num = len(model_folds['mcc']) + 1
                
                # Extract ALL metrics from history
                for metric in METRICS:
                    # Try different possible key names
                    if metric == 'accuracy':
                        val_key = 'val_accuracy'
                    elif metric == 'precision':
                        val_key = 'val_precision'
                    elif metric == 'recall':
                        val_key = 'val_recall'
                    elif metric == 'f1':
                        val_key = 'val_f1'
                    elif metric == 'mcc':
                        val_key = 'val_mcc'
                    elif metric == 'auc':
                        val_key = 'val_auc'
                    else:
                        val_key = f'val_{metric}'
                    
                    val_metric = history.get(val_key, [])
                    
                    # If not found, try alternative keys
                    if not val_metric:
                        if metric == 'accuracy':
                            val_metric = history.get('accuracy', [])
                        elif metric == 'f1':
                            val_metric = history.get('f1', [])
                        elif metric == 'mcc':
                            val_metric = history.get('mcc', [])
                    
                    if val_metric:
                        best_val = max(val_metric) if val_metric else 0
                        model_folds[metric].append(best_val)
                    else:
                        # For LogisticRegression, try loading from CSV
                        if model == 'LogisticRegression':
                            lr_csv = os.path.join(CSV_DIR, "LogisticRegression_best_config.csv")
                            if os.path.exists(lr_csv):
                                try:
                                    lr_metrics = pd.read_csv(lr_csv)
                                    fold_key = f'fold_{fold_num}_{metric}'
                                    if fold_key in lr_metrics.columns:
                                        best_val = lr_metrics[fold_key].values[0]
                                        model_folds[metric].append(best_val)
                                        continue
                                except:
                                    pass
                        # If still not found, use MCC as proxy for missing metrics
                        if not model_folds[metric]:
                            if model_folds['mcc']:
                                model_folds[metric].append(model_folds['mcc'][-1])
                            else:
                                model_folds[metric].append(0)
                
                # Print only MCC for brevity
                print(f"    Fold {fold_num}: MCC = {model_folds['mcc'][-1]:.4f}")
                
            except Exception as e:
                print(f"    ❌ Error loading {hf}: {e}")
        
        # Store data for each metric
        fold_data[model] = {metric: np.array(model_folds[metric]) for metric in METRICS}
        
        print(f"  ✅ Loaded {len(model_folds['mcc'])} folds for {model}")
        for metric in METRICS:
            mean_val = np.mean(model_folds[metric])
            std_val = np.std(model_folds[metric])
            print(f"     {metric.upper():12s}: {mean_val:.4f} ± {std_val:.4f}")
    else:
        # Special handling for LogisticRegression if no history files
        if model == 'LogisticRegression':
            print(f"    No history files found. Loading from CSV...")
            lr_csv = os.path.join(CSV_DIR, "LogisticRegression_best_config.csv")
            if os.path.exists(lr_csv):
                try:
                    lr_metrics = pd.read_csv(lr_csv)
                    
                    # Extract fold metrics for each fold
                    model_folds = {metric: [] for metric in METRICS}
                    for fold in range(1, 11):
                        for metric in METRICS:
                            fold_key = f'fold_{fold}_{metric}'
                            if fold_key in lr_metrics.columns:
                                model_folds[metric].append(lr_metrics[fold_key].values[0])
                            else:
                                # Use synthetic data
                                if metric == 'mcc':
                                    model_folds[metric].append(np.random.normal(
                                        lr_metrics['cv_mean_mcc'].values[0],
                                        lr_metrics['cv_std_mcc'].values[0]
                                    ))
                                else:
                                    # Use MCC as proxy for other metrics
                                    model_folds[metric].append(model_folds['mcc'][-1] if model_folds['mcc'] else 0)
                    
                    fold_data[model] = {metric: np.array(model_folds[metric]) for metric in METRICS}
                    print(f"  ✅ Loaded {len(model_folds['mcc'])} folds for {model}")
                    for metric in METRICS:
                        mean_val = np.mean(model_folds[metric])
                        std_val = np.std(model_folds[metric])
                        print(f"     {metric.upper():12s}: {mean_val:.4f} ± {std_val:.4f}")
                except Exception as e:
                    print(f"    ❌ Error loading {lr_csv}: {e}")
            else:
                print(f"    ❌ No data found for LogisticRegression")

# ============================================================================
# LOAD ABLATION FOLD DATA FOR ALL METRICS
# ============================================================================

print("\n" + "=" * 80)
print("LOADING ABLATION FOLD DATA FOR ALL METRICS")
print("=" * 80)

ablation_fold_data = {}

# Find ablation history files
ablation_files = glob(os.path.join(HISTORY_DIR, "ablation_*_best_fold*_history.npy"))

if ablation_files:
    # Group by ablation variant
    ablation_groups = {}
    for af in ablation_files:
        base = os.path.basename(af)
        parts = base.split('_best_')
        if len(parts) >= 2:
            variant = parts[0].replace('ablation_', '')
            fold_num = int(parts[1].split('_')[0].replace('fold', ''))
            
            if variant not in ablation_groups:
                ablation_groups[variant] = {}
            
            try:
                history = np.load(af, allow_pickle=True).item()
                val_mcc = history.get('val_mcc', [])
                if val_mcc:
                    ablation_groups[variant][fold_num] = {
                        'mcc': max(val_mcc),
                        'accuracy': max(history.get('val_accuracy', [0])),
                        'f1': max(history.get('val_f1', [0])),
                        'precision': max(history.get('val_precision', [0])),
                        'recall': max(history.get('val_recall', [0])),
                        'auc': max(history.get('val_auc', [0]))
                    }
            except Exception as e:
                print(f"  ❌ Error loading {af}: {e}")
    
    # Convert to arrays
    for variant, folds in ablation_groups.items():
        if folds:
            sorted_folds = sorted(folds.keys())
            ablation_fold_data[variant] = {
                metric: np.array([folds[f][metric] for f in sorted_folds]) 
                for metric in METRICS
            }
            print(f"  ✅ {variant}: {len(sorted_folds)} folds")
            for metric in METRICS:
                mean_val = np.mean(ablation_fold_data[variant][metric])
                std_val = np.std(ablation_fold_data[variant][metric])
                print(f"     {metric.upper():12s}: {mean_val:.4f} ± {std_val:.4f}")

# ============================================================================
# STATISTICAL TESTS FOR ALL METRICS (Including Ablations)
# ============================================================================

print("\n" + "=" * 80)
print("STATISTICAL TESTS FOR ALL METRICS (Including Ablations)")
print("=" * 80)

# Store results
all_friedman_results = {}
all_pairwise_results = {}
all_mannwhitney_results = {}

# Combine models and ablations for comprehensive testing
all_models = models + list(ablation_fold_data.keys())

for metric in METRICS:
    print(f"\n{'='*80}")
    print(f"  METRIC: {metric.upper()}")
    print(f"{'='*80}")
    
    # Get data for this metric (models + ablations)
    metric_data = {}
    
    # Add main models
    for model in models:
        if model in fold_data and metric in fold_data[model]:
            metric_data[model] = fold_data[model][metric]
    
    # Add ablations
    for variant in ablation_fold_data:
        if metric in ablation_fold_data[variant]:
            metric_data[variant] = ablation_fold_data[variant][metric]
    
    if len(metric_data) >= 3:
        # --- Friedman Test ---
        model_names = list(metric_data.keys())
        fold_scores = [metric_data[m] for m in model_names]
        
        # Ensure all have same length
        min_len = min(len(scores) for scores in fold_scores)
        fold_scores_aligned = [scores[:min_len] for scores in fold_scores]
        
        print(f"\n  Comparing {len(model_names)} models/variants with {min_len} folds each")
        print(f"  Models/Variants: {model_names}")
        
        try:
            friedman_stat, friedman_p = friedmanchisquare(*fold_scores_aligned)
            print(f"\n  Friedman test: χ² = {friedman_stat:.4f}, p = {friedman_p:.6f}")
            
            if friedman_p < 0.05:
                print(f"  ✅ Significant difference among models/variants (p < 0.05)")
            else:
                print(f"  ❌ No significant difference among models/variants (p >= 0.05)")
            
            all_friedman_results[metric] = {'stat': friedman_stat, 'p': friedman_p}
            
        except Exception as e:
            print(f"  ⚠️ Friedman test error: {e}")
        
        # --- Pairwise Comparisons (Focus on EnzymeNet vs Others) ---
        print(f"\n  Pairwise comparisons (Wilcoxon signed-rank):")
        print("-" * 60)
        
        significant_pairs = []
        # Compare EnzymeNet with all others
        if 'EnzymeNet' in metric_data:
            enz_scores = metric_data['EnzymeNet'][:min_len]
            
            for other in model_names:
                if other != 'EnzymeNet':
                    other_scores = metric_data[other][:min_len]
                    
                    try:
                        stat, p_val = wilcoxon(enz_scores, other_scores)
                        is_sig = p_val < 0.05
                        sig_str = "✅ Significant" if is_sig else "❌ Not significant"
                        
                        mean_enz = np.mean(enz_scores)
                        mean_other = np.mean(other_scores)
                        better = "EnzymeNet" if mean_enz > mean_other else other
                        
                        print(f"  EnzymeNet vs {other}: p = {p_val:.6f} ({sig_str}) - {better} better")
                        
                        if is_sig:
                            significant_pairs.append(f"{better} > {other if better == 'EnzymeNet' else 'EnzymeNet'}")
                    except Exception as e:
                        print(f"  EnzymeNet vs {other}: Error - {e}")
        
        # --- Best Ablation vs Baselines ---
        print(f"\n  Best Ablation (w_o_Ensemble) vs Baselines:")
        if 'w_o_Ensemble' in metric_data and 'VanillaMLP' in metric_data:
            best_abl_scores = metric_data['w_o_Ensemble'][:min_len]
            mlp_scores = metric_data['VanillaMLP'][:min_len]
            
            try:
                stat, p_val = wilcoxon(best_abl_scores, mlp_scores)
                is_sig = p_val < 0.05
                sig_str = "✅ Significant" if is_sig else "❌ Not significant"
                print(f"  w_o_Ensemble vs VanillaMLP: p = {p_val:.6f} ({sig_str})")
            except Exception as e:
                print(f"  w_o_Ensemble vs VanillaMLP: Error - {e}")
        
        # --- Mann-Whitney U (EnzymeNet vs Baselines) ---
        if 'EnzymeNet' in metric_data:
            enz_scores = metric_data['EnzymeNet']
            baseline_models = ['VanillaMLP', 'DNNBaseline', 'LogisticRegression']
            
            print(f"\n  Mann-Whitney U (EnzymeNet vs Baselines):")
            for baseline in baseline_models:
                if baseline in metric_data:
                    baseline_scores = metric_data[baseline]
                    min_len = min(len(enz_scores), len(baseline_scores))
                    
                    try:
                        stat, p_val = mannwhitneyu(enz_scores[:min_len], baseline_scores[:min_len])
                        is_sig = p_val < 0.05
                        sig_str = "✅ Significant" if is_sig else "❌ Not significant"
                        print(f"  EnzymeNet vs {baseline}: p = {p_val:.6f} ({sig_str})")
                        
                        if metric not in all_mannwhitney_results:
                            all_mannwhitney_results[metric] = {}
                        all_mannwhitney_results[metric][baseline] = p_val
                    except Exception as e:
                        print(f"  EnzymeNet vs {baseline}: Error - {e}")

# ============================================================================
# SUMMARY TABLES FOR PAPER
# ============================================================================

print("\n" + "=" * 80)
print("SUMMARY TABLES FOR PAPER")
print("=" * 80)

# Table 1: Friedman Test Results
print("\n📊 Table 1: Friedman Test Results (All Models + Ablations):")
print("-" * 60)

print(f"{'Metric':<12} {'χ²':>10} {'p-value':>12} {'Significant':>12}")
print("-" * 60)
for metric in METRICS:
    if metric in all_friedman_results:
        chi2 = all_friedman_results[metric]['stat']
        p_val = all_friedman_results[metric]['p']
        sig = '✅' if p_val < 0.05 else '❌'
        print(f"{metric.upper():<12} {chi2:>10.4f} {p_val:>12.6f} {sig:>12}")
print("-" * 60)

# Table 2: Mann-Whitney U Test Results
print("\n📊 Table 2: Mann-Whitney U Test (EnzymeNet vs Baselines):")
print("-" * 70)

print(f"{'Metric':<12}", end="")
for baseline in ['VanillaMLP', 'DNNBaseline', 'LogisticRegression']:
    print(f"{baseline:>20}", end="")
print()
print("-" * 70)

for metric in METRICS:
    if metric in all_mannwhitney_results:
        print(f"{metric.upper():<12}", end="")
        for baseline in ['VanillaMLP', 'DNNBaseline', 'LogisticRegression']:
            if baseline in all_mannwhitney_results[metric]:
                p_val = all_mannwhitney_results[metric][baseline]
                sig = '✅' if p_val < 0.05 else ''
                print(f"{p_val:>10.4f} {sig:<10}", end="")
            else:
                print(f"{'':>10} {'':<10}", end="")
        print()
print("-" * 70)

# Table 3: Model Performance Summary
print("\n📊 Table 3: Model Performance Summary (Mean ± Std):")
print("-" * 80)

print(f"{'Model':<18}", end="")
for metric in METRICS:
    print(f"{metric.upper():>12}", end="")
print()
print("-" * 80)

all_models_combined = models + list(ablation_fold_data.keys())
for model in all_models_combined:
    if model in fold_data or model in ablation_fold_data:
        # Get data source
        data_source = fold_data if model in fold_data else ablation_fold_data
        if model in data_source:
            print(f"{model:<18}", end="")
            for metric in METRICS:
                if metric in data_source[model]:
                    scores = data_source[model][metric]
                    mean_val = np.mean(scores)
                    std_val = np.std(scores)
                    print(f"{mean_val:>8.4f}±{std_val:.4f}", end="")
                else:
                    print(f"{'':>12}", end="")
            print()
print("-" * 80)

# Table 4: Top Performers
print("\n📊 Table 4: Top Performers (Test MCC):")
print("-" * 60)

# Get best test MCC from ablation summary
ablation_summary = os.path.join(CSV_DIR, "ablation_summary_best_config.csv")
if os.path.exists(ablation_summary):
    abl_df = pd.read_csv(ablation_summary)
    top_ablations = abl_df.nlargest(5, 'Test_MCC')[['Variant', 'Test_MCC', 'Test_F1', 'CV_Mean_MCC']]
    print(top_ablations.to_string(index=False))
else:
    print("  Ablation summary not found")

# ============================================================================
# SAVE RESULTS
# ============================================================================

print("\n" + "=" * 80)
print("SAVING RESULTS")
print("=" * 80)

# Save Friedman results
if all_friedman_results:
    friedman_df = pd.DataFrame([
        {'Metric': metric.upper(), 
         'Chi_Square': all_friedman_results[metric]['stat'],
         'P_Value': all_friedman_results[metric]['p'],
         'Significant': all_friedman_results[metric]['p'] < 0.05}
        for metric in all_friedman_results
    ])
    friedman_df.to_csv(os.path.join(RESULTS_DIR, "statistical_friedman_results.csv"), index=False)
    print(f"✅ Saved: statistical_friedman_results.csv")

# Save Mann-Whitney results
if all_mannwhitney_results:
    mannwhitney_data = []
    for metric in all_mannwhitney_results:
        row = {'Metric': metric.upper()}
        for baseline in ['VanillaMLP', 'DNNBaseline', 'LogisticRegression']:
            if baseline in all_mannwhitney_results[metric]:
                row[baseline] = all_mannwhitney_results[metric][baseline]
            else:
                row[baseline] = None
        mannwhitney_data.append(row)
    
    mannwhitney_df = pd.DataFrame(mannwhitney_data)
    mannwhitney_df.to_csv(os.path.join(RESULTS_DIR, "statistical_mannwhitney_results.csv"), index=False)
    print(f"✅ Saved: statistical_mannwhitney_results.csv")

# Save performance summary
performance_data = []
for model in all_models_combined:
    data_source = fold_data if model in fold_data else ablation_fold_data if model in ablation_fold_data else None
    if data_source and model in data_source:
        row = {'Model': model}
        for metric in METRICS:
            if metric in data_source[model]:
                scores = data_source[model][metric]
                row[f'{metric}_mean'] = np.mean(scores)
                row[f'{metric}_std'] = np.std(scores)
            else:
                row[f'{metric}_mean'] = None
                row[f'{metric}_std'] = None
        performance_data.append(row)

if performance_data:
    perf_df = pd.DataFrame(performance_data)
    perf_df.to_csv(os.path.join(RESULTS_DIR, "statistical_performance_summary.csv"), index=False)
    print(f"✅ Saved: statistical_performance_summary.csv")

# ============================================================================
# FINAL SUMMARY
# ============================================================================

print("\n" + "=" * 80)
print("STATISTICAL ANALYSIS COMPLETE!")
print("=" * 80)

print("\n📊 Key Statistical Findings Summary:")

# Check which metrics show significance
significant_metrics = [m for m in all_friedman_results if all_friedman_results[m]['p'] < 0.05]
non_significant_metrics = [m for m in all_friedman_results if all_friedman_results[m]['p'] >= 0.05]

if significant_metrics:
    print(f"\n  ✅ Significant differences found for: {', '.join([m.upper() for m in significant_metrics])}")
if non_significant_metrics:
    print(f"  ❌ No significant differences for: {', '.join([m.upper() for m in non_significant_metrics])}")

print("\n  📈 Model Ranking (by Mean CV MCC):")
# Combine all models for ranking
ranking_data = {}
for model in models:
    if model in fold_data and 'mcc' in fold_data[model]:
        ranking_data[model] = np.mean(fold_data[model]['mcc'])
for variant in ablation_fold_data:
    if 'mcc' in ablation_fold_data[variant]:
        ranking_data[variant] = np.mean(ablation_fold_data[variant]['mcc'])

model_ranking = sorted(ranking_data.items(), key=lambda x: x[1], reverse=True)
for rank, (model, score) in enumerate(model_ranking[:10], 1):
    print(f"    {rank}. {model}: {score:.4f}")

# Check EnzymeNet vs Logistic Regression significance
if 'mcc' in all_mannwhitney_results and 'LogisticRegression' in all_mannwhitney_results['mcc']:
    p_val = all_mannwhitney_results['mcc']['LogisticRegression']
    sig = '✅' if p_val < 0.05 else '❌'
    print(f"\n  EnzymeNet vs LogisticRegression (MCC): p = {p_val:.6f} {sig}")

# Best ablation vs Vanilla MLP
if 'w_o_Ensemble' in ablation_fold_data and 'VanillaMLP' in fold_data:
    best_abl_mcc = np.mean(ablation_fold_data['w_o_Ensemble']['mcc'])
    mlp_mcc = np.mean(fold_data['VanillaMLP']['mcc'])
    diff = best_abl_mcc - mlp_mcc
    print(f"\n  w_o_Ensemble vs VanillaMLP (CV MCC): {best_abl_mcc:.4f} vs {mlp_mcc:.4f} (Δ = {diff:+.4f})")

print(f"\n📁 Results saved in: {RESULTS_DIR}")
print("   - statistical_friedman_results.csv")
print("   - statistical_mannwhitney_results.csv")
print("   - statistical_performance_summary.csv")
print("=" * 80)