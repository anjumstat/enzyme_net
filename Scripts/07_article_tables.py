import pandas as pd
import os

BEST_CSV = r"D:\zebfish\new_class\paper3\Results\enzyme_net_results_fixed1\best_configuration\csv_files"

# ============================================================
# Helper: fetch a column that may go by either name
# ============================================================
def get_col(df, *candidates):
    """Return df[candidates[0]] if present, else try the next name."""
    for name in candidates:
        if name in df.columns:
            return df[name].iloc[0]
    raise KeyError(f"None of {candidates} found. Columns: {df.columns.tolist()}")

# ============================================================
# MAIN MODELS
# ============================================================
print("=" * 90)
print("MAIN MODELS — TEST METRICS")
print("=" * 90)
print(f"{'Model':<22} {'Acc':>8} {'Prec':>8} {'Rec':>8} {'F1':>8} {'MCC':>8} {'AUC':>8}")
print("-" * 90)

for model in ['LogisticRegression', 'VanillaMLP', 'DNNBaseline', 'EnzymeNet']:
    path = os.path.join(BEST_CSV, f"{model}_best_config.csv")
    if os.path.exists(path):
        df = pd.read_csv(path)
        print(f"{model:<22} "
              f"{get_col(df, 'accuracy'):>8.4f} "
              f"{get_col(df, 'precision'):>8.4f} "
              f"{get_col(df, 'recall'):>8.4f} "
              f"{get_col(df, 'f1'):>8.4f} "
              f"{get_col(df, 'mcc'):>8.4f} "
              f"{get_col(df, 'auc_roc', 'auc'):>8.4f}")

print("\n" + "=" * 90)
print("MAIN MODELS — CV METRICS (mean ± std)")
print("=" * 90)
print(f"{'Model':<22} {'Acc':>14} {'Prec':>14} {'Rec':>14} {'F1':>14} {'MCC':>14} {'AUC':>14}")
print("-" * 90)

for model in ['LogisticRegression', 'VanillaMLP', 'DNNBaseline', 'EnzymeNet']:
    path = os.path.join(BEST_CSV, f"{model}_best_config.csv")
    if os.path.exists(path):
        df = pd.read_csv(path)
        print(f"{model:<22} "
              f"{get_col(df, 'cv_mean_accuracy'):>6.4f}±{get_col(df, 'cv_std_accuracy'):.4f} "
              f"{get_col(df, 'cv_mean_precision'):>6.4f}±{get_col(df, 'cv_std_precision'):.4f} "
              f"{get_col(df, 'cv_mean_recall'):>6.4f}±{get_col(df, 'cv_std_recall'):.4f} "
              f"{get_col(df, 'cv_mean_f1'):>6.4f}±{get_col(df, 'cv_std_f1'):.4f} "
              f"{get_col(df, 'cv_mean_mcc'):>6.4f}±{get_col(df, 'cv_std_mcc'):.4f} "
              f"{get_col(df, 'cv_mean_auc_roc', 'cv_mean_auc'):>6.4f}±"
              f"{get_col(df, 'cv_std_auc_roc', 'cv_std_auc'):.4f}")

# ============================================================
# ABLATION VARIANTS
# ============================================================
ablation_variants = [
    'Full_EnzymeNet', 'w_o_MSFE', 'w_o_DFG', 'w_o_Attention',
    'w_o_Ensemble', 'w_o_Residuals',
    'Higher_Dropout_0.5', 'Lower_Dropout_0.1'
]

print("\n" + "=" * 90)
print("ABLATION VARIANTS — TEST METRICS")
print("=" * 90)
print(f"{'Variant':<35} {'Acc':>8} {'Prec':>8} {'Rec':>8} {'F1':>8} {'MCC':>8} {'AUC':>8}")
print("-" * 90)

for label in ablation_variants:
    path = os.path.join(BEST_CSV, f"ablation_{label}_best_config.csv")
    if os.path.exists(path):
        df = pd.read_csv(path)
        print(f"{label:<35} "
              f"{get_col(df, 'accuracy'):>8.4f} "
              f"{get_col(df, 'precision'):>8.4f} "
              f"{get_col(df, 'recall'):>8.4f} "
              f"{get_col(df, 'f1'):>8.4f} "
              f"{get_col(df, 'mcc'):>8.4f} "
              f"{get_col(df, 'auc_roc', 'auc'):>8.4f}")

print("\n" + "=" * 90)
print("ABLATION VARIANTS — CV METRICS (mean ± std)")
print("=" * 90)
print(f"{'Variant':<35} {'Acc':>14} {'Prec':>14} {'Rec':>14} {'F1':>14} {'MCC':>14} {'AUC':>14}")
print("-" * 90)

for label in ablation_variants:
    path = os.path.join(BEST_CSV, f"ablation_{label}_best_config.csv")
    if os.path.exists(path):
        df = pd.read_csv(path)
        print(f"{label:<35} "
              f"{get_col(df, 'cv_mean_accuracy'):>6.4f}±{get_col(df, 'cv_std_accuracy'):.4f} "
              f"{get_col(df, 'cv_mean_precision'):>6.4f}±{get_col(df, 'cv_std_precision'):.4f} "
              f"{get_col(df, 'cv_mean_recall'):>6.4f}±{get_col(df, 'cv_std_recall'):.4f} "
              f"{get_col(df, 'cv_mean_f1'):>6.4f}±{get_col(df, 'cv_std_f1'):.4f} "
              f"{get_col(df, 'cv_mean_mcc'):>6.4f}±{get_col(df, 'cv_std_mcc'):.4f} "
              f"{get_col(df, 'cv_mean_auc_roc', 'cv_mean_auc'):>6.4f}±"
              f"{get_col(df, 'cv_std_auc_roc', 'cv_std_auc'):.4f}")