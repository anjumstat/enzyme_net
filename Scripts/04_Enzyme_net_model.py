# -*- coding: utf-8 -*-
"""
ENZYME-NET: A NOVEL TRANSFORMER-MLP HYBRID FOR FISH ENZYME CLASSIFICATION (8 Classes)
Using UniProt Protein Embeddings (1024-dimensional)

NOVEL ARCHITECTURE INNOVATIONS:
1. Multi-Scale Feature Extraction (MSFE) with parallel pathways
2. Dynamic Feature Gating (DFG) for adaptive feature selection
3. Feature-Level Multi-Head Attention (treats features as tokens)
4. Residual Feature Processor with skip connections
5. Specialized Ensemble of Heads with learnable weights

IMPROVEMENTS OVER ORIGINAL:
  - lr is now passed through grid search (was hardcoded to 0.001 — bug fixed)
  - Grid search uses same N_FOLDS and EPOCHS as best-config (was inconsistent)
  - GPU support (cuda if available, else cpu — was hardcoded CPU)
  - warn_only=True logged so non-deterministic ops are visible
  - Stale save_test_predictions() call on untrained model removed
  - walrus-operator aliasing bug in final summary fixed
  - Output structure matches Vanilla+ exactly:
      ALL_CONFIGS_DIR/  — all 9 hyperparameter combinations
      BEST_CONFIG_DIR/  — best combination only
      Per-model: *_best_config.csv, *_fold_summary.csv
      Combined:  final_comprehensive_summary.csv, paper_results_table.csv
      NPY:       predictions + probabilities + true labels per fold
      Histories: complete training history per fold

Models Evaluated:
  1. Logistic Regression  (Baseline)
  2. Vanilla MLP          (Baseline)
  3. DNN Baseline         (Baseline)
  4. EnzymeNet            (Proposed Novel Model)
  5. EnzymeNet Ablation Studies (8 variants) — WITH 10-FOLD CV
"""

import os
import random
import warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, matthews_corrcoef,
)
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

warnings.filterwarnings("ignore")

# =============================================================================
# 1. CONFIGURATION
# =============================================================================

LEARNING_RATES           = [0.001, 0.0005, 0.0001]
BATCH_SIZES              = [64, 128, 256]
EPOCHS                   = 100
EARLY_STOPPING_PATIENCE  = 10
N_FOLDS                  = 10
RANDOM_STATE             = 42
SELECTION_METRIC         = "mcc"

# Training upgrades
LABEL_SMOOTHING  = 0.1
MIXUP_ALPHA      = 0.2
GRAD_ACCUM_STEPS = 2

DATA_DIR   = r"D:\zebfish\new_class\ml_ready_data_20percent"
OUTPUT_DIR = r"D:\zebfish\new_class\paper3\Results\enzyme_net_results_fixed1"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Main subdirectories
NPY_DIR     = os.path.join(OUTPUT_DIR, "npy_files")
CSV_DIR     = os.path.join(OUTPUT_DIR, "csv_files")
HISTORY_DIR = os.path.join(OUTPUT_DIR, "history_files")
MODELS_DIR  = os.path.join(OUTPUT_DIR, "models")
TEST_DIR    = os.path.join(OUTPUT_DIR, "test_predictions")

for d in [NPY_DIR, CSV_DIR, HISTORY_DIR, MODELS_DIR, TEST_DIR]:
    os.makedirs(d, exist_ok=True)

ALL_CONFIGS_DIR = os.path.join(OUTPUT_DIR, "all_configurations")
os.makedirs(ALL_CONFIGS_DIR, exist_ok=True)

BEST_CONFIG_DIR  = os.path.join(OUTPUT_DIR, "best_configuration")
os.makedirs(BEST_CONFIG_DIR, exist_ok=True)

BEST_NPY_DIR     = os.path.join(BEST_CONFIG_DIR, "npy_files")
BEST_CSV_DIR     = os.path.join(BEST_CONFIG_DIR, "csv_files")
BEST_HISTORY_DIR = os.path.join(BEST_CONFIG_DIR, "history_files")
BEST_MODELS_DIR  = os.path.join(BEST_CONFIG_DIR, "models")
BEST_TEST_DIR    = os.path.join(BEST_CONFIG_DIR, "test_predictions")

for d in [BEST_NPY_DIR, BEST_CSV_DIR, BEST_HISTORY_DIR, BEST_MODELS_DIR, BEST_TEST_DIR]:
    os.makedirs(d, exist_ok=True)

for lr in LEARNING_RATES:
    for bs in BATCH_SIZES:
        config_dir = os.path.join(ALL_CONFIGS_DIR, f"LR_{lr}_BS_{bs}")
        for sub in ["npy_files", "csv_files", "history_files", "models", "test_predictions"]:
            os.makedirs(os.path.join(config_dir, sub), exist_ok=True)

print(f"✅ All directories created in: {OUTPUT_DIR}")


# =============================================================================
# 2. REPRODUCIBILITY
# =============================================================================

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)

set_seed(RANDOM_STATE)

# FIX: GPU support instead of hardcoded CPU
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"  Device: {DEVICE}")


# =============================================================================
# 3. DATASET CLASS
# =============================================================================

class FishEmbeddingDataset(Dataset):
    def __init__(self, embeddings: np.ndarray, labels: np.ndarray):
        assert embeddings.shape[1] == 1024, "Expected 1024-d embeddings"
        self.X = torch.tensor(embeddings, dtype=torch.float32)
        self.y = torch.tensor(labels,     dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# =============================================================================
# 4. BASELINE MODELS
# =============================================================================

class VanillaMLP(nn.Module):
    def __init__(self, input_dim=1024, hidden_dims=None,
                 num_classes=8, dropout=0.3):
        super().__init__()
        hidden_dims = hidden_dims or [512, 256, 128, 64]
        layers, prev = [], input_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, num_classes))
        self.net = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.net(x)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class DNNBaseline(nn.Module):
    def __init__(self, input_dim=1024, hidden_dims=None,
                 num_classes=8, dropout=0.3):
        super().__init__()
        hidden_dims = hidden_dims or [512, 256, 128, 64]
        layers, prev = [], input_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.LayerNorm(h),
                       nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, num_classes))
        self.net = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.net(x)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# =============================================================================
# 5. NOVEL ENZYME-NET COMPONENTS
# =============================================================================

class MultiScaleFeatureExtractor(nn.Module):
    """3 parallel paths (shallow / medium / deep) → fusion."""
    def __init__(self, input_dim=1024, output_dim=256):
        super().__init__()
        self.scale1 = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.LayerNorm(output_dim), nn.GELU()
        )
        self.scale2 = nn.Sequential(
            nn.Linear(input_dim, output_dim // 2),
            nn.LayerNorm(output_dim // 2), nn.GELU(),
            nn.Linear(output_dim // 2, output_dim),
            nn.LayerNorm(output_dim), nn.GELU()
        )
        self.scale3 = nn.Sequential(
            nn.Linear(input_dim, output_dim // 4),
            nn.LayerNorm(output_dim // 4), nn.GELU(),
            nn.Linear(output_dim // 4, output_dim // 2),
            nn.LayerNorm(output_dim // 2), nn.GELU(),
            nn.Linear(output_dim // 2, output_dim),
            nn.LayerNorm(output_dim), nn.GELU()
        )
        self.fusion = nn.Sequential(
            nn.Linear(output_dim * 3, output_dim),
            nn.LayerNorm(output_dim), nn.GELU()
        )

    def forward(self, x):
        return self.fusion(torch.cat([self.scale1(x),
                                      self.scale2(x),
                                      self.scale3(x)], dim=-1))


class DynamicFeatureGating(nn.Module):
    """Input-adaptive sigmoid gate — suppresses irrelevant features."""
    def __init__(self, feature_dim, hidden_dim=128):
        super().__init__()
        self.gate_net = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.LayerNorm(hidden_dim), nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, feature_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        return x * self.gate_net(x)


class MultiHeadFeatureAttention(nn.Module):
    """
    Feature-level attention: projects the feature vector into
    num_tokens tokens of size token_dim, runs MHA, aggregates
    via a learnable CLS token, projects back.
    """
    def __init__(self, feature_dim=256, num_tokens=16,
                 token_dim=16, num_heads=4, dropout=0.1):
        super().__init__()
        self.num_tokens = num_tokens
        self.token_dim  = token_dim

        self.token_proj = nn.Linear(feature_dim, num_tokens * token_dim)
        self.attention  = nn.MultiheadAttention(
            embed_dim=token_dim, num_heads=num_heads,
            dropout=dropout, batch_first=True)
        self.cls_token  = nn.Parameter(torch.randn(1, 1, token_dim) * 0.02)
        self.output_proj = nn.Linear(token_dim, feature_dim)
        self.norm        = nn.LayerNorm(feature_dim)
        self.dropout     = nn.Dropout(dropout)

    def forward(self, x):
        residual  = x
        tokens    = self.token_proj(x).view(-1, self.num_tokens, self.token_dim)
        cls       = self.cls_token.expand(x.size(0), -1, -1)
        tokens    = torch.cat([cls, tokens], dim=1)
        attn_out, _ = self.attention(tokens, tokens, tokens)
        out       = self.output_proj(attn_out[:, 0, :])
        return self.norm(out + residual)


class FeatureProcessor(nn.Module):
    """Two residual MLP stages for stable refinement."""
    def __init__(self, dim, dropout=0.3):
        super().__init__()
        self.mlp1 = nn.Sequential(
            nn.Linear(dim, dim * 2), nn.LayerNorm(dim * 2),
            nn.GELU(), nn.Dropout(dropout),
            nn.Linear(dim * 2, dim), nn.LayerNorm(dim),
            nn.GELU(), nn.Dropout(dropout)
        )
        self.mlp2 = nn.Sequential(
            nn.Linear(dim, dim), nn.LayerNorm(dim),
            nn.GELU(), nn.Dropout(dropout)
        )

    def forward(self, x):
        x = x + self.mlp1(x)
        x = x + self.mlp2(x)
        return x


class SpecializedEnsembleHead(nn.Module):
    """3 expert classification heads with learnable ensemble weights."""
    def __init__(self, input_dim, num_classes=8):
        super().__init__()
        self.head1 = nn.Sequential(
            nn.Linear(input_dim, 256), nn.LayerNorm(256),
            nn.GELU(), nn.Dropout(0.2),
            nn.Linear(256, num_classes)
        )
        self.head2 = nn.Sequential(
            nn.Linear(input_dim, 512), nn.LayerNorm(512),
            nn.GELU(), nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
        self.head3 = nn.Sequential(
            nn.Linear(input_dim, 128), nn.LayerNorm(128),
            nn.GELU(),
            nn.Linear(128, num_classes)
        )
        self.ensemble_weights = nn.Parameter(torch.ones(3) / 3)

    def forward(self, x):
        w  = F.softmax(self.ensemble_weights, dim=0)
        return w[0] * self.head1(x) + w[1] * self.head2(x) + w[2] * self.head3(x)


class EnzymeNet(nn.Module):
    """
    NOVEL: Enzyme-Net — Transformer-MLP Hybrid for 8-class Enzyme Classification

    Pipeline:
        Input (1024-d)
          → MultiScaleFeatureExtractor  [3 parallel paths → fusion]
          → DynamicFeatureGating        [input-adaptive gating]
          → MultiHeadFeatureAttention   [feature-level attention]
          → FeatureProcessor            [residual MLP]
          → SpecializedEnsembleHead     [3 expert heads]
          → Output (8 classes)
    """
    def __init__(self, input_dim=1024, hidden_dim=256,
                 num_classes=8, dropout=0.3):
        super().__init__()
        self.msfe       = MultiScaleFeatureExtractor(input_dim, hidden_dim)
        self.dfg        = DynamicFeatureGating(hidden_dim)
        self.attention  = MultiHeadFeatureAttention(
            feature_dim=hidden_dim, num_tokens=16,
            token_dim=16, num_heads=4, dropout=dropout)
        self.processor  = FeatureProcessor(hidden_dim, dropout)
        self.classifier = SpecializedEnsembleHead(hidden_dim, num_classes)
        self.proj_head  = nn.Sequential(
            nn.Linear(hidden_dim, 128), nn.LayerNorm(128), nn.GELU())
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in',
                                        nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x, return_features=False):
        x = self.msfe(x)
        x = self.dfg(x)
        x = self.attention(x)
        x = self.processor(x)
        logits = self.classifier(x)
        if return_features:
            return logits, self.proj_head(x)
        return logits

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# =============================================================================
# 6. ABLATION VARIANTS
# =============================================================================

class EnzymeNetAblation(nn.Module):
    """Ablation wrapper — toggle each EnzymeNet component independently."""
    def __init__(self, input_dim=1024, hidden_dim=256, num_classes=8,
                 dropout=0.3, use_msfe=True, use_dfg=True,
                 use_attention=True, use_ensemble=True, use_residuals=True):
        super().__init__()
        self.use_dfg       = use_dfg
        self.use_attention = use_attention
        self.use_residuals = use_residuals   # stored but logic is in processor

        self.extractor = (MultiScaleFeatureExtractor(input_dim, hidden_dim)
                          if use_msfe else
                          nn.Sequential(nn.Linear(input_dim, hidden_dim),
                                        nn.LayerNorm(hidden_dim), nn.GELU()))

        if use_dfg:
            self.dfg = DynamicFeatureGating(hidden_dim)

        if use_attention:
            self.attention = MultiHeadFeatureAttention(
                feature_dim=hidden_dim, num_tokens=16,
                token_dim=16, num_heads=4, dropout=dropout)

        # Use the same FeatureProcessor class for residual variants
        # so ablation is truly toggling residuals, not changing the class
        self.processor = (FeatureProcessor(hidden_dim, dropout)
                          if use_residuals else
                          nn.Sequential(
                              nn.Linear(hidden_dim, hidden_dim),
                              nn.LayerNorm(hidden_dim), nn.GELU(),
                              nn.Dropout(dropout),
                              nn.Linear(hidden_dim, hidden_dim),
                              nn.LayerNorm(hidden_dim), nn.GELU(),
                              nn.Dropout(dropout)))

        self.classifier = (SpecializedEnsembleHead(hidden_dim, num_classes)
                           if use_ensemble else
                           nn.Linear(hidden_dim, num_classes))
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.extractor(x)
        if self.use_dfg:
            x = self.dfg(x)
        if self.use_attention:
            x = self.attention(x)
        x = self.processor(x)
        return self.classifier(x)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


ENZYME_ABLATION_VARIANTS = [
    ("Full_EnzymeNet",
     dict(use_msfe=True,  use_dfg=True,  use_attention=True,
          use_ensemble=True,  use_residuals=True,  dropout=0.3)),
    ("w_o_MSFE",
     dict(use_msfe=False, use_dfg=True,  use_attention=True,
          use_ensemble=True,  use_residuals=True,  dropout=0.3)),
    ("w_o_DFG",
     dict(use_msfe=True,  use_dfg=False, use_attention=True,
          use_ensemble=True,  use_residuals=True,  dropout=0.3)),
    ("w_o_Attention",
     dict(use_msfe=True,  use_dfg=True,  use_attention=False,
          use_ensemble=True,  use_residuals=True,  dropout=0.3)),
    ("w_o_Ensemble",
     dict(use_msfe=True,  use_dfg=True,  use_attention=True,
          use_ensemble=False, use_residuals=True,  dropout=0.3)),
    ("w_o_Residuals",
     dict(use_msfe=True,  use_dfg=True,  use_attention=True,
          use_ensemble=True,  use_residuals=False, dropout=0.3)),
    ("Higher_Dropout_0.5",
     dict(use_msfe=True,  use_dfg=True,  use_attention=True,
          use_ensemble=True,  use_residuals=True,  dropout=0.5)),
    ("Lower_Dropout_0.1",
     dict(use_msfe=True,  use_dfg=True,  use_attention=True,
          use_ensemble=True,  use_residuals=True,  dropout=0.1)),
]


# =============================================================================
# 7. MIXUP UTILITIES
# =============================================================================

def mixup_data(x: torch.Tensor, y: torch.Tensor,
               alpha: float = MIXUP_ALPHA):
    if alpha <= 0:
        return x, y, y, 1.0
    lam = float(np.random.beta(alpha, alpha))
    idx = torch.randperm(x.size(0), device=x.device)
    return lam * x + (1 - lam) * x[idx], y, y[idx], lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


# =============================================================================
# 8. TRAINER
# =============================================================================

class EnzymeTrainer:
    def __init__(self, model, optimizer, criterion,
                 scheduler=None, patience: int = EARLY_STOPPING_PATIENCE):
        self.model     = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.scheduler = scheduler
        self.patience  = patience

        self.history = {
            "train_loss":      [], "val_loss":      [],
            "val_mcc":         [], "val_auc":        [],
            "train_accuracy":  [], "val_accuracy":  [],
            "train_f1":        [], "val_f1":         [],
            "train_precision": [], "val_precision": [],
            "train_recall":    [], "val_recall":    [],
        }
        self.best_val_mcc = -1.0
        self.best_weights = None
        self._no_improve  = 0

    # ------------------------------------------------------------------
    def _epoch(self, loader, training: bool):
        self.model.train(training)
        total_loss = 0.0
        all_labels, all_probs, all_preds = [], [], []

        accum_steps = GRAD_ACCUM_STEPS if training else 1

        with torch.set_grad_enabled(training):
            for step, (X_b, y_b) in enumerate(loader):
                X_b = X_b.to(DEVICE)
                y_b = y_b.to(DEVICE)

                if training and MIXUP_ALPHA > 0:
                    X_b, y_a, y_b_mix, lam = mixup_data(X_b, y_b, MIXUP_ALPHA)
                    logits   = self.model(X_b)
                    loss     = mixup_criterion(self.criterion, logits, y_a, y_b_mix, lam)
                    y_metric = y_a
                else:
                    logits   = self.model(X_b)
                    loss     = self.criterion(logits, y_b)
                    y_metric = y_b

                if training:
                    (loss / accum_steps).backward()
                    if (step + 1) % accum_steps == 0 or (step + 1) == len(loader):
                        nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                        self.optimizer.step()
                        self.optimizer.zero_grad()

                total_loss += loss.item() * len(y_b)
                probs = torch.softmax(logits, dim=1).detach().cpu().numpy()
                preds = np.argmax(probs, axis=1)
                all_probs.extend(probs)
                all_preds.extend(preds)
                all_labels.extend(y_metric.cpu().numpy())

        avg_loss  = total_loss / len(loader.dataset)
        labels    = np.array(all_labels)
        probs     = np.array(all_probs)
        preds     = np.array(all_preds)

        accuracy  = accuracy_score(labels, preds)
        precision = precision_score(labels, preds, average="macro", zero_division=0)
        recall    = recall_score(labels,    preds, average="macro", zero_division=0)
        f1        = f1_score(labels,        preds, average="macro", zero_division=0)
        mcc       = matthews_corrcoef(labels, preds)
        try:
            auc = roc_auc_score(labels, probs, multi_class="ovr", average="macro")
        except Exception:
            auc = 0.0

        return avg_loss, accuracy, precision, recall, f1, mcc, auc, labels, probs, preds

    # ------------------------------------------------------------------
    def fit(self, train_loader, val_loader,
            epochs: int = EPOCHS, verbose: bool = True):
        if verbose:
            print(f"\n  Params: {self.model.count_parameters():,} | "
                  f"Epochs: {epochs} | Device: {DEVICE}")

        self.optimizer.zero_grad()   # required for grad-accum

        for epoch in range(1, epochs + 1):
            tr_loss, tr_acc, tr_prec, tr_rec, tr_f1, tr_mcc, tr_auc, _, _, _ = \
                self._epoch(train_loader, True)
            vl_loss, vl_acc, vl_prec, vl_rec, vl_f1, vl_mcc, vl_auc, _, _, _ = \
                self._epoch(val_loader, False)

            if self.scheduler:
                self.scheduler.step(vl_loss)

            self.history["train_loss"].append(tr_loss)
            self.history["val_loss"].append(vl_loss)
            self.history["val_mcc"].append(vl_mcc)
            self.history["val_auc"].append(vl_auc)
            self.history["train_accuracy"].append(tr_acc)
            self.history["val_accuracy"].append(vl_acc)
            self.history["train_f1"].append(tr_f1)
            self.history["val_f1"].append(vl_f1)
            self.history["train_precision"].append(tr_prec)
            self.history["val_precision"].append(vl_prec)
            self.history["train_recall"].append(tr_rec)
            self.history["val_recall"].append(vl_rec)

            if vl_mcc > self.best_val_mcc:
                self.best_val_mcc = vl_mcc
                self.best_weights = {k: v.clone()
                                     for k, v in self.model.state_dict().items()}
                self._no_improve  = 0
                flag = " ✓"
            else:
                self._no_improve += 1
                flag = ""

            if verbose and (epoch % 20 == 0 or epoch == 1):
                print(f"    Ep {epoch:03d} | TrLoss {tr_loss:.4f} | "
                      f"VlLoss {vl_loss:.4f} | VlAcc {vl_acc:.4f} | "
                      f"VlF1 {vl_f1:.4f} | VlMCC {vl_mcc:.4f}{flag}")

            if self._no_improve >= self.patience:
                if verbose:
                    print(f"    Early stop at epoch {epoch}.")
                break

        if self.best_weights:
            self.model.load_state_dict(self.best_weights)

    # ------------------------------------------------------------------
    def evaluate(self, loader):
        self.model.eval()
        loss, accuracy, precision, recall, f1, mcc, auc, labels, probs, preds = \
            self._epoch(loader, False)
        return {
            "accuracy":  accuracy,  "precision": precision,
            "recall":    recall,    "f1":        f1,
            "auc_roc":   auc,       "mcc":       mcc,
            "loss":      loss,
            "labels":    labels,    "probs":     probs,    "preds": preds,
        }

    def get_history(self):
        return self.history


# =============================================================================
# 9. HELPERS
# =============================================================================

def make_loaders(X_tr, y_tr, X_te, y_te, batch_size: int = 128):
    class_counts  = np.bincount(y_tr)
    if len(class_counts) < 8:
        class_counts = np.pad(class_counts, (0, 8 - len(class_counts)))
    class_weights  = 1.0 / class_counts.astype(np.float64)
    sample_weights = class_weights[y_tr]
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights),
                                    replacement=True)
    tr_ds = FishEmbeddingDataset(X_tr, y_tr)
    te_ds = FishEmbeddingDataset(X_te, y_te)
    return (DataLoader(tr_ds, batch_size=batch_size, sampler=sampler),
            DataLoader(te_ds, batch_size=batch_size, shuffle=False))


def save_predictions(model, loader, output_dir, prefix,
                     fold=None, data_type: str = "val"):
    model.eval()
    all_probs, all_preds, all_labels = [], [], []
    with torch.no_grad():
        for X_b, y_b in loader:
            X_b    = X_b.to(DEVICE)
            logits = model(X_b)
            probs  = torch.softmax(logits, dim=1).cpu().numpy()
            preds  = np.argmax(probs, axis=1)
            all_probs.extend(probs)
            all_preds.extend(preds)
            all_labels.extend(y_b.numpy())

    all_probs  = np.array(all_probs)
    all_preds  = np.array(all_preds)
    all_labels = np.array(all_labels)
    suffix     = f"_fold{fold}" if fold is not None else ""

    np.save(os.path.join(output_dir,
            f"{prefix}_{data_type}{suffix}_predictions.npy"),   all_preds)
    np.save(os.path.join(output_dir,
            f"{prefix}_{data_type}{suffix}_probabilities.npy"), all_probs)
    np.save(os.path.join(output_dir,
            f"{prefix}_{data_type}{suffix}_true_labels.npy"),   all_labels)
    return all_preds, all_probs, all_labels


# FIX: lr is now a proper parameter — grid search actually tests different LRs
def train_model_with_config(model_class, X_tr, y_tr, X_te, y_te,
                            epochs: int = EPOCHS,
                            batch_size: int = 128,
                            lr: float = 0.001,
                            verbose: bool = True,
                            model_name: str = "model",
                            fold=None,
                            save_outputs: bool = True,
                            save_predictions_flag: bool = True,
                            output_dirs=None,
                            **kwargs):
    tr_loader, te_loader = make_loaders(X_tr, y_tr, X_te, y_te, batch_size)

    model     = model_class(**kwargs).to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer, "min", factor=0.5, patience=7)
    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)

    trainer = EnzymeTrainer(model, optimizer, criterion, scheduler)
    trainer.fit(tr_loader, te_loader, epochs=epochs, verbose=verbose)
    eval_metrics = trainer.evaluate(te_loader)

    if save_outputs and output_dirs is not None:
        hist_name = (f"{model_name}_fold{fold}_history.npy"
                     if fold is not None
                     else f"{model_name}_final_history.npy")
        np.save(os.path.join(output_dirs['history'], hist_name),
                trainer.get_history())

        mdl_name = (f"{model_name}_fold{fold}.pt"
                    if fold is not None
                    else f"{model_name}_final.pt")
        torch.save(model.state_dict(),
                   os.path.join(output_dirs['models'], mdl_name))

        if save_predictions_flag:
            save_predictions(model, te_loader, output_dirs['npy'],
                             model_name, fold, "val")

    return eval_metrics, trainer.get_history()


# =============================================================================
# 10. GRID SEARCH
# =============================================================================

def run_grid_search(X_train, y_train, X_test, y_test):
    print(f"\n{'='*55}")
    print("  HYPERPARAMETER GRID SEARCH (ALL CONFIGURATIONS)")
    print(f"{'='*55}")

    all_results  = []
    best_score   = -1.0
    best_params  = {}
    best_cfg_dir = None

    for lr in LEARNING_RATES:
        for bs in BATCH_SIZES:
            print(f"\n  Testing LR={lr}, BS={bs}")

            config_dir  = os.path.join(ALL_CONFIGS_DIR, f"LR_{lr}_BS_{bs}")
            output_dirs = {
                'npy':     os.path.join(config_dir, "npy_files"),
                'csv':     os.path.join(config_dir, "csv_files"),
                'history': os.path.join(config_dir, "history_files"),
                'models':  os.path.join(config_dir, "models"),
                'test':    os.path.join(config_dir, "test_predictions"),
            }

            # FIX: same N_FOLDS and EPOCHS as best-config run (was 5 folds / 30 epochs)
            skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True,
                                  random_state=RANDOM_STATE)
            cv_scores, fold_results = [], []

            for fold, (tr_idx, val_idx) in enumerate(
                    skf.split(X_train, y_train), 1):
                X_tr, X_val = X_train[tr_idx], X_train[val_idx]
                y_tr, y_val = y_train[tr_idx], y_train[val_idx]

                m, _ = train_model_with_config(
                    EnzymeNet, X_tr, y_tr, X_val, y_val,
                    epochs=EPOCHS, batch_size=bs, lr=lr,   # ← lr passed
                    verbose=False, num_classes=8, hidden_dim=256,
                    model_name=f"EnzymeNet_LR{lr}_BS{bs}",
                    fold=fold, save_outputs=True,
                    save_predictions_flag=True,
                    output_dirs=output_dirs,
                )
                cv_scores.append(m['mcc'])
                fold_results.append(m)

            mean_mcc = float(np.mean(cv_scores))
            std_mcc  = float(np.std(cv_scores))
            print(f"    CV MCC: {mean_mcc:.4f} ± {std_mcc:.4f}")

            pd.DataFrame(fold_results).to_csv(
                os.path.join(output_dirs['csv'],
                             "grid_cv_fold_metrics.csv"), index=False)

            print("    Testing on unseen species...")
            m_final, _ = train_model_with_config(
                EnzymeNet, X_train, y_train, X_test, y_test,
                epochs=EPOCHS, batch_size=bs, lr=lr,       # ← lr passed
                verbose=False, num_classes=8, hidden_dim=256,
                model_name=f"EnzymeNet_LR{lr}_BS{bs}_final",
                fold=None, save_outputs=True,
                save_predictions_flag=True,
                output_dirs=output_dirs,
            )

            config_summary = {
                'learning_rate':  lr,       'batch_size':    bs,
                'cv_mean_mcc':    mean_mcc, 'cv_std_mcc':    std_mcc,
                'test_mcc':       m_final['mcc'],
                'test_f1':        m_final['f1'],
                'test_accuracy':  m_final['accuracy'],
                'test_precision': m_final['precision'],
                'test_recall':    m_final['recall'],
                'test_auc':       m_final['auc_roc'],
            }
            all_results.append(config_summary)
            pd.DataFrame([config_summary]).to_csv(
                os.path.join(output_dirs['csv'], "config_summary.csv"),
                index=False)

            if mean_mcc > best_score:
                best_score   = mean_mcc
                best_params  = {'lr': lr, 'bs': bs}
                best_cfg_dir = config_dir

    pd.DataFrame(all_results).to_csv(
        os.path.join(CSV_DIR, "full_grid_search_results.csv"), index=False)

    print(f"\n  Best params: LR={best_params['lr']}, "
          f"BS={best_params['bs']} (MCC={best_score:.4f})")
    print(f"  Best config dir: {best_cfg_dir}")
    return best_params, best_cfg_dir


# =============================================================================
# 11. RUN BEST CONFIGURATION WITH ALL MODELS
# =============================================================================

def run_best_configuration(X_train, y_train, X_test, y_test,
                           best_params, best_config_dir):
    print(f"\n{'='*55}")
    print(f"  BEST CONFIG: LR={best_params['lr']}, BS={best_params['bs']}")
    print(f"  Running ALL models with best hyperparameters")
    print(f"{'='*55}")

    best_lr = best_params['lr']
    bs      = best_params['bs']

    output_dirs = {
        'npy':     BEST_NPY_DIR,
        'csv':     BEST_CSV_DIR,
        'history': BEST_HISTORY_DIR,
        'models':  BEST_MODELS_DIR,
        'test':    BEST_TEST_DIR,
    }

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True,
                          random_state=RANDOM_STATE)

    # ------------------------------------------------------------------
    # Inner helper — collect N_FOLDS CV metrics for any model
    # ------------------------------------------------------------------
    def collect_all_fold_metrics(model_class, model_name,
                                 out_dirs, **kwargs):
        fold_metrics = {}
        cv_scores    = {k: [] for k in
                        ['mcc', 'accuracy', 'f1', 'precision', 'recall', 'auc']}
        fold_details = []

        print(f"\n    Running {N_FOLDS}-fold CV for {model_name}...")

        for fold, (tr_idx, val_idx) in enumerate(
                skf.split(X_train, y_train), 1):
            X_tr, X_val = X_train[tr_idx], X_train[val_idx]
            y_tr, y_val = y_train[tr_idx], y_train[val_idx]

            m_cv, _ = train_model_with_config(
                model_class, X_tr, y_tr, X_val, y_val,
                epochs=EPOCHS, batch_size=bs, lr=best_lr,   # ← lr passed
                verbose=False, num_classes=8,
                model_name=f"{model_name}_best_fold{fold}",
                fold=fold, save_outputs=True,
                save_predictions_flag=True,
                output_dirs=out_dirs,
                **kwargs,
            )

            fold_metrics[f'fold_{fold}_mcc']       = m_cv['mcc']
            fold_metrics[f'fold_{fold}_accuracy']   = m_cv['accuracy']
            fold_metrics[f'fold_{fold}_f1']         = m_cv['f1']
            fold_metrics[f'fold_{fold}_precision']  = m_cv['precision']
            fold_metrics[f'fold_{fold}_recall']     = m_cv['recall']
            fold_metrics[f'fold_{fold}_auc']        = m_cv['auc_roc']
            fold_metrics[f'fold_{fold}_loss']       = m_cv['loss']

            cv_scores['mcc'].append(m_cv['mcc'])
            cv_scores['accuracy'].append(m_cv['accuracy'])
            cv_scores['f1'].append(m_cv['f1'])
            cv_scores['precision'].append(m_cv['precision'])
            cv_scores['recall'].append(m_cv['recall'])
            cv_scores['auc'].append(m_cv['auc_roc'])

            fold_details.append({
                'fold':      fold,       'mcc':       m_cv['mcc'],
                'accuracy':  m_cv['accuracy'], 'f1':  m_cv['f1'],
                'precision': m_cv['precision'], 'recall': m_cv['recall'],
                'auc':       m_cv['auc_roc'],  'loss':   m_cv['loss'],
            })
            print(f"      Fold {fold:2d}: Acc={m_cv['accuracy']:.4f}, "
                  f"F1={m_cv['f1']:.4f}, MCC={m_cv['mcc']:.4f}, "
                  f"AUC={m_cv['auc_roc']:.4f}")

        cv_metrics = {
            'cv_mean_mcc':       np.mean(cv_scores['mcc']),
            'cv_std_mcc':        np.std(cv_scores['mcc']),
            'cv_mean_accuracy':  np.mean(cv_scores['accuracy']),
            'cv_std_accuracy':   np.std(cv_scores['accuracy']),
            'cv_mean_f1':        np.mean(cv_scores['f1']),
            'cv_std_f1':         np.std(cv_scores['f1']),
            'cv_mean_precision': np.mean(cv_scores['precision']),
            'cv_std_precision':  np.std(cv_scores['precision']),
            'cv_mean_recall':    np.mean(cv_scores['recall']),
            'cv_std_recall':     np.std(cv_scores['recall']),
            'cv_mean_auc':       np.mean(cv_scores['auc']),
            'cv_std_auc':        np.std(cv_scores['auc']),
        }
        return fold_metrics, cv_metrics, fold_details

    # ------------------------------------------------------------------
    # Logistic Regression
    # ------------------------------------------------------------------
    print("\n  Logistic Regression...")
    scaler     = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)
    clf = LogisticRegression(max_iter=1000, C=1.0, solver="lbfgs",
                             multi_class="multinomial",
                             class_weight="balanced",
                             random_state=RANDOM_STATE)
    clf.fit(X_train_sc, y_train)
    test_probs = clf.predict_proba(X_test_sc)
    test_preds = np.argmax(test_probs, axis=1)

    lr_test_metrics = {
        'accuracy':  accuracy_score(y_test, test_preds),
        'precision': precision_score(y_test, test_preds,
                                     average="macro", zero_division=0),
        'recall':    recall_score(y_test, test_preds,
                                  average="macro", zero_division=0),
        'f1':        f1_score(y_test, test_preds,
                              average="macro", zero_division=0),
        'mcc':       matthews_corrcoef(y_test, test_preds),
        'auc_roc':   roc_auc_score(y_test, test_probs,
                                   multi_class="ovr", average="macro"),
    }

    print(f"    Running {N_FOLDS}-fold CV for Logistic Regression...")
    lr_cv_scores    = {k: [] for k in
                       ['mcc', 'accuracy', 'f1', 'precision', 'recall', 'auc']}
    lr_fold_metrics = {}
    lr_fold_details = []

    for fold, (tr_idx, val_idx) in enumerate(
            skf.split(X_train, y_train), 1):
        X_tr, X_val = X_train[tr_idx], X_train[val_idx]
        y_tr, y_val = y_train[tr_idx], y_train[val_idx]

        sc_f    = StandardScaler()
        X_tr_s  = sc_f.fit_transform(X_tr)
        X_val_s = sc_f.transform(X_val)
        clf_f   = LogisticRegression(max_iter=1000, C=1.0, solver="lbfgs",
                                     multi_class="multinomial",
                                     class_weight="balanced",
                                     random_state=RANDOM_STATE)
        clf_f.fit(X_tr_s, y_tr)
        val_probs = clf_f.predict_proba(X_val_s)
        val_preds = np.argmax(val_probs, axis=1)

        f_mcc  = matthews_corrcoef(y_val, val_preds)
        f_acc  = accuracy_score(y_val, val_preds)
        f_f1   = f1_score(y_val, val_preds, average="macro", zero_division=0)
        f_prec = precision_score(y_val, val_preds,
                                 average="macro", zero_division=0)
        f_rec  = recall_score(y_val, val_preds,
                              average="macro", zero_division=0)
        try:
            f_auc = roc_auc_score(y_val, val_probs,
                                  multi_class="ovr", average="macro")
        except Exception:
            f_auc = 0.0

        lr_fold_metrics[f'fold_{fold}_mcc']       = f_mcc
        lr_fold_metrics[f'fold_{fold}_accuracy']   = f_acc
        lr_fold_metrics[f'fold_{fold}_f1']         = f_f1
        lr_fold_metrics[f'fold_{fold}_precision']  = f_prec
        lr_fold_metrics[f'fold_{fold}_recall']     = f_rec
        lr_fold_metrics[f'fold_{fold}_auc']        = f_auc

        lr_cv_scores['mcc'].append(f_mcc)
        lr_cv_scores['accuracy'].append(f_acc)
        lr_cv_scores['f1'].append(f_f1)
        lr_cv_scores['precision'].append(f_prec)
        lr_cv_scores['recall'].append(f_rec)
        lr_cv_scores['auc'].append(f_auc)

        lr_fold_details.append({
            'fold': fold, 'mcc': f_mcc, 'accuracy': f_acc,
            'f1': f_f1, 'precision': f_prec, 'recall': f_rec, 'auc': f_auc,
        })

        # History stub — keeps file layout identical to neural models
        np.save(os.path.join(BEST_HISTORY_DIR,
                f"LogisticRegression_best_fold{fold}_history.npy"),
                {'val_mcc': [f_mcc], 'val_auc': [f_auc],
                 'val_accuracy': [f_acc], 'val_f1': [f_f1],
                 'val_precision': [f_prec], 'val_recall': [f_rec],
                 'fold': fold})
        np.save(os.path.join(BEST_HISTORY_DIR,
                f"LogisticRegression_best_fold{fold}_predictions.npy"), val_preds)
        np.save(os.path.join(BEST_HISTORY_DIR,
                f"LogisticRegression_best_fold{fold}_probabilities.npy"), val_probs)
        np.save(os.path.join(BEST_HISTORY_DIR,
                f"LogisticRegression_best_fold{fold}_true_labels.npy"), y_val)

        print(f"      Fold {fold:2d}: Acc={f_acc:.4f}, F1={f_f1:.4f}, "
              f"MCC={f_mcc:.4f}, AUC={f_auc:.4f}")

    lr_metrics = {
        **lr_test_metrics,
        **lr_fold_metrics,
        'cv_mean_mcc':       np.mean(lr_cv_scores['mcc']),
        'cv_std_mcc':        np.std(lr_cv_scores['mcc']),
        'cv_mean_accuracy':  np.mean(lr_cv_scores['accuracy']),
        'cv_std_accuracy':   np.std(lr_cv_scores['accuracy']),
        'cv_mean_f1':        np.mean(lr_cv_scores['f1']),
        'cv_std_f1':         np.std(lr_cv_scores['f1']),
        'cv_mean_precision': np.mean(lr_cv_scores['precision']),
        'cv_std_precision':  np.std(lr_cv_scores['precision']),
        'cv_mean_recall':    np.mean(lr_cv_scores['recall']),
        'cv_std_recall':     np.std(lr_cv_scores['recall']),
        'cv_mean_auc':       np.mean(lr_cv_scores['auc']),
        'cv_std_auc':        np.std(lr_cv_scores['auc']),
    }
    pd.DataFrame([lr_metrics]).to_csv(
        os.path.join(BEST_CSV_DIR,
                     "LogisticRegression_best_config.csv"), index=False)
    pd.DataFrame(lr_fold_details).to_csv(
        os.path.join(BEST_CSV_DIR,
                     "LogisticRegression_fold_summary.csv"), index=False)
    print(f"    Test F1={lr_test_metrics['f1']:.4f} "
          f"Test MCC={lr_test_metrics['mcc']:.4f} | "
          f"CV MCC={lr_metrics['cv_mean_mcc']:.4f}"
          f"±{lr_metrics['cv_std_mcc']:.4f}")

    np.save(os.path.join(BEST_TEST_DIR,
            "LogisticRegression_test_predictions.npy"),  test_preds)
    np.save(os.path.join(BEST_TEST_DIR,
            "LogisticRegression_test_probabilities.npy"), test_probs)
    np.save(os.path.join(BEST_TEST_DIR,
            "LogisticRegression_test_true_labels.npy"),  y_test)

    # ------------------------------------------------------------------
    # Vanilla MLP
    # ------------------------------------------------------------------
    print("\n  Vanilla MLP...")
    m_vanilla, _ = train_model_with_config(
        VanillaMLP, X_train, y_train, X_test, y_test,
        epochs=EPOCHS, batch_size=bs, lr=best_lr, verbose=True,
        num_classes=8, model_name="VanillaMLP_best",
        fold=None, save_outputs=True, save_predictions_flag=True,
        output_dirs=output_dirs,
    )
    fm_v, cm_v, fd_v = collect_all_fold_metrics(
        VanillaMLP, "VanillaMLP", output_dirs)
    m_vanilla.update(fm_v)
    m_vanilla.update(cm_v)
    pd.DataFrame([m_vanilla]).to_csv(
        os.path.join(BEST_CSV_DIR, "VanillaMLP_best_config.csv"), index=False)
    pd.DataFrame(fd_v).to_csv(
        os.path.join(BEST_CSV_DIR, "VanillaMLP_fold_summary.csv"), index=False)
    print(f"    Test F1={m_vanilla['f1']:.4f} "
          f"Test MCC={m_vanilla['mcc']:.4f} | "
          f"CV MCC={cm_v['cv_mean_mcc']:.4f}±{cm_v['cv_std_mcc']:.4f}")

    # ------------------------------------------------------------------
    # DNN Baseline
    # ------------------------------------------------------------------
    print("\n  DNN Baseline...")
    m_dnn, _ = train_model_with_config(
        DNNBaseline, X_train, y_train, X_test, y_test,
        epochs=EPOCHS, batch_size=bs, lr=best_lr, verbose=True,
        num_classes=8, model_name="DNNBaseline_best",
        fold=None, save_outputs=True, save_predictions_flag=True,
        output_dirs=output_dirs,
    )
    fm_d, cm_d, fd_d = collect_all_fold_metrics(
        DNNBaseline, "DNNBaseline", output_dirs)
    m_dnn.update(fm_d)
    m_dnn.update(cm_d)
    pd.DataFrame([m_dnn]).to_csv(
        os.path.join(BEST_CSV_DIR, "DNNBaseline_best_config.csv"), index=False)
    pd.DataFrame(fd_d).to_csv(
        os.path.join(BEST_CSV_DIR, "DNNBaseline_fold_summary.csv"), index=False)
    print(f"    Test F1={m_dnn['f1']:.4f} "
          f"Test MCC={m_dnn['mcc']:.4f} | "
          f"CV MCC={cm_d['cv_mean_mcc']:.4f}±{cm_d['cv_std_mcc']:.4f}")

    # ------------------------------------------------------------------
    # EnzymeNet  (Proposed)
    # ------------------------------------------------------------------
    print("\n  EnzymeNet (Novel Proposed Model)...")
    m_enzyme, _ = train_model_with_config(
        EnzymeNet, X_train, y_train, X_test, y_test,
        epochs=EPOCHS, batch_size=bs, lr=best_lr, verbose=True,
        num_classes=8, hidden_dim=256, dropout=0.3,
        model_name="EnzymeNet_best",
        fold=None, save_outputs=True, save_predictions_flag=True,
        output_dirs=output_dirs,
    )
    fm_e, cm_e, fd_e = collect_all_fold_metrics(
        EnzymeNet, "EnzymeNet", output_dirs,
        hidden_dim=256, dropout=0.3)
    m_enzyme.update(fm_e)
    m_enzyme.update(cm_e)
    pd.DataFrame([m_enzyme]).to_csv(
        os.path.join(BEST_CSV_DIR, "EnzymeNet_best_config.csv"), index=False)
    pd.DataFrame(fd_e).to_csv(
        os.path.join(BEST_CSV_DIR, "EnzymeNet_fold_summary.csv"), index=False)
    print(f"    Test F1={m_enzyme['f1']:.4f} "
          f"Test MCC={m_enzyme['mcc']:.4f} | "
          f"CV MCC={cm_e['cv_mean_mcc']:.4f}±{cm_e['cv_std_mcc']:.4f}")

    # ------------------------------------------------------------------
    # Ablation Study
    # ------------------------------------------------------------------
    print("\n  Ablation Study...")
    ablation_results   = {}
    all_ablation_folds = []

    for label, kwargs in ENZYME_ABLATION_VARIANTS:
        print(f"\n    Running: {label}")
        m_abl, _ = train_model_with_config(
            EnzymeNetAblation, X_train, y_train, X_test, y_test,
            epochs=EPOCHS, batch_size=bs, lr=best_lr, verbose=False,
            num_classes=8,
            model_name=f"ablation_{label}_best",
            fold=None, save_outputs=True, save_predictions_flag=True,
            output_dirs=output_dirs,
            **kwargs,
        )
        fm_a, cm_a, fd_a = collect_all_fold_metrics(
            EnzymeNetAblation, f"ablation_{label}", output_dirs, **kwargs)
        m_abl.update(fm_a)
        m_abl.update(cm_a)
        ablation_results[label] = m_abl
        for fd in fd_a:
            fd['variant'] = label
            all_ablation_folds.append(fd)
        pd.DataFrame([m_abl]).to_csv(
            os.path.join(BEST_CSV_DIR,
                         f"ablation_{label}_best_config.csv"), index=False)
        pd.DataFrame(fd_a).to_csv(
            os.path.join(BEST_CSV_DIR,
                         f"ablation_{label}_fold_summary.csv"), index=False)
        print(f"      Test F1={m_abl['f1']:.4f} "
              f"Test MCC={m_abl['mcc']:.4f} | "
              f"CV MCC={cm_a['cv_mean_mcc']:.4f}±{cm_a['cv_std_mcc']:.4f}")

    ablation_summary = pd.DataFrame([{
        'Variant':            name,
        'Test_Accuracy':      m['accuracy'],
        'Test_Precision':     m['precision'],
        'Test_Recall':        m['recall'],
        'Test_F1':            m['f1'],
        'Test_MCC':           m['mcc'],
        'Test_AUC':           m['auc_roc'],
        'CV_Mean_MCC':        m['cv_mean_mcc'],
        'CV_Std_MCC':         m['cv_std_mcc'],
        'CV_Mean_Accuracy':   m['cv_mean_accuracy'],
        'CV_Std_Accuracy':    m['cv_std_accuracy'],
        'CV_Mean_F1':         m['cv_mean_f1'],
        'CV_Std_F1':          m['cv_std_f1'],
        'CV_Mean_Precision':  m['cv_mean_precision'],
        'CV_Std_Precision':   m['cv_std_precision'],
        'CV_Mean_Recall':     m['cv_mean_recall'],
        'CV_Std_Recall':      m['cv_std_recall'],
        'CV_Mean_AUC':        m['cv_mean_auc'],
        'CV_Std_AUC':         m['cv_std_auc'],
    } for name, m in ablation_results.items()])
    ablation_summary.to_csv(
        os.path.join(BEST_CSV_DIR,
                     "ablation_summary_best_config.csv"), index=False)
    pd.DataFrame(all_ablation_folds).to_csv(
        os.path.join(BEST_CSV_DIR,
                     "ablation_all_folds_combined.csv"), index=False)

    # ------------------------------------------------------------------
    # Final Comprehensive Summary  (matches Vanilla+ output exactly)
    # FIX: was using walrus-operator aliasing — all 4 model rows were identical
    # ------------------------------------------------------------------
    print("\n  Creating final comprehensive summary...")

    def _summary_row(name, m):
        return {
            'Model':             name,
            'Test_Accuracy':     m['accuracy'],
            'Test_Precision':    m['precision'],
            'Test_Recall':       m['recall'],
            'Test_F1':           m['f1'],
            'Test_MCC':          m['mcc'],
            'Test_AUC':          m['auc_roc'],
            'CV_Mean_Accuracy':  m.get('cv_mean_accuracy',  0),
            'CV_Std_Accuracy':   m.get('cv_std_accuracy',   0),
            'CV_Mean_F1':        m.get('cv_mean_f1',        0),
            'CV_Std_F1':         m.get('cv_std_f1',         0),
            'CV_Mean_MCC':       m.get('cv_mean_mcc',       0),
            'CV_Std_MCC':        m.get('cv_std_mcc',        0),
            'CV_Mean_Precision': m.get('cv_mean_precision', 0),
            'CV_Std_Precision':  m.get('cv_std_precision',  0),
            'CV_Mean_Recall':    m.get('cv_mean_recall',    0),
            'CV_Std_Recall':     m.get('cv_std_recall',     0),
            'CV_Mean_AUC':       m.get('cv_mean_auc',       0),
            'CV_Std_AUC':        m.get('cv_std_auc',        0),
        }

    # FIX: explicit named variables — no aliasing
    all_models_summary = [
        _summary_row("Logistic Regression", lr_metrics),
        _summary_row("Vanilla MLP",          m_vanilla),
        _summary_row("DNN Baseline",         m_dnn),
        _summary_row("EnzymeNet (Ours)",     m_enzyme),
    ]
    for name, m in ablation_results.items():
        all_models_summary.append(_summary_row(f"Ablation_{name}", m))

    final_df = pd.DataFrame(all_models_summary)
    final_df.to_csv(
        os.path.join(BEST_CSV_DIR, "final_comprehensive_summary.csv"),
        index=False)

    paper_cols = [
        'Model',
        'Test_Accuracy', 'Test_F1', 'Test_MCC', 'Test_AUC',
        'CV_Mean_Accuracy', 'CV_Mean_F1', 'CV_Mean_MCC', 'CV_Mean_AUC',
    ]
    final_df[paper_cols].to_csv(
        os.path.join(BEST_CSV_DIR, "paper_results_table.csv"), index=False)

    print(f"\n  ✅ All results saved to: {BEST_CSV_DIR}")

    return {
        'logistic':    lr_metrics,
        'vanilla_mlp': m_vanilla,
        'dnn':         m_dnn,
        'enzymenet':   m_enzyme,
        'ablation':    ablation_results,
    }


# =============================================================================
# 12. LOAD DATA
# =============================================================================

def load_data():
    print(f"\n📁 Loading data from: {DATA_DIR}")
    X_train = np.load(os.path.join(DATA_DIR, "X_train.npy"))
    X_test  = np.load(os.path.join(DATA_DIR, "X_test.npy"))
    y_train = np.load(os.path.join(DATA_DIR, "y_train.npy"))
    y_test  = np.load(os.path.join(DATA_DIR, "y_test.npy"))
    print(f"  X_train: {X_train.shape}  |  X_test: {X_test.shape}")
    print(f"  y_train: {y_train.shape}  |  y_test: {y_test.shape}")
    print(f"  Train class counts: {np.bincount(y_train)}")
    print(f"  Test  class counts: {np.bincount(y_test)}")
    return X_train, X_test, y_train, y_test


# =============================================================================
# 13. MAIN
# =============================================================================

def main():
    print("\n" + "=" * 70)
    print("  ENZYME-NET: NOVEL TRANSFORMER-MLP HYBRID")
    print("  FOR FISH ENZYME CLASSIFICATION (8 Classes)")
    print("  FULL REPRODUCIBILITY — ALL CONFIGURATIONS SAVED")
    print("=" * 70)
    print(f"\n  Configuration:")
    print(f"    CV Folds          : {N_FOLDS}")
    print(f"    Selection Metric  : {SELECTION_METRIC}")
    print(f"    Label Smoothing   : {LABEL_SMOOTHING}")
    print(f"    Mixup Alpha       : {MIXUP_ALPHA}")
    print(f"    Grad Accum Steps  : {GRAD_ACCUM_STEPS}")
    print(f"    Output Dir        : {OUTPUT_DIR}")
    print(f"    ALL Configs in    : {ALL_CONFIGS_DIR}")
    print(f"    BEST Config in    : {BEST_CONFIG_DIR}")

    X_train, X_test, y_train, y_test = load_data()

    # Species overlap check
    train_df = pd.read_csv(os.path.join(DATA_DIR, "train_data.csv"))
    test_df  = pd.read_csv(os.path.join(DATA_DIR, "test_data.csv"))
    train_sp = set(train_df['Organism'].unique())
    test_sp  = set(test_df['Organism'].unique())
    overlap  = train_sp.intersection(test_sp)
    print(f"\n  Species Split:")
    print(f"    Training species : {len(train_sp)}")
    print(f"    Test species     : {len(test_sp)}")
    print(f"    Overlap          : {len(overlap)} "
          f"({'✅ No overlap!' if len(overlap) == 0 else '⚠️ Overlap found!'})")

    best_params, best_config_dir = run_grid_search(
        X_train, y_train, X_test, y_test)

    best_results = run_best_configuration(
        X_train, y_train, X_test, y_test, best_params, best_config_dir)

    print("\n" + "=" * 70)
    print("  ✅ EXPERIMENTS COMPLETE!")
    print("=" * 70)
    print(f"\n  ALL Configurations saved in : {ALL_CONFIGS_DIR}")
    print(f"  BEST Configuration saved in : {BEST_CONFIG_DIR}")
    print(f"\n  Best Parameters: LR={best_params['lr']}, BS={best_params['bs']}")
    print("\n  📁 Output Files Summary:")
    print(f"    - BEST_CONFIG_DIR/csv_files/")
    print(f"      ├── final_comprehensive_summary.csv  (ALL models, ALL metrics)")
    print(f"      ├── paper_results_table.csv           (Formatted for paper)")
    print(f"      ├── *fold_summary.csv                 (Per-fold metrics per model)")
    print(f"      └── *best_config.csv                  (Full metrics + all folds)")
    print(f"    - BEST_CONFIG_DIR/history_files/")
    print(f"      └── *_fold*_history.npy               (Complete training history)")
    print("=" * 70)

    return {
        'best_params':     best_params,
        'best_config_dir': best_config_dir,
        'best_results':    best_results,
    }


if __name__ == "__main__":
    results = main()