# 🧬 Enzyme-Net: A Novel Transformer–MLP Hybrid Framework for Fish Enzyme Classification

## 📌 Overview

**Enzyme-Net** is a complete deep learning and bioinformatics pipeline designed for **enzyme function prediction in fish species** using **1024-dimensional UniProt protein embeddings**.

The framework introduces a **novel Transformer–MLP hybrid architecture** with advanced feature learning mechanisms and is evaluated using **species-aware splitting, cross-validation, ablation studies, and statistical significance testing**.

---

## 🚀 Key Contributions

### 🧠 Machine Learning Innovation
- Transformer–MLP hybrid architecture (Enzyme-Net)
- Multi-Scale Feature Extraction (MSFE)
- Dynamic Feature Gating (DFG)
- Feature-Level Multi-Head Attention
- Residual Feature Processing
- Ensemble-based classification head

---

### 🧬 Biological Rigor
- UniProt fish protein dataset
- 8-class EC enzyme classification (0–7)
- Species-aware train/test split (NO leakage)
- 19 unseen test species evaluation

---

### 📊 Experimental Design
- 10-fold stratified cross-validation
- Hyperparameter grid search
- 8 ablation variants
- Statistical significance testing:
  - Friedman Test
  - Wilcoxon Signed-Rank Test
  - Mann–Whitney U Test

---

### 📈 Publication-Ready Outputs
- 7 high-quality figures (PNG + TIFF)
- LaTeX-ready tables
- Model ranking analysis
- ROC analysis (8-class OVR)

---

## 🧬 Dataset

- Source: UniProt (reviewed fish proteins)
- Embeddings: 1024-d protein embeddings (HDF5)
- Classes:
  - 0 → Non-enzyme
  - 1–7 → EC enzyme classes

---

## 🏗️ Repository Structure

```
Enzyme-Net/
│
├── 01_fish_uniprot_analysis.py
│   → EC classification + dataset exploration
│
├── 02_species_distribution_analysis.py
│   → Fish species profiling + biological validation
│
├── 03_species_aware_split.py
│   → Train/test split (20% unseen species)
│
├── 04_enzyme_net_model.py
│   → Enzyme-Net architecture + baselines
│
├── 05_results_aggregation.py
│   → Combine CV + grid search results
│
├── 06_statistical_analysis.py
│   → Friedman + Wilcoxon + Mann–Whitney tests
│
├── 07_results_visualization_viewer.py
│   → Full results dashboard + tables
│
├── 08_generate_enzyme_net_figures.py
│   → Publication-ready figures generator
│
├── data/
│   ├── UniProt TSV files
│   ├── EC classification CSV
│   └── 1024-d protein embeddings (HDF5)
│
├── results/
│   ├── CV results
│   ├── ablation studies
│   ├── grid search outputs
│   └── final summaries
│
└── figures/
    ├── PNG/
    └── TIFF/
```

---

## ⚙️ Installation

```bash
git clone https://github.com/yourusername/Enzyme-Net.git
cd Enzyme-Net

pip install -r requirements.txt
```

---

## 📦 Requirements

```
numpy
pandas
scikit-learn
torch
matplotlib
seaborn
scipy
h5py
tabulate
```

---

## 🔄 Workflow Pipeline

### 1️⃣ Data Analysis
- UniProt dataset loading
- EC class mapping
- Species distribution analysis

### 2️⃣ Data Preparation
- Species-aware train/test split
- Embedding extraction (1024-d)
- Feature scaling

### 3️⃣ Model Training
- EnzymeNet training
- Baseline models:
  - Logistic Regression
  - Vanilla MLP
  - DNN Baseline

### 4️⃣ Evaluation
- 10-fold cross-validation
- Grid search optimization
- Performance metrics:
  - Accuracy
  - Precision
  - Recall
  - F1-score
  - MCC
  - AUC

### 5️⃣ Statistical Analysis
- Friedman test (multi-model comparison)
- Wilcoxon test (pairwise comparison)
- Mann–Whitney U test (distributional comparison)

### 6️⃣ Visualization
- Training curves
- ROC curves (8-class)
- Ablation study plots
- Model ranking charts

### 7️⃣ Figure Generation
- Publication-ready figures (300 DPI)
- PNG + TIFF formats
- Journal submission compliant outputs

---

## 📊 Key Results

| Model | Test MCC |
|------|----------|
| Logistic Regression | ~0.85 |
| Vanilla MLP | ~0.91 |
| DNN Baseline | ~0.90 |
| **EnzymeNet (Ours)** | **~0.93** |
| Best Ablation (w/o Ensemble) | **~0.93+** |

---

## 🧪 Ablation Study

The following components were evaluated:

- Without Multi-Scale Feature Extraction (MSFE)
- Without Dynamic Feature Gating (DFG)
- Without Attention Module
- Without Ensemble Head
- Without Residual Connections

👉 Result: Each component contributes significantly to final performance.

---

## 📈 Statistical Validation

Significance confirmed using:

- ✔ Friedman Test (global comparison)
- ✔ Wilcoxon Signed-Rank Test (pairwise)
- ✔ Mann–Whitney U Test (distribution comparison)

👉 EnzymeNet shows statistically significant improvement over baselines.

---

## 📊 Figures Generated

1. Model Performance Comparison  
2. Training Curves (Accuracy + Loss)  
3. 10-Fold Cross-Validation Results  
4. Hyperparameter Grid Search Heatmap  
5. Model Ranking with Error Bars  
6. ROC Curves (8-class classification)  
7. Ablation Study Results  

---

## 🧬 Biological Significance

- First species-aware enzyme classification system using fish proteomes
- Prevents dataset leakage via strict species separation
- Demonstrates generalization to unseen species
- Supports computational enzymology and functional annotation

---

## 🏆 Highlights

✔ Novel Transformer–MLP hybrid architecture  
✔ Species-level generalization evaluation  
✔ Strong statistical validation framework  
✔ Publication-ready figures and tables  
✔ Full reproducibility pipeline  

---

## 📌 Citation

If you use this work, please cite:

```
Enzyme-Net: A Novel Transformer–MLP Hybrid Framework for Fish Enzyme Classification

```

---

## 📬 Contact

For questions or collaboration:
- Author:Anjum Shahzad
- Email: anjumstat@yahoo.com
```

