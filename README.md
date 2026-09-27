# Enzyme-Net: A Species-Aware Feature-Level Attention Transformer-MLP Framework for Enzyme Classification from Protein Embeddings

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![Deep Learning](https://img.shields.io/badge/Deep%20Learning-PyTorch-red)
![Bioinformatics](https://img.shields.io/badge/Application-Computational%20Proteomics-green)
![License](https://img.shields.io/badge/License-MIT-yellow)

## Overview

This repository contains the implementation of **Enzyme-Net**, a novel deep learning framework for species-aware enzyme classification from protein sequence embeddings.

The framework utilizes pretrained protein language model embeddings and introduces a hybrid **Transformer-MLP architecture** with feature-level attention mechanisms for improved biological function prediction.

The proposed model is designed for **8-class enzyme classification**:

- Class 0: Non-enzyme proteins
- Classes 1–7: EC enzyme classes

The complete pipeline includes:

1. Fish UniProt dataset processing
2. Protein class annotation
3. Species-aware dataset splitting
4. Deep learning model training
5. Baseline comparison
6. Ablation studies
7. Statistical validation
8. Publication-quality result generation


---

# Model Overview

## Enzyme-Net Architecture

Enzyme-Net integrates several novel components:

### 1. Multi-Scale Feature Extraction (MSFE)

Extracts complementary patterns from high-dimensional protein embeddings using parallel feature processing pathways.

### 2. Dynamic Feature Gating (DFG)

Learns adaptive feature importance and selects biologically informative embedding dimensions.

### 3. Feature-Level Multi-Head Attention

Treats embedding features as tokens and learns interactions between protein representation dimensions.

### 4. Residual Feature Processor

Uses skip connections to improve feature transformation and gradient flow.

### 5. Learnable Ensemble Heads

Combines multiple prediction heads using adaptive weighting.

---

# Dataset

The framework uses fish protein sequences obtained from UniProt.

Dataset characteristics:

- Taxonomic group: Actinopterygii (fish)
- Protein source: UniProt reviewed proteins
- Representation: 1024-dimensional protein embeddings
- Task: 8-class enzyme classification

The dataset preparation scripts analyze protein distributions, EC classes, and species composition.

---

# Repository Structure
Enzyme-Net/
│
├── 01_fish_uniprot_analysis.py
│
├── 02_species_distribution_Analysis.py
│
├── 03_Species_aware_Split.py
│
├── 04_Enzyme_net_model.py
│
├── 05_results_aggrgation.py
│
├── 06_Statistical_Analysis.py
│
├── 07_article_tables.py
│
├── 08_article_figures.py
│
├── README.md
│
└── Results/

---

# Pipeline Workflow

UniProt Fish Protein Dataset
              |
              |
              v
Protein Sequence Processing
              |
              |
              v
1024-D Protein Embeddings
              |
              |
              v
Species-Aware Train/Test Split
              |
              |
              v
Baseline Models
(Logistic Regression,
 Vanilla MLP,
 DNN)
              |
              |
              v
Enzyme-Net
(MSFE + DFG + Attention + Residual + Ensemble)
              |
              |
              v
Performance Evaluation
              |
              |
              v
Statistical Analysis
              |
              |
              v
Publication Figures and Tables

---

# Installation

Clone repository:

```bash
git clone https://github.com/yourusername/Enzyme-Net.git

cd Enzyme-Net
conda create -n enzymenet python=3.9

conda activate enzymenet
pip install -r requirements.txt
Required Libraries
Main dependencies:
numpy
pandas
scikit-learn
scipy
matplotlib
seaborn
torch
h5py
Usage
Step 1: Analyze UniProt Dataset
Run:
python 01_fish_uniprot_analysis.py
This performs:
- Protein loading
- EC class extraction
- 8-class classification analysis
- Species statistics
Step 2: Species Distribution Analysis
python 02_species_distribution_Analysis.py

Generates:
- Species composition
- Protein distribution
- Fish group statistics
Step 3: Species-Aware Data Split
python 03_Species_aware_Split.py

Creates:
- Training dataset
- Independent species-aware test dataset
- Scaled 1024-dimensional embeddings
Step 4: Train Enzyme-Net
python 04_Enzyme_net_model.py

The model evaluates:
Baselines
- Logistic Regression
- Vanilla MLP
- DNN Baseline
Proposed Model
- Enzyme-Net
Ablation Variants
- Without MSFE
- Without DFG
- Without Attention
- Without Ensemble
- Without Residuals
- Higher Dropout
- Lower Dropout
Step 5: Aggregate Results
python 05_results_aggrgation.py

Produces:
- Combined CV results
- Best configurations
- Model comparison tables
Step 6: Statistical Analysis
python 06_Statistical_Analysis.py

Evaluates:
- Accuracy
- Precision
- Recall
- F1-score
- MCC
- AUC
Using fold-level statistical comparisons.
Step 7: Generate Article Tables
python 07_article_tables.py

Creates manuscript-ready tables.
Step 8: Generate Figures
python 08_article_figures.py

Generates:
- Model performance comparison
- Training curves
- Hyperparameter heatmaps
- Ranking plots
- ROC curves
- Ablation figures
Experimental Design
Validation Strategy
- Species-aware split
- 10-fold cross-validation
- Independent species testing
Evaluation Metrics
The framework reports:
- Accuracy
- Precision
- Recall
- F1-score
- Matthews Correlation Coefficient (MCC)
- ROC-AUC
Reproducibility
All experiments use:
- Fixed random seed
- Deterministic training settings
- Saved model configurations
- Complete fold-level histories
Citation
If you use this repository, please cite:
@article{enzymenet2026,
title={Enzyme-Net: A Species-Aware Feature-Level Attention Transformer-MLP Framework for Enzyme Classification from Protein Embeddings},
author={Your Name},
journal={},
year={2026}
}

Contact
For questions regarding this project:
Author: H.A.R
License
This project is released under the MIT License.
