import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from torch_geometric.data import DataLoader
from scipy.stats import pearsonr

# IMPROVE imports
from improvelib.applications.synergy.config import SynergyInferConfig
import improvelib.utils as frm
from model_params_def import infer_params
from utils_test import TestbedDataset, predicting

filepath = Path(__file__).resolve().parent

def run(params):
    modelpath = Path("exp_result/model.pt")

    # Load test data
    drug1_data_test = TestbedDataset(root="exp_result", dataset='drug1_test')
    drug2_data_test = TestbedDataset(root="exp_result", dataset='drug2_test')
    drug1_loader_test = DataLoader(drug1_data_test, batch_size=256, shuffle=None)
    drug2_loader_test = DataLoader(drug2_data_test, batch_size=256, shuffle=None)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load model
    best_model = torch.load(modelpath, map_location=device)

    # 1. Normal order prediction (Drug 1, Drug 2)
    print("Running normal order inference (Drug 1, Drug 2)...")
    _, S_normal, _ = predicting(best_model, device, drug1_loader_test, drug2_loader_test)

    # 2. Swapped order prediction (Drug 2, Drug 1)
    print("Running swapped order inference (Drug 2, Drug 1)...")
    _, S_swapped, _ = predicting(best_model, device, drug2_loader_test, drug1_loader_test)

    # Load real loewe scores from CSV
    ydf_fname = "test_y_data.csv"
    ydf_fpath = Path("exp_result") / ydf_fname
    rsp_df = pd.read_csv(ydf_fpath)
    real_loewe = rsp_df["loewe"].values

    # Set up styling
    sns.set_theme(style="whitegrid")
    
    # -------------------------------------------------------------
    # Plot (a): Real Synergy Score vs. Predicted Synergy Probability
    # -------------------------------------------------------------
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=real_loewe, y=S_normal, alpha=0.5, color="teal", edgecolor=None)
    
    # Add a trend line (lowess)
    sns.regplot(x=real_loewe, y=S_normal, scatter=False, color="red", label="Trend Line")
    
    plt.title("Plot (a): Real Synergy Score (Loewe) vs. Predicted Probability", fontsize=14, pad=15)
    plt.xlabel("Real Synergy Score (Loewe)", fontsize=12)
    plt.ylabel("Predicted Synergy Probability (DeepDDS)", fontsize=12)
    plt.ylim(-0.05, 1.05)
    plt.legend()
    plt.tight_layout()
    plt.savefig("exp_result/scatter_real_vs_pred.png", dpi=300)
    plt.close()
    print("Saved Plot (a) to exp_result/scatter_real_vs_pred.png")

    # -------------------------------------------------------------
    # Plot (b): Synergy score obtained from different input order
    # -------------------------------------------------------------
    plt.figure(figsize=(8, 6))
    
    # Calculate PCC between the two orders to show symmetry score
    pcc_order, _ = pearsonr(S_normal, S_swapped)
    
    sns.scatterplot(x=S_normal, y=S_swapped, alpha=0.5, color="coral", edgecolor=None)
    
    # Draw identity line y = x
    lims = [0, 1]
    plt.plot(lims, lims, 'r--', alpha=0.75, zorder=3, label="Ideal Symmetry (y = x)")
    
    plt.title(f"Plot (b): Symmetry Check (Drug Order Invariance)\nPearson Correlation: {pcc_order:.4f}", fontsize=14, pad=15)
    plt.xlabel("Predicted Prob. with Order (Drug 1, Drug 2)", fontsize=12)
    plt.ylabel("Predicted Prob. with Swapped Order (Drug 2, Drug 1)", fontsize=12)
    plt.xlim(-0.05, 1.05)
    plt.ylim(-0.05, 1.05)
    plt.legend()
    plt.tight_layout()
    plt.savefig("exp_result/scatter_drug_order.png", dpi=300)
    plt.close()
    print("Saved Plot (b) to exp_result/scatter_drug_order.png")
    
    print("\n✅ Ambos scatter plots generados correctamente en exp_result/")

if __name__ == "__main__":
    run(None)
