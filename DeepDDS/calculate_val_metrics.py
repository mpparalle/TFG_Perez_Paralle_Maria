import sys
from pathlib import Path
import json
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import DataLoader

# IMPROVE imports
from improvelib.applications.synergy.config import SynergyInferConfig
import improvelib.utils as frm
from model_params_def import infer_params
from utils_test import TestbedDataset, predicting

filepath = Path(__file__).resolve().parent

def run(params):
    modelpath = Path("exp_result/model.pt")

    # Load validation data
    drug1_data_val = TestbedDataset(root="exp_result", dataset='drug1_val')
    drug2_data_val = TestbedDataset(root="exp_result", dataset='drug2_val')
    drug1_loader_val = DataLoader(drug1_data_val, batch_size=256, shuffle=None)
    drug2_loader_val = DataLoader(drug2_data_val, batch_size=256, shuffle=None)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load model and predict
    best_model = torch.load(modelpath, map_location=device)
    T, S, Y = predicting(best_model, device, drug1_loader_val, drug2_loader_val)

    # Compute validation scores using IMPROVE's official function
    val_scores = frm.compute_performance_scores(
        y_true=T,
        y_pred=Y,
        stage="val",
        metric_type="classification",
        output_dir="exp_result",
        y_prob=S
    )

    print("\n--- METRICAS DE VALIDACION CALCULADAS ---")
    print(json.dumps(val_scores, indent=4))

if __name__ == "__main__":
    run(None)
