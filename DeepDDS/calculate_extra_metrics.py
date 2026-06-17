import sys
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_squared_error, r2_score
import torch
from torch_geometric.data import DataLoader

# IMPROVE imports
from improvelib.applications.synergy.config import SynergyInferConfig
import improvelib.utils as frm
from model_params_def import infer_params
from utils_test import TestbedDataset, predicting

filepath = Path(__file__).resolve().parent

def run(params):
    test_data_fname = frm.build_ml_data_file_name(data_format=params["data_format"], stage="test")
    modelpath = frm.build_model_path(
        model_file_name=params["model_file_name"],
        model_file_format=params["model_file_format"],
        model_dir=params["input_model_dir"])

    # Load data
    drug1_data_test = TestbedDataset(root=params['input_data_dir'], dataset='drug1_test')
    drug2_data_test = TestbedDataset(root=params['input_data_dir'], dataset='drug2_test')
    drug1_loader_test = DataLoader(drug1_data_test, batch_size=params["infer_batch"], shuffle=None)
    drug2_loader_test = DataLoader(drug2_data_test, batch_size=params["infer_batch"], shuffle=None)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load model and predict
    best_model = torch.load(modelpath, map_location=device)
    T, S, Y = predicting(best_model, device, drug1_loader_test, drug2_loader_test)

    # Load true continuous loewe scores from CSV
    ydf_fname = "test_y_data.csv"
    ydf_fpath = Path(params["input_data_dir"]) / ydf_fname
    rsp_df = pd.read_csv(ydf_fpath)

    # Ensure ydf matches T
    assert len(T) == rsp_df.shape[0], "Length mismatch between predictions and CSV!"
    
    # 1. Continuous targets
    y_true_loewe = rsp_df[params["y_col_name"]].values
    
    # 2. Binary targets
    y_true_binary = T  # Ground truth binary labels (0 or 1)
    
    # 3. Model predicted probabilities
    y_pred_prob = S   # Probability of being synergistic (class 1)
    
    # 4. Model predicted labels
    y_pred_binary = Y  # Binarized predictions (0 or 1)

    print("\n--- METRICAS DE REGRESION (Probabilidad de sinergia vs. Loewe continuo) ---")
    pcc_loewe, _ = pearsonr(y_true_loewe, y_pred_prob)
    scc_loewe, _ = spearmanr(y_true_loewe, y_pred_prob)
    print(f"PCC (Pearson) loewe vs. prob: {pcc_loewe:.4f}")
    print(f"SCC (Spearman) loewe vs. prob: {scc_loewe:.4f}")
    
    print("\n--- METRICAS DE REGRESION (Probabilidad de sinergia vs. Etiqueta binaria) ---")
    mse_bin = mean_squared_error(y_true_binary, y_pred_prob)
    rmse_bin = np.sqrt(mse_bin)
    pcc_bin, _ = pearsonr(y_true_binary, y_pred_prob)
    scc_bin, _ = spearmanr(y_true_binary, y_pred_prob)
    r2_bin = r2_score(y_true_binary, y_pred_prob)
    print(f"MSE: {mse_bin:.4f}")
    print(f"RMSE: {rmse_bin:.4f}")
    print(f"PCC (Pearson): {pcc_bin:.4f}")
    print(f"SCC (Spearman): {scc_bin:.4f}")
    print(f"R^2 (R-squared): {r2_bin:.4f}")

    print("\n--- METRICAS DE REGRESION (Etiqueta binaria predicha vs. Etiqueta binaria real) ---")
    mse_labels = mean_squared_error(y_true_binary, y_pred_binary)
    rmse_labels = np.sqrt(mse_labels)
    r2_labels = r2_score(y_true_binary, y_pred_binary)
    print(f"MSE: {mse_labels:.4f}")
    print(f"RMSE: {rmse_labels:.4f}")
    print(f"R^2 (R-squared): {r2_labels:.4f}")

    # Write scores to a json file
    results = {
        "loewe_vs_prob": {
            "pcc": float(pcc_loewe),
            "scc": float(scc_loewe)
        },
        "binary_vs_prob": {
            "mse": float(mse_bin),
            "rmse": float(rmse_bin),
            "pcc": float(pcc_bin),
            "scc": float(scc_bin),
            "r2": float(r2_bin)
        },
        "binary_vs_binary": {
            "mse": float(mse_labels),
            "rmse": float(rmse_labels),
            "r2": float(r2_labels)
        }
    }
    import json
    with open(Path(params["output_dir"]) / "extra_scores.json", "w") as f:
        json.dump(results, f, indent=4)
    print("\nMétricas guardadas con éxito en exp_result/extra_scores.json")

def main(args):
    cfg = SynergyInferConfig()
    params = cfg.initialize_parameters(
        pathToModelDir=filepath,
        default_config="deepdds_params.ini",
        additional_definitions=infer_params,
    )
    run(params)

if __name__ == "__main__":
    main(sys.argv[1:])
