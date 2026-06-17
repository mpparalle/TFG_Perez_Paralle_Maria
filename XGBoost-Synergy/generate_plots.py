import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Set aesthetic style
sns.set_theme(style="whitegrid")
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'figure.titlesize': 18
})

# Load the data
df = pd.read_csv("exp_result/test_y_data_predicted.csv")
y_true = df["loewe_true"]
y_pred = df["loewe_pred"]
residuals = y_true - y_pred

# Output directory for images (the artifact folder)
out_dir = r"C:\Users\mppar\.gemini\antigravity\brain\2e849998-c850-4c7b-b4fe-1f5568b60f7e"

# 1. Scatter Plot
plt.figure(figsize=(8, 6), dpi=300)
# Use a hexbin or scatter with alpha for dense plots
plt.scatter(y_true, y_pred, alpha=0.4, color="#34495e", edgecolors="w", linewidth=0.5, label="Combinaciones de fármacos")
# Perfect prediction line
min_val = min(y_true.min(), y_pred.min())
max_val = max(y_true.max(), y_pred.max())
plt.plot([min_val, max_val], [min_val, max_val], color="#e74c3c", linestyle="--", linewidth=2.5, label="Predicción Perfecta (y = x)")

plt.title("XGBoost-Synergy: Sinergia Real vs Predicha (Loewe)", pad=15)
plt.xlabel("Sinergia Real (Loewe)")
plt.ylabel("Sinergia Predicha (Loewe)")
plt.xlim(min_val - 2, max_val + 2)
plt.ylim(min_val - 2, max_val + 2)

# Metrics text box
metrics_text = "$R^2$ Score: 0.628\nCorr. Pearson (PCC): 0.798\nRMSE: 8.867"
plt.gca().text(0.05, 0.95, metrics_text, transform=plt.gca().transAxes, fontsize=12,
            verticalalignment='top', bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8, edgecolor='#bdc3c7'))

plt.legend(loc="lower right", frameon=True, facecolor="white", edgecolor="#bdc3c7")
plt.tight_layout()
plt.savefig(f"{out_dir}\\scatter_plot.png", dpi=300)
plt.close()
print("Saved scatter_plot.png")

# 2. Density Plot Comparison
plt.figure(figsize=(8, 6), dpi=300)
sns.kdeplot(y_true, fill=True, color="#e74c3c", alpha=0.3, linewidth=2, label="Distribución Real (Loewe)")
sns.kdeplot(y_pred, fill=True, color="#3498db", alpha=0.3, linewidth=2, label="Distribución Predicha (Loewe)")

plt.title("Comparación de Densidad de Distribución", pad=15)
plt.xlabel("Valor de Sinergia (Loewe)")
plt.ylabel("Densidad")
plt.legend(loc="upper right", frameon=True, facecolor="white", edgecolor="#bdc3c7")
plt.tight_layout()
plt.savefig(f"{out_dir}\\density_plot.png", dpi=300)
plt.close()
print("Saved density_plot.png")

# 3. Residual Distribution
plt.figure(figsize=(8, 6), dpi=300)
sns.histplot(residuals, kde=True, color="#1abc9c", stat="density", alpha=0.5, edgecolor="w", linewidth=0.5)
plt.axvline(0, color="#2c3e50", linestyle="--", linewidth=2, label="Residuo Cero (Sin Error)")

plt.title("Distribución de Residuos (Errores de Predicción)", pad=15)
plt.xlabel("Residuo (Real - Predicho)")
plt.ylabel("Densidad")
plt.legend(loc="upper right", frameon=True, facecolor="white", edgecolor="#bdc3c7")
plt.tight_layout()
plt.savefig(f"{out_dir}\\residuals_plot.png", dpi=300)
plt.close()
print("Saved residuals_plot.png")
