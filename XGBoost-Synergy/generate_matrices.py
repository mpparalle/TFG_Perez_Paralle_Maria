import pandas as pd
import numpy as np

# Load predictions
df = pd.read_csv("exp_result/test_y_data_predicted.csv")

print("--- Resumen de Predicciones ---")
print(f"Total de combinaciones evaluadas: {len(df)}")

# Find top 5 synergistic combinations predicted by XGBoost
print("\n--- TOP 5 Combinaciones Más Sinergistas Predictas (Mayor Loewe Predicho) ---")
top_syn = df.sort_values(by="loewe_pred", ascending=False).head(5)
for i, row in top_syn.iterrows():
    print(f"- Línea Celular: {row['DepMapID']} | {row['DrugID_row']} + {row['DrugID_col']} | Sinergia Predicha (Loewe): {row['loewe_pred']:.2f} (Real: {row['loewe_true']:.2f})")

# Let's select a cell line with many data points to build a matrix
cell_counts = df['DepMapID'].value_counts()
selected_cell = cell_counts.index[0] # The cell line with the most tests
cell_df = df[df['DepMapID'] == selected_cell]

print(f"\n--- Generando Matriz de Sinergia para la Línea Celular: {selected_cell} ({len(cell_df)} combinaciones) ---")

# Create a pivot table / matrix
# We handle both directions since drug combinations can be represented as DrugA + DrugB or DrugB + DrugA
matrix_data = []
drugs = sorted(list(set(cell_df['DrugID_row']).union(set(cell_df['DrugID_col']))))

# Initialize empty matrix
matrix = pd.DataFrame(np.nan, index=drugs, columns=drugs)

# Populate matrix
for i, row in cell_df.iterrows():
    d1, d2 = row['DrugID_row'], row['DrugID_col']
    val = row['loewe_pred']
    matrix.loc[d1, d2] = val
    matrix.loc[d2, d1] = val # symmetric

# Clean rows/columns with all NaNs to make it readable
matrix_clean = matrix.dropna(how='all', axis=0).dropna(how='all', axis=1)

# Save to CSV
matrix_file = "exp_result/synergy_matrix_selected.csv"
matrix_clean.to_csv(matrix_file)
print(f"Matriz guardada en: {matrix_file}")

# Display a dense snippet of the matrix (drugs with the most predictions)
print("\n--- Vista Parcial de la Matriz de Sinergia (Loewe Predicho) ---")
# Count non-NaNs per drug
non_nan_counts = matrix_clean.notna().sum()
# Get top 6 drugs with the most predictions
top_drugs = non_nan_counts.nlargest(6).index
snippet = matrix_clean.loc[top_drugs, top_drugs]
print(snippet.round(2))
