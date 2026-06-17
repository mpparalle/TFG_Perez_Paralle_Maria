import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

print("=== INICIANDO PROCESAMIENTO DE RESULTADOS PARA EL TFG ===")

# 1. BUSCAR TODAS LAS MÉTRICAS DISPONIBLES
directorio_actual = "."
resultados = []

# Rastreamos todas las carpetas buscando archivos de puntuaciones (scores)
for raiz, carpetas, archivos in os.walk(directorio_actual):
    for archivo in archivos:
        if archivo.endswith("scores.json") or (archivo.startswith("val_scores") and archivo.endswith(".json")):
            ruta_completa = os.path.join(raiz, archivo)
            try:
                with open(ruta_completa, 'r', encoding='utf-8') as f:
                    datos = json.load(f)
                    # Identificamos de qué proceden según la carpeta
                    nombre_origen = os.path.basename(raiz) if os.path.basename(raiz) != "LGBM" else "Raiz"
                    datos["Origen"] = f"{nombre_origen} ({archivo})"
                    resultados.append(datos)
            except Exception as e:
                print(f"No se pudo leer el archivo {ruta_completa}: {e}")

if not resultados:
    print("No se encontraron datos validos de metricas en formato JSON.")
    exit()

# 2. GENERAR TABLA DE RESULTADOS
df = pd.DataFrame(resultados)
# Reordenar columnas para que el origen salga primero
columnas = ["Origen"] + [col for col in df.columns if col != "Origen"]
df = df[columnas]

print("\nTABLA DE METRICAS ENCONTRADAS:")
print(df.to_string(index=False))

# Guardar en CSV para Excel
df.to_csv("tabla_metricas_tfg.csv", index=False, encoding='utf-8')
print("\nTabla guardada con exito como 'tabla_metricas_tfg.csv' (puedes abrirla en Excel).")

# 3. GENERAR FIGURA EN FORMATO CIENTÍFICO (SCATTER PLOT REAL VS PREDICHO)
# Vamos a buscar si existen archivos de predicciones guardados (.csv o .tsv)
pred_archivo = None
for raiz, carpetas, archivos in os.walk(directorio_actual):
    for archivo in archivos:
        if "pred" in archivo.lower() and (archivo.endswith(".csv") or archivo.endswith(".tsv")):
            pred_archivo = os.path.join(raiz, archivo)
            break

if pred_archivo:
    print(f"\nGenerando grafico de dispersion usando predicciones de: {pred_archivo}")
    try:
        # Leer el archivo detectando si usa comas o tabuladores
        separador = "\t" if pred_archivo.endswith(".tsv") else ","
        df_pred = pd.read_csv(pred_archivo, sep=separador)
        
        # El framework IMPROVE suele llamar a las columnas 'y_true' e 'y_pred' o similar.
        # Intentamos detectar los nombres de forma automática:
        col_real = [c for c in df_pred.columns if "true" in c.lower() or "actual" in c.lower() or "target" in c.lower()][0]
        col_pred = [c for c in df_pred.columns if "pred" in c.lower()][0]
        
        # Configurar estilo visual para el TFG (estilo paper científico)
        sns.set_theme(style="ticks")
        plt.figure(figsize=(7, 6))
        
        # Dibujar nube de puntos con transparencia para evitar solapamiento masivo
        sns.scatterplot(x=df_pred[col_real], y=df_pred[col_pred], alpha=0.4, color="#2b5c8f", edgecolor="none")
        
        # Línea de identidad ideal (45 grados)
        min_val = min(df_pred[col_real].min(), df_pred[col_pred].min())
        max_val = max(df_pred[col_real].max(), df_pred[col_pred].max())
        plt.plot([min_val, max_val], [min_val, max_val], color="#d95f02", linestyle="--", linewidth=2, label="Prediccion Perfecta")
        
        # Etiquetas y diseño formal
        plt.xlabel("Valor Real de Respuesta a la Droga (AUC)", fontsize=12, fontweight="bold")
        plt.ylabel("Valor Predicho por LightGBM (AUC)", fontsize=12, fontweight="bold")
        plt.title("Grafico de Dispersion: Valores Reales vs. Predicciones\n(Modelo LightGBM - Prediccion de Respuesta a Farmacos)", fontsize=13, pad=15)
        plt.legend(loc="upper left", frameon=True)
        sns.despine() # Quita los bordes superior y derecho para estética limpia
        plt.tight_layout()
        
        # Guardar la imagen en alta definición para tu documento de Word/LaTeX
        plt.savefig("grafico_dispersion_tfg.png", dpi=300)
        plt.savefig("grafico_dispersion_tfg.pdf") # Formato vectorial ideal por si tu TFG requiere máxima calidad
        print("¡Grafico cientifico guardado como 'grafico_dispersion_tfg.png' y 'grafico_dispersion_tfg.pdf'!")
        
    except Exception as e:
        print(f"No se pudo generar el gráfico de dispersión de forma automática: {e}")
        print("No te preocupes, esto ocurre si los nombres de las columnas del CSV de predicciones varían.")
else:
    print("\nNo se localizo ningun archivo de predicciones (*pred*.csv/tsv) para dibujar el scatter plot.")
    print("Para graficarlo manualmente, indícame si tienes archivos que terminen en '_predictions.csv'.")