# TFG: Predicción de Sinergia de Drogas (Framework IMPROVE)

Este repositorio contiene el espacio de trabajo para el Trabajo de Fin de Grado (TFG) de **María Pérez Parallé**, centrado en la implementación y comparación de modelos de Predicción de Respuesta a Drogas (DRP) y sinergia de medicamentos bajo el framework estandarizado **IMPROVE**.

---

## 📂 Estructura del Proyecto

El espacio de trabajo está compuesto por las siguientes carpetas y repositorios independientes:

* **[IMPROVE_repo](./IMPROVE_repo)**: Contiene la librería principal `improvelib`. Es el framework que estandariza la entrada de datos, la estructura del preprocesamiento y las métricas de evaluación para comparar los distintos modelos de manera rigurosa.
* **[LGBM-Synergy](./LGBM-Synergy)**: Modelo de predicción de sinergia de drogas basado en **LightGBM**.
* **[XGBoost-Synergy](./XGBoost-Synergy)**: Modelo de predicción de sinergia de drogas basado en **XGBoost**.
* **[DeepDDS](./DeepDDS)**: Modelo de aprendizaje profundo (Deep Learning) basado en redes neuronales sobre grafos (GNN) para la predicción de sinergia.
* **[LGBM](./LGBM)**: Modelo base de LightGBM (clasificación y regresión estándar).

> [!NOTE]
> Cada una de estas subcarpetas contiene su propio archivo `README.md` detallado con instrucciones específicas sobre hiperparámetros, configuración y particularidades de dicho modelo.

---

## ⚙️ Entornos Virtuales (`venv`)

Para ejecutar los modelos, se han configurado entornos virtuales locales en Python (versión de sistema recomendada: **Python 3.9.13**):

### 1. Entorno de LightGBM Synergy (`LGBM-Synergy/venv`)
Este entorno contiene todas las dependencias científicas necesarias y está enlazado en modo de desarrollo (editable) con la librería local `improvelib` de `IMPROVE_repo`.
* **Activación**:
  ```powershell
  # En PowerShell (Windows):
  .\LGBM-Synergy\venv\Scripts\Activate.ps1
  ```

### 2. Entorno de DeepDDS (`DeepDDS/venv`)
Entorno virtual limpio preparado para las dependencias específicas de Deep Learning (como PyTorch y PyTorch Geometric).
* **Activación**:
  ```powershell
  # En PowerShell (Windows):
  .\DeepDDS\venv\Scripts\Activate.ps1
  ```

---

## 🚀 Flujo de Trabajo Común (Workflow)

El framework **IMPROVE** define un flujo de trabajo estándar en tres fases para todos los modelos. A continuación se muestra un ejemplo de uso con el modelo de LightGBM (asegúrate de tener el entorno virtual activo):

### 1. Preprocesamiento de datos
Toma los datos del dataset bruto (por ejemplo, `synergy_data_v0.2.0`) y genera archivos estructurados en formato `.parquet` y `.csv` para el entrenamiento.
```powershell
cd LGBM-Synergy
.\venv\Scripts\python lgbmsynergy_preprocess_improve.py --input_dir ./synergy_data_v0.2.0 --output_dir exp_result
```

### 2. Entrenamiento del modelo
Entrena el modelo (por ejemplo, LightGBM) usando el set de entrenamiento generado en la fase anterior y guarda el modelo entrenado (`model.txt`).
```powershell
.\venv\Scripts\python lgbmsynergy_train_improve.py --input_dir exp_result --output_dir exp_result
```

### 3. Inferencia y Evaluación (Test)
Aplica el modelo entrenado sobre los datos de test, calcula las métricas de rendimiento y guarda las predicciones.
```powershell
.\venv\Scripts\python lgbmsynergy_infer_improve.py --input_data_dir exp_result --input_model_dir exp_result --output_dir exp_result --calc_infer_score true
```

*Los demás modelos (`XGBoost-Synergy`, `DeepDDS`) siguen esta misma estructura de archivos de ejecución (ej: `xgboostsynergy_preprocess_improve.py`, etc.).*

---

## 📊 Verificación del Entorno

Para comprobar que todo está correctamente configurado y que los enlaces de las librerías funcionan, puedes situarte en la carpeta del modelo y ejecutar la inferencia de prueba con los datos precalculados:

```powershell
cd LGBM-Synergy
.\venv\Scripts\python lgbmsynergy_infer_improve.py --input_data_dir exp_result --input_model_dir exp_result --output_dir exp_result --calc_infer_score true
```

Si todo está bien, la terminal mostrará la finalización exitosa del proceso de inferencia y las puntuaciones del modelo (como `mse`, `rmse`, `r2`, etc.).
