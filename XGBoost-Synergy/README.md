# XGBoost-Synergy

---

Este repositorio muestra cómo utilizar la [librería IMPROVE](https://jdacs4c-improve.github.io/docs/) para construir un modelo de predicción de sinergia de fármacos utilizando XGBoost.


## Dependencias
Las instrucciones de instalación se detallan a continuación en [Instrucciones paso a paso](#instrucciones-paso-a-paso).


Framework de ML:
+ [SciKit-Learn](https://scikit-learn.org/)

Dependencias de IMPROVE:
+ [IMPROVE](https://github.com/JDACS4C-IMPROVE/IMPROVE)

## Dataset
Los datos de referencia (benchmark) para Synergy se pueden descargar de este [sitio](https://web.cels.anl.gov/projects/IMPROVE_FTP/candle/public/improve/benchmarks/synergy_data_v0.2.0). En este experimento, se utiliza específicamente la partición **Split 0 del dataset de O'Neil** (`ONEIL_split_0`), el cual viene preconfigurado en el archivo [xgboostsynergy_params.ini](file:///C:/Users/mppar/Antigravity%20TFG/XGBoost-Synergy/xgboostsynergy_params.ini).



# Instrucciones paso a paso (usando Docker en Windows)

### 1. Clonar el repositorio del modelo y cambiar a la rama develop (o etiqueta de su elección)
```powershell
git clone https://github.com/JDACS4C-IMPROVE/XGBoost-Synergy
cd XGBoost-Synergy
git checkout develop
```

### 2. Configurar el entorno computacional (Construir la imagen de Docker)
Construya la imagen de Docker utilizando el [Dockerfile](file:///C:/Users/mppar/Antigravity%20TFG/XGBoost-Synergy/Dockerfile) provisto:
```powershell
docker build -t xgboost-synergy .
```

### 3. Preprocesar los datos de referencia para construir los datos de entrada del modelo
Ejecute el script de preprocesamiento montando las carpetas locales de la librería, el dataset y el espacio de trabajo actual:
```powershell
docker run --rm `
  -v "C:\Users\mppar\Antigravity TFG\IMPROVE_repo:/usr/src/IMPROVE_repo" `
  -v "C:\Users\mppar\Antigravity TFG\DeepDDS\synergy_data_v0.2.0:/usr/src/synergy_data_v0.2.0" `
  -v "C:\Users\mppar\Antigravity TFG\XGBoost-Synergy:/usr/src/app" `
  xgboost-synergy python xgboostsynergy_preprocess_improve.py --input_dir /usr/src/synergy_data_v0.2.0 --output_dir exp_result
```

Preprocesa los datos para la partición configurada en [xgboostsynergy_params.ini](file:///C:/Users/mppar/Antigravity%20TFG/XGBoost-Synergy/xgboostsynergy_params.ini) (por defecto, **Split 0 de O'Neil**: `ONEIL_split_0_train.txt`, `ONEIL_split_0_val.txt`, `ONEIL_split_0_test.txt`) y crea los conjuntos de datos de entrenamiento (train), validación (val) y prueba (test).

Genera:
* Tres archivos de datos de entrada para el modelo (`train_data.parquet`, `val_data.parquet`, `test_data.parquet`)
* Tres archivos de datos tabulares que contienen los valores de sinergia y metadatos correspondientes: `train_y_data.csv`, `val_y_data.csv`, `test_y_data.csv`

### 4. Entrenar el modelo
Entrene el modelo utilizando los datos preprocesados dentro del contenedor Docker:
```powershell
docker run --rm `
  -v "C:\Users\mppar\Antigravity TFG\IMPROVE_repo:/usr/src/IMPROVE_repo" `
  -v "C:\Users\mppar\Antigravity TFG\DeepDDS\synergy_data_v0.2.0:/usr/src/synergy_data_v0.2.0" `
  -v "C:\Users\mppar\Antigravity TFG\XGBoost-Synergy:/usr/src/app" `
  xgboost-synergy python xgboostsynergy_train_improve.py --input_dir exp_result --output_dir exp_result
```

Entrena un modelo utilizando los datos de entrada del modelo.

Genera:
* Modelo entrenado (`model.json`)
* Predicciones sobre los datos de validación (datos tabulares): `val_y_data_predicted.csv`
* Puntuaciones de rendimiento de predicción sobre los datos de validación: `val_scores.json`

### 5. Ejecutar inferencia en datos de prueba con el modelo entrenado
Evalúe el rendimiento del modelo entrenado sobre el conjunto de datos de prueba:
```powershell
docker run --rm `
  -v "C:\Users\mppar\Antigravity TFG\IMPROVE_repo:/usr/src/IMPROVE_repo" `
  -v "C:\Users\mppar\Antigravity TFG\DeepDDS\synergy_data_v0.2.0:/usr/src/synergy_data_v0.2.0" `
  -v "C:\Users\mppar\Antigravity TFG\XGBoost-Synergy:/usr/src/app" `
  xgboost-synergy python xgboostsynergy_infer_improve.py --input_data_dir exp_result --input_model_dir exp_result --output_dir exp_result --calc_infer_score true
```

Genera:
* Predicciones sobre los datos de prueba (datos tabulares): `test_y_data_predicted.csv`
* Puntuaciones de rendimiento de predicción sobre los datos de prueba: `test_scores.json`
