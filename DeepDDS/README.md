# IMPROVE - DeepDDS: Drug Synergy Prediction

---

This repository demonstrates how to use the [IMPROVE library](https://jdacs4c-improve.github.io/docs/) for building a synergy prediction model using DeepDDS.


## Dependencies
Installation instructions are detailed below in [Step-by-step instructions](#step-by-step-instructions).


ML framework:
+ [PyTorch](https://pytorch.org/)

IMPROVE dependencies:
+ [IMPROVE](https://github.com/JDACS4C-IMPROVE/IMPROVE)

## Dataset
Benchmark data for Synergy can be downloaded from this [site](https://web.cels.anl.gov/projects/IMPROVE_FTP/candle/public/improve/benchmarks/synergy_data_v0.2.0).



# Step-by-step instructions

### 1. Clone the model repository and checkout the develop branch (or tag of your choice)
```bash
git clone https://github.com/JDACS4C-IMPROVE/DeepDDs
cd DeepDDs
git checkout develop
```

---

## Option 1: Using Local Conda Environment

### 2. Set computational environment
```bash
conda create -n deepdds python pytorch-gpu scikit-learn pandas pytorch_geometric pytorch_scatter seaborn rdkit pyyaml
conda activate deepdds
```

### 3. Preprocess benchmark data to construct model input data 
```bash
python deepdds_preprocess_improve.py --input_dir ./synergy_data_v0.2.0 --output_dir exp_result
```

Preprocesses the data and creates train, validation (val), and test datasets.

Generates:
* three model input data files
* three tabular data files, each containing the synergy values and corresponding metadata: `train_y_data.csv`, `val_y_data.csv`, `test_y_data.csv`

### 4. Train model
```bash
python deepdds_train_improve.py --input_dir exp_result --output_dir exp_result
```

Trains a model using the model input data.

Generates:
* trained model
* predictions on val data (tabular data): `val_y_data_predicted.csv`
* prediction performance scores on val data: `val_scores.json`

### 5. Run inference on test data with the trained model
```bash
python deepdds_infer_improve.py --input_data_dir exp_result --input_model_dir exp_result --output_dir exp_result --calc_infer_score true
```

Evaluates the performance on a test dataset with the trained model.

Generates:
* predictions on test data (tabular data): `test_y_data_predicted.csv`
* prediction performance scores on test data: `test_scores.json`

---

## Option 2: Using Docker (Windows PowerShell)

### 2. Build the Docker Image
Build the Docker image using the provided [Dockerfile](file:///c:/Users/mppar/TFG_Perez_Paralle_Maria/DeepDDS/Dockerfile):
```powershell
docker build -t deepdds .
```

### 3. Preprocess benchmark data to construct model input data
Run the preprocessing script by mounting the local library, dataset, and workspace directories:
```powershell
docker run --rm `
  -v "C:\Users\mppar\TFG_Perez_Paralle_Maria\IMPROVE_repo:/usr/src/IMPROVE_repo" `
  -v "C:\Users\mppar\TFG_Perez_Paralle_Maria\DeepDDS\synergy_data_v0.2.0:/usr/src/synergy_data_v0.2.0" `
  -v "C:\Users\mppar\TFG_Perez_Paralle_Maria\DeepDDS:/usr/src/app" `
  deepdds python deepdds_preprocess_improve.py --input_dir /usr/src/synergy_data_v0.2.0 --output_dir exp_result
```

### 4. Train DeepDDS model
Train the model using the preprocessed data inside the Docker container:
```powershell
docker run --rm `
  -v "C:\Users\mppar\TFG_Perez_Paralle_Maria\IMPROVE_repo:/usr/src/IMPROVE_repo" `
  -v "C:\Users\mppar\TFG_Perez_Paralle_Maria\DeepDDS\synergy_data_v0.2.0:/usr/src/synergy_data_v0.2.0" `
  -v "C:\Users\mppar\TFG_Perez_Paralle_Maria\DeepDDS:/usr/src/app" `
  deepdds python deepdds_train_improve.py --input_dir exp_result --output_dir exp_result
```

### 5. Run inference on test data with the trained DeepDDS model
Evaluate the performance on a test dataset using the trained model inside Docker:
```powershell
docker run --rm `
  -v "C:\Users\mppar\TFG_Perez_Paralle_Maria\IMPROVE_repo:/usr/src/IMPROVE_repo" `
  -v "C:\Users\mppar\TFG_Perez_Paralle_Maria\DeepDDS\synergy_data_v0.2.0:/usr/src/synergy_data_v0.2.0" `
  -v "C:\Users\mppar\TFG_Perez_Paralle_Maria\DeepDDS:/usr/src/app" `
  deepdds python deepdds_infer_improve.py --input_data_dir exp_result --input_model_dir exp_result --output_dir exp_result --calc_infer_score true
```

## References

Original GitHub: https://github.com/Sinwang404/DeepDDs
Original Paper: https://academic.oup.com/bib/article-abstract/23/1/bbab390/6375262

