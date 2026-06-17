# LGBM-Synergy

This repository demonstrates how to use the [IMPROVE library](https://jdacs4c-improve.github.io/docs/) for building a synergy prediction model using LightGBM (LGBM).


## Dependencies
Installation instructions are detailed below in [Step-by-step instructions](#step-by-step-instructions).

Conda `yml` file [lgbm_environment.yml](./lgbm_environment.yml)

ML framework:
+ [LightGBM](https://lightgbm.readthedocs.io/en/stable/) - machine learning framework for building the prediction model

IMPROVE dependencies:
+ [IMPROVE](https://github.com/JDACS4C-IMPROVE/IMPROVE)

## Dataset
Benchmark data for Synergy can be downloaded from this [site](https://web.cels.anl.gov/projects/IMPROVE_FTP/candle/public/improve/benchmarks/synergy_data_v0.2.0).


## Model scripts and parameter file
+ `lgbmsynergy_preprocess_improve.py` - takes benchmark data files and transforms into files for trianing and inference
+ `lgbmsynergy_train_improve.py` - trains a LightGBM-based DRP model
+ `lgbmsynergy_infer_improve.py` - runs inference with the trained LightGBM model
+ `model_params_def.py` - definitions of parameters that are specific to the model
+ `lgbmsynergy_params.txt` - default parameter file (parameter values specified in this file override the defaults)



# Step-by-step instructions

### 1. Clone the model repository and checkout the develop branch (or tag of your choice)
```bash
git clone https://github.com/JDACS4C-IMPROVE/LGBM-Synergy
cd LGBM-Synergy
git checkout develop
```


### 2. Set computational environment
Create conda env using `yml`
```bash
conda env create -f lgbm_environment.yml
```



### 4. Preprocess benchmark data to construct model input data 
```bash
python lgbmsynergy_preprocess_improve.py --input_dir ./synergy_data_v0.2.0 --output_dir exp_result
```

Preprocesses the data and creates train, validation (val), and test datasets.

Generates:
* three model input data files: `train_data.parquet`, `val_data.parquet`, `test_data.parquet`
* three tabular data files, each containing the synergy values and corresponding metadata: `train_y_data.csv`, `val_y_data.csv`, `test_y_data.csv`



### 5. Train LightGBM model
```bash
python lgbmsynergy_train_improve.py --input_dir exp_result --output_dir exp_result
```

Trains a LightGBM model using the model input data: `train_data.parquet` (training), `val_data.parquet` (early stopping).

Generates:
* trained model: `model.txt`
* predictions on val data (tabular data): `val_y_data_predicted.csv`
* prediction performance scores on val data: `val_scores.json`


### 6. Run inference on test data with the trained LightGBM model
```bash
python lgbmsynergy_infer_improve.py --input_data_dir exp_result --input_model_dir exp_result --output_dir exp_result --calc_infer_score true
```

Evaluates the performance on a test dataset with the trained model.

Generates:
* predictions on test data (tabular data): `test_y_data_predicted.csv`
* prediction performance scores on test data: `test_scores.json`
