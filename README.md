# Freight Rate Prediction Challenge

This repository contains a machine-learning solution for forecasting freight rates based on lane, equipment, distance, weight, and market conditions. The project uses labeled historical data to train a regression model, generate predictions for a validation set, and create the fixed December forecast chart required by the challenge scorer.

## Overview

The objective is to:

- train and validate a model on the provided historical freight data
- predict rates for every load in `dataset/validation.csv`
- complete the template file `dataset/validation-predictions-template.csv` as `validation_predictions.csv`
- generate predicted rates for the fixed December scenario in `dataset/december-chart-inputs.csv`
- validate the final outputs with the provided scoring script

This is a data-science challenge project rather than a web application, so the repository is organized around scripts, datasets, models, and notebooks.

## Project Structure

```text
ML_Assessment/
├── dataset/
│   ├── train-test.csv
│   ├── validation.csv
│   ├── validation-predictions-template.csv
│   └── december-chart-inputs.csv
├── models/
│   └── freight_rate_model_lgb.pkl
├── notebooks/
│   ├── eda.ipynb
│   └── Experiments.ipynb
├── scorer_results/
├── main.py
├── generate_predictions.py
├── score.py
├── requirements.txt
├── README.md
├── validation_predictions.csv
├── freight-rate-ml-assessment.pdf
└── .gitignore
```

## Data Description

The core training dataset is `dataset/train-test.csv`. It contains historical freight shipments with the following key fields:

- `load_id`: unique shipment identifier
- `pickup` and `delivery`: origin and destination cities
- `pickup_lat`, `pickup_lon`, `delivery_lat`, `delivery_lon`: exact coordinates
- `distance`: route distance in miles
- `equipment`: trailer type such as Dry Van, Reefer, or Flatbed
- `weight`: shipment weight
- `date`: trip date
- `market_index`: market-condition indicator
- `quote_signal`: pricing signal or demand pressure indicator
- `posted_rate`: target variable to predict

The validation dataset `dataset/validation.csv` contains the 12,000 loads to score and includes the same feature structure without `posted_rate`. The template `dataset/validation-predictions-template.csv` contains the required `load_id,predicted_rate` format to be filled.

The December prediction file `dataset/december-chart-inputs.csv` is a fixed scenario designed for a single route and date range. The challenge scorer validates that it contains exactly 31 daily rows from 2025-12-01 to 2025-12-31 with fixed values for pickup, delivery, distance, equipment, and weight.

## Modeling Approach

The implementation in `main.py` follows a straightforward ML pipeline:

1. Load the labeled freight dataset.
2. Remove extreme outliers in `posted_rate` to prevent the model from being dominated by a few unusually large shipments.
3. Engineer additional features:
   - haversine distance between pickup and delivery locations
   - expected base rate based on distance and quote signal
   - market pressure based on market index and quote signal
   - temporal features from the date such as month, day, day-of-week, and day-of-year
4. Convert categorical fields (`pickup`, `delivery`, `equipment`) to category types for LightGBM compatibility.
5. Split the data into train and validation sets with a fixed random state.
6. Train a LightGBM regressor on `log1p(posted_rate)` and convert predictions back with `expm1`.
7. Evaluate with MAE, RMSE, and R².
8. Save the trained model to `models/freight_rate_model_lgb.pkl`.

This is a strong choice for structured tabular data because LightGBM handles non-linear relationships, categorical features, and large datasets efficiently.

## Prediction Workflow

The repository includes a prediction script at `generate_predictions.py` that:

- loads the trained model
- loads the training reference data
- fills missing fields for the validation or December scenarios
- applies the same feature engineering as the training script
- predicts rates for the validation and December datasets
- saves `validation_predictions.csv`
- updates the December prediction file as required by the challenge

The score script `score.py` validates both expected output files and generates the chart `scorer_results/candidate_december.png`.

## Setup

Use a Python virtual environment to keep dependencies isolated.

```bash
cd ML_Assessment
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Usage

### 1) Train the model

```bash
python main.py
```

This script reads the training data, fits the LightGBM model, prints validation metrics, and saves the trained model.

### 2) Generate final predictions

```bash
python generate_predictions.py
```

This creates:

- `validation_predictions.csv`
- the updated December prediction dataset in `dataset/december-chart-inputs.csv`

### 3) Validate the outputs

```bash
python score.py --predictions validation_predictions.csv --december-predictions dataset/december-chart-inputs.csv
```

This checks:

- the validation file has the exact required row count and `load_id,predicted_rate` format
- the December file contains one row per day in the fixed range and all required constraints
- the chart is generated in `scorer_results/candidate_december.png`

## Outputs and Deliverables

The project is designed to produce these required outputs:

- `validation_predictions.csv`: complete final predictions for the 12,000 validation loads
- `scorer_results/candidate_december.png`: chart of the fixed December predictions
- a trained model in `models/freight_rate_model_lgb.pkl`

## Notebooks

The notebooks folder contains exploratory work and experiments used during development:

- `notebooks/eda.ipynb`: exploratory data analysis and quick inspection of dataset patterns
- `notebooks/Experiments.ipynb`: modeling experiments and feature-engineering trials

## Notes

- The project uses a fixed split and deterministic random state for reproducibility.
- Predictions are generated on the original scale using the inverse log transformation (`expm1`).
- The scoring script enforces strict validation rules, so the file format and values must be exact.

## Summary

This repository demonstrates a complete end-to-end freight-rate prediction workflow: data preparation, feature engineering, model training, prediction generation, and final validation. It is organized for easy reproduction and is ready to use for the challenge submission process.
