# Spaceflights

[![Powered by Kedro](https://img.shields.io/badge/powered_by-kedro-ffc900?logo=kedro)](https://kedro.org)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![MLflow](https://img.shields.io/badge/MLflow-tracking-blue)](https://mlflow.org/)
[![DVC](https://img.shields.io/badge/DVC-data%20versioning-purple)](https://dvc.org/)

## Overview

**Spaceflights** is a machine learning project that predicts spaceflight shuttle prices using Linear Regression. Built with the Kedro framework, it demonstrates a complete ML pipeline including data processing, model training, evaluation, and experiment tracking.

The project uses spaceflight-themed datasets containing:
- **Shuttle specifications** - capacity, engines, crew, certifications
- **Company information** - ratings, locations, fleet count, approvals
- **Customer reviews** - ratings and pricing data

## Tech Stack

| Category | Technologies |
|----------|-------------|
| **ML Framework** | Kedro 1.1.1, scikit-learn 1.5.1 |
| **Data Processing** | pandas, NumPy |
| **Visualization** | matplotlib, seaborn, Plotly, Kedro-viz |
| **Experiment Tracking** | MLflow, kedro-mlflow |
| **Data Versioning** | DVC (Google Drive remote) |
| **Testing** | pytest |
| **Code Quality** | ruff |

## Project Structure

```
spaceflights/
├── src/spaceflights/
│   ├── pipelines/
│   │   ├── data_processing/    # Data cleaning and preprocessing
│   │   ├── data_science/       # Model training and evaluation
│   │   └── reporting/          # Visualizations
│   ├── hooks.py                # MLflow hooks
│   └── pipeline_registry.py    # Pipeline registration
├── conf/
│   ├── base/                   # Shared configuration
│   │   ├── catalog.yml         # Data catalog
│   │   └── parameters_*.yml    # Pipeline parameters
│   └── local/                  # Local configuration
│       └── mlflow.yml          # MLflow settings
├── data/
│   ├── 01_raw/                 # Raw datasets
│   ├── 02_intermediate/        # Preprocessed data
│   ├── 05_model_input/         # Train/test splits
│   ├── 06_models/              # Trained models
│   └── 08_reporting/           # Generated visualizations
├── tests/                      # Unit and integration tests
└── notebooks/                  # Jupyter notebooks
```

## Installation

### Prerequisites
- Python >= 3.10
- pip or conda

### Setup

```bash
# Clone the repository
git clone https://github.com/NabinRai4017/spaceflights.git
cd spaceflights

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Or install with development dependencies
pip install -e ".[dev]"
```

## Usage

### Run the Complete Pipeline

```bash
kedro run
```

### Run Specific Pipelines

```bash
# Data processing only
kedro run --pipeline=data_processing

# Model training and evaluation
kedro run --pipeline=data_science

# Generate reports and visualizations
kedro run --pipeline=reporting
```

### Visualize Pipeline

```bash
kedro viz run
```

### View MLflow Experiments

```bash
mlflow ui
# Opens at http://127.0.0.1:5000
```

## Pipelines

### 1. Data Processing Pipeline

Cleans and preprocesses raw data:
- Converts rating percentages to floats
- Parses price strings to numeric values
- Handles boolean certification flags
- Merges datasets into a single model input table

**Input:** `companies.csv`, `reviews.csv`, `shuttles.xlsx`
**Output:** `model_input_table` (Parquet)

### 2. Data Science Pipeline

Trains and evaluates the regression model:
- 80/20 train/test split
- Linear Regression model
- Comprehensive evaluation metrics:
  - R², MAE, RMSE, MAPE
  - Max Error, Median Absolute Error
  - Explained Variance

**Features used:**
- `engines`, `passenger_capacity`, `crew`
- `d_check_complete`, `moon_clearance_complete`
- `iata_approved`, `company_rating`, `review_scores_rating`

**Visualizations generated:**
- Feature coefficients bar chart
- Residuals vs predicted values
- Predictions vs actuals scatter plot
- Residuals distribution histogram

### 3. Reporting Pipeline

Generates additional visualizations:
- Passenger capacity comparison by shuttle type
- Interactive Plotly charts

## MLflow Integration

All experiments are tracked with MLflow:

- **Experiment name:** `spaceflights`
- **Registered model:** `spaceflights-regressor`
- **Tracked items:**
  - Model parameters and hyperparameters
  - Evaluation metrics
  - Visualization artifacts
  - System metrics (CPU, RAM, disk)

## Data Version Control (DVC)

Data files are version-controlled using DVC with Google Drive as remote storage:

```bash
# Pull data from remote
dvc pull

# Push data changes
dvc push
```

## Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov

# Run specific test
pytest tests/pipelines/data_science/test_pipeline.py
```

## Development

### Jupyter Notebooks

```bash
# Start Jupyter notebook with Kedro context
kedro jupyter notebook

# Or JupyterLab
kedro jupyter lab

# Or IPython
kedro ipython
```

Available variables in notebooks: `catalog`, `context`, `pipelines`, `session`

### Code Quality

```bash
# Lint code
ruff check src/

# Format code
ruff format src/
```

## Configuration

| File | Purpose |
|------|---------|
| `conf/base/catalog.yml` | Data input/output definitions |
| `conf/base/parameters_data_science.yml` | Model hyperparameters |
| `conf/local/mlflow.yml` | MLflow server configuration |

## License

This project is licensed under the MIT License.
