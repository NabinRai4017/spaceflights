import logging
import time

import mlflow
from mlflow.models.signature import infer_signature
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    max_error,
    mean_absolute_error,
    r2_score,
    mean_squared_error,
    explained_variance_score,
    median_absolute_error,
    mean_absolute_percentage_error,
)
from sklearn.model_selection import train_test_split

# Enable MLflow's built-in system metrics logging
# This will show metrics in the "System Metrics" section of MLflow UI
mlflow.enable_system_metrics_logging()

@mlflow.trace(name="split_data", span_type="DATA_PROCESSING")
def split_data(data: pd.DataFrame, parameters: dict) -> tuple:
    """Splits data into features and targets training and test sets.

    Args:
        data: Data containing features and target.
        parameters: Parameters defined in parameters/data_science.yml.
    Returns:
        Split data.
    """
    start_time = time.time()

    X = data[parameters["features"]]
    y = data["price"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=parameters["test_size"], random_state=parameters["random_state"]
    )

    # Log data statistics directly to MLflow
    data_stats = {
        "data.n_samples_total": len(data),
        "data.n_samples_train": len(X_train),
        "data.n_samples_test": len(X_test),
        "data.n_features": len(parameters["features"]),
    }
    mlflow.log_params(data_stats)

    # Log target statistics as metrics
    target_stats = {
        "target_mean": float(y.mean()),
        "target_std": float(y.std()),
        "target_min": float(y.min()),
        "target_max": float(y.max()),
    }
    mlflow.log_metrics(target_stats)

    # Log execution time
    execution_time = time.time() - start_time
    mlflow.log_metric("split_data_time_seconds", execution_time)

    logger = logging.getLogger(__name__)
    logger.info("Data split: %d train, %d test samples (%.2fs)", len(X_train), len(X_test), execution_time)

    return X_train, X_test, y_train, y_test


@mlflow.trace(name="train_model", span_type="MODEL_TRAINING")
def train_model(X_train: pd.DataFrame, y_train: pd.Series) -> tuple:
    """Trains the linear regression model.

    Args:
        X_train: Training data of independent features.
        y_train: Training data for price.

    Returns:
        Trained model and feature names for evaluation.
    """
    start_time = time.time()

    regressor = LinearRegression()
    regressor.fit(X_train, y_train)

    # Log model parameters directly to MLflow
    model_params = {
        "model.type": "LinearRegression",
        "model.fit_intercept": regressor.fit_intercept,
        "model.n_features_in": int(regressor.n_features_in_),
    }
    mlflow.log_params(model_params)

    # Log intercept as metric
    mlflow.log_metric("model_intercept", float(regressor.intercept_))

    # Log individual coefficients as metrics
    feature_names = X_train.columns.tolist()
    for name, coef in zip(feature_names, regressor.coef_):
        mlflow.log_metric(f"coef_{name}", float(coef))

    # Log training time
    training_time = time.time() - start_time
    mlflow.log_metric("train_model_time_seconds", training_time)

    logger = logging.getLogger(__name__)
    logger.info("Model trained with %d features (%.2fs)", len(feature_names), training_time)

    return regressor, feature_names


@mlflow.trace(name="evaluate_model", span_type="MODEL_EVALUATION")
def evaluate_model(
    regressor: LinearRegression,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    feature_names: list,
) -> tuple:
    """Calculates metrics and generates evaluation artifacts.

    Args:
        regressor: Trained model.
        X_test: Testing data of independent features.
        y_test: Testing data for price.
        feature_names: List of feature names for plotting.

    Returns:
        Tuple of metrics and matplotlib figures for artifacts.
    """
    start_time = time.time()

    y_pred = regressor.predict(X_test)
    residuals = y_test - y_pred

    # Calculate all metrics
    score = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    me = max_error(y_test, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
    mape = mean_absolute_percentage_error(y_test, y_pred) * 100  # as percentage
    explained_var = explained_variance_score(y_test, y_pred)
    median_ae = median_absolute_error(y_test, y_pred)

    logger = logging.getLogger(__name__)
    logger.info("Model has a coefficient R^2 of %.3f on test data.", score)
    logger.info("RMSE: %.3f, MAE: %.3f, MAPE: %.2f%%", rmse, mae, mape)

    # Create Feature Coefficients Plot
    fig_coefficients, ax1 = plt.subplots(figsize=(10, 6))
    coeffs = regressor.coef_
    colors = ['green' if c >= 0 else 'red' for c in coeffs]
    bars = ax1.barh(feature_names, coeffs, color=colors)
    ax1.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
    ax1.set_xlabel('Coefficient Value')
    ax1.set_ylabel('Feature')
    ax1.set_title('Linear Regression Feature Coefficients')
    for i, (coef, bar) in enumerate(zip(coeffs, bars)):
        ax1.text(coef, bar.get_y() + bar.get_height()/2, f'{coef:.2f}',
                va='center', ha='left' if coef >= 0 else 'right', fontsize=9)
    plt.tight_layout()

    # Create Residuals Plot
    fig_residuals, ax2 = plt.subplots(figsize=(10, 6))
    ax2.scatter(y_pred, residuals, alpha=0.5, edgecolors='k', linewidth=0.5)
    ax2.axhline(y=0, color='red', linestyle='--', linewidth=2)
    ax2.set_xlabel('Predicted Values')
    ax2.set_ylabel('Residuals')
    ax2.set_title('Residuals vs Predicted Values')
    ax2.grid(True, alpha=0.3)
    plt.tight_layout()

    # Create Predictions vs Actuals Plot
    fig_predictions, ax3 = plt.subplots(figsize=(10, 6))
    ax3.scatter(y_test, y_pred, alpha=0.5, edgecolors='k', linewidth=0.5)
    min_val = min(y_test.min(), y_pred.min())
    max_val = max(y_test.max(), y_pred.max())
    ax3.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
    ax3.set_xlabel('Actual Values')
    ax3.set_ylabel('Predicted Values')
    ax3.set_title(f'Predictions vs Actuals (R² = {score:.3f})')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    plt.tight_layout()

    # Create Residuals Distribution Histogram
    fig_residuals_hist, ax4 = plt.subplots(figsize=(10, 6))
    ax4.hist(residuals, bins=30, edgecolor='black', alpha=0.7)
    ax4.axvline(x=0, color='red', linestyle='--', linewidth=2)
    ax4.axvline(x=residuals.mean(), color='blue', linestyle='-', linewidth=2, label=f'Mean: {residuals.mean():.2f}')
    ax4.set_xlabel('Residual Value')
    ax4.set_ylabel('Frequency')
    ax4.set_title('Distribution of Residuals')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    plt.tight_layout()

    # Log evaluation time
    evaluation_time = time.time() - start_time
    mlflow.log_metric("evaluate_model_time_seconds", evaluation_time)

    logger.info("Model evaluation completed (%.2fs)", evaluation_time)

    # Return as tuple of simple lists for metrics and figures for artifacts
    return (
        [float(score)],         # r2_score
        [float(mae)],           # mae
        [float(me)],            # max_error
        [float(rmse)],          # rmse
        [float(mape)],          # mape
        [float(explained_var)], # explained_variance
        [float(median_ae)],     # median_absolute_error
        fig_coefficients,       # coefficients plot
        fig_residuals,          # residuals plot
        fig_predictions,        # predictions vs actuals plot
        fig_residuals_hist,     # residuals histogram
    )
