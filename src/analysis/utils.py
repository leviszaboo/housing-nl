import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

dir_path = os.path.dirname(os.path.realpath(__file__))

phase_1_vars = ['avg_income', 'pop_density', 'unemp_rate', 'net_labor_participation',
                'homes_per_capita', 'multy_family', 'distance_to_urban_center', 'station_count']

phase_1_log = ['has_station', 'log_avg_income', 'log_homes_per_capita', 'log_multy_family', 'log_distance']

def calculate_vif_and_condition_indices(X: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    """
    Calculate the Variance Inflation Factor (VIF) and condition indices for a set of predictors.

    Parameters:
        X (pd.DataFrame): The predictors.
    
    Returns:
        pd.DataFrame: The VIF values for each predictor.
        np.ndarray: The condition indices.
    """
    _, singular_values, _ = np.linalg.svd(X)
    condition_indices = singular_values[0] / singular_values

    condition_indices = np.round(condition_indices, decimals=2)

    vif_data = pd.DataFrame()
    vif_data["feature"] = X.columns
    vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

    return vif_data, condition_indices

def residual_analysis(model: sm.regression.linear_model.RegressionResultsWrapper, fig_name: str) -> None:
    # Residuals vs. Fitted Values
    fitted_vals = model.fittedvalues
    residuals = model.resid

    plt.figure(figsize=(8, 12))

    # Residuals vs. Fitted Values
    plt.subplot(3, 1, 1)
    plt.scatter(fitted_vals, residuals, alpha=0.5)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('Fitted Values')
    plt.ylabel('Residuals')
    plt.title('Residuals vs Fitted Values')

    # Q-Q Plot
    plt.subplot(3, 1, 2)
    sm.qqplot(residuals, line='45', ax=plt.gca())
    plt.title('Q-Q Plot')

    # Cook's Distance
    cooks = model.get_influence().cooks_distance[0]
    plt.subplot(3, 1, 3)
    plt.stem(np.arange(len(cooks)), cooks, markerfmt=",")
    plt.xlabel('Observation')
    plt.ylabel("Cook's Distance")
    plt.title("Cook's Distance for Each Observation")

    plt.tight_layout()

    plt.savefig(os.path.join(dir_path, f'../../output/figures/{fig_name}'))
    