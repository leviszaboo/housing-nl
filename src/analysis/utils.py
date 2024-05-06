import os
import pandas as pd
import numpy as np
from statsmodels.stats.outliers_influence import variance_inflation_factor

dir_path = os.path.dirname(os.path.realpath(__file__))

phase_1_vars = ['avg_income', 'pop_density', 'unemp_rate', 'net_labor_participation',
                'homes_per_capita', 'multy_family', 'distance_to_urban_center', 'station_count']

phase_1_log = ['avg_income', 'pop_density', 'unemp_rate', 'net_labor_participation', 'homes_per_capita', 'multy_family',]

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