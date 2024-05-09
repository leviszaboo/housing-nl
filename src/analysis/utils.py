import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

dir_path = os.path.dirname(os.path.realpath(__file__))

phase_1_vars = ['avg_income', 'homes_per_capita', 'multy_family', 'distance_to_urban_center']

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
    """
    Create a set of plots to analyze the residuals of a regression model.

    Parameters:
        model (sm.regression.linear_model.RegressionResultsWrapper): The regression model.
        fig_name (str): The name of the output figure.
    
    Returns:
        None
    """
    fitted_vals = model.fittedvalues
    residuals = model.resid
    mean_residuals = np.mean(residuals)

    plt.figure(figsize=(12, 14)) 
    gs = plt.GridSpec(2, 2, height_ratios=[1, 1])

    # Residuals vs. Fitted Values
    ax1 = plt.subplot(gs[0, 0])
    sns.scatterplot(x=fitted_vals, y=residuals, alpha=0.5, ax=ax1)
    ax1.axhline(y=mean_residuals, color='#404040', linestyle='--')
    ax1.set_xlabel('Fitted Values')
    ax1.set_ylabel('Residuals')
    ax1.set_title('Residuals vs Fitted Values')

    # Histogram of residuals
    ax2 = plt.subplot(gs[0, 1], sharey=ax1)
    sns.histplot(y=residuals, ax=ax2)
    ax2.axhline(y=mean_residuals, color='#404040', linestyle='--')
    ax2.set_title('Residuals Distribution')
    ax2.set_xlabel('Frequency')
    ax2.set_ylabel('')  
    ax2.yaxis.set_label_position("right")
    ax2.yaxis.tick_right()

    # Cook's Distance
    ax3 = plt.subplot(gs[1, :])
    cooks = model.get_influence().cooks_distance[0]
    ax3.stem(np.arange(len(cooks)), cooks, markerfmt=",")
    ax3.get_lines()[1].set_color("#404040")
    ax3.set_xlabel('Observation')
    ax3.set_ylabel("Cook's Distance")
    ax3.set_title("Cook's Distance for Each Observation")

    plt.tight_layout()

    plt.savefig(os.path.join(dir_path, f'../../output/figures/{fig_name}'))
    
def get_largest_cooks_indices(model: sm.regression.linear_model.RegressionResultsWrapper, x: float) -> list[int]:
    """
    Get the indices of the observations with the largest Cook's distance.

    Parameters:
        model (sm.regression.linear_model.RegressionResultsWrapper): The regression model.
        x (float): The x largest cooks index to return.
    
    Returns:
        list[int]: The indices of the observations with the largest Cook's distance.
    """
    cooks = model.get_influence().cooks_distance[0]
    return np.argsort(cooks)[-(x):]