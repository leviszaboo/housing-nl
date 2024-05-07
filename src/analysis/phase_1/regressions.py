import pandas as pd
from src.analysis.utils import calculate_vif_and_condition_indices, residual_analysis
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler
import numpy as np

def regression(X: pd.DataFrame, y: pd.Series) -> tuple[sm.regression.linear_model.RegressionResultsWrapper, pd.DataFrame, np.ndarray]:
    """
    Perform a multiple linear regression.

    Parameters:
        X (pd.DataFrame): The predictors.
        y (pd.Series): The target variable.

    Returns:
        sm.regression.linear_model.RegressionResultsWrapper: The regression results.
        pd.DataFrame: The VIF values for each predictor.
        np.ndarray: The condition indices.
    """
    X = sm.add_constant(X)
    model = sm.OLS(y, X).fit()
    vif, cond_indices = calculate_vif_and_condition_indices(X)
    
    return model, vif, cond_indices

def simple(df: pd.DataFrame) -> None:
    """
    Perform a simple linear regression with the 'has_station' variable.

    Parameters:
        df (pd.DataFrame): The phase 1 dataset.
    
    Returns:
        None
    """
    X = df[['has_station', 'multy_family', 'homes_per_capita', 'distance_to_urban_center']]
    X = sm.add_constant(X)

    y = df['m2_price']

    model, vif, cond_indices = regression(X, y)

    model_summary = model.summary()

    residual_analysis(model, 'phase_1/simple_residuals.png')
    
    return model_summary, vif, cond_indices

def standardized(df: pd.DataFrame) -> None:
    """
    Perform a multiple linear regression with standardized predictors.

    Parameters:
        df (pd.DataFrame): The phase 1 dataset.
    
    Returns:
        None
    """
    X = df[['has_station', 'avg_income', 'multy_family', 'homes_per_capita', 'distance_to_urban_center']]

    scaler = StandardScaler()

    cols = X.columns

    X = scaler.fit_transform(X)
    X = pd.DataFrame(X, columns=cols)
    X = sm.add_constant(X)

    y = df['m2_price']

    model, vif, cond_indices = regression(X, y)

    model_summary = model.summary()
    
    return model_summary, vif, cond_indices

def log_transformed(df: pd.DataFrame) -> None:
    """
    Perform a multiple linear regression with log-transformed predictors and dependent variable.

    Parameters:
        df (pd.DataFrame): The phase 1 dataset.

    Returns:
        None
    """
    X = df[['has_station', 'log_avg_income', 'log_multy_family', 'log_homes_per_capita', 'log_distance']]

    y = df['log_m2_price']

    model, vif, cond_indices = regression(X, y)

    model_summary = model.summary()

    residual_analysis(model, 'phase_1/controls_log_residuals.png')
    
    return model_summary, vif, cond_indices

def log_standardized(df: pd.DataFrame) -> None:
    """
    Perform a multiple linear regression with standardized log-transformed predictors and dependent variable.

    Parameters:
        df (pd.DataFrame): The phase 1 dataset.
    
    Returns:
        None
    """
    X = df[['has_station', 'log_avg_income', 'log_multy_family', 'log_homes_per_capita', 'log_distance']]

    scaler = StandardScaler()

    cols = X.columns

    X = scaler.fit_transform(X)
    X = pd.DataFrame(X, columns=cols)
    X = sm.add_constant(X)

    y = df['log_m2_price']

    model, vif, cond_indices = regression(X, y)

    model_summary = model.summary()
    
    return model_summary, vif, cond_indices

def interaction_terms(df: pd.DataFrame) -> None:
    """
    Perform a multiple linear regression with interaction terms.

    Parameters:
        df (pd.DataFrame): The phase 1 dataset.
    
    Returns:
        None
    """
    X = df[['has_station', 'log_avg_income', 'log_multy_family', 'log_homes_per_capita', 'log_distance', 'station_x_multy_fam', 'station_x_distance']]

    y = df['log_m2_price']

    model, vif, cond_indices = regression(X, y)

    model_summary = model.summary()
    
    return model_summary, vif, cond_indices

def interaction_terms_standardized(df: pd.DataFrame) -> None:
    """
    Perform a multiple linear regression with standardized interaction terms.

    Parameters:
        df (pd.DataFrame): The phase 1 dataset.
    
    Returns:
        None
    """
    X = df[['has_station', 'log_avg_income', 'log_multy_family', 'log_homes_per_capita', 'log_distance', 'station_x_multy_fam', 'station_x_distance']]

    scaler = StandardScaler()

    cols = X.columns

    X = scaler.fit_transform(X)
    X = pd.DataFrame(X, columns=cols)
    X = sm.add_constant(X)

    y = df['log_m2_price']

    model, vif, cond_indices = regression(X, y)

    model_summary = model.summary()
    
    return model_summary, vif, cond_indices

def run_phase_1_regressions(df: pd.DataFrame) -> None:
    """
    Run the regressions for the Phase 1 analysis.

    Parameters:
        df (pd.DataFrame): The phase 1 dataset.
    
    Returns:
        None
    """
    print("Running phase 1 regressions...")

    # just to make sure the variable is binary
    df['has_station'] = df['has_station'].astype(int)

    simple_summary, simple_vif, simple_cond_indices = simple(df)
    controls_standardized_summary, controls_standardized_vif, controls_standardized_cond_indices = standardized(df)
    controls_log_summary, controls_log_vif, controls_log_cond_indices = log_transformed(df)
    controls_log_standardized_summary, controls_log_standardized_vif, controls_log_standardized_cond_indices = log_standardized(df)
    interaction_terms_summary, interaction_terms_vif, interaction_terms_cond_indices = interaction_terms(df)
    interaction_terms_standardized_summary, interaction_terms_standardized_vif, interaction_terms_standardized_cond_indices = interaction_terms_standardized(df)
    
    # TODO: Save the results

