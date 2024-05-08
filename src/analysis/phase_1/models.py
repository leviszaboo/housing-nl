import pandas as pd
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler
import numpy as np
import os
import logging
from src.analysis.phase_1.tables import create_reg_summaries
from src.analysis.utils import calculate_vif_and_condition_indices, residual_analysis, dir_path

def regression(X: pd.DataFrame, y: pd.Series) -> tuple[sm.regression.linear_model.RegressionResultsWrapper, pd.DataFrame, np.ndarray, float]:
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
    # make sure 'has_station' is an integer
    X['has_station'] = X['has_station'].astype(int)

    X = sm.add_constant(X)
    model = sm.OLS(y, X).fit()
    bic = model.bic
    vif, cond_indices = calculate_vif_and_condition_indices(X)
    
    return model, vif, cond_indices, bic

def simple(df: pd.DataFrame) -> None:
    """
    Perform a simple linear regression with the 'has_station' variable.

    Parameters:
        df (pd.DataFrame): The phase 1 dataset.
    
    Returns:
        None
    """
    X = df[['has_station','avg_income', 'multy_family']]
    X = sm.add_constant(X)

    y = df['m2_price']

    model, vif, cond_indices, bic = regression(X, y)

    residual_analysis(model, 'phase_1/simple_residuals.png')
    
    return model, vif, cond_indices, bic

def controls(df: pd.DataFrame) -> None:
    """
    Perform a multiple linear regression with the control variables.

    Parameters:
        df (pd.DataFrame): The phase 1 dataset.
    
    Returns:
        None
    """
    X = df[['has_station', 'avg_income', 'multy_family', 'homes_per_capita', 'distance_to_urban_center']]
    X = sm.add_constant(X)

    y = df['m2_price']

    model, vif, cond_indices, bic = regression(X, y)
    
    return model, vif, cond_indices, bic

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

    model, vif, cond_indices, bic = regression(X, y)
    
    return model, vif, cond_indices, bic

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

    model, vif, cond_indices, bic = regression(X, y)

    residual_analysis(model, 'phase_1/controls_log_residuals.png')
    
    return model, vif, cond_indices, bic

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

    model, vif, cond_indices, bic = regression(X, y)
    
    return model, vif, cond_indices, bic

def log_interaction_terms(df: pd.DataFrame) -> None:
    """
    Perform a multiple linear regression with interaction terms.

    Parameters:
        df (pd.DataFrame): The phase 1 dataset.
    
    Returns:
        None
    """
    X = df[['has_station', 'log_avg_income', 'log_multy_family', 'log_homes_per_capita', 'log_distance', 'station_x_multy_fam', 'station_x_distance']]

    y = df['log_m2_price']

    model, vif, cond_indices, bic = regression(X, y)
    
    return model, vif, cond_indices, bic

def log_interaction_terms_standardized(df: pd.DataFrame) -> None:
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

    model, vif, cond_indices, bic = regression(X, y)
    
    return model, vif, cond_indices, bic

output_path = os.path.join(dir_path, '../../output/tables/phase_1')

def create_score_summaries(df: pd.DataFrame) -> tuple[dict, dict]:
    """
    Create summaries of VIFs, condition indices, and BIC scores for the models.

    Parameters:
        df (pd.DataFrame): The phase 1 dataset.
    
    Returns:
        dict: Dictionary with the results for the non-log models.
        dict: Dictionary with the results for the log models.
    """
    non_log_results = {}
    log_results = {}
    models = {
        "simple": simple,
        "controls": controls,
        "standardized": standardized,
        "log_transformed": log_transformed,
        "log_standardized": log_standardized,
        "log_interaction_terms": log_interaction_terms,
        "log_interaction_terms_standardized": log_interaction_terms_standardized
    }

    logging.info("Summarizing VIFs, condition indices, and BIC scores for the models...")

    for name, func in models.items():
        model, vif, cond_indices, bic = func(df)
        # remove const from vif and cond_indices
        vif = vif[vif['feature'] != 'const']
        cond_indices = cond_indices[cond_indices != cond_indices[0]]

        if "log" in name:
            log_results[name] = {"model": model, "vif": vif, "cond_indices": cond_indices, "bic": bic}
        else:
            non_log_results[name] = {"model": model, "vif": vif, "cond_indices": cond_indices, "bic": bic}

    return non_log_results, log_results


def run_phase_1_regressions(df: pd.DataFrame) -> None:
    """
    Run the regressions for the Phase 1 analysis.

    Parameters:
        df (pd.DataFrame): The phase 1 dataset.
    
    Returns:
        None
    """
    logging.info("Running Phase 1 regressions...")

    # extract only model from tuple
    simple_model, _, _, _= simple(df)
    controls_model, _, _, _ = controls(df)
    controls_standardized_model, _, _, _ = standardized(df)

    logging.info("Creating summaries for Phase 1 simple models...")
    create_reg_summaries(simple_model, controls_model, controls_standardized_model, output_file=os.path.join(output_path, 'regression_summaries.tex'))

    controls_log_model, _, _, _ = log_transformed(df)
    controls_log_standardized_model, _, _, _= log_standardized(df)
    interaction_terms_model, _, _, _= log_interaction_terms(df)
    interaction_terms_standardized_model, _, _, _ = log_interaction_terms_standardized(df)

    logging.info("Creating summaries for Phase 1 log models...")
    create_reg_summaries(controls_log_model, controls_log_standardized_model, interaction_terms_model, interaction_terms_standardized_model, output_file=os.path.join(output_path, 'regression_summaries_log.tex'))

    logging.info("Phase 1 regressions complete.")

