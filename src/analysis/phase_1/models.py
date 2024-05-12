import pandas as pd
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler
from scipy import stats
import numpy as np
import os
import logging
from src.analysis.tables import create_reg_summaries, create_test_score_summary
from src.analysis.utils import calculate_vif_and_condition_indices, get_largest_cooks_indices, residual_analysis, dir_path

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

    X = X.reset_index(drop=True)
    y = y.reset_index(drop=True)

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

    residual_analysis(model, 'phase_1/resid/simple_residuals.png')
    
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
    X = df[['has_station', 'log_avg_income', 'log_multy_family', 'homes_per_capita', 'log_distance']]

    y = df['log_m2_price']

    model, vif, cond_indices, bic = regression(X, y)

    residual_analysis(model, 'phase_1/resid/controls_log_residuals.png')
    
    return model, vif, cond_indices, bic

def log_standardized(df: pd.DataFrame) -> None:
    """
    Perform a multiple linear regression with standardized log-transformed predictors and dependent variable.

    Parameters:
        df (pd.DataFrame): The phase 1 dataset.
    
    Returns:
        None
    """
    X = df[['has_station', 'log_avg_income', 'log_multy_family', 'homes_per_capita', 'log_distance']]

    scaler = StandardScaler()

    cols = X.columns

    X = scaler.fit_transform(X)
    X = pd.DataFrame(X, columns=cols)
    X = sm.add_constant(X)

    y = df['log_m2_price']

    model, vif, cond_indices, bic = regression(X, y)
    
    return model, vif, cond_indices, bic

def log_interaction(df: pd.DataFrame) -> None:
    """
    Perform a multiple linear regression with interaction terms.

    Parameters:
        df (pd.DataFrame): The phase 1 dataset.
    
    Returns:
        None
    """
    X = df[['has_station', 'log_avg_income', 'log_multy_family', 'homes_per_capita', 'log_distance', 'station_x_multy_fam', 'station_x_distance']]

    y = df['log_m2_price']

    model, vif, cond_indices, bic = regression(X, y)
    
    return model, vif, cond_indices, bic

def log_int_standardized(df: pd.DataFrame) -> None:
    """
    Perform a multiple linear regression with standardized interaction terms.

    Parameters:
        df (pd.DataFrame): The phase 1 dataset.
    
    Returns:
        None
    """
    X = df[['has_station', 'log_avg_income', 'log_multy_family', 'homes_per_capita', 'log_distance', 'station_x_multy_fam', 'station_x_distance']]

    scaler = StandardScaler()

    cols = X.columns

    X = scaler.fit_transform(X)
    X = pd.DataFrame(X, columns=cols)
    X = sm.add_constant(X)

    y = df['log_m2_price']

    model, vif, cond_indices, bic = regression(X, y)
    
    return model, vif, cond_indices, bic

def log_dropped(df: pd.DataFrame, cooks_indices: list[float]) -> None:
    """
    Perform a multiple linear regression with log-transformed predictors and dependent variable, dropping outliers.

    Parameters:
        df (pd.DataFrame): The phase 1 dataset.
    
    Returns:
        None
    """
    df = df.drop(df.index[cooks_indices])

    X = df[['has_station', 'log_avg_income', 'log_multy_family', 'homes_per_capita', 'log_distance']]

    y = df['log_m2_price']

    model, vif, cond_indices, bic = regression(X, y)

    residual_analysis(model, 'phase_1/resid/log_dropped_residuals.png')
    
    return model, vif, cond_indices, bic

def log_standardized_dropped(df: pd.DataFrame, cooks_indices) -> None:
    """
    Perform a multiple linear regression with standardized log-transformed predictors and dependent variable, dropping outliers.

    Parameters:
        df (pd.DataFrame): The phase 1 dataset.
    
    Returns:
        None
    """
    df = df.drop(df.index[cooks_indices])

    X = df[['has_station', 'log_avg_income', 'log_multy_family', 'homes_per_capita', 'log_distance']]

    scaler = StandardScaler()

    cols = X.columns

    X = scaler.fit_transform(X)
    X = pd.DataFrame(X, columns=cols)
    X = sm.add_constant(X)

    y = df['log_m2_price']

    model, vif, cond_indices, bic = regression(X, y)
    
    return model, vif, cond_indices, bic

def log_int_dropped(df: pd.DataFrame, cooks_indices) -> None:
    """
    Perform a multiple linear regression with interaction terms, dropping outliers.

    Parameters:
        df (pd.DataFrame): The phase 1 dataset.
    
    Returns:
        None
    """
    df = df.drop(df.index[cooks_indices])

    X = df[['has_station', 'log_avg_income', 'log_multy_family', 'homes_per_capita', 'log_distance', 'station_x_multy_fam', 'station_x_distance']]

    y = df['log_m2_price']

    model, vif, cond_indices, bic = regression(X, y)
    
    return model, vif, cond_indices, bic

def log_int_standardized_dropped(df: pd.DataFrame, cooks_indices) -> None:
    """
    Perform a multiple linear regression with standardized interaction terms, dropping outliers.

    Parameters:
        df (pd.DataFrame): The phase 1 dataset.
    
    Returns:
        None
    """
    df = df.drop(df.index[cooks_indices])

    X = df[['has_station', 'log_avg_income', 'log_multy_family', 'homes_per_capita', 'log_distance', 'station_x_multy_fam', 'station_x_distance']]

    scaler = StandardScaler()

    cols = X.columns

    X = scaler.fit_transform(X)
    X = pd.DataFrame(X, columns=cols)
    X = sm.add_constant(X)

    y = df['log_m2_price']

    model, vif, cond_indices, bic = regression(X, y)
    
    return model, vif, cond_indices, bic

def log_int_cent(df: pd.DataFrame) -> None:
    """
    Perform a multiple linear regression with interaction terms and centered interaction variables to avoid multicollinearity.

    Parameters:
        df (pd.DataFrame): The phase 1 dataset.
    
    Returns:
        None
    """
    X = df[['has_station', 'log_avg_income', 'c_log_multy_fam', 'c_log_distance', 'homes_per_capita', 'station_x_c_multy_fam', 'station_x_c_distance']]

    y = df['log_m2_price']

    model, vif, cond_indices, bic = regression(X, y)
    
    return model, vif, cond_indices, bic

def log_int_std_cent(df: pd.DataFrame) -> None:
    """
    Perform a multiple linear regression with standardized interaction terms and centered interaction variables to avoid multicollinearity.

    Parameters:
        df (pd.DataFrame): The phase 1 dataset.
    
    Returns:
        None
    """
    X = df[['has_station', 'log_avg_income', 'c_log_multy_fam', 'c_log_distance', 'homes_per_capita', 'station_x_c_multy_fam', 'station_x_c_distance']]

    scaler = StandardScaler()

    cols = X.columns

    X = scaler.fit_transform(X)
    X = pd.DataFrame(X, columns=cols)
    X = sm.add_constant(X)

    y = df['log_m2_price']

    model, vif, cond_indices, bic = regression(X, y)
    
    return model, vif, cond_indices, bic

def log_int_dropped_cent(df: pd.DataFrame, cooks_indices) -> None:
    """
    Perform a multiple linear regression with interaction terms and centered interaction variables to avoid multicollinearity, dropping outliers.

    Parameters:
        df (pd.DataFrame): The phase 1 dataset.
    
    Returns:
        None
    """
    df = df.drop(df.index[cooks_indices])

    X = df[['has_station', 'log_avg_income', 'c_log_multy_fam', 'c_log_distance', 'homes_per_capita', 'station_x_c_multy_fam', 'station_x_c_distance']]

    y = df['log_m2_price']

    model, vif, cond_indices, bic = regression(X, y)
    
    return model, vif, cond_indices, bic

def log_int_std_dropped_cent(df: pd.DataFrame, cooks_indices) -> None:
    """
    Perform a multiple linear regression with standardized interaction terms and centered interaction variables to avoid multicollinearity, dropping outliers.

    Parameters:
        df (pd.DataFrame): The phase 1 dataset.
    
    Returns:
        None
    """
    df = df.drop(df.index[cooks_indices])

    X = df[['has_station', 'log_avg_income', 'c_log_multy_fam', 'c_log_distance', 'homes_per_capita', 'station_x_c_multy_fam', 'station_x_c_distance']]

    scaler = StandardScaler()

    cols = X.columns

    X = scaler.fit_transform(X)
    X = pd.DataFrame(X, columns=cols)
    X = sm.add_constant(X)

    y = df['log_m2_price']

    model, vif, cond_indices, bic = regression(X, y)
    
    return model, vif, cond_indices, bic

output_path = os.path.join(dir_path, '../../output/tables/phase_1')

def create_score_summaries(df: pd.DataFrame, cooks: list[float], cooks_int: list[float]) -> tuple[dict, dict, dict]:
    """
    Create summaries of VIFs, condition indices, and BIC scores for the models.

    Parameters:
        df (pd.DataFrame): The phase 1 dataset.
    
    Returns:
        dict: Dictionary with the results for the non-log models.
        dict: Dictionary with the results for the log models.
        dict: Dictionary with the results for dropped models.
        dict: Dictionary with the results for centered models.
    """
    non_log_results = {}
    log_results = {}
    dropped_results = {}
    centered_results = {}

    models = {
        "simple": simple,
        "controls": controls,
        "standardized": standardized,
        "log_transformed": log_transformed,
        "log_standardized": log_standardized,
        "log_interaction": log_interaction,
        "log_int_standardized": log_int_standardized,
        "log_dropped": log_dropped,
        "log_standardized_dropped": log_standardized_dropped,
        "log_int_dropped": log_int_dropped,
        "log_int_standardized_dropped": log_int_standardized_dropped,
        "log_int_cent": log_int_cent,
        "log_int_std_cent": log_int_std_cent,
        "log_int_dropped_cent": log_int_dropped_cent,
        "log_int_std_dropped_cent": log_int_std_dropped_cent
    }

    logging.info("Summarizing VIFs, condition indices, and BIC scores for the models...")

    for name, func in models.items():
        if "dropped" in name:
            args = (df, cooks_int) if "int" in name else (df, cooks)
        else:
            args = (df,)

 
        model, vif, cond_indices, bic = func(*args)

        vif = vif[vif['feature'] != 'const']
        cond_indices = cond_indices[cond_indices != cond_indices[0]]

        results_dict = centered_results if "cent" in name else \
                       dropped_results if "dropped" in name else \
                       log_results if "log" in name else non_log_results

        results_dict[name] = {"model": model, "vif": vif, "cond_indices": cond_indices, "bic": bic}

    return non_log_results, log_results, dropped_results, centered_results


def get_model_scores(df: pd.DataFrame) -> tuple[dict, dict, dict]:
    """
    Get the model scores for the Phase 1 analysis.

    Parameters:
        df (pd.DataFrame): The phase 1 dataset.

    Returns:
        dict: Dictionary with the results for the non-log models.
        dict: Dictionary with the results for the log models.
        dict: Dictionary with the results for the models with outliers removed.
    """
    logging.info("Getting model scores...")

    controls_log_model, _, _, _ = log_transformed(df)
    interaction_model, _, _, _ = log_interaction(df)

    cooks_indices = get_largest_cooks_indices(controls_log_model, 10)
    cooks_indices_int = get_largest_cooks_indices(interaction_model, 10)

    non_log_results, log_results, dropped_results, centered_results = create_score_summaries(df, cooks_indices, cooks_indices_int)

    return non_log_results, log_results, dropped_results, centered_results

def extract_test_scores(*results) -> dict:
    """
    Extract the Omnibus, Jarque-Bera, R2, and Adjusted R2 scores from the statsmodels results objects.

    Parameters:
        *results (RegressionResultsWrapper): The regression results objects from statsmodels.
    
    Returns:
        dict: Dictionary with the test scores for each model.
    """
    logging.info("Extracting test scores...")
    test_scores = {}

    for i, result in enumerate(results):
        residuals = result.resid
        omnibus_stat, omnibus_pvalue = stats.normaltest(residuals)
        jarque_bera_stat, jarque_bera_pvalue = stats.jarque_bera(residuals)

        model_name = f'model_{i+1}'  
        test_scores[model_name] = {
            'Omnibus': omnibus_stat,
            'Omnibus_p_value': omnibus_pvalue,
            'Jarque_Bera': jarque_bera_stat,
            'Jarque_Bera_p_value': jarque_bera_pvalue,
            'Durbin_Watson': sm.stats.stattools.durbin_watson(residuals),  
            'R2': result.rsquared,  
            'Adjusted_R2': result.rsquared_adj  
        }

    return test_scores

def run_phase_1_regressions(df: pd.DataFrame) -> None:
    """
    Run the regressions for the Phase 1 analysis.

    Parameters:
        df (pd.DataFrame): The phase 1 dataset.
    
    Returns:
        None
    """
    logging.info("Running Phase 1 regressions...")

    # SIMPLE MODELS
    simple_model, _, _, _= simple(df)
    controls_model, _, _, _ = controls(df)
    controls_standardized_model, _, _, _ = standardized(df)

    logging.info("Creating summaries for Phase 1 simple models...")
    create_reg_summaries(simple_model, controls_model, controls_standardized_model, 
                         output_file=os.path.join(output_path, 'reg_tables/regression_summaries.tex'))

    simple_scores = extract_test_scores(simple_model, controls_model)  

    create_test_score_summary(simple_scores, output_file=os.path.join(output_path, 'test_tables/test_scores_simple.tex'))

    # LOG MODELS
    controls_log_model, _, _, _ = log_transformed(df)
    controls_log_standardized_model, _, _, _= log_standardized(df)
    interaction_model, _, _, _= log_interaction(df)
    int_standardized_model, _, _, _ = log_int_standardized(df)

    logging.info("Creating summaries for Phase 1 log models...")
    create_reg_summaries(controls_log_model, controls_log_standardized_model, interaction_model, 
                        int_standardized_model, output_file=os.path.join(output_path, 'reg_tables/regression_summaries_log.tex'))

    cooks_indices = get_largest_cooks_indices(controls_log_model, 10)
    cooks_indices_int = get_largest_cooks_indices(interaction_model, 10)

    # LOG MODELS WITH OUTLIERS REMOVED
    log_dropped_model, _, _, _ = log_dropped(df, cooks_indices)
    log_standardized_dropped_model, _, _, _ = log_standardized_dropped(df, cooks_indices)
    log_int_dropped_model, _, _, _ = log_int_dropped(df, cooks_indices_int)
    log_int_standardized_dropped_model, _, _, _ = log_int_standardized_dropped(df, cooks_indices_int)

    logging.info("Creating summaries for Phase 1 log models with outliers removed...")
    create_reg_summaries(log_dropped_model, log_standardized_dropped_model, 
                         log_int_dropped_model, log_int_standardized_dropped_model, 
                         output_file=os.path.join(output_path, 'reg_tables/regression_summaries_dropped.tex'))

    log_scores = extract_test_scores(controls_log_model, interaction_model,
                                            log_dropped_model, log_int_dropped_model)
    
    create_test_score_summary(log_scores, output_file=os.path.join(output_path, 'test_tables/test_scores_log.tex'))
    
    # LOG MODELS WITH CENTERED INTERACTION TERMS
    log_cent_model, _, _, _ = log_int_cent(df)
    log_std_cent_model, _, _, _ = log_int_std_cent(df)
    log_dropped_cent_model, _, _, _ = log_int_dropped_cent(df, cooks_indices_int)
    log_std_dropped_cent_model, _, _, _ = log_int_std_dropped_cent(df, cooks_indices_int)

    logging.info("Creating summaries for Phase 1 log models with centered interaction terms...")
    create_reg_summaries(log_cent_model, log_std_cent_model, log_dropped_cent_model, 
                         log_std_dropped_cent_model, output_file=os.path.join(output_path, 'reg_tables/regression_summaries_cent_log.tex'))

    logging.info("Phase 1 regressions complete.")

