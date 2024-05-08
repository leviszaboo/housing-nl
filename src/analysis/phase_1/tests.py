import pandas as pd
import os 
import logging
from scipy.stats import shapiro, levene, ttest_ind, mannwhitneyu
from src.analysis.utils import dir_path

def run_tests(df: pd.DataFrame) -> dict:
    """
    Run the tests for the Phase 1 analysis.
    Checks for normality via a Shapiro-Wilk test.
    Checks for homogeneity of variance via a Levene test.
    Runs a t-test or a Mann-Whitney U test based on the normality test
    to compare the m² prices of municipalities with and without train stations.

    Parameters:
        df (pd.DataFrame): The main dataset.
    
    Returns:
        dict: A dictionary with test names, statistics, and p-values.
    """
    logging.info("Running tests...")

    with_station = df[df['has_station']]['m2_price']
    without_station = df[~df['has_station']]['m2_price']

    shapiro_with_station = shapiro(with_station)
    shapiro_without_station = shapiro(without_station)

    levene_test = levene(with_station, without_station)

    if shapiro_with_station.pvalue > 0.05 and shapiro_without_station.pvalue > 0.05 and levene_test.pvalue > 0.05:
        test_result = ttest_ind(with_station, without_station, equal_var=True)
        test_type = 'Independent t-test'
    else:
        test_result = mannwhitneyu(with_station, without_station, alternative='two-sided')
        test_type = 'Mann-Whitney U test'

    results = {
        'Normality Test (With Station) Statistic': shapiro_with_station.statistic,
        'Normality Test (With Station) P-value': shapiro_with_station.pvalue,
        'Normality Test (Without Station) Statistic': shapiro_without_station.statistic,
        'Normality Test (Without Station) P-value': shapiro_without_station.pvalue,
        'Levene Test for Equal Variances Statistic': levene_test.statistic,
        'Levene Test for Equal Variances P-value': levene_test.pvalue,
        'Statistical Test': test_type,
        'Test Statistic': test_result.statistic,
        'P-value': test_result.pvalue
    }

    logging.info("Tests complete.")
    return results


def save_test_results(results: dict, output_file: str) -> None:
    """
    Save the test results to a LaTeX-formatted table.

    Parameters:
        results (dict): The test results.
        output_filename (str): The filename to save the LaTeX output.

    Returns:
        None
    """
    logging.info("Saving test results as LaTeX table...")

    latex_table = """
    \\begin{table}
        \\centering
        \\caption{Statistical Test Results}
        \\label{tab:test_results}
        \\vspace{10pt}
        \\begin{tabular}{@{\extracolsep{5pt}} l c c}
            \\hline
            \\hline \\\[-1.8ex]
            \\textbf{Test Name} & \\textbf{Test Statistic} & \\textbf{P-value} \\\\
            \\hline \\\[-1.8ex]
    """

    latex_table += f'            Normality Test (With Station) & {results["Normality Test (With Station) Statistic"]:.4f} & {results["Normality Test (With Station) P-value"]:.4f} \\\\\n'
    latex_table += f'            Normality Test (Without Station) & {results["Normality Test (Without Station) Statistic"]:.4f} & {results["Normality Test (Without Station) P-value"]:.4f} \\\\\n'
    latex_table += f'            Levene Test for Equal Variances & {results["Levene Test for Equal Variances Statistic"]:.4f} & {results["Levene Test for Equal Variances P-value"]:.4f} \\\\\n'
    latex_table += f'            {results["Statistical Test"]} & {results["Test Statistic"]:.4f} & {results["P-value"]:.4f} \\\\\n'

    latex_table += """
            \\hline
            \\hline
        \\end{tabular}
    \\end{table}
    """

    with open(output_file, 'w') as f:
        f.write(latex_table)


output_path = os.path.join(dir_path, '../../output/tables/phase_1')

def run_and_save_tests(df: pd.DataFrame) -> None:
    """
    Run the tests and save the results to a LaTeX-formatted table.

    Parameters:
        df (pd.DataFrame): The main dataset.
        output_file (str): The filename to save the LaTeX output.

    Returns:
        None
    """
    results = run_tests(df)
    save_test_results(results, os.path.join(output_path, 'tests.tex'))

    return
