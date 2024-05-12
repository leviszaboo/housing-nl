import pandas as pd
import os
import statsmodels.api as sm
from src.analysis.utils import dir_path

def create_summary_statistics(df: pd.DataFrame, output_file: str) -> None:
    """
    Create a LaTeX table with summary statistics for all numeric columns.

    Parameters:
        df (pd.DataFrame): The DataFrame containing the data.
        output_file (str): The file path where the LaTeX table will be written.

    Returns:
        None
    """
    summary_stats = df.describe().T

    latex_table = """
    \\begin{table}
      \\centering 
      \\caption{Summary statistics for all numeric columns} 
      \\vspace{10pt}
      \\label{tab:summary_statistics} 
      \\begin{tabular}{@{\\extracolsep{5pt}} lccccccc} 
      \\hline 
      \\hline \\\\[-1.8ex] 
      \\textbf{Column} & \\textbf{mean} & \\textbf{std} & \\textbf{min} & \\textbf{25\%} & \\textbf{50\%} & \\textbf{75\%} & \\textbf{max} \\\\ 
      \\hline \\\\[-1.8ex] 
    """

    for index, row in summary_stats.iterrows():
        escaped_index = index.replace('_', '\_')
        latex_table += f"{escaped_index} & {row['mean']:.2f} & {row['std']:.2f} & {row['min']:.2f} & {row['25%']:.2f} & {row['50%']:.2f} & {row['75%']:.2f} & {row['max']:.2f} \\\\ \n"

    latex_table += """
      \\hline 
      \\hline 
      \\end{tabular} 
    \\end{table} 
    """

    with open(output_file, 'w') as file:
        file.write(latex_table)

    return

def create_test_score_summary(test_scores: dict, output_file: str) -> None:
    """
    Create a LaTeX table with test scores for multiple regression models.

    Parameters:
        models (sm.regression.linear_model.RegressionResultsWrapper): The regression models.
        output_file (str): The file path where the LaTeX table will be written.

    Returns:
        None
    """
    latex_table = """
    \\begin{table}
        \\centering
        \\caption{Summary of Test Scores for Regression Models}
        \\vspace{10pt}
        \\label{tab:test_scores}
        \\begin{tabular}{l%s}
        \\hline
        \\hline \\\\[-1.8ex]
    """ % ("c" * len(test_scores.keys()))

    headers = ["Omnibus", "Omnibus p-value", "Jarque-Bera", "Jarque-Bera p-value", "Durbin Watson", "R2", "Adjusted R2"]
    model_headers = " & ".join([f"\\textbf{{Model {i + 1}}}" for i in range(len(test_scores.keys()))])
    latex_table += "Metric & " + model_headers + " \\\\\n\\hline \\\\[-1.8ex] \n"

   # Iterate over each metric and each model
    for test in headers:
        row = f"\\textbf{{{test}}}"
        for model_name in test_scores:
            score = test_scores[model_name][test.replace(" ", "_").replace("-", "_")]
            row += f" & {score:.3f}"
            
        row += " \\\\\n"
        latex_table += row


    latex_table += """
        \\hline
        \\hline
        \\end{tabular}
    \\end{table}
    """

    with open(output_file, 'w') as file:
        file.write(latex_table)
    
    return

def create_reg_summaries(*models: sm.regression.linear_model.RegressionResultsWrapper, output_file: str) -> None:
    """
    Create a LaTeX table with regression models.

    Parameters:
        models (sm.regression.linear_model.RegressionResultsWrapper): The regression models.
        output_file (str): The file path where the LaTeX table will be written.

    Returns:
        None
    """

    all_results = {}

    # Extract results from each model
    for idx, model in enumerate(models):
        model_results = model.summary2().tables[1]
        for index, row in model_results.iterrows():
            coef_name = index.replace('_', '\_')
            coef_val = row['Coef.']
            std_err = row['Std.Err.']
            p_val = row['P>|t|']
            stars = '***' if p_val < 0.01 else '**' if p_val < 0.05 else '*' if p_val < 0.1 else ''

            if coef_name not in all_results:
                all_results[coef_name] = {}
            all_results[coef_name][idx] = f"{coef_val:.4f} ({std_err:.4f}){stars}"

    for coef in all_results:
        for i in range(len(models)):
            if i not in all_results[coef]:
                all_results[coef][i] = '-'

    latex_table = """
    \\begin{table}
        \\centering
        \\caption{Regression models}
        \\vspace{10pt}
        \\label{tab:regression_models}
        \\begin{tabular}{l%s}
        \\hline
        \\hline \\\\[-1.8ex]
    """ % ("c" * len(models))

    # Adding column headers
    latex_table += " & " + " & ".join([f"\\textbf{{Model {i + 1}}}" for i in range(len(models))]) + " \\\\\n"
    latex_table += "\\hline \\\\[-1.8ex] \n"

    # Fill in rows for each coefficient
    for coef, models in all_results.items():
        row = coef
        for i in range(len(models)):
            row += " & " + models[i]
        row += " \\\\\n"
        latex_table += row

    latex_table += """
        \\hline
        \\hline
        \\end{tabular}
    \\end{table}
    """

    with open(output_file, 'w') as file:
        file.write(latex_table)

    return 

output_path = os.path.join(dir_path, '../../output/tables/')

def create_main_tables(df: pd.DataFrame) -> None:
    """
    Create tables for the summary statistics of the main dataset.

    Returns:
        None
    """
    create_summary_statistics(df, os.path.join(output_path, 'summary_statistics.tex'))  

    return
