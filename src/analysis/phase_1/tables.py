import pandas as pd
import os
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
    \\begin{table}[H] 
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

output_path = os.path.join(dir_path, '../../output/tables/phase_1')

def create_tables(df: pd.DataFrame) -> None:
    """
    Create tables for the summary statistics of the main dataset.

    Returns:
        None
    """
    create_summary_statistics(df, os.path.join(output_path, 'summary_statistics.tex'))  