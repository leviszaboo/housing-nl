import pandas as pd
import os

from src.analysis.phase_1.plots import plots_station_no_station
from src.analysis.phase_1.tables import create_tables
from src.analysis.utils import dir_path

def run_phase_1(df: pd.DataFrame) -> None:
    """
    Run the Phase 1 analysis.

    Parameters:
        df (pd.DataFrame): The main dataset.
    
    Returns:
        None
    """
    print("Running Phase 1 analysis...")

    # Create tables for the summary statistics of the main dataset
    create_tables(df)

    # Create figures for comparison of m² prices by presence of train stations
    plots_station_no_station(df)

    print("Phase 1 analysis complete")
    return 

def run_analysis() -> None:
    """
    Run the analysis.

    Returns:
        None
    """
    print("Running analysis")

    # Load the main dataset
    df = pd.read_csv(os.path.join(dir_path, '../../output/data/main.csv'))
    
    run_phase_1(df)

