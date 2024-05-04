import pandas as pd
import os

from src.analysis.phase_1.tables import create_tables
from src.analysis.utils import dir_path

def run_phase_1(df: pd.DataFrame) -> None:
    print("Running Phase 1 analysis...")

    # Create tables for the summary statistics of the main dataset
    create_tables(df)

    print("Phase 1 analysis complete")
    return 

def run_analysis() -> None:
    print("Running analysis")

    # Load the main dataset
    df = pd.read_csv(os.path.join(dir_path, '../../output/data/main.csv'))
    
    run_phase_1(df)

