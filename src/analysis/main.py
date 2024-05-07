import json
import pandas as pd
import os

from src.analysis.maps import create_maps
from src.analysis.phase_1.plots import create_plots
from src.analysis.phase_1.regressions import run_phase_1_regressions
from src.analysis.phase_1.tables import create_main_tables
from src.analysis.phase_1.tests import run_and_save_tests
from src.analysis.utils import dir_path

def run_phase_1(df: pd.DataFrame, geojson_data) -> None:
    """
    Run the Phase 1 analysis.

    Parameters:
        df (pd.DataFrame): The main dataset.
    
    Returns:
        None
    """
    print("Running Phase 1 analysis...")

    # Create chloropleth maps
    create_maps(df, geojson_data)

    # Create figures 
    create_plots(df)

    # Run tests and save the results
    run_and_save_tests(df)

    # Run phase 1 regressions
    run_phase_1_regressions(df)

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
    main = pd.read_csv(os.path.join(dir_path, '../../output/data/main.csv'))

    # Load the phase 1 dataset
    phase_1 = pd.read_csv(os.path.join(dir_path, '../../output/data/phase1.csv'))

    with open(os.path.join(dir_path, '../../unprocessed/gemeente.geojson'), 'r') as geojson_file:
        geojson_data = json.load(geojson_file)

    # summary statistics
    create_main_tables(main)
    
    run_phase_1(phase_1, geojson_data)

