import json
import pandas as pd
import seaborn as sns
import os
import logging

from src.analysis.maps import create_maps

from src.analysis.phase_1.plots import create_plots as create_phase_1_plots
from src.analysis.phase_1.models import run_phase_1_regressions
from src.analysis.phase_2.models import run_phase_2_regressions
from src.analysis.tables import create_main_tables
from src.analysis.phase_1.tests import run_and_save_tests
from src.analysis.utils import dir_path

from src.analysis.phase_2.plots import create_plots as create_phase_2_plots

def run_phase_1(df: pd.DataFrame, geojson_data) -> None:
    """
    Run the Phase 1 analysis.

    Parameters:
        df (pd.DataFrame): The main dataset.
    
    Returns:
        None
    """
    logging.info("Running Phase 1 analysis...")

    # Create chloropleth maps
    create_maps(df, geojson_data)

    # Run tests and save the results
    run_and_save_tests(df)

    # Run phase 1 regressions
    run_phase_1_regressions(df)

    # Create figures 
    create_phase_1_plots(df)

    logging.info("Phase 1 analysis complete.")
    return 

def run_phase_2(df: pd.DataFrame) -> None:
    """
    Run the Phase 2 analysis.

    Parameters:
        df (pd.DataFrame): The main dataset.
    
    Returns:
        None
    """
    logging.info("Running Phase 2 analysis...")

    # Create figures
    create_phase_2_plots(df)

    # Run Phase 2 regressions
    run_phase_2_regressions(df)

    logging.info("Phase 2 analysis complete.")
    return

def run_analysis(phase: list[int]) -> None:
    """
    Run the analysis.

    Parameters:
        phase (list[int]): The phase(s) to run.

    Returns:
        None
    """
    logging.info("Running analysis...")

    # Load the main dataset
    main = pd.read_csv(os.path.join(dir_path, '../../output/data/main.csv'))

    # Load the phase 1 dataset
    phase_1 = pd.read_csv(os.path.join(dir_path, '../../output/data/phase1.csv'))

    # Load the phase 2 dataset
    phase_2 = pd.read_csv(os.path.join(dir_path, '../../output/data/phase2.csv'))

    with open(os.path.join(dir_path, '../../unprocessed/gemeente.geojson'), 'r') as geojson_file:
        geojson_data = json.load(geojson_file)

    # summary statistics
    create_main_tables(main)

    # Set the color palette
    sns.set_palette('Set2')
    
    if 1 in phase:
        run_phase_1(phase_1, geojson_data)

    if 2 in phase:
        run_phase_2(phase_2)

