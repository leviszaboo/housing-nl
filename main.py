from logging_config import setup_logging
import pandas as pd
pd.options.mode.chained_assignment = None

# Setup colored logging
setup_logging()

# Continue with your application logic
import logging
import time
import argparse

from src.dataset.stations import create_stations_data
from src.dataset.main import create_main_dataset
from src.analysis.main import run_analysis

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

def main() -> None:
    """
    Main entry point of the application.

    Returns:
        None
    """
    parser = argparse.ArgumentParser(description="Run specific parts of the application or the full pipeline.")
    parser.add_argument(
        '--dataset_only',
        action='store_true',
        help='Flag to run data creation tasks only.'
    )
    parser.add_argument(
        '--analysis_only',
        action='store_true',
        help='Flag to run analysis tasks only.'
    )
    parser.add_argument(
        '--skip_station_data',
        action='store_true',
        help='Flag to skip the creation of the stations dataset.'
    )
    parser.add_argument(
        '--phase_1',
        action='store_true',
        help='Only run Phase 1 of the analysis.'
    )
    parser.add_argument(
        '--phase_2',
        action='store_true',
        help='Only run Phase 2 of the analysis.'
    )

    args = parser.parse_args()

    # Default behavior: execute both tasks if no arguments are specified
    run_data_creation = args.dataset_only or not (args.dataset_only or args.analysis_only)
    analysis_only_tasks = args.analysis_only or not (args.dataset_only or args.analysis_only)


    if run_data_creation:
        logging.info('Starting data processing...')
        start_time = time.time()

        try:
            # Create the stations dataset
            if not args.skip_station_data:
                create_stations_data()

            # Create the main dataset
            create_main_dataset()

            logging.info(f'Data processing completed in {time.time() - start_time:.2f} seconds.')
        except Exception as e:
            logging.error(f'An error occurred during data processing: {e.with_traceback()}')
            return

    if analysis_only_tasks:
        logging.info('Starting analysis...')
        start_time = time.time()

        try:
            run_analysis(phase=[1, 2] if not (args.phase_1 or args.phase_2) else [1] if args.phase_1 else [2])
            logging.info(f'Analysis completed in {time.time() - start_time:.2f} seconds.')
        except Exception as e:
            logging.error(f'An error occurred during analysis: {e.with_traceback()}')
            return

if __name__ == '__main__':
    main()
