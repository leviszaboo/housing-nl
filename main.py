import logging
import time

from src.dataset.stations import create_stations_data
from src.dataset.main import create_main_dataset

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

def main():
    """
    Main entry point of the application.
    """
    logging.info('Starting data processing...')
    start_time = time.time()

    try:
        # Create the stations dataset
        create_stations_data()

        # Create the main dataset
        create_main_dataset()

        logging.info(f'Data processing completed in {time.time() - start_time:.2f} seconds.')
    except Exception as e:
        logging.error(f'An error occurred during data processing: {e}')
        return

if __name__ == '__main__':
    main()
