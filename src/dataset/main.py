import pandas as pd
import numpy as np
import os
from src.dataset.cbs import clean_house_type_data, clean_income_data, clean_labor_data, clean_municipality_data, clean_price_data, clean_surface_data
from src.dataset.utils import municipality_name_mapping, \
    apply_name_mapping, dir_path
from src.dataset.distances import load_and_process_geojson

def process_stations_data(stations_path: str, municipalities_df: pd.DataFrame) -> pd.DataFrame:
    """
    Counts the number of stations and sums up the traffic in each municipality, including municipalities with 0 stations.
    Counts the number of stations for each category in the 'type' column.
    Sums of train traffic in each municipality.

    Parameters:
        stations_path (str): Path to the CSV file containing stations data with 'municipality', 'type', and 'traffic' columns.
        municipalities_df (pd.DataFrame): DataFrame containing a list of all municipalities.

    Returns:
        pd.DataFrame: A DataFrame containing the number of stations and total traffic in each municipality, including those with 0 stations.
    """
    station_df = pd.read_csv(stations_path)

    station_df = apply_name_mapping(station_df, 'municipality', municipality_name_mapping)
    municipalities_df = apply_name_mapping(municipalities_df, 'municipality', municipality_name_mapping)

    municipalities_df = municipalities_df[['municipality']].drop_duplicates().reset_index(drop=True)

    station_count = station_df.groupby('municipality').size().reset_index(name='station_count')

    # station_type_count = station_df.groupby(['municipality', 'type']).size().unstack(fill_value=0).reset_index()
    # station_type_count.columns = ['municipality'] + [f'{col}_count' for col in station_type_count.columns[1:]]

    traffic_sum = station_df.groupby('municipality')['traffic_count'].sum().reset_index(name='traffic')

    # Merge counts and traffic data with the complete list of municipalities
    merged_df = pd.merge(municipalities_df, station_count, on='municipality', how='left')
    # merged_df = pd.merge(merged_df, station_type_count, on='municipality', how='left')
    merged_df = pd.merge(merged_df, traffic_sum, on='municipality', how='left')

    # Fill NaN values with 0 and convert counts/traffic to integers
    merged_df.fillna(0, inplace=True)
    count_columns = [col for col in merged_df.columns if '_count' in col] + ['traffic']
    merged_df[count_columns] = merged_df[count_columns].astype(int)

    merged_df['has_station'] = (merged_df['station_count'] > 0).astype(int)

    return merged_df

def create_phase_1_dataset(df: pd.DataFrame, ) -> None:
    """
    Creates the Phase 1 dataset by including logarithmic and interaction columns.

    Parameters:
        df (pd.DataFrame): The main dataset.
    
    Returns:
        None
    """

    df['log_avg_income'] = np.log(df['avg_income'])
    df['log_distance'] = np.log(df['distance_to_urban_center'])
    df['log_homes_per_capita'] = np.log(df['homes_per_capita'])
    df['log_multy_family'] = np.log(df['multy_family'])
    df['log_m2_price'] = np.log(df['m2_price'])

    df['station_x_count'] = df['has_station'] * df['station_count']
    df['station_x_multy_fam'] = df['has_station'] * df['log_multy_family']
    df['station_x_pop_den'] = df['has_station'] * df['pop_density']
    df['station_x_distance'] = df['has_station'] * df['log_distance']

    df.to_csv(os.path.join(dir_path, '../../output/data/phase1.csv'), index=False)

def create_phase_2_dataset(df: pd.DataFrame) -> None:
    """
    Creates the Phase 2 dataset by limiting to municipalities with a station 
    and including logarithmic and interaction columns.

    Parameters:
        df (pd.DataFrame): The main dataset.
    
    Returns:
        None
    """
    df = df[df['has_station'] == 1]

    df['log_traffic'] = np.log(df['traffic'])
    df['log_avg_income'] = np.log(df['avg_income'])
    df['log_distance'] = np.log(df['distance_to_urban_center'])
    df['log_homes_per_capita'] = np.log(df['homes_per_capita'])
    df['log_multy_family'] = np.log(df['multy_family'])
    df['log_m2_price'] = np.log(df['m2_price'])

    df['traffic_x_multy_fam'] = df['log_traffic'] * df['log_multy_family']  

    df.to_csv(os.path.join(dir_path, '../../output/data/phase2.csv'), index=False)

def merge_datasets(*datasets: pd.DataFrame) -> pd.DataFrame:
    """
    Merges multiple datasets on the 'municipality' column.

    Parameters:
        *datasets (pd.DataFrame): The datasets to merge.
    
    Returns:
        pd.DataFrame: The merged dataset.
    """
    datasets = [apply_name_mapping(data, 'municipality', municipality_name_mapping) for data in datasets]

    merged_data = datasets[0]
    for data in datasets[1:]:
        merged_data = pd.merge(merged_data, data, on='municipality', how='inner')
    return merged_data

def process_merged_data(data: pd.DataFrame) -> pd.DataFrame:
    """
    Processes the merged data to calculate the price per square meter.

    Parameters:
        data (pd.DataFrame): The merged data.
    
    Returns:
        pd.DataFrame: The processed data.
    """
    data.insert(1, 'm2_price', data['avg_price'] / data['avg_surface'])
    data.insert(8, 'homes_per_capita', data['total_homes'] / data['population'])

    data.drop(columns=['avg_price', 'avg_surface', 'total_homes'], inplace=True)

    return data

# Define paths to the data files
base_path = os.path.join(dir_path, '../../unprocessed/')

prices_path = os.path.join(base_path, 'cbs/prices.csv')
surface_path = os.path.join(base_path, 'cbs/surface.csv')
mun_size_path = os.path.join(base_path, 'cbs/mun_size.csv')
incomes_path = os.path.join(base_path, 'cbs/incomes.csv')
labor_path = os.path.join(base_path, 'cbs/labor.csv')
house_types_path = os.path.join(base_path, 'cbs/property_types.csv')
geojson_path = os.path.join(base_path, 'gemeente.geojson')

stations_path = os.path.join(dir_path, '../../output/data/stations.csv')

main_output = os.path.join(dir_path, '../../output/data/main.csv')

def create_main_dataset() -> None:
    """
    Creates the main dataset by processing and merging the individual datasets.

    Returns:
        None
    """
    print('Creating main dataset...')

    # Process each dataset
    print('Processing CBS datasets...')
    data_prices = clean_price_data(prices_path)
    data_surface = clean_surface_data(surface_path)
    data_mun_size = clean_municipality_data(mun_size_path)
    data_incomes = clean_income_data(incomes_path)
    data_labor = clean_labor_data(labor_path)
    data_house_types = clean_house_type_data(house_types_path)

    print('Processing GeoJSON data...')
    data_cities = load_and_process_geojson(geojson_path)

    data_stations_count = process_stations_data(stations_path, data_prices[['municipality']])

    # Merge data
    print('Merging final datasets...')
    final_data = merge_datasets(data_prices, data_surface, data_mun_size, 
                                data_incomes, data_labor, data_house_types, 
                                data_cities, data_stations_count)

    # Process the merged data
    final_data = process_merged_data(final_data)

    phase_1 = final_data.copy()
    phase_2 = final_data.copy()

    # Create the analysis datasets
    create_phase_1_dataset(phase_1)
    create_phase_2_dataset(phase_2)

    # Save the final data to a CSV file
    final_data.to_csv(main_output, index=False)

    print('Data processing completed successfully.')