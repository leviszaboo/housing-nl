import pandas as pd
import os
from src.dataset.utils import municipality_name_mapping, \
    missing_incomes, apply_name_mapping
from src.dataset.distances import load_and_process_geojson

def clean_price_data(file_path: str) -> pd.DataFrame:
    """
    Cleans the price data and creates a Pandas DataFrame from the CSV file.

    Parameters:
        file_path (str): The path to the CSV file containing price data.

    Returns:
        pd.DataFrame: A cleaned DataFrame with price data.
    """
    data = pd.read_csv(file_path, sep=';', header=2)
    data.columns = ['municipality', 'Subject', 'Currency', 'avg_price']
    data = data.drop(columns=['Subject', 'Currency'])
    data['avg_price'] = pd.to_numeric(data['avg_price'], errors='coerce')
    data.dropna(subset=['avg_price'], inplace=True)

    return data

def clean_surface_data(file_path: str) -> pd.DataFrame:
    """
    Cleans the surface area data and creates a Pandas DataFrame from CSV file.

    Parameters:
        file_path (str): The path to the CSV file containing surface area data.

    Returns:
        pd.DataFrame: A cleaned DataFrame with surface area data.
    """
    data = pd.read_csv(file_path, delimiter=';', skiprows=4)
    corrected_columns = {'Unnamed: 0': 'municipality', 'Totaal.1': 'avg_surface'}
    data = data.rename(columns=corrected_columns)
    data = data[['municipality', 'avg_surface']].dropna()
    data = data[1:]  
    data.reset_index(drop=True, inplace=True)
    data['avg_surface'] = pd.to_numeric(data['avg_surface'], errors='coerce')

    return data

def clean_municipality_data(file_path: str) -> pd.DataFrame:
    """
    Cleans the data related to municipality size, population density, and total population.

    Parameters:
        file_path (str): The path to the CSV file containing the relevant data.

    Returns:
        pd.DataFrame: A cleaned DataFrame with columns for municipality, size, population density, and total population.
    """
    data = pd.read_csv(file_path, delimiter=';', skiprows=4)

    data.columns = ['municipality', 'year', 'population', 'pop_density', 'size']

    data = data.drop(columns=['year'])

    # Strip whitespace and potential non-numeric characters
    for col in data.columns[1:]:
        data[col] = data[col].replace({',': '', ' km²': '', ' aantal': '', r'\s+': ''}, regex=True)

    data['population'] = pd.to_numeric(data['population'], errors='coerce')
    data['pop_density'] = pd.to_numeric(data['pop_density'], errors='coerce')
    data['size'] = pd.to_numeric(data['size'], errors='coerce')

    data.dropna(subset=['population', 'pop_density', 'size'], inplace=True)
    data.reset_index(drop=True, inplace=True)
    
    return data

def clean_income_data(file_path: str) -> pd.DataFrame:
    """
    Cleans the income data and creates a Pandas DataFrame from the CSV file.

    Parameters:
        file_path (str): The path to the CSV file containing income data.

    Returns:
        pd.DataFrame: A cleaned DataFrame with income data.
    """
    data = pd.read_csv(file_path, delimiter=';', skiprows=6)
    data.columns = ['municipality', 'avg_income']
    data.reset_index(drop=True, inplace=True)

    for col in data.columns[1:]:
        data[col] = data[col].replace({',': '.', r'\s+': ''}, regex=True)

    data['avg_income'] = pd.to_numeric(data['avg_income'], errors='coerce')

    # Fill missing income data
    data['avg_income'] = data['avg_income'].fillna(data['municipality'].map(missing_incomes))

    data.dropna(subset=['avg_income'], inplace=True)

    return data

def clean_labor_data(file_path: str) -> pd.DataFrame:
    """
    Cleans the labor data and creates a Pandas DataFrame from the CSV file.

    Parameters:
        file_path (str): The path to the CSV file containing labor data.

    Returns:
        pd.DataFrame: A cleaned DataFrame with labor data.
    """
    data = pd.read_csv(file_path, delimiter=';', skiprows=5)
    data.columns = ['municipality', 'unemp_rate', 'net_labor_participation']
    data.reset_index(drop=True, inplace=True)

    for col in data.columns[1:]:
        data[col] = data[col].replace({',': '.', r'\s+': ''}, regex=True)

    data['unemp_rate'] = pd.to_numeric(data['unemp_rate'], errors='coerce')
    data['net_labor_participation'] = pd.to_numeric(data['net_labor_participation'], errors='coerce')


    return data

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

    station_type_count = station_df.groupby(['municipality', 'type']).size().unstack(fill_value=0).reset_index()
    station_type_count.columns = ['municipality'] + [f'{col}_count' for col in station_type_count.columns[1:]]

    traffic_sum = station_df.groupby('municipality')['traffic_count'].sum().reset_index(name='total_traffic')

    # Merge counts and traffic data with the complete list of municipalities
    merged_df = pd.merge(municipalities_df, station_count, on='municipality', how='left')
    merged_df = pd.merge(merged_df, station_type_count, on='municipality', how='left')
    merged_df = pd.merge(merged_df, traffic_sum, on='municipality', how='left')

    # Fill NaN values with 0 and convert counts/traffic to integers
    merged_df.fillna(0, inplace=True)
    count_columns = [col for col in merged_df.columns if '_count' in col] + ['total_traffic']
    merged_df[count_columns] = merged_df[count_columns].astype(int)

    return merged_df

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
    data['m2_price'] = data['avg_price'] / data['avg_surface']
    return data

# Define paths to the data files
dir_path = os.path.dirname(os.path.realpath(__file__))
base_path = os.path.join(dir_path, '../../data/unprocessed/')

prices_path = os.path.join(base_path, 'cbs/prices.csv')
surface_path = os.path.join(base_path, 'cbs/surface.csv')
mun_size_path = os.path.join(base_path, 'cbs/mun_size.csv')
incomes_path = os.path.join(base_path, 'cbs/incomes.csv')
labor_path = os.path.join(base_path, 'cbs/labor.csv')
geojson_path = os.path.join(base_path, 'gemeente.geojson')

stations_path = os.path.join(dir_path, '../../data/output/stations.csv')

main_output = os.path.join(dir_path, '../../data/output/main.csv')

def create_main_dataset():
    """
    Creates the main dataset by processing and merging the individual datasets.
    """
    print('Creating main dataset...')

    # Process each dataset
    print('Processing CBS datasets...')
    data_prices = clean_price_data(prices_path)
    data_surface = clean_surface_data(surface_path)
    data_mun_size = clean_municipality_data(mun_size_path)
    data_incomes = clean_income_data(incomes_path)
    data_labor = clean_labor_data(labor_path)

    print('Processing GeoJSON data...')
    data_cities = load_and_process_geojson(geojson_path)

    data_stations_count = process_stations_data(stations_path, data_prices[['municipality']])

    # Merge data
    print('Merging final datasets...')
    final_data = merge_datasets(data_prices, data_surface, data_mun_size, 
                                data_incomes, data_labor, data_cities, 
                                data_stations_count)

    # Process the merged data
    final_data = process_merged_data(final_data)

    # Save the final data to a CSV file
    final_data.to_csv(main_output, index=False)

    print('Data processing completed successfully.')