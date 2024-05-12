import logging
import pandas as pd
import geopandas as gpd
import os
from src.dataset.traffic import aggregate_traffic_data
from src.dataset.utils import apply_name_mapping, municipality_name_mapping, station_type_translation, dir_path


def clean_station_data(file_path: str) -> pd.DataFrame:
    """
    Cleans the station data and creates a Pandas DataFrame from the CSV file.

    Parameters:
        file_path (str): The path to the CSV file containing station data.

    Returns:
        pd.DataFrame: A cleaned DataFrame with station data.
    """
    station_data = pd.read_csv(file_path)

    # Filter for stations in the Netherlands
    data_nl = station_data[station_data['country'] == 'NL']

    data = data_nl[['code', 'name_long', 'type', 'geo_lat', 'geo_lng']].copy()
    data['type'] = data['type'].map(station_type_translation)
    data.rename(columns={'name_long': 'station_name'}, inplace=True)

    # Add missing code and traffic for Nieuw Amsterdam - pandas might confuse NA with NaN
    data.loc[data['station_name'] == 'Nieuw Amsterdam', 'code'] = 'NA'

    return data

def find_municipalities_for_stations(station_df: pd.DataFrame, geojson_path: str) -> pd.DataFrame:
    """
    Enhances station DataFrame with municipality information based on lat/lon columns.
    
    Parameters:
        station_df (pd.DataFrame): DataFrame containing stations with 'geo_lat' and 'geo_lng' columns.
        geojson_path (str): The path to the GeoJSON file containing municipality boundaries.
        
    Returns:
        pd.DataFrame: The input DataFrame enhanced with a 'municipality' column.
    """
    # Load the municipalities GeoDataFrame
    municipalities_gdf = gpd.read_file(geojson_path)
    
    # Convert station DataFrame to GeoDataFrame
    station_gdf = gpd.GeoDataFrame(
        station_df, 
        geometry=gpd.points_from_xy(station_df.geo_lng, station_df.geo_lat),
        crs="EPSG:4326"
    )
    
    # Perform spatial join between stations and municipalities
    joined_gdf = gpd.sjoin(station_gdf, municipalities_gdf, how="left", predicate="within")
    
    # Extract the 'statnaam' from the joined GeoDataFrame to the original DataFrame
    station_df['municipality'] = joined_gdf['statnaam']

    # Fill missing municipalitiy 
    station_df.loc[station_df['station_name'] == 'Eemshaven', 'municipality'] = 'Het Hogeland'
    
    return station_df

def merge_traffic_data(stations_df: pd.DataFrame) -> pd.DataFrame:
    """
    Merges the station DataFrame with the aggregated traffic data.

    Parameters:
        stations_df (pd.DataFrame): The DataFrame containing the station data.
        traffic_path (str): The path to the CSV file containing the aggregated traffic data.
    
    Returns:
        pd.DataFrame: The merged DataFrame.
    """
    months = range(1, 13)
    
    traffic_df = aggregate_traffic_data(months)
    merged_df = pd.merge(stations_df, traffic_df, left_on='code', right_on='station_code', how='left')

    # Add missing traffic data for 'NA' (Nieuw Amsterdam)
    merged_df.loc[merged_df['station_name'] == 'Nieuw Amsterdam', 'traffic_count'] = 27180 / 365

    return merged_df

# Define paths to the data files
base_path = os.path.join(dir_path, '../../unprocessed/')

stations_path = os.path.join(base_path, 'rdt/stations.csv')
geojson_path = os.path.join(base_path, 'gemeente.geojson')

stations_output = os.path.join(dir_path, '../../output/data/stations.csv')

def create_stations_data() -> None:
    """
    Creates the stations dataset by processing the station data and merging it with traffic data.

    Returns:
        None
    """
    logging.info('Processing stations data...')

    data_stations = clean_station_data(stations_path)
    data_stations = find_municipalities_for_stations(data_stations, geojson_path)
    data_stations = apply_name_mapping(data_stations, 'municipality', municipality_name_mapping)
    data_stations = merge_traffic_data(data_stations)
    data_stations.to_csv(stations_output, index=False)

    logging.info('Stations data processing complete.')