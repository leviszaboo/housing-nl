import pandas as pd
import geopandas as gpd
import json
from shapely.geometry import shape
from pyproj import Transformer
import math

# Mapping for municipality name corrections and encoding issues between the geojson and the NBS datasets
municipality_name_mapping = {
    'Noardeast-FryslÃ¢n': 'Noardeast-Fryslân',  
    'Utrecht (gemeente)': 'Utrecht',  
    'Groningen (gemeente)': 'Groningen',  
    "'s-Gravenhage (gemeente)": "'s-Gravenhage",  
    'Hengelo (O.)': 'Hengelo',  
    'Rijswijk (ZH.)': 'Rijswijk',  
    'Laren (NH.)': 'Laren',  
    'Beek (L.)': 'Beek',  
    'Stein (L.)': 'Stein',  
    'Middelburg (Z.)': 'Middelburg',  
    'SÃºdwest-FryslÃ¢n': 'Súdwest-Fryslân'  
}

# Mapping Dutch station types to English
station_type_translation = {
    'megastation': 'mega_station',
    'knooppuntIntercitystation': 'interchange_intercity_station',
    'intercitystation': 'intercity_station',
    'knoppuntSneltreinstation': 'interchange_express_train_station',
    'sneltreinstation': 'express_train_station',
    'knooppuntStoptreinstation': 'interchange_local_train_station',
    'stoptreinstation': 'local_train_station',
    'facultatief station': 'optional_station', 
}

# Missing income data - values are from 2021
missing_incomes = {
    'Ameland': 47.5,
    'Renswoude': 59.4,
    'Rozendaal': 81.4,
    'Schiermonnikoog': 42.4,
    'Vlieland': 42.7
}

def clean_price_data(file_path):
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

def clean_surface_data(file_path):
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

def clean_municipality_data(file_path):
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

def clean_station_data(file_path):
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

    # Create a new DataFrame with the specified columns and translated types
    data = data_nl[['code', 'name_long', 'type', 'geo_lat', 'geo_lng']].copy()
    data['type'] = data['type'].map(station_type_translation)
    data.rename(columns={'name_long': 'station_name'}, inplace=True)

    return data

def clean_income_data(file_path):
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

def calculate_centroid_lat_lon(geometry):
    """
    Calculates the centroid latitude and longitude from a geometry object.

    Parameters:
        geometry (dict): The geometry object in GeoJSON format.

    Returns:
        tuple: Latitude and longitude of the centroid.
    """
    shapely_geometry = shape(geometry)
    transformer_to_3857 = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
    transformer_to_4326 = Transformer.from_crs("EPSG:3857", "EPSG:4326", always_xy=True)
    try:
        centroid_3857 = transformer_to_3857.transform(*shapely_geometry.centroid.coords[0])
        centroid_4326 = transformer_to_4326.transform(*centroid_3857)
        return centroid_4326[1], centroid_4326[0]
    except Exception as e:
        return None, None

def haversine(lat1, lon1, lat2, lon2):
    """
    Calculates the distance between two points on the Earth using the Haversine formula.

    Parameters:
        lat1 (float): Latitude of the first point.
        lon1 (float): Longitude of the first point.
        lat2 (float): Latitude of the second point.
        lon2 (float): Longitude of the second point.

    Returns:
        float: The distance between the two points in kilometers.
    """
    R = 6371.0
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)
    a = math.sin(delta_phi / 2)**2 + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    distance = R * c
    return distance

def load_and_process_geojson(file_path):
    """
    Loads GeoJSON data and processes it to calculate distances to airports.

    Parameters:
        file_path (str): The path to the GeoJSON file.

    Returns:
        pd.DataFrame: A DataFrame containing municipalities and their distances to airports.
    """
    with open(file_path, 'r') as geojson_file:
        geojson_data = json.load(geojson_file)
    schiphol_coords = (52.3158, 4.7480)
    list_data = []
    for feature in geojson_data['features']:
        original_name = feature['properties']['statnaam']
        corrected_name = municipality_name_mapping.get(original_name, original_name)
        geometry = feature['geometry']
        lat, lon = calculate_centroid_lat_lon(geometry)
        schiphol_distance = haversine(lat, lon, *schiphol_coords)
        list_data.append({
            'municipality': corrected_name,
            'schiphol_distance': schiphol_distance,
        })
    return pd.DataFrame(list_data)

def find_municipalities_for_stations(station_df, geojson_path):
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

def count_stations_per_municipality(station_df, municipalities_df):
    """
    Counts the number of stations in each municipality, including municipalities with 0 stations.
    Counts the number of stations for each category in the 'type' column.

    Parameters:
        station_df (pd.DataFrame): DataFrame containing stations with 'municipality' and 'type' columns.
        municipalities_df (pd.DataFrame): DataFrame containing a list of all municipalities.

    Returns:
        pd.DataFrame: A DataFrame containing the number of stations in each municipality, including those with 0 stations.
    """
    # Apply name mapping for municipalities
    station_df = apply_name_mapping(station_df, 'municipality', municipality_name_mapping)
    municipalities_df = apply_name_mapping(municipalities_df, 'municipality', municipality_name_mapping)

    # Preparing the municipalities DataFrame
    municipalities_df = municipalities_df[['municipality']].drop_duplicates().reset_index(drop=True)

    # Step A: Count of stations per municipality
    station_count = station_df.groupby('municipality').size().reset_index(name='station_count')

    # Step B: Count of each type of station per municipality
    station_type_count = station_df.groupby(['municipality', 'type']).size().unstack(fill_value=0).reset_index()
    station_type_count.columns = ['municipality'] + [f'{col}_count' for col in station_type_count.columns[1:]]

    # Merging with municipalities to include those without any station
    merged_df = pd.merge(municipalities_df, station_count, on='municipality', how='left')
    merged_df = pd.merge(merged_df, station_type_count, on='municipality', how='left')

    # Fill missing values with 0
    merged_df.fillna(0, inplace=True)

    # Correct the data types for count columns to integer
    count_columns = [col for col in merged_df.columns if '_count' in col]
    merged_df[count_columns] = merged_df[count_columns].astype(int)

    return merged_df


def apply_name_mapping(data, col, mapping):
    """
    Applies a name mapping to a DataFrame column.

    Parameters:
        data (pd.DataFrame): The DataFrame to process.
        col (str): The column to apply the mapping to.
        mapping (dict): The mapping to apply to the column.
    
    Returns:
        pd.DataFrame: The processed DataFrame.
    """
    data[col] = data[col].map(mapping).fillna(data[col])
    return data

def merge_datasets(*datasets):
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

def process_merged_data(data):
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
prices_path = './data/unprocessed/prices.csv'
surface_path = './data/unprocessed/surface.csv'
mun_size_path = './data/unprocessed/mun_size.csv'
incomes_path = './data/unprocessed/incomes.csv'
geojson_path = './data/unprocessed/gemeente.geojson'
stations_path = './data/unprocessed/stations.csv'

# Process each data set
data_stations = clean_station_data(stations_path)
data_stations = find_municipalities_for_stations(data_stations, geojson_path)
data_stations = apply_name_mapping(data_stations, 'municipality', municipality_name_mapping)

data_prices = clean_price_data(prices_path)
data_surface = clean_surface_data(surface_path)
data_mun_size = clean_municipality_data(mun_size_path)
data_incomes = clean_income_data(incomes_path)
data_schiphol = load_and_process_geojson(geojson_path)

data_stations_count = count_stations_per_municipality(data_stations, data_prices[['municipality']])

# Merge data
final_data = merge_datasets(data_prices, data_surface, data_mun_size, data_incomes, data_schiphol, data_stations_count)

# Process the merged data
final_data = process_merged_data(final_data)

# Save the final data to a CSV file
final_data.to_csv('./data/main.csv', index=False)
data_stations.to_csv('./data/stations.csv', index=False)