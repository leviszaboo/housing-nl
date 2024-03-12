import pandas as pd
import json
from shapely.geometry import shape
from pyproj import Transformer
import math
import numpy as np

# Mapping for municipality name corrections and encoding issues between the geojson and the NBS datasets
name_mapping = {
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

def clean_price_data(file_path):
    """
    Cleans the price data from a given CSV file.

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
    data['municipality'] = data['municipality'].map(name_mapping).fillna(data['municipality'])
    return data

def clean_surface_data(file_path):
    """
    Cleans the surface area data from a given CSV file.

    Parameters:
        file_path (str): The path to the CSV file containing surface area data.

    Returns:
        pd.DataFrame: A cleaned DataFrame with surface area data.
    """
    data = pd.read_csv(file_path, delimiter=';', skiprows=4)
    corrected_columns = {'Unnamed: 0': 'municipality', 'Totaal.1': 'avg_surface'}
    data = data.rename(columns=corrected_columns)
    data = data[['municipality', 'avg_surface']].dropna()
    data = data[1:]  # Skip the first row which might be headers or summary
    data.reset_index(drop=True, inplace=True)
    data['avg_surface'] = pd.to_numeric(data['avg_surface'], errors='coerce')
    data['municipality'] = data['municipality'].map(name_mapping).fillna(data['municipality'])
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
    eindhoven_coords = (51.4501, 5.3745)
    list_data = []
    for feature in geojson_data['features']:
        original_name = feature['properties']['statnaam']
        corrected_name = name_mapping.get(original_name, original_name)
        geometry = feature['geometry']
        lat, lon = calculate_centroid_lat_lon(geometry)
        schiphol_distance = haversine(lat, lon, *schiphol_coords)
        eindhoven_distance = haversine(lat, lon, *eindhoven_coords)
        list_data.append({
            'municipality': corrected_name,
            'schiphol_distance': schiphol_distance,
            'eindhoven_distance': eindhoven_distance
        })
    return pd.DataFrame(list_data)

def merge_data_and_calculate_distances(data_prices, data_surface, data_airports):
    """
    Merges datasets and calculates distances to the nearest airport.

    Parameters:
        data_prices (pd.DataFrame): The cleaned price data.
        data_surface (pd.DataFrame): The cleaned surface area data.
        data_airports (pd.DataFrame): Data containing distances to airports.

    Returns:
        pd.DataFrame: The merged data with added distance information.
    """
    data = pd.merge(data_prices, data_surface, on='municipality', how='inner')
    data['m2_price'] = data['avg_price'] / data['avg_surface']
    data_airports['closest_airport'] = np.where(data_airports['schiphol_distance'] < data_airports['eindhoven_distance'], 'Schiphol', 'Eindhoven')
    data_airports['distance_to_closest_airport'] = np.where(data_airports['schiphol_distance'] < data_airports['eindhoven_distance'], data_airports['schiphol_distance'], data_airports['eindhoven_distance'])
    final_data = pd.merge(data, data_airports[['municipality', 'closest_airport', 'distance_to_closest_airport']], on='municipality', how='inner')
    return final_data

# Define paths to the data files
prices_path = './data/prices.csv'
surface_path = './data/surface.csv'
geojson_path = './data/gemeente.geojson'

# Process each data set
data_prices = clean_price_data(prices_path)
data_surface = clean_surface_data(surface_path)
data_airports = load_and_process_geojson(geojson_path)

# Merge data and calculate distances to the nearest airport
final_data = merge_data_and_calculate_distances(data_prices, data_surface, data_airports)

# Optionally, save the final data to a CSV file
final_data.to_csv('main.csv', sep='\t', index=False)