import pandas as pd
import json
from shapely.geometry import shape
from pyproj import Transformer
import math
from src.dataset.utils import city_coords, municipality_name_mapping

def calculate_centroid_lat_lon(geometry: dict) -> tuple:
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

def haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
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

def load_and_process_geojson(file_path: str) -> pd.DataFrame:
    """
    Loads GeoJSON data and processes it to calculate distances to key cities.

    Parameters:
        file_path (str): The path to the GeoJSON file.

    Returns:
        pd.DataFrame: A DataFrame containing municipalities and their distances to the nearest key city.
    """
    with open(file_path, 'r') as geojson_file:
        geojson_data = json.load(geojson_file)

    list_data = []

    for feature in geojson_data['features']:
        original_name = feature['properties']['statnaam']
        corrected_name = municipality_name_mapping.get(original_name, original_name)
        geometry = feature['geometry']
        lat, lon = calculate_centroid_lat_lon(geometry)

        # Calculate distances to each city and find the nearest one
        nearest_city = None
        min_distance = float('inf')
        for city, coords in city_coords.items():
            distance = haversine(lat, lon, coords[0], coords[1])
            if distance < min_distance:
                min_distance = distance
                nearest_city = city

        list_data.append({
            'municipality': corrected_name,
            'nearest_city': nearest_city,
            'distance_to_nearest_city': min_distance
        })

    return pd.DataFrame(list_data)