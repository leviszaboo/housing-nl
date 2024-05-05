import pandas as pd
import plotly.express as px
import os
from src.analysis.utils import dir_path

output_path = os.path.join(dir_path, '../../output/maps')

def create_price_map(df: pd.DataFrame, geojson_data) -> None:
    """
    Create a map of the average m² price by municipality.

    Parameters:
        df (pd.DataFrame): The DataFrame containing the data.
        output_file (str): The file path where the map will be saved.

    Returns:
        None
    """
    print("Creating price map...")

    fig = px.choropleth_mapbox(df,
                           geojson=geojson_data,
                           locations='municipality',
                           color='m2_price',
                           featureidkey="properties.statnaam", 
                           color_continuous_scale='cividis_r',  
                           mapbox_style="carto-positron",  
                           zoom=6.5, center = {"lat": 52.370216, "lon": 4.895168},
                           opacity=0.8,
                           labels={'m2_price': 'Euros'}
                          )

    fig.update_layout(mapbox_style="white-bg", mapbox_layers=[])

    fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})

    fig.write_html(os.path.join(output_path, 'price_map.html'))

    print("Price map created.")
    return

def create_station_count_map(df: pd.DataFrame, geojson_data) -> None:
    """
    Create a map of the number of train stations by municipality.

    Parameters:
        df (pd.DataFrame): The DataFrame containing the data.

    Returns:
        None
    """
    print("Creating station count map...")

    fig = px.choropleth_mapbox(df,
                           geojson=geojson_data,
                           locations='municipality',
                           color='station_count',
                           featureidkey="properties.statnaam", 
                           color_continuous_scale='cividis_r',  
                           mapbox_style="carto-positron",  
                           zoom=6.5, center = {"lat": 52.370216, "lon": 4.895168},
                           opacity=0.8,
                           labels={'station_count': 'Number of Stations'}
                          )

    fig.update_layout(mapbox_style="white-bg", mapbox_layers=[])

    fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})

    fig.write_html(os.path.join(output_path, 'station_count_map.html'))

    print("Station count map created.")
    return

def create_distances_map(df: pd.DataFrame, geojson_data) -> None:
    """
    Create a map of the distances to the closest urban center by municipality.

    Parameters:
        df (pd.DataFrame): The DataFrame containing the data.

    Returns:
        None
    """
    print("Creating distances map...")

    fig = px.choropleth_mapbox(df,
                           geojson=geojson_data,
                           locations='municipality',
                           color='distance_to_nearest_city',
                           featureidkey="properties.statnaam", 
                           color_continuous_scale='cividis',  
                           mapbox_style="carto-positron",  
                           zoom=6.5, center = {"lat": 52.370216, "lon": 4.895168},
                           opacity=0.8,
                           labels={'distance_to_nearest_city': 'Kilometers'}
                          )

    fig.update_layout(mapbox_style="white-bg", mapbox_layers=[])

    fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})

    fig.write_html(os.path.join(output_path, 'distances_map.html'))

    print("Distances map created.")
    return

def create_traffic_map(df: pd.DataFrame, geojson_data) -> None:
    """
    Create a map of the total traffic by municipality.

    Parameters:
        df (pd.DataFrame): The DataFrame containing the data.

    Returns:
        None
    """
    print("Creating traffic map...")

    fig = px.choropleth_mapbox(df,
                           geojson=geojson_data,
                           locations='municipality',
                           color='traffic',
                           featureidkey="properties.statnaam", 
                           color_continuous_scale='cividis_r',  
                           mapbox_style="carto-positron",  
                           zoom=6.5, center = {"lat": 52.370216, "lon": 4.895168},
                           opacity=0.8,
                           labels={'traffic': 'Trains per Day'}
                          )

    fig.update_layout(mapbox_style="white-bg", mapbox_layers=[])

    fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})

    fig.write_html(os.path.join(output_path, 'traffic_map.html'))

    print("Traffic map created.")
    return

def create_pop_density_map(df: pd.DataFrame, geojson_data) -> None:
    """
    Create a map of the population density by municipality.

    Parameters:
        df (pd.DataFrame): The DataFrame containing the data.

    Returns:
        None
    """
    print("Creating population density map...")

    fig = px.choropleth_mapbox(df,
                           geojson=geojson_data,
                           locations='municipality',
                           color='pop_density',
                           featureidkey="properties.statnaam", 
                           color_continuous_scale='cividis_r',  
                           mapbox_style="carto-positron",  
                           zoom=6.5, center = {"lat": 52.370216, "lon": 4.895168},
                           opacity=0.8,
                           labels={'pop_density': 'People per km²'}
                          )

    fig.update_layout(mapbox_style="white-bg", mapbox_layers=[])

    fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})

    fig.write_html(os.path.join(output_path, 'pop_density_map.html'))

    print("Population density map created.")
    return

def create_income_map(df: pd.DataFrame, geojson_data) -> None:
    """
    Create a map of the average income level by municipality.

    Parameters:
        df (pd.DataFrame): The DataFrame containing the data.

    Returns:
        None
    """
    print("Creating income map...")

    fig = px.choropleth_mapbox(df,
                           geojson=geojson_data,
                           locations='municipality',
                           color='avg_income',
                           featureidkey="properties.statnaam", 
                           color_continuous_scale='cividis_r',  
                           mapbox_style="carto-positron",  
                           zoom=6.5, center = {"lat": 52.370216, "lon": 4.895168},
                           opacity=0.8,
                           labels={'avg_income': '1000 Euros'}
                          )

    fig.update_layout(mapbox_style="white-bg", mapbox_layers=[])

    fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})

    fig.write_html(os.path.join(output_path, 'income_map.html'))

    print("Income map created.")
    return

def create_maps(df: pd.DataFrame, geojson_data) -> None:
    """
    Create maps for the average m² price by municipality.

    Parameters:
        df (pd.DataFrame): The DataFrame containing the data.

    Returns:
        None
    """
    create_price_map(df, geojson_data)

    create_distances_map(df, geojson_data)

    create_traffic_map(df, geojson_data)

    create_pop_density_map(df, geojson_data)

    create_income_map(df, geojson_data)

    create_station_count_map(df, geojson_data)

    return