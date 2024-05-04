import pandas as pd
import os

dir_path = os.path.dirname(os.path.realpath(__file__))

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

# Coordinates for major cities and urban centers
city_coords = {
    'Amsterdam': (52.3676, 4.9041),
    'Rotterdam': (51.9244, 4.4777),
    'The Hague': (52.0705, 4.3007),
    'Utrecht': (52.0907, 5.1214),
    'Eindhoven': (51.4231, 5.4623),
    'Tilburg': (51.5853, 5.0564),
    'Breda': (51.5719, 4.7683),
    "'s-Hertogenbosch": (51.6978, 5.3037),
    'Maastricht': (50.8514, 5.6910),
    'Groningen': (53.2194, 6.5665)
}

# Missing income data - values are from 2021
missing_incomes = {
    'Ameland': 47.5,
    'Renswoude': 59.4,
    'Rozendaal': 81.4,
    'Schiermonnikoog': 42.4,
    'Vlieland': 42.7
}

def apply_name_mapping(data: pd.DataFrame, col: str, mapping: dict) -> pd.DataFrame:
    """
    Applies a name mapping to a DataFrame column.

    Parameters:
        data (pd.DataFrame): The DataFrame to process.
        col (str): The column to apply the mapping to.
        mapping (dict): The mapping to apply to the column.
    
    Returns:
        pd.DataFrame: The processed DataFrame.
    """
    data.loc[:, col] = data[col].map(mapping).fillna(data[col])

    return data
