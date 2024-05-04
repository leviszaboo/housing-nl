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
    data.loc[:, col] = data[col].map(mapping).fillna(data[col])

    return data
