import pandas as pd
import logging

from src.dataset.utils import missing_incomes

def clean_price_data(file_path: str) -> pd.DataFrame:
    """
    Cleans the price data and creates a Pandas DataFrame from the CSV file.

    Parameters:
        file_path (str): The path to the CSV file containing price data.

    Returns:
        pd.DataFrame: A cleaned DataFrame with price data.
    """
    logging.info("Cleaning house price data...")

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
    logging.info("Cleaning surface area data...")

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
    logging.info("Cleaning municipality data...")

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
    logging.info("Cleaning income data...")

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
    logging.info("Cleaning labor data...")

    data = pd.read_csv(file_path, delimiter=';', skiprows=5)
    data.columns = ['municipality', 'unemp_rate', 'net_labor_participation']
    data.reset_index(drop=True, inplace=True)

    for col in data.columns[1:]:
        data[col] = data[col].replace({',': '.', r'\s+': ''}, regex=True)

    data['unemp_rate'] = pd.to_numeric(data['unemp_rate'], errors='coerce')
    data['net_labor_participation'] = pd.to_numeric(data['net_labor_participation'], errors='coerce')


    return data

def clean_house_type_data(file_path: str) -> pd.DataFrame:
    """
    Cleans the house type data and creates a Pandas DataFrame from the CSV file.

    Parameters:
        file_path (str): The path to the CSV file containing house type data.

    Returns:
        pd.DataFrame: A cleaned DataFrame with house type data.
    """
    logging.info("Cleaning house type data...")

    data = pd.read_csv(file_path, delimiter=';', skiprows=4)
    data.columns = ['municipality', 'period', 'total_homes', 'multy_family', 'single_family', 'detached']
    data.reset_index(drop=True, inplace=True)

    data = data.drop(columns=['period'])

    for col in data.columns[1:]:
        data[col] = data[col].replace({',': '.', r'\s+': ''}, regex=True)

    data['total_homes'] = pd.to_numeric(data['total_homes'], errors='coerce')
    data['multy_family'] = pd.to_numeric(data['multy_family'], errors='coerce')
    data['single_family'] = pd.to_numeric(data['single_family'], errors='coerce')
    data['detached'] = pd.to_numeric(data['detached'], errors='coerce')

    data.dropna(subset=['total_homes', 'multy_family', 'single_family', 'detached'], inplace=True)

    data['multy_family'] = (data['multy_family'] / data['total_homes']) * 100

    data = data.drop(columns=['single_family', 'detached'])

    return data
