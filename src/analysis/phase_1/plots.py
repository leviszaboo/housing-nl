import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import os
from src.analysis.utils import dir_path

output_path = os.path.join(dir_path, '../../output/figures/phase_1')

def plots_station_no_station(df: pd.DataFrame) -> None:
    """
    Create boxplot, swarmplot, and violin plot to compare m² prices by presence of train stations
    Also saves the plots to the output folder.

    Parameters:
        df: DataFrame with the main dataset
    
    Returns:
        None
    """
    # Categorize municipalities by presence of train stations
    df['has_station'] = df['station_count'] > 0

    sns.set_palette('Set2')

    # Create boxplot to visualize the difference in m² prices
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df, x='has_station', y='m2_price', hue='has_station')
    plt.xlabel('Has Train Station')
    plt.ylabel('Average m² Price')
    plt.title('Comparison of m² Prices by Presence of Train Stations')
    plt.grid(True)

    plt.savefig(os.path.join(output_path, 'boxplot.png'))

    # Swarm Plot
    plt.figure(figsize=(8, 6))
    sns.swarmplot(data=df, x='has_station', y='m2_price', hue='has_station')
    plt.xlabel('Has Train Station')
    plt.ylabel('Average m² Price')
    plt.title('Comparison of m² Prices by Presence of Train Stations')
    plt.grid(True)

    plt.savefig(os.path.join(output_path, 'swarm.png'))

    # Violin Plot
    plt.figure(figsize=(10, 6))
    sns.violinplot(data=df, x='has_station', y='m2_price', hue='has_station', inner='box')
    plt.xlabel('Has Train Station')
    plt.ylabel('Average m² Price')
    plt.title('Comparison of m² Prices by Presence of Train Stations')
    plt.grid(True)

    plt.savefig(os.path.join(output_path, 'violin.png'))

