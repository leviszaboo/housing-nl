import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import os
import logging
from src.analysis.utils import dir_path, phase_1_vars, phase_1_log

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

def scatter_plots(df: pd.DataFrame) -> None:
    """
    Create scatter plots of variables against m2_price.

    Parameters:
        df: DataFrame with the main dataset
    
    Returns:
        None
    """
    variables = phase_1_vars

    fig, axes = plt.subplots(nrows=4, ncols=2, figsize=(15, 25))
    fig.suptitle("Scatter Plots of Variables Against m2_price", fontsize=16)

    axes = axes.flatten()

    for i, var in enumerate(variables):
        axes[i].scatter(df[var], df['m2_price'], alpha=0.7, edgecolor='k')
        axes[i].set_xlabel(var)
        axes[i].set_ylabel('m2_price')
        axes[i].set_title(f'{var} vs m2_price')

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    plt.savefig(os.path.join(output_path, 'scatter_plots.png'))

def plot_distribution(df: pd.DataFrame) -> None:
    """
    Create a distribution plot of the m2_price variable.

    Parameters:
        df: DataFrame with the main dataset
    
    Returns:
        None
    """
    plt.figure(figsize=(10, 6))
    sns.histplot(df['m2_price'], kde=True, bins=30)
    plt.xlabel('m2_price')
    plt.ylabel('Frequency')
    plt.title('Distribution of m2_price')

    plt.savefig(os.path.join(output_path, 'm2_price_distribution.png'))

def correlation_heatmap(df: pd.DataFrame) -> None:
    """
    Create a heatmap of the correlation matrix of the variables.

    Parameters:
        df: DataFrame with the main dataset
    
    Returns:
        None
    """
    variables = phase_1_vars

    plt.figure(figsize=(14, 12))
    sns.heatmap(df[variables].corr(), annot=True, fmt='.2f', 
                cmap='BrBG', linewidths=.5, vmin=-1, vmax=1)
    plt.xticks(rotation=45)
    plt.title('Correlation Matrix of Variables')

    plt.savefig(os.path.join(output_path, 'corr_heatmap.png'))

def updated_log_corr_heatmap(df: pd.DataFrame) -> None:
    """
    Create a heatmap of the correlation matrix of the log-transformed variables.

    Parameters:
        df: DataFrame with the main dataset
    
    Returns:
        None
    """
    variables = phase_1_log

    plt.figure(figsize=(14, 12))
    sns.heatmap(df[variables].corr(), annot=True, fmt='.2f', 
                cmap='BrBG', linewidths=.5, vmin=-1, vmax=1)
    plt.xticks(rotation=45)
    plt.title('Correlation Matrix of Log-Transformed Variables')

    plt.savefig(os.path.join(output_path, 'log_corr_heatmap.png'))

def create_plots(df: pd.DataFrame) -> None:
    """
    Create plots for the Phase 1 analysis.

    Parameters:
        df: DataFrame with the main dataset
    
    Returns:
        None
    """
    logging.info("Creating and saving figures...")
    plots_station_no_station(df)
    scatter_plots(df)
    correlation_heatmap(df)
    updated_log_corr_heatmap(df)
    plot_distribution(df)
