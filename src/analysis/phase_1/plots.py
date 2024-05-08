import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import os
import logging
from src.analysis.phase_1.models import create_score_summaries
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
    sns.set_palette('Set2')

    df['has_station'] = df['has_station'].astype(int)

    # Create boxplot to visualize the difference in m² prices
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df, x='has_station', y='m2_price', hue='has_station')
    plt.xlabel('Has Train Station')
    plt.ylabel('Average m² Price')
    plt.title('Comparison of m² Prices by Presence of Train Stations')
    plt.grid(True)
    plt.legend(title='Has Train Station', loc='upper right', labels=['No', 'Yes'])

    plt.savefig(os.path.join(output_path, 'boxplot.png'))

    # Swarm Plot
    plt.figure(figsize=(8, 6))
    sns.swarmplot(data=df, x='has_station', y='m2_price', hue='has_station')
    plt.xlabel('Has Train Station')
    plt.ylabel('Average m² Price')
    plt.title('Comparison of m² Prices by Presence of Train Stations')
    plt.grid(True)
    plt.legend(title='Has Train Station', loc='upper right', labels=['No', 'Yes'])

    plt.savefig(os.path.join(output_path, 'swarm.png'))

    # Violin Plot
    plt.figure(figsize=(10, 6))
    sns.violinplot(data=df, x='has_station', y='m2_price', hue='has_station', inner='box')
    plt.xlabel('Has Train Station')
    plt.ylabel('Average m² Price')
    plt.title('Comparison of m² Prices by Presence of Train Stations')
    plt.grid(True)
    plt.legend(title='Has Train Station', loc='upper right', labels=['No', 'Yes'])

    plt.savefig(os.path.join(output_path, 'violin.png'))

def scatter_plots(df: pd.DataFrame) -> None:
    """
    Create scatter plots of variables against m2_price.

    Parameters:
        df: DataFrame with the main dataset
    
    Returns:
        None
    """
    variables = ['avg_income', 'homes_per_capita', 'multy_family', 'distance_to_urban_center']

    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(15, 15))
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
    variables = ['avg_income', 'homes_per_capita', 'multy_family', 'unemp_rate', 'pop_density', 'net_labor_participation', 'distance_to_urban_center']

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

def visualize_model_scores(results: dict, title: str) -> None:
    """
    Visualize the VIFs, condition indices, and BIC scores for the models.

    Parameters:
        results: Dictionary with the results of the models.
        title: Title for the plots
    
    Returns:
        None
    """
    fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(10, 15), dpi=120)

    # VIFs
    vif_data_list = []
    for name, data in results.items():
        if 'vif' in data and not pd.DataFrame(data['vif']).empty:
            vif_data = pd.DataFrame(data['vif'])
            vif_data['log_VIF'] = np.log10(vif_data['VIF'])
            vif_data['Model'] = name
            vif_data_list.append(vif_data)

    if vif_data_list:
        full_vif_data = pd.concat(vif_data_list)
        sns.barplot(x='log_VIF', y='feature', hue='Model', data=full_vif_data, ax=axes[0])
        axes[0].set_title(f'Log-Scaled VIFs for {title}')
        axes[0].set_xlabel("Log10 (VIF)")
        axes[0].set_ylabel(" ")

        axes[0].axvline(x=0.699, color='#404040', linewidth=2, linestyle='--')

    # Condition Indices
    cond_indices_list = [{
        "Model": name,
        "Condition Index": np.log10(data['cond_indices']).max()
    } for name, data in results.items() if 'cond_indices' in data]

    if cond_indices_list:
        cond_indices_data = pd.DataFrame(cond_indices_list)
        sns.barplot(x='Condition Index', y='Model', data=cond_indices_data, ax=axes[1], width=0.6 if 'Non-Log' in title else 0.8)
        axes[1].set_title(f'Log-Scaled Condition Indices for {title}')
        axes[1].set_xlabel("Log10 (Condition Index)")
        axes[1].set_ylabel(" ")
        axes[1].axvline(x=1.477, color='gray', linewidth=2, linestyle='--')

    # BIC Scores
    bic_list = [{
        "Model": name,
        "BIC": data['bic']
    } for name, data in results.items() if 'bic' in data]

    if bic_list:
        bic_data = pd.DataFrame(bic_list)
        sns.barplot(x='BIC', y='Model', data=bic_data, ax=axes[2], width=0.6 if 'Non-Log' in title else 0.8)
        axes[2].set_title(f'BIC Scores for {title}')
        axes[2].set_xlabel("BIC Score")
        axes[2].set_ylabel(" ")
        axes[2].set_xlim(bic_data['BIC'].min() - 5, bic_data['BIC'].max() + 5)
    
    plt.tight_layout()
    
    filename = title.lower().replace('-', '_').replace(' ', '_')

    plt.savefig(os.path.join(output_path, f'{filename}_scores.png'))

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

    non_log_scores, log_scores = create_score_summaries(df)
    visualize_model_scores(non_log_scores, 'Non-Log Models')
    visualize_model_scores(log_scores, 'Log Models')
