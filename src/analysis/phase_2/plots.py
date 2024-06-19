import logging
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os
from src.analysis.phase_2.models import get_model_scores
from src.analysis.utils import dir_path, phase_2_log

output_path = os.path.join(dir_path, '../../output/figures/phase_2')

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

    logging.info(f"Visualizing scores for {title}...")

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

        axes[0].axvline(x=0.699, color='#404040', linewidth=2, linestyle='--', label='VIF = 5')
        axes[0].legend()

    # Condition Indices
    cond_indices_list = [{
        "Model": name,
        "Condition Index": np.log10(data['cond_indices']).max()
    } for name, data in results.items() if 'cond_indices' in data]

    if cond_indices_list:
        cond_indices_data = pd.DataFrame(cond_indices_list)
        sns.barplot(x='Condition Index', y='Model', data=cond_indices_data, ax=axes[1], width=0.6, hue='Model')
        axes[1].set_title(f'Log-Scaled Condition Indices for {title}')
        axes[1].set_xlabel("Log10 (Condition Index)")
        axes[1].set_ylabel(" ")
        axes[1].axvline(x=1.477, color='gray', linewidth=2, linestyle='--', label='Condition Index = 30')
        axes[1].legend()

    # BIC Scores
    bic_list = [{
        "Model": name,
        "BIC": data['bic']
    } for name, data in results.items() if 'bic' in data]

    if bic_list:
        bic_data = pd.DataFrame(bic_list)
        sns.barplot(x='BIC', y='Model', data=bic_data, ax=axes[2], width=0.6, hue='Model')
        axes[2].set_title(f'BIC Scores for {title}')
        axes[2].set_xlabel("BIC Score")
        axes[2].set_ylabel(" ")
        axes[2].set_xlim(bic_data['BIC'].min() - 5, bic_data['BIC'].max() + 5)
    
    plt.tight_layout()
    
    filename = title.lower().replace('-', '_').replace(' ', '_')

    plt.savefig(os.path.join(output_path, f'model_scores/{filename}_scores.png'))

def traffic_plots(df: pd.DataFrame) -> None:
    """
    Create a plot with traffic vs m2_price and log_traffic vs log_m2_price along with their distributions.

    Parameters:
        df: DataFrame with the main dataset
        output_path: Path to save the resulting plot

    Returns:
        None
    """
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(15, 12))

    palette = sns.color_palette('Set2')

    # First row for traffic vs m2_price
    # Scatter plot
    sns.scatterplot(x=df['traffic'], y=df['m2_price'], ax=axes[0, 0], color=palette[1], edgecolor='k', alpha=0.7)
    axes[0, 0].set_xlabel('traffic')
    axes[0, 0].set_ylabel('m2_price')
    axes[0, 0].set_title('traffic vs m2_price')
    
    # Distribution plot
    sns.histplot(df['traffic'], kde=True, ax=axes[0, 1], color=palette[0])
    axes[0, 1].set_xlabel('traffic')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].set_title('Distribution of traffic')

    # Second row for log_traffic vs log_m2_price
    # Scatter plot
    sns.scatterplot(x=df['log_traffic'], y=df['log_m2_price'], ax=axes[1, 0], color=palette[1], edgecolor='k', alpha=0.7)
    axes[1, 0].set_xlabel('log_traffic')
    axes[1, 0].set_ylabel('log_m2_price')
    axes[1, 0].set_title('log_traffic vs log_m2_price')
    
    # Distribution plot
    sns.histplot(df['log_traffic'], kde=True, ax=axes[1, 1], color=palette[0])
    axes[1, 1].set_xlabel('log_traffic')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].set_title('Distribution of log_traffic')

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(os.path.join(output_path, 'traffic_plots.png'))

def create_plots(df: pd.DataFrame) -> None:
    """
    Create plots for the Phase 2 analysis.

    Parameters:
        df: DataFrame with the Phase 2 dataset
    
    Returns:
        None
    """
    logging.info("Creating and saving figures for Phase 2...")
    traffic_plots(df)

    log_results, dropped_results, centered_results = get_model_scores(df)

    visualize_model_scores(log_results, 'Log Models')
    visualize_model_scores(dropped_results, 'Log Models with Dropped Outliers')
    visualize_model_scores(centered_results, 'Centered Log Models')

    plt.close('all')

    logging.info("Figures saved.")

    return