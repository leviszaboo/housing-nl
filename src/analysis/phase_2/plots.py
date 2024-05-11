import logging
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os
from src.analysis.phase_2.models import get_model_scores
from src.analysis.utils import dir_path, phase_2_log

output_path = os.path.join(dir_path, '../../output/figures/phase_2')

def log_corr_heatmap(df: pd.DataFrame) -> None:
    """
    Create a heatmap of the correlation matrix of the log-transformed variables.

    Parameters:
        df: DataFrame with the Phase 2 dataset
    
    Returns:
        None
    """
    variables = phase_2_log

    plt.figure(figsize=(14, 12))
    sns.heatmap(df[variables].corr(), annot=True, fmt='.2f', 
                cmap='BrBG', linewidths=.5, vmin=-1, vmax=1)
    plt.xticks(rotation=45)
    # plt.title('Correlation Matrix of Log-Transformed Variables')

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

    plt.savefig(os.path.join(output_path, f'model_scores/{filename}_scores.png'))

def create_plots(df: pd.DataFrame) -> None:
    """
    Create plots for the Phase 2 analysis.

    Parameters:
        df: DataFrame with the Phase 2 dataset
    
    Returns:
        None
    """
    logging.info("Creating and saving figures for Phase 2...")
    log_corr_heatmap(df)

    log_results, dropped_results, centered_results = get_model_scores(df)

    visualize_model_scores(log_results, 'Log Models')
    visualize_model_scores(dropped_results, 'Log Models Dropped Outliers')
    visualize_model_scores(centered_results, 'Centered Log Models')

    logging.info("Figures saved.")

    return