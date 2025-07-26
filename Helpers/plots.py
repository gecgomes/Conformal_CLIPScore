import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import random
import pandas as pd
import seaborn as sns


def compute_and_save_confusion_heatmap(clipscores: pd.Series, stds: pd.Series, avg_clip:float,avg_std:float , save_path: str) -> dict:
    if len(clipscores) != len(stds):
        raise ValueError("The 'clipscores' and 'stds' series must have the same length.")

    high_clip = clipscores >= avg_clip
    high_std = stds >= avg_std

    matrix = {
        'high_clip_high_std': ((high_clip) & (high_std)).sum(),
        'high_clip_low_std':  ((high_clip) & (~high_std)).sum(),
        'low_clip_high_std':  ((~high_clip) & (high_std)).sum(),
        'low_clip_low_std':   ((~high_clip) & (~high_std)).sum()
    }

    # Convert to 2x2 DataFrame for plotting
    df_matrix = pd.DataFrame([
        [matrix['high_clip_low_std'],matrix['high_clip_high_std'] ],
        [matrix['low_clip_low_std'], matrix['low_clip_high_std'] ]
    ], index=['CLIP ≥ avg', 'CLIP < avg'], columns=['STD < avg', 'STD ≥ avg'])

    # Plot heatmap
    plt.figure(figsize=(6, 5))
    sns.heatmap(df_matrix, annot=True, fmt='d', cmap='Blues', cbar=False,
                linewidths=0.5, linecolor='gray')
    plt.title("CLIPScore vs Confidence (STD) Confusion Matrix")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

    return matrix

def plot_violin(df, scores_list, save_path):
    """
    Selects a random instance, a high variance instance, and a low variance instance 
    from the dataset, prints their image URLs and captions, and saves a violin plot.

    Parameters:
    - df: pandas DataFrame with columns ['image_url', 'caption']
    - scores_list: List of NumPy arrays (each should be 1D)
    - save_path: String, path to save the violin plot

    Returns:
    - None (Prints image URLs and captions, and saves the plot)
    """

    # Compute variances for all scores
    variances = np.array([np.var(scores) for scores in scores_list])

    # Get sorted indices based on variance
    sorted_indices = np.argsort(variances)  # Sorts in ascending order

    # Select a random instance
    random_idx = random.randint(0, len(scores_list) - 1)
    # Select third lowest variance instance
    low_var_idx = sorted_indices[530]  # Third lowest

    # Select third highest variance instance
    high_var_idx = sorted_indices[-13]  # Third highest

    # Extract scores and ensure they are 1D
    random_scores = scores_list[random_idx].flatten()
    high_var_scores = scores_list[high_var_idx].flatten()
    low_var_scores = scores_list[low_var_idx].flatten()

    # Print URL and captions
    print("\nRandom Instance:")
    print(f"URL: {df.iloc[random_idx]['image_path']}")
    print(f"Caption: {df.iloc[random_idx]['candidates']}\n")

    print("High Variance Instance:")
    print(f"URL: {df.iloc[high_var_idx]['image_path']}")
    print(f"Caption: {df.iloc[high_var_idx]['candidates']}\n")

    print("Low Variance Instance:")
    print(f"URL: {df.iloc[low_var_idx]['image_path']}")
    print(f"Caption: {df.iloc[low_var_idx]['candidates']}\n")

    # Create a dictionary for plotting
    data = {
        'Random': random_scores,
        'High Variance': high_var_scores,
        'Low Variance': low_var_scores
    }

    # Convert to DataFrame for Seaborn
    df_plot = pd.DataFrame({k: pd.Series(v) for k, v in data.items()})

    # Plot violin plots
    plt.figure(figsize=(12, 7))
    sns.violinplot(data=df_plot, palette="pastel")

    # Customize font sizes and add grid
    plt.ylabel("CLIPScore", fontsize=24)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)

    # Save the plot
    plt.savefig(save_path, format='png', dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Violin plot saved at: {save_path}")

def plot_violin_2(df, scores_list, save_path):
    """
    Selects a random instance, a high variance instance, and a low variance instance 
    from the dataset, prints their image URLs and captions, and saves three separate 
    violin plots in a single figure.

    Parameters:
    - df: pandas DataFrame with columns ['image_url', 'caption']
    - scores_list: List of NumPy arrays (each should be 1D)
    - save_path: String, path to save the violin plot

    Returns:
    - None (Prints image URLs and captions, and saves the plot)
    """

    # Compute variances for all scores
    variances = np.array([np.var(scores) for scores in scores_list])

    # Get sorted indices based on variance
    sorted_indices = np.argsort(variances)

    # Select a random instance
    random_idx = random.randint(0, len(scores_list) - 1)
    # Select a low variance instance
    low_var_idx = sorted_indices[530]
    # Select a high variance instance
    high_var_idx = sorted_indices[-13]

    # Extract scores and ensure they are 1D
    random_scores = scores_list[random_idx].flatten()
    high_var_scores = scores_list[high_var_idx].flatten()
    low_var_scores = scores_list[low_var_idx].flatten()

    # Print URL and captions
    print("\nRandom Instance:")
    print(f"URL: {df.iloc[random_idx]['image_path']}")
    print(f"Caption: {df.iloc[random_idx]['candidates']}\n")

    print("High Variance Instance:")
    print(f"URL: {df.iloc[high_var_idx]['image_path']}")
    print(f"Caption: {df.iloc[high_var_idx]['candidates']}\n")

    print("Low Variance Instance:")
    print(f"URL: {df.iloc[low_var_idx]['image_path']}")
    print(f"Caption: {df.iloc[low_var_idx]['candidates']}\n")

    # Create subplots for each instance
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Define data, labels, and colors
    data_labels = [
        ('Random Example', random_scores, 'lightblue'),
        ('High Variance Example', high_var_scores, 'peachpuff'),
        ('Low Variance Example', low_var_scores, 'lightgreen')
    ]

    # Plot each violin separately
    for ax, (label, scores, color) in zip(axes, data_labels):
        sns.violinplot(y=scores, ax=ax, color=color)
        ax.set_title(label, fontsize=24)
        ax.set_ylabel("CLIPScore", fontsize=20)
        ax.set_ylim(0, 1.8)
        ax.tick_params(axis='both', which='major', labelsize=18)

    plt.tight_layout()
    plt.savefig(save_path, format='png', dpi=600, bbox_inches='tight')
    plt.close()

    print(f"Violin plot saved at: {save_path}")


def plot_risks_and_pvalues(risks, pvalues, lambdas, alpha, delta, N, upper_bound):
    # Compute delta/N
    delta_N = delta / N

    # Find the point in risks that exceeds alpha
    exceeding_idx = np.where(risks > alpha)[0]

    if len(exceeding_idx) > 0:
        chosen_idx = exceeding_idx[0] - 50  # Take the first occurrence where risk > alpha
        chosen_idx = np.argmin(pvalues)
        chosen_lambda = lambdas[chosen_idx]
        chosen_risk = risks[chosen_idx]
        chosen_pvalue = pvalues[chosen_idx]
    else:
        chosen_idx = None
        chosen_lambda = None
        chosen_risk = None
        chosen_pvalue = None

    fig, axs = plt.subplots(1, 2, figsize=(14, 7), dpi=100)

    # Risks subplot
    axs[0].scatter(lambdas, risks, color='dodgerblue', s=50, alpha=0.7, label='Risks')
    axs[0].axhline(y=alpha, color='red', linestyle='--', linewidth=1.5, label=f'alpha = {alpha:.2f}')

    if chosen_idx is not None:
        axs[0].scatter(chosen_lambda, chosen_risk, color='orange', s=100, edgecolor='black', label='Chosen Point')
        axs[0].text(
            chosen_lambda, 
            chosen_risk - 0.01, 
            f"$\mathcal{{H}}_\lambda: R(\lambda) > {alpha:.3f}$",
            fontsize=10, color='black',
            ha='left', va='center',
            bbox=dict(boxstyle="round,pad=0.3", edgecolor="gray", facecolor="white", alpha=0.8)
        )

    axs[0].set_xlabel(f'$\lambda$', fontsize=12)
    axs[0].set_ylabel(f'$R(\lambda)$', fontsize=12)
    axs[0].grid(True, linestyle='--', alpha=0.6)
    axs[0].legend(fontsize=10)
    axs[0].xaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
    axs[0].ticklabel_format(style='sci', axis='both', scilimits=(-2, 2))

    # Pvalues subplot with -log10 transformation
    transformed_pvalues = np.log10(pvalues)
    transformed_delta_N = np.log10(delta_N)

    axs[1].scatter(lambdas, transformed_pvalues, color='dodgerblue', s=50, alpha=0.7, label='Pvalues')
    axs[1].axhline(y=transformed_delta_N, color='green', linestyle='--', linewidth=1.5, label=f'log10(delta/N) = log10({delta:.2f}/{N:.2f}) {transformed_delta_N:.2f}')

    if chosen_idx is not None:
        transformed_chosen_pvalue = np.log10(chosen_pvalue)
        axs[1].scatter(chosen_lambda, transformed_chosen_pvalue, color='orange', s=100, edgecolor='black', label='Chosen Point')
        axs[1].text(
            chosen_lambda, 
            transformed_chosen_pvalue - np.log10(0.01),
            f"$\mathcal{{H}}_\lambda: R(\lambda) > {alpha:.3f}$",
            fontsize=10, color='black',
            ha='left', va='center',
            bbox=dict(boxstyle="round,pad=0.3", edgecolor="gray", facecolor="white", alpha=0.8)
        )

    axs[1].set_xlabel(f'$\lambda$', fontsize=12)
    axs[1].set_ylabel('log10(Pvalues)', fontsize=12)
    axs[1].grid(True, linestyle='--', alpha=0.6)
    axs[1].legend(fontsize=10)

    plt.tight_layout()
    plt.savefig("/cfs/home/u021414/PhD/ConformalFoil/cache/UPearsonS_UB{}.png".format(upper_bound), format='png', dpi=300, bbox_inches='tight')
    