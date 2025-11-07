import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import argparse
import glob
from datetime import datetime


def get_parser():
    
    parser = argparse.ArgumentParser(description="Generate figures from classification reports.")
    parser.add_argument("--path-results", type=str, required=True,
        help="Path to the cache_amr directory.",)
    parser.add_argument("--pathogen", type=str, required=True,
        help="Pathogen to filter results by (e.g., 'p_aeruginosa').")
    parser.add_argument("--drug", type=str, required=True,
        help="Drug to filter results by (e.g., 'cefepime').")
    parser.add_argument("--model", type=str, default="XGB", choices=["XGB", "LGR", "all"],
        help="Model type to filter results by (e.g., 'XGB'). Default is 'XGB'.")
    parser.add_argument("--path-out", type=str, default="figures",
        help="Path to the output directory for figures. Default is 'figures'.")
    
    return parser



def plot_metrics_classification_average(df_combined_metrics, pathogen, drug, metric_type, path_out_dir):
    """
    Plots grouped bar charts for precision, recall, and F1-score (macro and weighted)
    across different experimental settings (e.g., feature thresholds, k-mer sizes).

    Args:
        df_combined_metrics (pd.DataFrame): DataFrame containing metrics from multiple reports.
                                            Expected columns: 'Experiment ID', 'Metric Type', 'Metric', 'Score'.
        pathogen (str): Name of the pathogen.
        drug (str): Name of the drug.
        path_out_dir (str): Directory to save the plot.
    """

    # Use a custom order for Metric Type and Metric for consistent plotting
    metric_type_order = ['Macro Average', 'Weighted Average']
    metric_order = ['Precision', 'Recall', 'F1-score']

    if metric_type == 'macro_avg':
        df = df_combined_metrics[df_combined_metrics['Metric Type'] == 'Macro Average'].copy()
    elif metric_type == 'weighted_avg':
        df = df_combined_metrics[df_combined_metrics['Metric Type'] == 'Weighted Average'].copy()
    else:
        raise ValueError("Invalid metric type. Use 'Macro Average' or 'Weighted Average'.")
    
    # Plot metric
    if not df.empty:
        plt.figure(figsize=(12, 7)) # Adjust figure size as needed
        sns.barplot(
            data=df,
            x='Experiment ID',
            y='Score',
            hue='Metric',
            hue_order=metric_order,
            palette='viridis',
            errorbar=None
        )
        plt.title(f'{metric_type} Metrics for {pathogen} x {drug}', fontsize=14)
        plt.xlabel("Experiment Configuration", fontsize=12)
        plt.ylabel("Score", fontsize=12)
        plt.ylim(0, 1)
        plt.xticks(fontsize=10)
        plt.legend(title='Metric')
        plt.grid(axis='y', linestyle='--', alpha=0.8)
        
        fname = f"{pathogen}_{drug}_metrics_{metric_type}.png"
        output_path = os.path.join(path_out_dir, f"{fname}")
        plt.savefig(output_path, bbox_inches='tight', dpi=300)
        print(f"Saved {metric_type} metrics plot to {output_path}")
        plt.close() # Close the plot to free memory
    else:
        print(f"No data to plot for {metric_type}.")


def plot_accuracy_by_feature_matrix_size(csv_file: str, output_file_name: str, error_shaded: bool):
    """
    Generates and saves a line plot of balanced accuracy vs. feature matrix size.

    This function reads a CSV file, calculates the mean and standard deviation of
    'balanced_accuracy' for different 'max_features' and 'split_method' values,
    and plots the results as line graphs on a single figure.

    Args:
        csv_file (str): The path to the input CSV file.
        output_file_name (str): The name of the file to save the plot to (e.g., 'plot.png').
        error_shaded (bool): If True, a shaded region representing the standard
                             deviation will be plotted around the mean accuracy line.
    """
    try:
        df = pd.read_csv(csv_file)
    except FileNotFoundError:
        print(f"Error: The file '{csv_file}' was not found.")
        return

    # Helper function to convert 'K' notation (e.g., '5000K') to numeric values
    def convert_k_to_numeric(val):
        val = str(val).upper().replace('K', '')
        return int(val) * 1000

    # Apply the conversion and create a numeric column for sorting
    df['max_features_numeric'] = df['max_features'].apply(convert_k_to_numeric)

    # Group by split method and feature size, then calculate mean and std dev
    agg_df = df.groupby(['split_method', 'max_features', 'max_features_numeric'])['balanced_accuracy'].agg(['mean', 'std']).reset_index()

    # Sort the aggregated data by the numeric feature size
    agg_df = agg_df.sort_values('max_features_numeric')
    
    # Plotting
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(12, 8))

    split_methods = agg_df['split_method'].unique()
    # Create a color map for plotting multiple lines
    colors = plt.cm.viridis(np.linspace(0, 1, len(split_methods)))

    for i, method in enumerate(split_methods):
        method_df = agg_df[agg_df['split_method'] == method]
        
        # Plot the mean accuracy line
        ax.plot(
            method_df['max_features_numeric'], 
            method_df['mean'], 
            marker='o', linestyle='-', 
            label=f'Split Method: {method}', 
            color=colors[i]
        )

        # Add the shaded error region if requested
        if error_shaded:
            y_upper = method_df['mean'] + method_df['std']
            y_lower = method_df['mean'] - method_df['std']
            ax.fill_between(method_df['max_features_numeric'], y_lower, y_upper, alpha=0.2, color=colors[i])

    # --- Formatting the plot ---
    combination = f"{df['pathogen_short'].iloc[0]} x {df['antibiotic'].iloc[0]}"
    ax.set_xscale('log') # Set the x-axis to a logarithmic scale
    ax.set_title(f"{combination}: Accuracy vs. Feature Matrix Size (k=17)", fontsize=13, fontweight='bold')
    ax.set_xlabel('Feature Matrix Size (max_features)', fontsize=12)
    ax.set_ylabel('Averaged Balanced Accuracy', fontsize=12)
    
    # Create a sorted mapping of numeric values to their string labels
    tick_map = agg_df[['max_features_numeric', 'max_features']].drop_duplicates().sort_values('max_features_numeric')
    # replace '1000K' with '1M', 5000K with 5M
    for feat in tick_map['max_features']:
        if feat.endswith('K'):
            num = int(feat[:-1])
            if num >= 1000:
                tick_map['max_features'] = tick_map['max_features'].replace({feat: f"{num//1000}M"})
    
    # Apply the correctly sorted ticks and labels
    ax.set_ylim(0.3, 1)
    ax.set_xticks(tick_map['max_features_numeric'])
    ax.set_xticklabels(tick_map['max_features'], rotation=45, ha='right')

    ax.legend(title='Split Method', fontsize=10)
    # Save the figure to a file
    plt.tight_layout() # Ensure everything fits without overlapping
    plt.savefig(output_file_name, bbox_inches='tight', dpi=300)
    print(f"Log scale plot successfully saved to '{output_file_name}'")


def plot_per_class_metrics(df_per_class_metrics, pathogen, drug, path_out_dir):
    """
    Plots per-class precision, recall, and F1-score across different experimental settings,
    with separate plots for each class (e.g., Resistant vs. Non-Resistant).

    Args:
        df_per_class_metrics (pd.DataFrame): DataFrame containing per-class metrics.
                                             Expected columns: 'Experiment ID', 'Class', 'Metric', 'Score'.
        pathogen (str): Name of the pathogen.
        drug (str): Name of the drug.
        path_out_dir (str): Directory to save the plots.
    """
    if df_per_class_metrics.empty:
        print("No data to plot for per-class metrics.")
        return

    metric_order = ['Precision', 'Recall', 'F1-score']
    # Get unique classes, assuming they are '0' and '1' or named categories
    # Convert numeric classes to string for plotting if they are ints/floats
    df_per_class_metrics['Class'] = df_per_class_metrics['Class'].astype(str)
    unique_classes = sorted(df_per_class_metrics['Class'].unique())


    for class_label in unique_classes:
        df_class = df_per_class_metrics[df_per_class_metrics['Class'] == class_label].copy()
        if not df_class.empty:
            plt.figure(figsize=(12, 7))
            sns.barplot(
                data=df_class,
                x='Experiment ID',
                y='Score',
                hue='Metric',
                hue_order=metric_order,
                palette='viridis',
                errorbar=None
            )
            # You might want to map '0' and '1' to more descriptive names
            class_name = "SUSCEPTIBLE" if class_label == '0' else ("RESISTANT" if class_label == '1' else class_label)
            plt.title(f'Per-Class Metrics for {pathogen} x {drug} - Class: {class_name}', fontsize=14)
            plt.xlabel("Experiment Configuration", fontsize=12)
            plt.ylabel("Score", fontsize=12)
            plt.ylim(0, 1)
            plt.xticks(fontsize=10)
            plt.legend(title='Metric')
            plt.grid(axis='y', linestyle='--', alpha=0.7)

            output_path_class = os.path.join(path_out_dir, f"{pathogen}_{drug}_class_{class_label}_metrics.png")
            plt.savefig(output_path_class, bbox_inches='tight', dpi=300)
            print(f"Saved per-class metrics plot for Class {class_label} to {output_path_class}")
            plt.close()
        else:
            print(f"No data to plot for Class {class_label}.")


def main():
    parser = get_parser()
    args = parser.parse_args()
    
    now = datetime.now().strftime(f"%Y%m%d-%H%M%S")
    specie_drug_combination = f"{args.pathogen.split('_')[0][0]}_{args.pathogen.split('_')[1]}__{args.drug}"
    
    figs_dir = os.path.join(args.path_out, f"figs_{specie_drug_combination}") #_{now}")
    # Create output directory if it doesn't exist
    os.makedirs(figs_dir, exist_ok=True)

    # Fetch all pickle files in the results directory
    path_csvs = os.path.join(args.path_results, args.pathogen, "results")
    print(f"Searching for CSV files in {path_csvs}")
    # csvs = glob.glob(os.path.join(path_csvs, "*.csv"))
    if args.model == 'all':
        csvs = glob.glob(os.path.join(path_csvs, f"*{args.drug}*.csv"), recursive=True)
    else:
        csvs = glob.glob(os.path.join(path_csvs, f"*{args.model}*{args.drug}*.csv"), recursive=True)
    print(f"Found {len(csvs)} files for combination: {specie_drug_combination}")

    # keep only those CSVs where k=17
    csvs = [c for c in csvs if "k_17" in c]
    print(f"Filtered to {len(csvs)} files with k=17")

    dfs = []
    # Load the results into a DataFrame
    for file in csvs:
        df = pd.read_csv(file)
        if 'split_method' in df.columns:
            df['split_method'] = df['split_method'].replace({'rand': 'random'})
        dfs.append(df)

    df_mega = pd.concat(dfs, ignore_index=True)
    print(f"Combined DataFrame shape: {df_mega.shape}")
    # save combined dataframe
    combined_csv_path = os.path.join(figs_dir, f"{specie_drug_combination}_combined.csv")
    df_mega.to_csv(combined_csv_path, index=False)
    print(f"Saved combined CSV to {combined_csv_path}")

    # Plot accuracy vs. feature matrix size
    output_plot_path = os.path.join(figs_dir, f"{specie_drug_combination}_acc_vs_fMat.png")
    plot_accuracy_by_feature_matrix_size(combined_csv_path, output_plot_path, error_shaded=True)

        
if __name__ == "__main__":
    main()