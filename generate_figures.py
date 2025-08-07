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
        help="Path to the directory containing classification reports.",)
    parser.add_argument("--pathogen", type=str, required=True,
        help="Pathogen to filter results by (e.g., 'p_aeruginosa').")
    parser.add_argument("--drug", type=str, required=True,
        help="Drug to filter results by (e.g., 'cefepime').")
    parser.add_argument("--path-out", type=str, default="figures",
        help="Path to the output directory for figures. Default is 'figures'.")
    
    return parser

def fetch_experiment_info(path_pkl):
    """
    Extracts pathogen, drug, kmer size, and feature threshold from the file name.
    e.g. results_xgb__p_aeruginosa__cefepime__k_31__mode_count__feat_100K.pkl
    """
    filename = os.path.basename(path_pkl)
    pattern = r'results_(?P<classifier>[\w_]+)__' \
          r'(?P<pathogen>[\w_]+)__' \
          r'(?P<drug>[\w_]+)__' \
          r'(?P<kmer>k_\d+)__' \
          r'mode_(?P<mode>[\w_]+)__' \
          r'feat_(?P<feat_thr>\d+K)\.pkl'
    match = re.match(pattern, filename)
    if match:
        return {
            "classifier": match.group("classifier"),
            "pathogen": match.group("pathogen"),
            "drug": match.group("drug"),
            "kmer": match.group("kmer"),
            "mode": match.group("mode"),
            "feature_threshold": match.group("feat_thr")
        }
    else:
        raise ValueError(f"Filename {filename} does not match expected pattern.")


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


def plot_line_metrics(df_line_plot, pathogen, drug, path_out_dir):
    """
    Plots balanced accuracy score as line plots against feature threshold.

    Args:
        df_line_plot (pd.DataFrame): DataFrame containing 'feature_threshold', 'Metric', and 'Score'.
        pathogen (str): Name of the pathogen.
        drug (str): Name of the drug.
        path_out_dir (str): Directory to save the plot.
    """
    if df_line_plot.empty:
        print("No data to plot for the line plot.")
        return

    plt.figure(figsize=(10, 6))
    sns.lineplot(
        data=df_line_plot,
        x="feature_threshold_numeric", # Use the numeric version for plotting
        y="Score",
        hue="Metric",
        marker='o',
        palette='deep',
        linewidth=2,
        errorbar=None  # Disable seaborn's default errorbar since we'll add custom ones
    )
    
    # Add custom error bars for standard deviation
    for metric in df_line_plot['Metric'].unique():
        metric_df = df_line_plot[df_line_plot['Metric'] == metric]
        plt.errorbar(
            x=metric_df['feature_threshold_numeric'],
            y=metric_df['Score'],
            yerr=metric_df['Score_std'],
            fmt='none',  # Don't plot markers, just error bars
            capsize=4,   # Size of error bar caps
            capthick=1.5,
            alpha=0.7
        )
    
    # Add text labels on top of each marker for readability
    for metric in df_line_plot['Metric'].unique():
        metric_df = df_line_plot[df_line_plot['Metric'] == metric]
        for _, row in metric_df.iterrows():
            score_text = f"{row['Score']:.3f}" # Format score to 3 decimal places
            feat_thr = row['feature_threshold']
            plt.text(
                x=row['feature_threshold_numeric'], 
                y=row['Score'], 
                s=feat_thr,
                ha='center',      # Horizontal alignment
                va='bottom',      # Vertically align text above the marker
                fontsize=9,
                fontweight='bold'
            )

    # Set x-ticks to be the original string labels, but based on the sorted numeric values
    x_ticks_numeric = sorted(df_line_plot["feature_threshold_numeric"].unique())
    x_labels_str = [f"{int(x // 1000)}K" if x < 1000000 else f"{int(x // 1000000)}M" for x in x_ticks_numeric]

    plt.xticks(ticks=x_ticks_numeric, labels=x_labels_str, rotation=45, ha='right')
    
    # Optional: Adjust Y-axis limits to make space for the text labels above the markers
    y_min, y_max = plt.ylim()
    # plt.ylim(y_min, y_max + (y_max - y_min) * 0.1)
    plt.ylim(0.65, 1)  # Ensure Y-axis is between 0 and 1 for scores 

    plt.title(f"Performance Metrics vs. Feature Threshold for {pathogen} x {drug}", fontsize=14, pad=20)
    plt.xlabel("Feature Threshold", fontsize=12)
    plt.ylabel("Score", fontsize=12)
    plt.legend(title='Metric')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout() # Adjust layout to prevent labels from being cut off

    output_path_line = os.path.join(path_out_dir, f"{pathogen}_{drug}_metrics_line.png")
    plt.savefig(output_path_line, bbox_inches='tight', dpi=300)
    print(f"Saved line plot to {output_path_line}")
    plt.close()


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
    figs_gridsearch_dir = os.path.join(args.path_out, f"figs_{args.pathogen}_{args.drug}_{now}")
    # Create output directory if it doesn't exist
    os.makedirs(figs_gridsearch_dir, exist_ok=True)

    # Fetch all pickle files in the results directory
    path_results = args.path_results
    pkl_files = glob.glob(os.path.join(path_results, "**", "*.pkl"), recursive=True)    # recursively search subdirectories 
    print(f"Found {len(pkl_files)} pickle files")

    pathogen, drug = args.pathogen, args.drug
    specie_drug_reports_paths = [] 

    # Filter pickle files for the specified pathogen and drug
    for path_pkl in pkl_files:
        try:
            exp_info = fetch_experiment_info(path_pkl)
            if exp_info["pathogen"] == pathogen and exp_info["drug"] == drug:
                specie_drug_reports_paths.append(path_pkl)
        except ValueError as e:
            print(f"Skipping file due to naming pattern mismatch: {path_pkl} - {e}")

    print(f"Number of relevant reports for {pathogen} and {drug}: {len(specie_drug_reports_paths)}")
    for path_pkl in specie_drug_reports_paths:
        print(f" - {path_pkl}")

    all_line_plot_data = []
    all_combined_metrics_for_plot = []
    all_per_class_metrics_data = []

    for path_pkl in specie_drug_reports_paths:
        # Load the classification report from the pickle file
        with open(path_pkl, 'rb') as f:
            try:
                report_data = pickle.load(f)
                report_dict = None # Initialize to None

                if isinstance(report_data, pd.DataFrame) and 'classification_report' in report_data.columns:
                    report_dict = report_data['classification_report'].iloc[0] # Assuming dict is in first row
                else:
                    report_dict = report_data['best_classification_report']

                # Try to extract balanced_accuracy_scores and f1_macro_scores for line plot
                if isinstance(report_data, dict) and "balanced_accuracy_scores" in report_data.keys():
                    # TODO: define list of metrics for line plot (per each fold), store each metric's data in a loop
                    mean_balanced_acc = np.mean(report_data["balanced_accuracy_scores"])
                    std_balanced_acc = np.std(report_data["balanced_accuracy_scores"])
                    #  mean_f1_macro = np.mean(report_data["f1_macro_scores"])

                    exp_info = fetch_experiment_info(path_pkl) # Ensure exp_info is available
                    feature_threshold_str = exp_info["feature_threshold"]
                    feature_threshold_numeric = int(feature_threshold_str.replace("K", "")) * 1000

                    all_line_plot_data.append({
                         'feature_threshold': feature_threshold_str,
                         'feature_threshold_numeric': feature_threshold_numeric,
                         'Metric': 'Balanced Accuracy',
                         'Score': mean_balanced_acc,
                         'Score_std': std_balanced_acc
                    })
                    #  all_line_plot_data.append({
                    #      'feature_threshold': feature_threshold_str,
                    #      'feature_threshold_numeric': feature_threshold_numeric,
                    #      'Metric': 'F1 Macro',
                    #      'Score': mean_f1_macro
                    #  })
                else:
                    print(f"Warning: {path_pkl} dict does not contain 'balanced_accuracy_scores'. Skipping for line plot data collection.")

            except Exception as e:
                print(f"Error loading pickle file {path_pkl}: {e}. Skipping all plots for this file.")
                continue
        
        # Extract information from the file name
        exp_info = fetch_experiment_info(path_pkl)

        feature_threshold = exp_info["feature_threshold"]
        kmer_size = exp_info["kmer"]
        classifier_type = exp_info["classifier"] # Added classifier for uniqueness if needed

        # Prepare a unique ID for each experiment configuration
        experiment_id = f"K={kmer_size.split('_')[1]} Feat={feature_threshold}"
        # If you have different classifiers, you might add f"{classifier_type} - K={kmer_size.split('_')[1]} Feat={feature_threshold}"

        # --- New Logic for grouped bar chart ---
        # Ensure 'report_dict' is the actual classification_report dictionary
        if 'macro avg' in report_dict and 'weighted avg' in report_dict:
            # Add Macro Average metrics
            all_combined_metrics_for_plot.append({
                'Experiment ID': experiment_id,
                'Metric Type': 'Macro Average',
                'Metric': 'Precision', 'Score': report_dict['macro avg']['precision']
            })
            all_combined_metrics_for_plot.append({
                'Experiment ID': experiment_id,
                'Metric Type': 'Macro Average',
                'Metric': 'Recall', 'Score': report_dict['macro avg']['recall']
            })
            all_combined_metrics_for_plot.append({
                'Experiment ID': experiment_id,
                'Metric Type': 'Macro Average',
                'Metric': 'F1-score', 'Score': report_dict['macro avg']['f1-score']
            })
            # Add Weighted Average metrics
            all_combined_metrics_for_plot.append({
                'Experiment ID': experiment_id,
                'Metric Type': 'Weighted Average',
                'Metric': 'Precision', 'Score': report_dict['weighted avg']['precision']
            })
            all_combined_metrics_for_plot.append({
                'Experiment ID': experiment_id,
                'Metric Type': 'Weighted Average',
                'Metric': 'Recall', 'Score': report_dict['weighted avg']['recall']
            })
            all_combined_metrics_for_plot.append({
                'Experiment ID': experiment_id,
                'Metric Type': 'Weighted Average',
                'Metric': 'F1-score', 'Score': report_dict['weighted avg']['f1-score']
            })
        else:
            print(f"Warning: {path_pkl} does not contain 'macro avg' or 'weighted avg' in its report dictionary. Skipping for grouped bar plot.")


        # ========================
        # Logic for per-class metrics grouped bar chart
        # ========================
        if report_dict:
            # Iterate through classes '0', '1', etc. (assuming binary classification)
            for class_label in report_dict.keys():
                if isinstance(report_dict[class_label], dict) and class_label in ['0', '1']: # Ensure it's a class dictionary
                    all_per_class_metrics_data.append({
                        'Experiment ID': experiment_id,
                        'Class': class_label,
                        'Metric': 'Precision', 'Score': report_dict[class_label]['precision']
                    })
                    all_per_class_metrics_data.append({
                        'Experiment ID': experiment_id,
                        'Class': class_label,
                        'Metric': 'Recall', 'Score': report_dict[class_label]['recall']
                    })
                    all_per_class_metrics_data.append({
                        'Experiment ID': experiment_id,
                        'Class': class_label,
                        'Metric': 'F1-score', 'Score': report_dict[class_label]['f1-score']
                    })
            if not any(isinstance(report_dict[k], dict) and 'precision' in report_dict[k] for k in report_dict.keys() if k not in ['accuracy', 'macro avg', 'weighted avg']):
                 print(f"Warning: {path_pkl} report_dict does not contain valid per-class metrics. Skipping for per-class bar plot.")
        else:
            print(f"Warning: {path_pkl} has no valid report_dict. Skipping for per-class bar plot.")

    # ======================
    # Line Plot 
    # ======================
    df_line_plot = pd.DataFrame(all_line_plot_data)
    if not df_line_plot.empty:
        df_line_plot.sort_values(by="feature_threshold_numeric", inplace=True)
        plot_line_metrics(
            df_line_plot,
            pathogen,
            drug,
            figs_gridsearch_dir
        )
    else:
        print("No data collected for the line plot (balanced_accuracy and f1_macro).")

    # ======================
    # Grouped Bar Chart for Macro and Weighted Average Metrics
    # ======================
    if all_combined_metrics_for_plot:
        df_combined_metrics_for_plot = pd.DataFrame(all_combined_metrics_for_plot)

        # Sort the 'Experiment ID' for better visualization if it contains numbers
        # This assumes format like "K=X Feat=YK"
        def sort_key(exp_id):
            parts = exp_id.replace('K=', '').replace('Feat=', '').replace('K', '').split(' ')
            k_val = int(parts[0])
            feat_val = int(parts[1])
            return (k_val, feat_val)

        df_combined_metrics_for_plot['Experiment ID_sort_key'] = df_combined_metrics_for_plot['Experiment ID'].apply(sort_key)
        df_combined_metrics_for_plot.sort_values(by='Experiment ID_sort_key', inplace=True)
        df_combined_metrics_for_plot.drop(columns='Experiment ID_sort_key', inplace=True)

        plot_metrics_classification_average(
            df_combined_metrics_for_plot,
            pathogen,
            drug,
            'macro_avg',  # Plot Macro Average metrics
            figs_gridsearch_dir # Pass the directory for saving
        )
    else:
        print("No data found for the grouped bar chart (macro/weighted average metrics).")
    
    # ======================
    # Grouped Bar Chart for Per-Class Metrics
    # ======================
    if all_per_class_metrics_data:
        df_per_class_metrics = pd.DataFrame(all_per_class_metrics_data)
        # Apply the same sorting to ensure consistency across plots
        def sort_key(exp_id):
            parts = exp_id.replace('K=', '').replace('Feat=', '').replace('K', '').split(' ')
            k_val = int(parts[0])
            feat_val = int(parts[1])
            return (k_val, feat_val)

        df_per_class_metrics['Experiment ID_sort_key'] = df_per_class_metrics['Experiment ID'].apply(sort_key)
        df_per_class_metrics.sort_values(by='Experiment ID_sort_key', inplace=True)
        df_per_class_metrics.drop(columns='Experiment ID_sort_key', inplace=True)

        plot_per_class_metrics(
            df_per_class_metrics,
            pathogen,
            drug,
            figs_gridsearch_dir
        )
    else:
        print("No data found for per-class metrics, even though --plot-per-class was specified.")
        
        
if __name__ == "__main__":
    main()