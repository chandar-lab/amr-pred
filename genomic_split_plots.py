import os
import logging
import glob
from data.data_utils import load_data


import umap
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from sklearn.manifold import TSNE

from data.generate_phylogeny_folds import get_matrix


def plot_folds_umap(folds, dist_table):
    """
    Plot the folds using a scatter plot.
    Use Umap to turn the distance table into a 2D representation.
    parameters:
    - folds: dict, mapping of assembly names to fold numbers
    - dist_table: pd.DataFrame, distance table with assemblies as index and columns
    """
    # Convert to NumPy array
    distance_matrix = dist_table.values

    # Create list of fold nums for labels
    fold_labels = [folds[id] for id in dist_table.index]

    # Run UMAP
    reducer = umap.UMAP(metric="precomputed", random_state=42)
    embedding = reducer.fit_transform(distance_matrix)

    # Plot with colors based on folds
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(embedding[:, 0], embedding[:, 1], c=fold_labels, cmap='tab10', s=40)
    plt.title("UMAP Projection Colored by Fold")
    plt.xlabel("UMAP 1")
    plt.ylabel("UMAP 2")

    # Add legend for folds
    handles, labels = scatter.legend_elements(prop="colors")
    unique_folds = sorted(set(fold_labels))
    plt.legend(handles, [f"Fold {i}" for i in unique_folds], title="Fold", bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.tight_layout()
    # Save plt to disk
    plt.savefig('genomic_folds_umap.png')

def plot_folds_tsne(folds, dist_table):
    """
    Plot the folds using a scatter plot.
    Use t-SNE to turn the distance table into a 2D representation.
    parameters:
    - folds: dict, mapping of assembly names to fold numbers
    - dist_table: pd.DataFrame, distance table with assemblies as index and columns
    """
    # Convert to NumPy array
    distance_matrix = dist_table.values

    # Create list of fold nums for labels
    fold_labels = [folds[id] for id in dist_table.index]

    # Run t-SNE
    tsne = TSNE(n_components=2, metric="precomputed", init="random", random_state=42)
    embedding = tsne.fit_transform(distance_matrix)

    # Plot with colors based on folds
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(embedding[:, 0], embedding[:, 1], c=fold_labels, cmap='tab10', s=40)
    plt.title("t-SNE Projection Colored by Fold")
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")

    # Add legend for folds
    handles, labels = scatter.legend_elements(prop="colors")
    unique_folds = sorted(set(fold_labels))
    plt.legend(handles, [f"Fold {i}" for i in unique_folds], title="Fold", bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.tight_layout()
    # Save plt to disk
    plt.savefig('genomic_folds_tsne.png')


def get_genomic_folds(pathogen, antibiotic, path_dataroot, path_csv, num_folds=5, dist_method='dashing', force_regenerate=False):
    """
    Generate genomic distance folds for a given pathogen and antibiotic.
    parameters:
    - pathogen: str, name of the pathogen (e.g., 'pseudomonas_aeruginosa')
    - antibiotic: str, name of the antibiotic (e.g., 'cefepime')
    - path_dataroot: str, root path of the dataset genomic assemblies
    - path_csv: str, path to the folder containing CSV files for each specie-drug combination
    - num_folds: int, number of folds to create (default is 5
    - dist_method: str, method to use for distance calculation ('mash', 'ani', or 'dashing', default is 'dashing')
    - force_regenerate: bool, whether to force regeneration of the assemblies list and distance matrix
    returns:
    - fold_assignments: dict, mapping of assembly names to fold numbers
    """
    fold_assignments_csv = os.path.join(path_dataroot, f"{pathogen}__{antibiotic}__{dist_method}__phylogeny_{num_folds}folds.csv")
    dist_table_csv = os.path.join(path_dataroot, f"{pathogen}__{antibiotic}__{dist_method}__distance_table.csv")
    logger = logging.getLogger(__name__) # Ensures the messages sent to the logger have the function name

    # Return the fold assignments as a dict if the file already exists
    # if os.path.isfile(fold_assignments_csv) and os.path.isfile(dist_table_csv) and not force_regenerate:
    #     logger.info(f"Fold assignments already exist at {fold_assignments_csv}, loading from file.")
    #     fold_assignments_df = pd.read_csv(fold_assignments_csv)
    #     fold_assignments = dict(zip(fold_assignments_df['assembly'], fold_assignments_df['fold']))

    #     logger.info(f"Distance table already exists at {dist_table_csv}, loading from file.")
    #     dist_table = pd.read_csv(dist_table_csv, index_col=0)
    #     return fold_assignments, dist_table

    # Task 1: Generate files with list of assemblies for the pathogen
    path_root_fna = path_dataroot
    path_csv = path_csv

    dataset = f"{pathogen}__{antibiotic}_SR_filt.csv"
    path_data_csv = os.path.join(path_csv, dataset)
    if not os.path.isfile(path_data_csv):
        raise FileNotFoundError(f"CSV file not found: {path_data_csv}. Please check the name of the pathogen and drug.")
    
    # Generate the file with paths to all assemblies for the pathogen, if needed
    assemblies_txt = os.path.join(path_root_fna, "all_assemblies.txt")
    if (not os.path.isfile(os.path.join(assemblies_txt))) or force_regenerate:
        logger.info(f"Generating list of assemblies for {pathogen}...")
        # Get all assemblies from the path_root_fna folder, and put them in a list
        subfolders = os.listdir(path_root_fna)
        assembly_files = [glob.glob(os.path.join(path_root_fna, folder, "ncbi_dataset", "data", folder, "*.fna"))[0] for folder in subfolders if os.path.isdir(os.path.join(path_root_fna, folder))]
 
        # Save the list of assemblies to a file in the path_root_fna folder
        with open(assemblies_txt, "w") as f:
            for assembly in assembly_files:
                f.write(f"{assembly}\n")
    else:
        logger.info(f"List of assemblies already exists at {assemblies_txt}, skipping regeneration.")

    # Make the assemblies list hold only the names of the assemblies, not the full paths
    with open(assemblies_txt, "r") as f:
        # Extract the name of the parent folder from each assembly path
        all_pathogen_assemblies = [line.strip().split('/')[-2] for line in f.readlines() if line.strip()]


    # Task 2: Generate and save distance matrix for the assemblies using the desired distance metric
    full_pathogen_matrix = get_matrix(assemblies_txt, path_root_fna, method=dist_method, force_regenerate=force_regenerate)
    logger.info(f"Full pathogen distance matrix shape: {full_pathogen_matrix.shape}")


    # Task 3: determine which assemblies are relevant for the antibiotic
    # Load the data using the load_data function
    dataset_list = load_data(path_data_csv, path_root_fna)

    # Create a set of all assemblies from the dataset_list
    unordered_dataset_assemblies = [entry['assembly'] for entry in dataset_list]

    # Create a list of indices for the assemblies that are also in the dataset_list
    relevant_indices = []
    ordered_dataset_assemblies = []
    for i, assembly in enumerate(all_pathogen_assemblies):
        if assembly in unordered_dataset_assemblies:
            relevant_indices.append(i)
            ordered_dataset_assemblies.append(assembly)
    print(f"Number of relevant assemblies for {pathogen} and {antibiotic}: {len(relevant_indices)}")

    # Filter the matrix to only include the relevant assemblies
    specie_drug_dist_matrix = full_pathogen_matrix[np.ix_(relevant_indices, relevant_indices)]
    logger.info(f"Filtered distance matrix shape: {specie_drug_dist_matrix.shape}")

    # Turn into a dataframe indexed by the assembly names on both axes
    specie_drug_dist_df = pd.DataFrame(specie_drug_dist_matrix, index=ordered_dataset_assemblies, columns=ordered_dataset_assemblies)
    # Save the specie-drug distance table to a CSV file
    specie_drug_dist_matrix_csv = os.path.join(path_root_fna, f"{pathogen}__{antibiotic}__{dist_method}__distance_table.csv")
    specie_drug_dist_df.to_csv(specie_drug_dist_matrix_csv)
    logger.info(f"Saved distance matrix to {specie_drug_dist_matrix_csv}")

    # Task 4: Create folds based on the distance matrix, with a csv
    ### Use the method described in Learning From Limited Data: "Towards Best Practice Techniques for Antimicrobial Resistance Prediction From Whole Genome Sequencing Data", 2021

    logger.info(f"Creating {num_folds} folds based on the distance matrix...")
    # Begin by selecting seeds, equal to the number of folds, which act as the centroids of the folds.

    fold_seeds = [] # List of indices of the assemblies that will be used as seeds for the folds
    # Begin by selecting the pair of seeds with the maximum distance between them
    # This can be done by finding the single biggest entry in the matrix
    index = np.argmax(specie_drug_dist_matrix)
    row, col = np.unravel_index(index, specie_drug_dist_matrix.shape)
    fold_seeds.append(row) 
    fold_seeds.append(col)
    # Find the location (row, column) of the max value
    max_idx = specie_drug_dist_df.stack().idxmax()

    # Access the row and column names
    row_name, col_name = max_idx
    print(f"Initial seeds by df: {row_name}, {col_name} (indices: {fold_seeds[0]}, {fold_seeds[1]})")
    
    # Now additional seeds can be selected by iteratively finding the assembly with the largest minimum distance to an existing seed
    while len(fold_seeds) < num_folds:
        # Calculate the minimum distance to an existing seed for each assembly
        min_distances = np.min(specie_drug_dist_matrix[:, fold_seeds], axis=1)
        # Find the assembly with the maximum minimum distance
        new_seed = np.argmax(min_distances)
        fold_seeds.append(new_seed)
    
    # === Step 1: Compute 2D projection ===
    tsne = TSNE(metric='precomputed', init='random', random_state=42)
    embedding = tsne.fit_transform(specie_drug_dist_matrix)

    # === Step 2: Setup plot ===
    fig, ax = plt.subplots(figsize=(8, 6))
    initial_colors = ['lightgray'] * len(ordered_dataset_assemblies)

    # Color seeds immediately
    colors = ['red', 'blue', 'green', 'orange', 'magenta', 'peru', 'black', 'yellow', 'turquoise', 'springgreen']
    for i, seed_idx in enumerate(fold_seeds):
        initial_colors[seed_idx] = colors[i % len(colors)]

    sc = ax.scatter(embedding[:, 0], embedding[:, 1], c=initial_colors, s=40)
    ax.set_title("Fold Assignments (t-SNE Projection with each fold in a different color)")
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")

    # === Step 3: Assign points and capture frame colors ===
    assigned_indices = set(fold_seeds)  # Start with seeds as already assigned
    fold_assignments = {ordered_dataset_assemblies[i]: i % num_folds for i in fold_seeds}
    fold_sizes = [0] * num_folds
    for i in fold_seeds:
        fold_sizes[i % num_folds] += 1

    frames = [initial_colors.copy()]

    logger.info("Starting fold assignment...")
    seed_idx = 0
    while len(assigned_indices) < len(ordered_dataset_assemblies):
        distances = specie_drug_dist_matrix[seed_idx, :]
        unassigned_indices = [i for i in range(len(ordered_dataset_assemblies)) if i not in assigned_indices]
        if not unassigned_indices:
            break

        min_distance_index = np.argmin(distances[unassigned_indices])
        actual_index = unassigned_indices[min_distance_index]
        fold_id = len(fold_assignments) % num_folds

        fold_assignments[ordered_dataset_assemblies[actual_index]] = fold_id
        assigned_indices.add(actual_index)
        fold_sizes[fold_id] += 1

        # Create new color frame
        current_colors = frames[-1].copy()
        current_colors[actual_index] = colors[fold_id % len(colors)]
        frames.append(current_colors)
        # Update the seed_idx
        seed_idx = (seed_idx + 1) % len(fold_seeds)

    logger.info("Fold assignment completed.")
    # === Step 4: Animate ===
    def update(frame_idx):
        sc.set_color(frames[frame_idx])
        return sc,

    ani = animation.FuncAnimation(
        fig, update, frames=len(frames), interval=200, blit=False
    )

    # === Step 5: Save to file ===
    ani.save("fold_assignment_animation.mp4", fps=5)


    # print("Fold assignments:")
    # for assembly, fold in fold_assignments.items():
    #     print(f"{assembly}: Fold {fold}")

    # Print the size of each fold 
    logger.info(f"Fold sizes: {fold_sizes}")

    # Save the fold assignments to a CSV file
    fold_assignments_df = pd.DataFrame(list(fold_assignments.items()), columns=['assembly', 'fold'])
    # Save to csv in the path_root_fna folder
    fold_assignments_df.to_csv(fold_assignments_csv, index=False)
    logger.info(f"Fold assignments saved to {fold_assignments_csv}")

    # Return the fold assignments as a dict
    fold_assignments = dict(zip(fold_assignments_df['assembly'], fold_assignments_df['fold']))
    return fold_assignments, specie_drug_dist_df

if __name__ == "__main__":
    console_handler = logging.StreamHandler()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[console_handler]
    )

    pathogen = 'pseudomonas_aeruginosa'
    antibiotic = 'cefepime'
    # Example usage
    # Assuming `folds` is a dictionary mapping assembly names to fold numbers
    # and `dist_table` is a DataFrame with the distance table
    folds, dist_table = get_genomic_folds(pathogen, antibiotic, "../amr_genomic_data/"+pathogen, "./datasets/"+pathogen+"/filtered/", num_folds=10, dist_method='mash')

    #plot_folds_umap(folds, dist_table)
