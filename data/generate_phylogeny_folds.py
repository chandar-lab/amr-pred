''' Example usage:
python generate_phylogeny_folds.py --path-dataroot ../amr_genomic_data/pseudomonas_aeruginosa --path-csv ./datasets --pathogen pseudomonas_aeruginosa --antibiotic cefepime'''

import os
import sys
import argparse
import logging
import time

import pandas as pd
import numpy as np
from datetime import datetime
from collections import defaultdict
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from scipy.spatial.distance import squareform

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.genomic_dst_utils import mash_distance_txt, ani_distance_txt, dashing_distance_txt, get_assemblies_txt
from data.data_utils import load_data, get_folds_stats


def parse_args():
    parser = argparse.ArgumentParser(description="Create genomic distance folds for a dataset")

    # Alex's path-dataroot: ../amr-genomic-data/<pathogen>, assuming current root is scratch/amr
    parser.add_argument("--path-dataroot", type=str, help="Root path of the dataset genomic sequences for all species")
    # Alex's path-csv: ./datasets, assuming current root is scratch/amr
    parser.add_argument("--path-csv", type=str, default="", help="Path to the folder containing CSV files for each specie-drug combination")
    parser.add_argument("--path-out", type=str, default="", help="Path to the output folder where results will be saved. If not provided, will save in the current directory.")
    parser.add_argument("--pathogen", type=str, help="Name of the pathogen of interest")
    parser.add_argument("--antibiotic", type=str, help="Name of the antibiotic of interest")
    parser.add_argument("--folds", type=int, default=5, help="Number of folds to create")
    parser.add_argument("--dist_method", type=str, default='dashing', choices=['mash', 'ani', 'dashing'], 
                        help="Method to use for distance calculation: 'mash', 'ani', or 'dashing'. Default is 'dashing'.")
    parser.add_argument("--clustering", type=str, default='unordered', choices=['unordered', 'hierarchical'], 
                        help="Method to use for splitting the data into folds: 'unordered' or 'hierarchical'. Default is 'unordered'.")
    parser.add_argument("--force_regenerate", action='store_true', help="Force regeneration of the assemblies list and distance matrix, even if they already exist")
    return parser.parse_args()


def get_matrix(assemblies_txt, method='dashing', force_regenerate=False):
    # Map the three methods to their respective functions
    method_map = {
        'mash': mash_distance_txt,
        'ani': ani_distance_txt,
        'dashing': dashing_distance_txt
    }

    if method not in method_map:
        raise ValueError(f"Invalid method '{method}'. Choose from {list(method_map.keys())}.")
    else:
        logging.info(f"Generating distance matrix using {method} method...")
        dist_func = method_map[method]

    # Get the number of assemblies from the assemblies_txt file
    with open(assemblies_txt, 'r') as f:
        assemblies = [line.strip() for line in f.readlines() if line.strip()]
    
    # Start the timer to measure the time taken to generate the distance matrix
    start_time = time.time()

    # Generate the distance matrix using the specified method
    logging.info(f"Generating {method} distance matrix for {len(assemblies)} assemblies...")    
    dist_matrix = dist_func(assemblies_txt)
    
    elapsed_time = time.time() - start_time
    logging.info(f"Distance matrix generated in {elapsed_time:.2f} seconds.")
    
    return dist_matrix

def generate_folds(logger, path_fold_assignments_csv, assemblies, specie_drug_dist_matrix, clustering, num_folds=5):

    fold_assignments = {}

    if clustering == 'unordered':
        logger.info(f"Creating {num_folds} folds based on the distance matrix...")
        # Begin by selecting seeds, equal to the number of folds, which act as the centroids of the folds.

        fold_seeds = [] # List of indices of the assemblies that will be used as seeds for the folds
        # Begin by selecting the pair of seeds with the maximum distance between them
        # This can be done by finding the single biggest entry in the matrix
        index = np.argmax(specie_drug_dist_matrix)
        row, col = np.unravel_index(index, specie_drug_dist_matrix.shape)
        fold_seeds.append(row) 
        fold_seeds.append(col)
        
        # Now additional seeds can be selected by iteratively finding the assembly with the largest minimum distance to an existing seed
        while len(fold_seeds) < num_folds:
            # Calculate the minimum distance to an existing seed for each assembly
            min_distances = np.min(specie_drug_dist_matrix[:, fold_seeds], axis=1)
            # Find the assembly with the maximum minimum distance
            new_seed = np.argmax(min_distances)
            fold_seeds.append(new_seed)
        
        logger.info("Assigning assemblies to folds based on the selected seeds...")
        # Create a list to hold the fold assignments
        assigned_indices = set()
        fold_sizes = [1] * num_folds  # Used to logging, to ensure equal distribution of assemblies.

        # Begin by assigning the seeds to their respective folds
        for i, seed in enumerate(fold_seeds):
            fold_assignments[assemblies[seed]] = i
            assigned_indices.add(seed)

        # Take turns assigning assemblies to folds, starting with the first seed
        curr_seed = 0
        while len(assigned_indices) < len(assemblies):
            # Calculate the distances from the current fold's seed to all assemblies not yet assigned to a fold
            distances = specie_drug_dist_matrix[fold_seeds[curr_seed], :]
            unassigned_indices = [i for i in list(range(len(assemblies))) if i not in assigned_indices]
            # Find the assembly with the minimum distance to the current seed
            min_distance_index = np.argmin(distances[unassigned_indices])
            # Assign this assembly to the current fold
            fold_assignments[assemblies[unassigned_indices[min_distance_index]]] = curr_seed
            assigned_indices.add(unassigned_indices[min_distance_index])
            fold_sizes[curr_seed] += 1
            # Move to the next seed
            curr_seed = (curr_seed + 1) % num_folds

    elif clustering == 'hierarchical':
        # --- Step 1: Hierarchical Clustering --- 
        # 1. Convert the square distance matrix to the condensed format
        # The linkage function requires a 1D array representing the upper triangle of the matrix.
        condensed_dist_mat = squareform(specie_drug_dist_matrix)
        print(f"Condensed Distance Matrix Shape: {condensed_dist_mat.shape}")

        # 2. Perform hierarchical clustering (e.g., using UPGMA which is 'average' linkage)
        # UPGMA is often used with genomic distance data.
        Z = linkage(condensed_dist_mat, method='ward', metric='jaccard')
        # print(f"Z shape: {Z.shape}")

        # --- Step 2: Form Granular Flat Clusters ---
        # Create more clusters than folds to allow for flexible, balanced grouping.
        # A good rule of thumb is 4-5 times the number of folds.
        num_intermediate_clusters = num_folds * 5  # Adjust this multiplier as needed for your dataset.
        min_diff_between_folds = 5  # Minimum size difference between folds to consider them balanced.
        
        while True:
            print(f"Attempting to create {num_intermediate_clusters} intermediate clusters...")    
            flat_clusters = fcluster(Z, num_intermediate_clusters, criterion='maxclust')
            # print(f"Number of flat clusters formed: {len(set(flat_clusters))}")
            # print(f"Flat clusters: {flat_clusters}")

            # --- Step 3: Group Sample Indices by Cluster ---
            # Create a dictionary where keys are cluster IDs and values are lists of sample indices.
            cluster_groups = defaultdict(list)
            for i, cluster_id in enumerate(flat_clusters):
                cluster_groups[cluster_id].append(i)

            # --- Step 4: Distribute Clusters into Folds ---
            # Assign entire clusters to folds to keep related samples together.
            # This loop assigns each cluster to the fold that is currently the smallest.
            folds = [[] for _ in range(num_folds)]
            # Sort clusters by size (largest first) to improve balance.
            sorted_clusters = sorted(cluster_groups.values(), key=len, reverse=True)

            for cluster in sorted_clusters:
                # Find the fold with the fewest samples.
                smallest_fold = min(folds, key=len)
                # Add all samples from the current cluster to that fold.
                smallest_fold.extend(cluster)

            # --- Verification ---
            fold_sizes = [len(fold) for fold in folds]
            print(f"Sizes of folds: {fold_sizes}")
            # Check if the folds are balanced enough
            if max(fold_sizes) - min(fold_sizes) < min_diff_between_folds:
                print("Folds are balanced enough, stopping the clustering process.")
                break
            else:
                print("Folds are not balanced enough, continuing to form clusters.")
                num_intermediate_clusters += 5

        # --- Step 5: Populate fold_assignments dictionary ---
        # Convert the folds list (containing indices) to the fold_assignments dict (assembly names to fold numbers)
        for fold_idx, fold_indices in enumerate(folds):
            for assembly_idx in fold_indices:
                fold_assignments[assemblies[assembly_idx]] = fold_idx

    # Print the size of each fold 
    logger.info(f"Clustering method: {clustering} \t Fold sizes: {fold_sizes}")

    # Save the fold assignments to a CSV file
    fold_assignments_df = pd.DataFrame(list(fold_assignments.items()), columns=['assembly', 'fold'])
    # Save to csv in the path_root_fna folder
    fold_assignments_df.to_csv(path_fold_assignments_csv, index=False)
    logger.info(f"Fold assignments saved to {path_fold_assignments_csv}")

    # Return the fold assignments as a dict
    fold_assignments = dict(zip(fold_assignments_df['assembly'], fold_assignments_df['fold']))
    
    return fold_assignments


def save_dist_matrix(path_base, matrix, names):
    """
    Save a square distance matrix as float32 <path_base>.npy, with the assembly names in <path_base>_names.npy.
    """
    np.save(f"{path_base}.npy", np.asarray(matrix, dtype=np.float32))
    np.save(f"{path_base}_names.npy", np.asarray(names, dtype=str))


def load_dist_matrix(path_base, mmap_mode=None):
    """
    Load a distance matrix saved by save_dist_matrix.
    If only a legacy <path_base>.csv exists, it is converted to .npy once and then loaded.
    parameters:
    - path_base: str, path of the matrix without extension
    - mmap_mode: passed to np.load (e.g. 'r' to only read the rows/columns that are indexed)
    returns:
    - (matrix, names) tuple, or None if no saved matrix exists
    """
    path_npy, path_names = f"{path_base}.npy", f"{path_base}_names.npy"
    if not (os.path.isfile(path_npy) and os.path.isfile(path_names)):
        path_csv = f"{path_base}.csv"
        if not os.path.isfile(path_csv):
            return None
        logging.info(f"Converting legacy distance matrix {path_csv} to .npy")
        dist_df = pd.read_csv(path_csv, index_col=0)
        save_dist_matrix(path_base, dist_df.values, dist_df.index.astype(str))
    return np.load(path_npy, mmap_mode=mmap_mode), np.load(path_names).tolist()


def get_genomic_folds(logger, pathogen, antibiotic, path_dataroot, path_csv, path_out, num_folds=5, 
                      dist_method='dashing', clustering_method='unordered', force_regenerate=False):
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
    path_fold_assignments_csv = os.path.join(
        path_out, f"{pathogen}__{antibiotic}__{dist_method}__phyloFolds_{clustering_method}.csv")
    dist_table_path = os.path.join(path_out, f"{pathogen}__{antibiotic}__{dist_method}__distMat")

    # Return the fold assignments as a dict if the file already exists
    if os.path.isfile(path_fold_assignments_csv) and not force_regenerate:
        loaded_dist = load_dist_matrix(dist_table_path)
        if loaded_dist is not None:
            logger.info(f"Fold assignments already exist at {path_fold_assignments_csv}, loading from file.")
            fold_assignments_df = pd.read_csv(path_fold_assignments_csv)
            fold_assignments = dict(zip(fold_assignments_df['assembly'], fold_assignments_df['fold']))

            logger.info(f"Distance table already exists at {dist_table_path}.npy, loading from file.")
            dist_mat, dist_names = loaded_dist
            return fold_assignments, pd.DataFrame(dist_mat, index=dist_names, columns=dist_names)

    # Task 1: Generate files with list of assemblies for the pathogen
    path_root_fna = path_dataroot
    
    # Generate the file with paths to all assemblies for the pathogen, if needed
    assemblies_txt = get_assemblies_txt(path_root_fna, force_regenerate=force_regenerate)

    # Make the assemblies list hold only the names of the assemblies, not the full paths
    with open(assemblies_txt, "r") as f:
        # Extract the name of the parent folder from each assembly path
        all_pathogen_assemblies = [line.strip().split('/')[-2] for line in f.readlines() if line.strip()]


    # Task 2: Generate and save distance matrix for the assemblies using the desired distance metric
    full_pathogen_matrix_path = os.path.join(path_out, f"{pathogen}__{dist_method}__distMatAll")
    # Check if the distance matrix already exists. If not, generate it.
    # Memory-mapped, so only the rows/columns relevant to this antibiotic are read from disk
    loaded_full = None if force_regenerate else load_dist_matrix(full_pathogen_matrix_path, mmap_mode='r')
    if loaded_full is None:
        # Cast to float32 so a fresh run gives the same values as a run loading the saved matrix
        full_pathogen_matrix = get_matrix(assemblies_txt, method=dist_method, force_regenerate=force_regenerate).astype(np.float32)
        logger.info(f"Full pathogen distance matrix shape: {full_pathogen_matrix.shape}")
        save_dist_matrix(full_pathogen_matrix_path, full_pathogen_matrix, all_pathogen_assemblies)
        logger.info(f"Saved full distance matrix to {full_pathogen_matrix_path}.npy")
    else:
        logging.info(f"{dist_method} distance matrix already exists at {full_pathogen_matrix_path}.npy, skipping regeneration.")
        # Take the assembly names from the saved matrix, as assemblies_txt may have been regenerated since
        # (different order or contents), which would misalign the names with the matrix rows
        full_pathogen_matrix, all_pathogen_assemblies = loaded_full


    # Task 3: determine which assemblies are relevant for the antibiotic
    dataset = f"{pathogen}__{antibiotic}_SR_filt.csv"
    path_data_csv = os.path.join(path_csv, dataset)
    if not os.path.isfile(path_data_csv):
        raise FileNotFoundError(f"CSV file not found: {path_data_csv}. Please check the name of the pathogen and drug.")

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
    logger.info(f"Number of relevant assemblies for {pathogen} and {antibiotic}: {len(relevant_indices)}")

    # Filter the matrix to only include the relevant assemblies
    specie_drug_dist_matrix = full_pathogen_matrix[np.ix_(relevant_indices, relevant_indices)]
    logger.info(f"Filtered distance matrix shape: {specie_drug_dist_matrix.shape}")

    # Turn into a dataframe indexed by the assembly names on both axes
    specie_drug_dist_df = pd.DataFrame(specie_drug_dist_matrix, index=ordered_dataset_assemblies, columns=ordered_dataset_assemblies)
    # Check if the distance matrix already exists. If not, save it.
    loaded_dist = None if force_regenerate else load_dist_matrix(dist_table_path, mmap_mode='r')
    if loaded_dist is None:
        logger.info(f"Saving distance matrix ...")
        save_dist_matrix(dist_table_path, specie_drug_dist_matrix, ordered_dataset_assemblies)
        logger.info(f"Saved distance matrix to {dist_table_path}.npy")
    else:
        logger.info(f"Distance matrix already exists at {dist_table_path}.npy, skipping regeneration.")
        # Check if the shape of the saved matrix matches the expected shape
        if loaded_dist[0].shape != specie_drug_dist_matrix.shape:
            logger.error(f"Loaded distance matrix shape {loaded_dist[0].shape} does not match expected shape {specie_drug_dist_matrix.shape}.")
            logger.info("Regenerating the distance matrix...")
            save_dist_matrix(dist_table_path, specie_drug_dist_matrix, ordered_dataset_assemblies)
            logger.info(f"Saved distance matrix to {dist_table_path}.npy")

    # Task 4: Create folds based on the distance matrix, with a csv
    ### If clustering method is "unordered":  Use the method described in 
    ### Learning From Limited Data: "Towards Best Practice Techniques for Antimicrobial Resistance Prediction From Whole Genome Sequencing Data", 2021
    fold_assignments = generate_folds(
        logger,
        path_fold_assignments_csv,
        ordered_dataset_assemblies,
        specie_drug_dist_matrix, 
        clustering=clustering_method, 
        num_folds=num_folds
    )

    return fold_assignments, specie_drug_dist_df


if __name__ == "__main__":
    args = parse_args()
    # Example usage to pre-generate folds:
    # --path-dataroot ../amr_genomic_data/pseudomonas_aeruginosa --path-csv ./datasets --pathogen pseudomonas_aeruginosa --antibiotic cefepime

    # Set up logging (copied from run_baseline_models.py)
    now = datetime.now().strftime(f"%Y%m%d_%H%M%S")
    path_out = os.path.join(args.path_out, args.pathogen, "phylogeny")
    os.makedirs(path_out, exist_ok=True)
    file_handler = logging.FileHandler(
        os.path.join(path_out, f"log__{args.pathogen}__{args.antibiotic}__phyloFoldsLog_{args.clustering}.txt"))
    console_handler = logging.StreamHandler()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[file_handler, console_handler]
    )
    logger = logging.getLogger(__name__) # Ensures the messages sent to the logger have the function name
    logger.info(f"Arguments: {args}")

    # Generate the genomic folds based on the provided arguments
    folds, dist_table_df = get_genomic_folds(
        logger=logger,
        pathogen=args.pathogen,
        antibiotic=args.antibiotic,
        path_dataroot=args.path_dataroot,
        path_csv=os.path.join(args.path_csv, args.pathogen, "filtered"),
        path_out=path_out,
        num_folds=args.folds,
        dist_method=args.dist_method,
        clustering_method=args.clustering,
        force_regenerate=args.force_regenerate
    )

    # Print the stats for the folds
    fold_stats, fold_distance_matrix = get_folds_stats(folds, dist_table_df)
    
    # Print the statistics for each fold
    logger.info("Minimum, maximum, and average distances between assemblies in each fold:")
    for fold, stats in fold_stats.items():
        logger.info(f"Fold {fold}: Min: {stats['intra_min']:.6f}, Max: {stats['intra_max']:.6f}, Avg: {stats['intra_avg']:.6f}")
    logger.info("Minimum, maximum, and average distances to the other folds:")
    for fold, stats in fold_stats.items():
        logger.info(f"Fold {fold}: Min: {stats['inter_min']:.6f}, Max: {stats['inter_max']:.6f}, Avg: {stats['inter_avg']:.6f}")
    logger.info("\nPairwise minimum distances between folds:")
    logger.info(f"\n{fold_distance_matrix}")
