''' Example usage:
python generate_phylogeny_folds.py --path-dataroot ../amr_genomic_data/pseudomonas_aeruginosa --path-csv ./datasets --pathogen pseudomonas_aeruginosa --antibiotic cefepime'''

import os
import argparse
import logging
import time

import pandas as pd
import numpy as np
from datetime import datetime

from genomic_dst_utils import mash_distance_txt, ani_distance_txt, dashing_distance_txt, get_assemblies_txt
from data_utils import load_data


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
    parser.add_argument("--dist_method", type=str, default='dashing', choices=['mash', 'ani', 'dashing'], help="Method to use for distance calculation: 'mash', 'ani', or 'dashing'. Default is 'dashing'.")
    parser.add_argument("--force_regenerate", action='store_true', help="Force regeneration of the assemblies list and distance matrix, even if they already exist")
    return parser.parse_args()


def get_matrix(assemblies_txt, path_out, method='dashing', force_regenerate=False):
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


def get_genomic_folds(pathogen, antibiotic, path_dataroot, path_csv, path_out, num_folds=5, dist_method='dashing', force_regenerate=False):
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
    fold_assignments_csv = os.path.join(path_out, f"{pathogen}__{antibiotic}__{dist_method}__phylogeny_folds.csv")
    dist_table_csv = os.path.join(path_out, f"{pathogen}__{antibiotic}__{dist_method}__distMat.csv")
    logger = logging.getLogger(__name__) # Ensures the messages sent to the logger have the function name

    # Return the fold assignments as a dict if the file already exists
    if os.path.isfile(fold_assignments_csv) and os.path.isfile(dist_table_csv) and not force_regenerate:
        logger.info(f"Fold assignments already exist at {fold_assignments_csv}, loading from file.")
        fold_assignments_df = pd.read_csv(fold_assignments_csv)
        fold_assignments = dict(zip(fold_assignments_df['assembly'], fold_assignments_df['fold']))

        logger.info(f"Distance table already exists at {dist_table_csv}, loading from file.")
        dist_table = pd.read_csv(dist_table_csv, index_col=0)
        return fold_assignments, dist_table

    # Task 1: Generate files with list of assemblies for the pathogen
    path_root_fna = path_dataroot
    
    # Generate the file with paths to all assemblies for the pathogen, if needed
    assemblies_txt = get_assemblies_txt(path_root_fna, force_regenerate=force_regenerate)

    # Make the assemblies list hold only the names of the assemblies, not the full paths
    with open(assemblies_txt, "r") as f:
        # Extract the name of the parent folder from each assembly path
        all_pathogen_assemblies = [line.strip().split('/')[-2] for line in f.readlines() if line.strip()]


    # Task 2: Generate and save distance matrix for the assemblies using the desired distance metric
    full_pathogen_matrix_csv = os.path.join(path_out, f"{pathogen}__{dist_method}__distMatAll.csv")
    # Check if the distance matrix already exists. If not, generate it.
    if (not os.path.isfile(full_pathogen_matrix_csv)) or force_regenerate:
        full_pathogen_matrix = get_matrix(assemblies_txt, path_root_fna, method=dist_method, force_regenerate=force_regenerate)
        logger.info(f"Full pathogen distance matrix shape: {full_pathogen_matrix.shape}")
        # save the full distance matrix to a csv file
        pd.DataFrame(full_pathogen_matrix, index=all_pathogen_assemblies, columns=all_pathogen_assemblies).to_csv(full_pathogen_matrix_csv)
        logger.info(f"Saved full distance matrix to {full_pathogen_matrix_csv}")
    else:
        logging.info(f"{dist_method} distance matrix already exists at {full_pathogen_matrix_csv}, skipping regeneration.")
        # Load the distance matrix
        dist_matrix = pd.read_csv(full_pathogen_matrix_csv, index_col=0)
        full_pathogen_matrix = dist_matrix.values


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
    # Save the specie-drug distance table to a CSV file
    specie_drug_dist_matrix_csv = os.path.join(path_out, f"{pathogen}__{antibiotic}__{dist_method}__distMat.csv")
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
    
    # Now additional seeds can be selected by iteratively finding the assembly with the largest minimum distance to an existing seed
    while len(fold_seeds) < num_folds:
        # Calculate the minimum distance to an existing seed for each assembly
        min_distances = np.min(specie_drug_dist_matrix[:, fold_seeds], axis=1)
        # Find the assembly with the maximum minimum distance
        new_seed = np.argmax(min_distances)
        fold_seeds.append(new_seed)
      
    logger.info("Assigning assemblies to folds based on the selected seeds...")
    # Create a list to hold the fold assignments
    fold_assignments = {}
    assigned_indices = set()
    fold_sizes = [1] * num_folds  # Used to logging, to ensure equal distribution of assemblies.

    # Begin by assigning the seeds to their respective folds
    for i, seed in enumerate(fold_seeds):
        fold_assignments[ordered_dataset_assemblies[seed]] = i
        assigned_indices.add(seed)

    # Take turns assigning assemblies to folds, starting with the first seed
    curr_seed = 0
    while len(assigned_indices) < len(ordered_dataset_assemblies):
        # Find the assembly with the maximum distance to the current fold assignments
        # Calculate the distances from the seed to all assemblies not yet assigned to a fold
        distances = specie_drug_dist_matrix[seed, :]
        unassigned_indices = [i for i in list(range(len(ordered_dataset_assemblies))) if i not in assigned_indices]
        # Find the assembly with the minimum distance to the current seed
        min_distance_index = np.argmin(distances[unassigned_indices])
        # Assign this assembly to the current fold
        fold_assignments[ordered_dataset_assemblies[unassigned_indices[min_distance_index]]] = len(fold_assignments) % num_folds
        assigned_indices.add(unassigned_indices[min_distance_index])
        fold_sizes[len(fold_assignments) % num_folds] += 1
        # Move to the next seed
        curr_seed = (curr_seed + 1) % num_folds

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

def folds_stats(folds, dist_table):
    """
    Print the minimum, maximum, and average distance between assemblies in each fold.
    parameters:
    - dist_table: pd.DataFrame, distance table with assemblies as index and columns
    - folds: dict, mapping of assembly names to fold numbers
    """
    fold_stats = {}
    for fold in set(folds.values()):
        # Get the assemblies in the current fold
        fold_assemblies = [assembly for assembly, f in folds.items() if f == fold]
        # Get all the assemblies in the other folds
        other_assemblies = [assembly for assembly, f in folds.items() if f != fold]
        # Get the distance matrix for the current fold
        fold_distances = dist_table.loc[fold_assemblies, fold_assemblies].values
        # Calculate the minimum, maximum, and average distance
        min_distance = np.min(fold_distances)
        max_distance = np.max(fold_distances)
        avg_distance = np.mean(fold_distances)
        fold_stats[fold] = {
            'intra_min': min_distance,
            'intra_max': max_distance,
            'intra_avg': avg_distance
        }
        # Calculate the minimum, maximum and average distance to the other folds
        other_distances = dist_table.loc[fold_assemblies, other_assemblies].values
        if other_distances.size > 0:  # Check if there are any distances
            min_distance_other = np.min(other_distances)
            max_distance_other = np.max(other_distances)
            avg_distance_other = np.mean(other_distances)
            fold_stats[fold].update({
                'inter_min': min_distance_other,
                'inter_max': max_distance_other,
                'inter_avg': avg_distance_other
            })
        else:
            fold_stats[fold].update({
                'inter_min': None,
                'inter_max': None,
                'inter_avg': None
            })

    # Create a matrix with minimum distances between each pair of folds
    fold_distance_matrix = pd.DataFrame(index=fold_stats.keys(), columns=fold_stats.keys())
    for fold1 in fold_stats.keys():
        for fold2 in fold_stats.keys():
            if fold1 == fold2:
                fold_distance_matrix.loc[fold1, fold2] = 0.0
            else:
                # Get the assemblies in the current folds
                fold1_assemblies = [assembly for assembly, f in folds.items() if f == fold1]
                fold2_assemblies = [assembly for assembly, f in folds.items() if f == fold2]
                # Get the distance matrix for the current folds
                distances = dist_table.loc[fold1_assemblies, fold2_assemblies].values
                if distances.size > 0:
                    # Calculate the minimum distance between the two folds
                    min_distance = np.min(distances)
                    fold_distance_matrix.loc[fold1, fold2] = min_distance
                else:
                    fold_distance_matrix.loc[fold1, fold2] = None


    # Print the statistics for each fold
    print("Minimum, maximum, and average distances between assemblies in each fold:")
    for fold, stats in fold_stats.items():
        print(f"Fold {fold}: Min: {stats['intra_min']:.6f}, Max: {stats['intra_max']:.6f}, Avg: {stats['intra_avg']:.6f}")
    print("Minimum, maximum, and average distances to the other folds:")
    for fold, stats in fold_stats.items():
        print(f"Fold {fold}: Min: {stats['inter_min']:.6f}, Max: {stats['inter_max']:.6f}, Avg: {stats['inter_avg']:.6f}")
    print("\nPairwise minimum distances between folds:")
    print(fold_distance_matrix)



if __name__ == "__main__":
    args = parse_args()
    # Example usage to pre-generate folds:
    # --path-dataroot ../amr_genomic_data/pseudomonas_aeruginosa --path-csv ./datasets --pathogen pseudomonas_aeruginosa --antibiotic cefepime

    # Set up logging (copied from run_baseline_models.py)
    now = datetime.now().strftime(f"%Y%m%d_%H%M%S")
    path_out = os.path.join(args.path_out, args.pathogen, "phylogeny")
    os.makedirs(path_out, exist_ok=True)
    file_handler = logging.FileHandler(os.path.join(path_out, f"log__{args.pathogen}__{args.antibiotic}__phylogeny_splits.txt"))
    console_handler = logging.StreamHandler()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[file_handler, console_handler]
    )

    # Generate the genomic folds based on the provided arguments
    folds, dist_table_df = get_genomic_folds(args.pathogen, args.antibiotic, args.path_dataroot, os.path.join(args.path_csv, args.pathogen, "filtered"), args.folds, args.dist_method, args.force_regenerate)

    # Print the stats for the folds
    folds_stats(folds, dist_table_df)