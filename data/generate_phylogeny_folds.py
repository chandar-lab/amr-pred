''' Example usage:
python generate_phylogeny_folds.py --path-dataroot ../amr_genomic_data/pseudomonas_aeruginosa --path-csv ./datasets --pathogen pseudomonas_aeruginosa --antibiotic cefepime'''

import os
import argparse
import glob
import logging

import pandas as pd
import numpy as np
from datetime import datetime

from genomic_dst_utils import mash_distance_txt, ani_distance_txt, dashing_distance_txt
from data_utils import load_data


def parse_args():
    parser = argparse.ArgumentParser(description="Create genomic distance folds for a dataset")

    # Alex's path-dataroot: ../amr-genomic-data/<pathogen>, assuming current root is scratch/amr
    parser.add_argument("--path-dataroot", type=str, help="Root path of the dataset genomic sequences for all species")
    # Alex's path-csv: ./datasets, assuming current root is scratch/amr
    parser.add_argument("--path-csv", type=str, default="", help="Path to the folder containing CSV files for each specie-drug combination")
    parser.add_argument("--pathogen", type=str, help="Name of the pathogen of interest")
    parser.add_argument("--antibiotic", type=str, help="Name of the antibiotic of interest")
    parser.add_argument("--folds", type=int, default=5, help="Number of folds to create")
    parser.add_argument("--dist_k", type=int, default=31, help="k-mer size for distance calculation (where applicable). Higher is more accurate but takes more time. Default is 31.")
    parser.add_argument("--dist_method", type=str, default='dashing', choices=['mash', 'ani', 'dashing'], help="Method to use for distance calculation: 'mash', 'ani', or 'dashing'. Default is 'dashing'.")
    parser.add_argument("--force_regenerate", action='store_true', help="Force regeneration of the assemblies list and distance matrix, even if they already exist")
    return parser.parse_args()

def get_matrix(assemblies_txt, path_root_fna, method='dashing', force_regenerate=False):
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
    
    # Generate the distance matrix using the specified method
    dist_matrix = None
    dist_matrix_file = os.path.join(path_root_fna, f"{method}_distance_matrix.csv")
    # Check if the distance matrix already exists. If not, generate it.
    if (not os.path.isfile(dist_matrix_file)) or force_regenerate:
        logging.info(f"Generating {method} distance matrix for {len(assemblies)} assemblies...")
        dist_matrix = dist_func(assemblies_txt)
        # Save the matrix to a csv file
        np.savetxt(dist_matrix_file, dist_matrix, delimiter=',', fmt='%.6f') # Use 6 decimal places, probably enough precision at our scale
        logging.info(f"Saved {method} distance matrix to {dist_matrix_file}")
    else:
        logging.info(f"{method} distance matrix already exists at {dist_matrix_file}, skipping regeneration.")
        # Load the distance matrix
        dist_matrix = np.loadtxt(dist_matrix_file, delimiter=',')
    
    return dist_matrix

def main():
    # Example run command:
    # --path-dataroot ../amr_genomic_data/pseudomonas_aeruginosa --path-csv ./datasets --pathogen pseudomonas_aeruginosa --antibiotic cefepime
    args = parse_args()

    # Set up logging (copied from run_baseline_models.py)
    now = datetime.now().strftime(f"%Y%m%d_%H%M%S")
    path_out = os.path.join("/home/mila/a/aselstya/scratch/amr/results", now.split('_')[0], now.split('_')[1])
    os.makedirs(path_out, exist_ok=True)
    file_handler = logging.FileHandler(os.path.join(path_out, f"log__{args.pathogen}__{args.antibiotic}__phylogeny_split.txt"))
    console_handler = logging.StreamHandler()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[file_handler, console_handler]
    )

    # Task 1: Generate files with list of assemblies for the pathogen
    path_root_fna = args.path_dataroot
    path_csv = args.path_csv

    dataset = f"{args.pathogen}/filtered/{args.pathogen}__{args.antibiotic}_SR_filt.csv"
    path_data_csv = os.path.join(path_csv, dataset)
    if not os.path.isfile(path_data_csv):
        raise FileNotFoundError(f"CSV file not found: {path_data_csv}. Please check the name of the pathogen and drug.")
    
    # Generate the file with paths to all assemblies for the pathogen, if needed
    assemblies_txt = os.path.join(path_root_fna, "all_assemblies.txt")
    if (not os.path.isfile(os.path.join(assemblies_txt))) or args.force_regenerate:
        logging.info(f"Generating list of assemblies for {args.pathogen}...")
        # Get all assemblies from the path_root_fna folder, and put them in a list
        subfolders = os.listdir(path_root_fna)
        assembly_files = [glob.glob(os.path.join(path_root_fna, folder, "ncbi_dataset", "data", folder, "*.fna"))[0] for folder in subfolders if os.path.isdir(os.path.join(path_root_fna, folder))]
 
        # Save the list of assemblies to a file in the path_root_fna folder
        with open(assemblies_txt, "w") as f:
            for assembly in assembly_files:
                f.write(f"{assembly}\n")
    else:
        logging.info(f"List of assemblies already exists at {assemblies_txt}, skipping regeneration.")

    # Make the assemblies list hold only the names of the assemblies, not the full paths
    with open(assemblies_txt, "r") as f:
        # Extract the name of the parent folder from each assembly path
        all_pathogen_assemblies = [line.strip().split('/')[-2] for line in f.readlines() if line.strip()]




    # Task 2: Generate and save distance matrix for the assemblies using the desired distance metric
    dashing_matrix = get_matrix(assemblies_txt, path_root_fna, method=args.dist_method, force_regenerate=args.force_regenerate)
    logging.info(f"Distance matrix shape: {dashing_matrix.shape}")

    # Task 3: determine which assemblies are relevant for the antibiotic
    # Load the data using the load_data function
    dataset_list = load_data(path_data_csv, path_root_fna)

    # Create a set of all assemblies from the dataset_list
    dataset_assemblies = [entry['assembly'] for entry in dataset_list]

    # Create a list of indices for the assemblies that are also in the dataset_list
    relevant_indices = [i for i, assembly in enumerate(all_pathogen_assemblies) if assembly in dataset_assemblies]
    print(f"Number of relevant assemblies for {args.pathogen} and {args.antibiotic}: {len(relevant_indices)}")

    # Filter the matrix to only include the relevant assemblies
    specie_drug_dist_matrix = dashing_matrix[np.ix_(relevant_indices, relevant_indices)]
    logging.info(f"Filtered distance matrix shape: {specie_drug_dist_matrix.shape}")


    # Task 4: Create folds based on the distance matrix, with a csv
    # which maximizes genomic distance between the folds (hard case),
    # the file will have a row for each assembly, and then the associated fold number.
    # Print the number of susceptible, intermediate, and resistant isolates in each fold.

    ### Use the method described in Learning From Limited Data: "Towards Best Practice Techniques for Antimicrobial Resistance Prediction From Whole Genome Sequencing Data", 2021

    logging.info(f"Creating {args.folds} folds based on the distance matrix...")
    # Begin by selecting seeds, equal to the number of folds, which act as the centroids of the folds.
    num_folds = args.folds
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
    
    logging.info("Assigning assemblies to folds based on the selected seeds...")
    # Create a list to hold the fold assignments
    fold_assignments = {}
    assigned_indices = set()
    fold_sizes = [0] * num_folds  # Used to logging, to ensure equal distribution of assemblies.
    # Take turns assigning assemblies to folds, starting with the first seed
    while len(assigned_indices) < len(dataset_assemblies):
        for seed in fold_seeds:
            # Find the assembly with the maximum distance to the current fold assignments
            # Calculate the distances from the seed to all assemblies not yet assigned to a fold
            distances = specie_drug_dist_matrix[seed, :]
            unassigned_indices = [i for i in list(range(len(dataset_assemblies))) if i not in assigned_indices]
            # Find the assembly with the minimum distance to the current seed
            min_distance_index = np.argmin(distances[unassigned_indices])
            # Assign this assembly to the current fold
            fold_assignments[dataset_assemblies[unassigned_indices[min_distance_index]]] = len(fold_assignments) % num_folds
            assigned_indices.add(unassigned_indices[min_distance_index])
            fold_sizes[len(fold_assignments) % num_folds] += 1


    print("Fold assignments:")
    for assembly, fold in fold_assignments.items():
        print(f"{assembly}: Fold {fold + 1}")

    # Print the size of each fold 
    logging.info(f"Fold sizes: {fold_sizes}")

    # Save the fold assignments to a CSV file
    fold_assignments_df = pd.DataFrame(list(fold_assignments.items()), columns=['assembly', 'fold'])
    # Save to csv in the path_root_fna folder
    fold_assignments_csv = os.path.join(path_root_fna, f"{args.pathogen}__{args.antibiotic}__phylogeny_folds.csv")
    fold_assignments_df.to_csv(fold_assignments_csv, index=False)
    logging.info(f"Fold assignments saved to {fold_assignments_csv}")



if __name__ == "__main__":
    main()