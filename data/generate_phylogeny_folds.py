from genomic_dst_utils import mash_distance_txt, ani_distance_txt, dashing_distance_txt
import os
import argparse

import pandas as pd
import glob
import logging

LABEL_MAPPING = {
    # NOTE: convention is to use 0 for susceptible and 1 for resistant
    "resistant": 1,
    "susceptible": 0
}

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
    parser.add_argument("--force_regenerate", action='store_true', help="Force regeneration of the assemblies list and distance matrix, even if they already exist")
    return parser.parse_args()

def load_data(path_data_csv, path_root_fna):

    df_data = pd.read_csv(path_data_csv)
    specie_drug_assemblies = df_data['Assembly'].tolist()
    # get the corresponding phenotypic labels for each assembly
    phenotypes = df_data['Phenotype_revised'].tolist()
    # print(phenotypes)

    dataset_list = []
    for assembly, phenotype in zip(specie_drug_assemblies, phenotypes):
        temp_dict = {}
        # if assembly not in dataset_dict:
        fname = glob.glob(os.path.join(path_root_fna, assembly, "ncbi_dataset", "data", assembly, "*.fna"))
        if len(fname) == 0:
            logging.info(f"WARNING: No .fna file found for assembly {assembly}. Skipping this entry.")
            continue
        
        temp_dict["seq"] = fname[0] 
        temp_dict["label"] = LABEL_MAPPING[phenotype]
        temp_dict["assembly"] = assembly
        dataset_list.append(temp_dict)
    
    print("Number of samples in the dataset: ", len(dataset_list))
    
    return dataset_list  # For testing, limit to 100 samples

def main():
    # Example run command:
    # --path-dataroot ../amr_genomic_data/pseudomonas_aeruginosa --path-csv ./datasets --pathogen pseudomonas_aeruginosa --antibiotic cefepime
    args = parse_args()

    # Task 1: Generate files with list of assemblies for the pathogen
    path_root_fna = args.path_dataroot
    path_csv = args.path_csv

    dataset = f"{args.pathogen}/filtered/{args.pathogen}__{args.antibiotic}_SR_filt.csv"
    path_data_csv = os.path.join(path_csv, dataset)
    if not os.path.isfile(path_data_csv):
        raise FileNotFoundError(f"CSV file not found: {path_data_csv}. Please check the name of the pathogen and drug.")
    
    assemblies_txt = os.path.join(path_root_fna, "all_assemblies.txt")
    if (not os.path.isfile(os.path.join(assemblies_txt))) or args.force_regenerate:
        # Get all assemblies from the path_root_fna folder, and put them in a list
        subfolders = os.listdir(path_root_fna)
        assemblies = [glob.glob(os.path.join(path_root_fna, folder, "ncbi_dataset", "data", folder, "*.fna"))[0] for folder in subfolders if os.path.isdir(os.path.join(path_root_fna, folder))]
        
        assemblies = assemblies
        # Save the list of assemblies to a file in the path_root_fna folder
        with open(assemblies_txt, "w") as f:
            for assembly in assemblies:
                f.write(f"{assembly}\n")
            


    # Task 2: Generate distance matrix for the assemblies using the desired distance metric
    # mash_matrix = mash_distance(assemblies_txt, assemblies_txt)

    # Subtask 2.1: save the distance matrix to a file

    # Task 3: determine which assemblies are relevant for the antibiotic

    # Task 4: Create folds based on the distance matrix, with two output csvs:
    # 1. maximum genomic distance between the folds (hard case),
    # 2. identical genomic distance distribution within each fold (easier case)
    # Each file will a row for each assembly, and then the associated fold number.
    # Print the number of susceptible, intermediate, and resistant isolates in each fold.

if __name__ == "__main__":
    main()