"""
Script for downloading genomic data from NCBI for AMR prediction. It assumes that the user has 
already downloaded the isolates and ASTs databases from NCBI. 

The script does the following:
1. Loads the isolates and ASTs databases from NCBI.
2. Filters the isolates database to keep only those isolates that are present in the ASTs database.
3. Merges the isolates and ASTs databases on the 'Isolate' column to get the corresponding assembly (containing genomic data) for each isolate.
4. Downloads the genomic data for the filtered isolates from NCBI using the datasets command line tool.

"""

import pandas as pd
import os
from tqdm import tqdm
import argparse
from loguru import logger
import glob


def parse_args():
    parser = argparse.ArgumentParser(description="Download genomes from NCBI")
    parser.add_argument(
        "--path-csvs",
        type=str,
        # default="ncbi_SR_phenotpye_asts_20250505.tsv",
        help="Path to the folder containing CSV files for each specie-drug combination",
    )
    parser.add_argument(
        "--path-out",
        type=str,
        default="/home/mila/n/naga-karthik.enamundram/scratch/dataset_ncbi_asts",
        help="Path to the output directory where the genomes will be downloaded",
    )
    
    return parser.parse_args()


def main():
    
    args = parse_args()
    # get the organism name from the path_csvs
    organism = args.path_csvs.split("/")[-2]

    # download the genomic data for each organism in separate folders
    path_output = os.path.join(args.path_out, organism)

    # create the output directory if it doesn't exist
    if not os.path.exists(path_output):
        os.makedirs(path_output, exist_ok=True)

    # add logging
    logger.add(os.path.join(path_output, "download_dataset.log"), rotation="10 MB")

    # get all the csv files in the path_csvs directory
    specie_drug_csvs = glob.glob(os.path.join(args.path_csvs, "*.csv"))
    logger.info(f"Number of specie-drug CSV files found: {len(specie_drug_csvs)}")
    
    mega_df = pd.DataFrame()
    for path_csv in specie_drug_csvs:
        # load the csv file
        df = pd.read_csv(path_csv)
        # append to the mega dataframe
        mega_df = pd.concat([mega_df, df], ignore_index=True)

    # # load isolates tsv file
    # df_isolates = pd.read_csv(args.path_isolates, sep="\t", header=0)
    # logger.info(f"Number of isolates: {len(df_isolates)}")
    
    # # remove rows where 'Assembly' is NaN (data can be only downloaded where assembly info is available)
    # df_isolates = df_isolates.dropna(subset=["Assembly"])
    # logger.info(f"Number of isolates after removing NaN assemblies: {len(df_isolates)}")

    # # # print unique organisms
    # # print(f"Unique organism groups: {len(df_isolates['#Organism group'].unique())}")


    # # load the ASTs tsv file
    # df_asts = pd.read_csv(args.path_asts, sep="\t", header=0)
    # logger.info(f"Number of ASTs: {len(df_asts)}")
    
    # # NOTE: the ASTs file and Isolates file come from different databases and the Isolates database has much more organisms.
    # # The common column in both dataframes is the 'Isolate' column and we're filtering for only those 'Isolate' that are present in the ASTs file.
    # # We will match the corresponding assembly to the ASTs dataframe for each isolate and that will be used to download the genomes.
    
    # # get the list of unique isolates     
    # unique_isolates_asts = df_asts['Isolate'].unique()
    # logger.info(f"Number of unique isolates in ASTs: {len(unique_isolates_asts)}")
    
    # # keep only the isolates that are in the ASTs file
    # df_isolates_common = df_isolates[df_isolates['Isolate'].isin(unique_isolates_asts)]
    # logger.info(f"Number of common isolates: {len(df_isolates_common)}")


    # # add the corresponding assembly to the ASTs dataframe for each isolate from the isolates dataframe
    # df_asts['Assembly'] = df_asts['Isolate'].map(df_isolates_common.set_index('Isolate')['Assembly'])

    # columns_to_keep = [
    #     '#BioSample', 'Organism group', 'Scientific name', 'Isolation type', 
    #     'Location', 'Isolation source', 'Isolate', 'Antibiotic',
    #     'Resistance phenotype', 'Measurement sign', 'MIC (mg/L)', 'Assembly'
    #     ]

    # # filter the ASTs dataframe to keep only the relevant columns
    # df_asts = df_asts[columns_to_keep]
    # logger.info(f"Length of the ASTs dataframe: {len(df_asts)}")
    # # NOTE: this length should match the number in the ASTs website for S/R phenotypes before 2025-05-04

    # # save the updated ASTs dataframe to a new tsv file
    # df_asts.to_csv(os.path.join(os.path.dirname(args.path_asts), '20250505_ncbi_asts_filtered_all_organisms.tsv'), sep='\t', index=False)

    # # get the unique organism groups
    # unique_organisms = df_asts['Organism group'].unique()
    # logger.info(f"Unique organism groups in ASTs: {unique_organisms}")
    # logger.info(f"Number of unique organism groups in ASTs: {len(unique_organisms)}")

    # # # NOTE: Not all organisms have the sufficent number of S/R phenotypes to be used for training.
    # # # Hence, we will filter the organisms to keep only those that have more than 1000 samples combining all specie x drug combinations
    # # # drop columns where counts are less than 1000
    # # df_counts = df_asts['Organism group'].value_counts()
    # # organism_gt_1000 = df_counts[df_counts > 1000].index
    # # # print(df_counts[df_counts > 1000].index)

    # # # filter the dataframe to keep only the relevant columns
    # # df_asts = df_asts[df_asts['Organism group'].isin(organism_gt_1000)]

    # # # save the updated ASTs dataframe to a new tsv file
    # # df_asts.to_csv(os.path.join(os.path.dirname(args.path_asts), '20250505_ncbi_asts_filtered_greater_1000.tsv'), sep='\t', index=False)
    # # logger.info(f"Length of the ASTs dataframe after filtering: {len(df_asts)}")

    # ---
    # DOWNLOAD THE ASSEMBLIES
    # ---

    # get the assembly names in a list
    # NOTE: the assembly are duplicated because for each organism, there are multiple drugs and hence multiple rows
    # hence, we need to get the unique assembly names
    assemblies = mega_df['Assembly'].unique()
    logger.info(f"Number of unique assemblies: {len(assemblies)}")

    # run a terminal command to download the assemblies
    # for more info: https://www.ncbi.nlm.nih.gov/datasets/docs/v2/how-tos/genomes/large-download/
    for assembly in tqdm(assemblies, desc="Downloading genomes from NCBI"):
        # 1. run the command to download the assembly
        os.system(f"datasets download genome accession {assembly} --dehydrated --include genome,seq-report --filename {assembly}.zip")
        # 2. unzip the file
        os.system(f"unzip {assembly}.zip -d {path_output}/{assembly}")
        # 3. rehydrate the file
        os.system(f"datasets rehydrate --directory {path_output}/{assembly}")
        # 4. remove the zip file
        os.system(f"rm {assembly}.zip")


if __name__ == "__main__":
    # run the main function
    main()