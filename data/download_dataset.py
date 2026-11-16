"""
Script for downloading genomic data from NCBI for AMR prediction. It assumes that the user has 
already downloaded the isolates and ASTs databases from NCBI. 

The script does the following:
1. Loads the isolates and ASTs databases from NCBI.
2. Filters the isolates database to keep only those isolates that are present in the ASTs database.
3. Merges the isolates and ASTs databases on the 'Isolate' column to get the corresponding assembly (containing genomic data) for each isolate.
4. Downloads the genomic data for the filtered isolates from NCBI using the datasets command line tool.

"""

import hydra
import pandas as pd
import os
from loguru import logger
import glob
from omegaconf import DictConfig
from data_utils import download_assemblies, register_config_resolvers


register_config_resolvers()


@hydra.main(config_path="../config", config_name="main", version_base=None)
def main(cfg: DictConfig):
    # get the organism name from the hydra config file
    organism = cfg.dataset.pathogen

    # download the genomic data for each organism in separate folders
    path_output = os.path.join(cfg.path_assemblies_root, organism)

    # create the output directory if it doesn't exist
    if not os.path.exists(path_output):
        os.makedirs(path_output, exist_ok=True)

    # add logging
    logger.add(os.path.join(path_output, "download_dataset.log"), rotation="10 MB")

    # get all the csv files in the path_csvs directory
    specie_drug_csvs = glob.glob(os.path.join(cfg.path_datasets_reinterpreted, organism, "filtered", "*.csv"))
    logger.info(f"Number of specie-drug CSV files found: {len(specie_drug_csvs)}")
    
    mega_df = pd.DataFrame()
    for path_csv in specie_drug_csvs:
        # load the csv file
        df = pd.read_csv(path_csv)
        # append to the mega dataframe
        mega_df = pd.concat([mega_df, df], ignore_index=True)

    # ---
    # DOWNLOAD THE ASSEMBLIES
    # ---

    # get the assembly names in a list
    # NOTE: the assembly are duplicated because for each organism, there are multiple drugs and hence multiple rows
    # hence, we need to get the unique assembly names
    assemblies = mega_df['Assembly'].unique()
    logger.info(f"Number of unique assemblies: {len(assemblies)}")

    download_assemblies(list(assemblies), path_output, logger)


if __name__ == "__main__":
    # run the main function
    main()