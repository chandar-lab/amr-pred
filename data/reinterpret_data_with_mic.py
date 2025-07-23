"""
Script for preprocessing the NCBI ASTs dataset. The script does the following:
1. Loads the NCBI ASTs dataset containing all species, phenotypic AST labels, and assemblies.
    NOTE: the csv file is downloaded from https://www.ncbi.nlm.nih.gov/pathogens/asts browser by 
    selecting the "Organism group", checking "Susceptible", "Resistant", and "Not defined" phenotypes,
    1.1 Filters the dataset with MIC values from CLSI breakpoints
2. (optional) Generates barplots showing genome counts per species and antibiotic.
3. Store the individual species-antibiotic dataset combination in a separate CSV file for training.
"""

import pandas as pd
import numpy as np
import os
import logging
from tqdm import tqdm
import argparse
from data_utils import (plot_antibiotic_counts, plot_antibiotic_counts_per_organism, 
    get_drugs_for_organism_clsi, get_mic_breakpoints, clean_breakpoints_file)



def get_parser():
    parser = argparse.ArgumentParser(description="Filter the NCBI resistance phenotypes ASTs dataset with MIC values from CLSI breakpoints")
    parser.add_argument(
        "--path-clsi-bps",
        type=str,
        default="breakpoints_per_species_20250527.csv",
        help="Path to the CLSI breakpoints file",
    )
    parser.add_argument(
        "--path-data",
        type=str,
        default="acinetobacter_baumannii_SRND_20250527.csv",
        help="Path to the single-specie dataset downloaded from NCBI ASTs database",
    )

    parser.add_argument(
        "--path-isolates",
        type=str,
        default="isolates.tsv",
        help="Path to the isolates file downloaded from https://www.ncbi.nlm.nih.gov/pathogens/isolates",   
    )
    parser.add_argument(
        "--path-out",
        type=str,
        default="datasets",
        help="Path to the output directory where the individual pathogen-antiobiotic datasets will be saved",
    )
    parser.add_argument(
        "--save-plots",
        action="store_true",
        help="Save the plots showing dataset statistics (genome counts, antibiotic counts, etc.)",
    )

    return parser


def is_resistant(row, bp_r):
    return row['MIC_revised'] >= bp_r


def is_susceptible(row, bp_s):
    return row['MIC_revised'] <= bp_s


def clean_ecoli_data(df):
    """
    Clean the E. coli data by removing Shigella samples and keeping only E. coli samples.
    """
    # lowercase the 'Scientific name' column for case-insensitive comparison
    df['Scientific name'] = df['Scientific name'].str.lower()

    # filter out Shigella samples
    df = df[~df['Scientific name'].str.contains('shigella', na=False)]

    return df


def main():
    parser = get_parser()
    args = parser.parse_args()

    # create the output directory if it doesn't exist
    if not os.path.exists(args.path_out):
        os.makedirs(args.path_out, exist_ok=True)

    # get the organism name from the path_data
    organism = os.path.basename(os.path.dirname(args.path_data))

    # logging setup
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)  # Set level for the logger itself

    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    # Create stream handler (console)
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(formatter)
    # Ensure log directory exists
    log_dir = os.path.join(args.path_out, organism, "filtered")
    os.makedirs(log_dir, exist_ok=True)

    # Create file handler
    log_file = os.path.join(log_dir, "log.txt")    
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)

    # Add both handlers to the logger
    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)

    logger.info(f"Species: {organism.upper()}")

    # load the data
    df = pd.read_csv(args.path_data, low_memory=False)

    if organism == "escherichia_coli":
        orig_df_len = len(df)
        logger.info(f"Length of the original E. coli dataset: {orig_df_len} samples")
        # we need to only keep E.coli samples and discard the Shigella ones
        df = clean_ecoli_data(df)
        # # save the cleaned dataframe
        # df.to_csv(os.path.join(args.path_out, organism, 'filtered', f"{organism}_cleaned.csv"), index=False)
        logger.info(f"Cleaned E. coli data: {len(df)} samples")
        logger.info(f"Number of Shigella samples removed: {orig_df_len - len(df)}")
        # NOTE: difference between the original and cleaned E. coli dataset should match the number in the AST database
        # browser when E.coli and Shigella --> scientific name; shows the samples for E.coli and Shigella separately.

        # rename organism group to 'escherichia_coli'
        df['Organism group'] = 'escherichia_coli'

    phenotypic_unique_isolates = df['Isolate'].unique()
    # logger.info(f"Number of isolates in phenotypic data: {len(df['Isolate'].unique())}")

    # ================================
    # Store the assembly identifier in the dataframe 
    # ================================
    
    # add the assembly column to the dataframe
    df_isolates = pd.read_csv(args.path_isolates, sep="\t", low_memory=False)

    # keep the data for the organism
    df_isolates['#Organism group'] = df_isolates['#Organism group'].str.replace(' ', '_', regex=False)
    df_isolates['#Organism group'] = df_isolates['#Organism group'].str.lower()
    if organism == "escherichia_coli":
        df_isolates = df_isolates[df_isolates['#Organism group'] == 'e.coli_and_shigella']
        df_isolates['#Organism group'] = 'escherichia_coli'
    else:
        df_isolates = df_isolates[df_isolates['#Organism group'] == organism]

    # drop the rows where 'Assembly' is NaN
    df_isolates = df_isolates.dropna(subset=['Assembly'])
    # logger.info(f"Number of isolates after dropping empty assemblies: {len(df_isolates['Isolate'].unique())}")

    # keep only the isolates that are in the ASTs file
    df_isolates_common = df_isolates[df_isolates['Isolate'].isin(phenotypic_unique_isolates)]
    # keep only the isolates and assembly columns
    df_isolates_common = df_isolates_common[['Isolate', 'Assembly']]
    # logger.info(f"Number of common isolates: {len(df_isolates_common)}")

    # merge the isolates dataframe with the main dataframe on 'Isolate' column
    df = df.merge(df_isolates_common, on='Isolate', how='inner')
    # logger.info(f"Number of samples after merging with isolates: {len(df)}")

    # make sure the 'Antibiotic' column is lowercase
    df['Antibiotic'] = df['Antibiotic'].str.lower()
    # replace space or '-' in drug names with underscore
    df['Antibiotic'] = df['Antibiotic'].str.replace(' ', '_', regex=False)
    df['Antibiotic'] = df['Antibiotic'].str.replace('-', '_', regex=False)

    # same for Organism group
    df['Organism group'] = df['Organism group'].str.lower()
    df['Organism group'] = df['Organism group'].str.replace(' ', '_', regex=False)

    # print number of resistant and susceptible samples (RAW DATASET)
    logger.info(f"Number of samples (RAW DATA): {len(df)}")
    logger.info(f"\tNumber of resistant samples: {len(df[df['Resistance phenotype'] == 'resistant'])}")
    logger.info(f"\tNumber of susceptible samples: {len(df[df['Resistance phenotype'] == 'susceptible'])}")
    logger.info(f"\tNumber of unknown samples: {len(df[df['Resistance phenotype'] == 'not defined'])}")


    # ================================
    # Filter the dataset with CLSI breakpoints
    # ================================

    # get cleaned breakpoints df
    df_bps = clean_breakpoints_file(args.path_clsi_bps)

    # filter out to keep only the drugs that are in the breakpoints file
    drugs_in_bps = get_drugs_for_organism_clsi(df_bps, organism)
    df = df[df['Antibiotic'].isin(drugs_in_bps)]
    # df.to_csv(os.path.join(args.path_out, organism, 'temp_data.csv'), index=True)

    # add a check to ensure that list of drugs match
    if not set(df['Antibiotic'].unique()).issubset(set(drugs_in_bps)):
        raise ValueError("Some drugs in the dataset are not found in the breakpoints file.")
    
    # remove nan values from the 'MIC (mg/L)' column (i.e. where no MIC value was found in the database)
    logger.info(f"Removing samples with NaN MIC values")
    df = df.dropna(subset=['MIC (mg/L)'])

    # print number of resistant and susceptible samples (after 1st level of filtering -- CLSI drugs)
    logger.info(f"Number of samples (AFTER FILTERING FOR CLSI DRUGS): {len(df)}")
    logger.info(f"\tNumber of resistant samples: {len(df[df['Resistance phenotype'] == 'resistant'])}")
    logger.info(f"\tNumber of susceptible samples: {len(df[df['Resistance phenotype'] == 'susceptible'])}")
    logger.info(f"\tNumber of unknown samples: {len(df[df['Resistance phenotype'] == 'not defined'])}")

    # create a new column 'MIC_revised' that will contain the MIC values
    # that are revised based on the CLSI breakpoints
    df['MIC_revised'] = df['MIC (mg/L)'].copy()
    # iterate through the dataframe and fill the 'MIC_revised' column
    for i, (index, row) in enumerate(df.iterrows()):
        # get the drug and organism for the current row
        drug = row['Antibiotic']
        organism = row['Organism group']

        # check if the sample is resistant or susceptible
        # NOTE: we do the reinterpretation for samples that are strictly > or < considering the double dilutions
        # for samples with >= or == or <= we keep the original MIC value as they are typically at the breakpoint
        if row['Measurement sign'] == '>':
            df.at[index, 'MIC_revised'] = row['MIC (mg/L)'] * 2
            # NOTE: why * 2? because phenotypic tests are tested for double dilutions. 
            # e.g. if the test says MIC >32 then, it means that the measured MIC value is >= 64, 
            # so with a breakpoint of 64 the category is resistant.
            # But, if the measured MIC value is >16, then the test is inconclusive.

        elif row['Measurement sign'] in ['<']:
            df.at[index, 'MIC_revised'] = row['MIC (mg/L)'] / 2
            

    drugs = sorted(df['Antibiotic'].unique().tolist())

    organism_groups = [organism]
    dataset_counts = {}
    for organism in tqdm(organism_groups, desc="Organism groups", unit="group"):
        
        drug_count = {}
        num_resistant_before, num_susceptible_before = 0, 0
        num_resistant_after, num_susceptible_after = 0, 0
        for drug in drugs:
            # logger.info(f"======================")
            # logger.info(f"Drug {drug}")
            # logger.info(f"======================")

            # filter out for the current drug
            df_new = df[df['Antibiotic'] == drug].copy()

            # initialize the 'Phenotype_revised' column with NaN values
            df_new['Phenotype_revised'] = np.nan
            
            # get breakpoints for the current drug and organism
            bp_s, bp_r = get_mic_breakpoints(df_bps, organism, drug)
            if bp_s is None or bp_r is None:
                logger.info(f"No breakpoints found for {organism} and {drug}. Skipping...")
                continue
            
            # iterate through all the samples and classify them based on revised MIC values
            indices_to_drop = []
            for i, (index, row) in enumerate(df_new.iterrows()):
                if is_resistant(row, bp_r):
                    df_new.at[index, 'Phenotype_revised'] = 'resistant'
                elif is_susceptible(row, bp_s):
                    df_new.at[index, 'Phenotype_revised'] = 'susceptible'
                else:
                    # logger.info(f"Raw data show: '{row['Resistance phenotype']}' with {row['Measurement sign']} {row['MIC (mg/L)']} MIC (mg/L), but BP for {drug} is S: {bp_s}, R: {bp_r}. Dropping...")
                    indices_to_drop.append(index)
            
            # Drop samples outside the loop
            if indices_to_drop:
                df_new.drop(indices_to_drop, inplace=True)
                # logger.info(f"Number of samples dropped for {drug} for CLSI breakpoint criteria mismatch: {len(indices_to_drop)}")
            
            # check if there any nan values in the 'Phenotype_revised' column
            if df_new['Phenotype_revised'].isna().any():
                logger.info(f"Warning: There are {df_new['Phenotype_revised'].isna().sum()} samples with NaN phenotype for {drug}.")

            num_resistant = len(df_new[df_new['Phenotype_revised'] == 'resistant'])
            num_susceptible = len(df_new[df_new['Phenotype_revised'] == 'susceptible'])
            
            # logger.info(f"Number of resistant samples for {drug}: {num_resistant}")
            # logger.info(f"Number of susceptible samples for {drug}: {num_susceptible}")
            num_resistant_after += num_resistant
            num_susceptible_after += num_susceptible
            num_resistant_before += len(df_new[df_new['Resistance phenotype'] == 'resistant'])
            num_susceptible_before += len(df_new[df_new['Resistance phenotype'] == 'susceptible'])

            # save the filtered dataframe
            output_path = os.path.join(args.path_out, organism, 'filtered', f"{organism}__{drug}_SR_filt.csv")
            if not os.path.exists(output_path):
                # create the directory for saving the filtered data if it doesn't exist
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # save the filtered dataframe to a CSV file
            df_new.to_csv(output_path, index=False)
            
            # # keep only the relevant columns
            # cols_to_keep = ['#BioSample', 'Organism group', 'Isolate', 'Assembly', 'Antibiotic',
            #                     'Measurement sign', 'MIC (mg/L)', 'Resistance phenotype', 'MIC_revised', 'Phenotype_revised']
            # df_to_compare = df_new[cols_to_keep]
            
            # # get the samples where 'resistance phenotype' is not the same as 'Phenotype_revised'
            # df_compare = df_to_compare[df_to_compare['Resistance phenotype'] != df_to_compare['Phenotype_revised']]
            # logger.info(f"Samples with different resistance phenotype: {len(df_compare)}")

            # create the dictionary
            drug_count[drug] = {
                "resistant": num_resistant,
                "susceptible": num_susceptible
            }
    
            # # get the number of antibiotics per organism (i.e. total number of "datasets")
            # dataset_counts[organism] = dataset_counts.get(organism, 0) + 1

        # compute fractional change in resistant and susceptible samples
        logger.info(f"")
        logger.info(f"REINTERPRETATION WITH CLSI BREAKPOINTS: ")
        change_r = num_resistant_after - num_resistant_before
        change_s = num_susceptible_after - num_susceptible_before
        
        if change_r > 0 and change_s > 0:
            fractional_change_r = change_r / num_resistant_before * 100
            fractional_change_s = change_s / num_susceptible_before * 100
            logger.info(f"\tResistant samples:\tRaw data (before): {num_resistant_before}\t Revised (after): {num_resistant_after}\tChange: {change_r} ({fractional_change_r:.2f}% increase)")
            logger.info(f"\tSusceptible samples:\tRaw data (before): {num_susceptible_before}\t Revised (after): {num_susceptible_after}\tChange: {change_s} ({fractional_change_s:.2f}% increase)")
        elif change_r < 0 and change_s < 0:
            fractional_change_r = abs(change_r) / num_resistant_before * 100
            fractional_change_s = abs(change_s) / num_susceptible_before * 100
            logger.info(f"\tResistant samples:\tRaw data (before): {num_resistant_before}\t Revised (after): {num_resistant_after}\tChange: {change_r} ({fractional_change_r:.2f}% decrease)")
            logger.info(f"\tSusceptible samples:\tRaw data (before): {num_susceptible_before}\t Revised (after): {num_susceptible_after}\tChange: {change_s} ({fractional_change_s:.2f}% decrease)")
        elif change_r > 0 and change_s < 0:
            fractional_change_r = change_r / num_resistant_before * 100
            fractional_change_s = abs(change_s) / num_susceptible_before * 100
            logger.info(f"\tResistant samples:\tRaw data (before): {num_resistant_before}\t Revised (after): {num_resistant_after}\tChange: {change_r} ({fractional_change_r:.2f}% increase)")
            logger.info(f"\tSusceptible samples:\tRaw data (before): {num_susceptible_before}\t Revised (after): {num_susceptible_after}\tChange: {change_s} ({fractional_change_s:.2f}% decrease)")
        elif change_r < 0 and change_s > 0:
            fractional_change_r = abs(change_r) / num_resistant_before * 100
            fractional_change_s = change_s / num_susceptible_before * 100
            logger.info(f"\tResistant samples:\tRaw data (before): {num_resistant_before}\t Revised (after): {num_resistant_after}\tChange: {change_r} ({fractional_change_r:.2f}% decrease)")
            logger.info(f"\tSusceptible samples:\tRaw data (before): {num_susceptible_before}\t Revised (after): {num_susceptible_after}\tChange: {change_s} ({fractional_change_s:.2f}% increase)")
        
        
        if len(drug_count) == 0:
            logger.info(f"No drugs with S/R genomes found for {organism}")
        else:
            if args.save_plots:
                save_path = os.path.join(args.path_out, organism, "filtered", "plots_single_species")
                # create the directory for saving the plots if it doesn't exist
                if not os.path.exists(save_path):
                    os.makedirs(save_path, exist_ok=True)
                # plot the antibiotic counts
                plot_antibiotic_counts(drug_count, organism, save_path)
        
                # # create a dataframe from the dictionary
                # df_counts = pd.DataFrame.from_dict(dataset_counts, orient="index")
                # df_counts.reset_index(inplace=True)
                # df_counts.columns = ["Organism", "Antibiotic Count"]
                # # sort the dataframe by antibiotic count
                # df_counts.sort_values(by="Antibiotic Count", ascending=False, inplace=True)

                # # plot the antibiotic counts per organism
                # plot_antibiotic_counts_per_organism(df_counts, save_path)


if __name__ == "__main__":
    main()
