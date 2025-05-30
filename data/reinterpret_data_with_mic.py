"""
Script for preprocessing the NCBI ASTs dataset. The script does the following:
1. Loads the NCBI ASTs dataset containing all species, phenotypic AST labels, and assemblies.
    1.1 Filters the dataset with MIC values from CLSI breakpoints
2. (optional) Generates barplots showing genome counts per species and antibiotic.
3. Store the individual species-antibiotic dataset combination in a separate CSV file for training.
"""

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import argparse


# global parameters
FONTSIZE = 12


def get_parser():
    parser = argparse.ArgumentParser(description="Filter the NCBI resistance phenotypes ASTs dataset with MIC values from CLSI breakpoints")
    parser.add_argument(
        "--path-clsi-bps",
        type=str,
        default="breakpoints_per_species_20250527.csv",
        help="Path to the filtered ncbi dataset (output of `download_dataset.py`) containing all species, "
                "phenotypics AST labels, and assemblies",
    )
    parser.add_argument(
        "--path-data",
        type=str,
        default="acinetobacter_baumannii_SRND_20250527.csv",
        help="Path to the ncbi dataset (output of `download_dataset.py`)",
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


def clean_breakpoints_file(path_clsi_bps):
    """
    This function takes as input the path to the CLSI breakpoints file and performs some minor cleaning:
    """

    df_bps = pd.read_csv(path_clsi_bps, sep=';')

    # drop rows where "BP_S_CLSI (<=)" and "BP_R_CLSI (>=)" are both NA
    df_bps = df_bps.dropna(subset=["BP_S_CLSI (<=)", "BP_R_CLSI (>=)"], how='all')

    # make all the drug names lowercase
    df_bps['drug'] = df_bps['drug'].str.lower()

    # replace space in taxon names with underscore
    df_bps['taxon'] = df_bps['taxon'].str.replace(' ', '_', regex=False)
    # make sure the taxon names are lowercase
    df_bps['taxon'] = df_bps['taxon'].str.lower()

    # replace acinetobacter_calcoaceticus/baumannii_complex with acinetobacter_baumannii
    df_bps['taxon'] = df_bps['taxon'].str.replace('acinetobacter_calcoaceticus/baumannii_complex', 'acinetobacter_baumannii', regex=False)

    # # save the dataframe to a new CSV file
    # df_bps.to_csv(path_file.replace(".csv", "_filt.csv"), index=False)
    return df_bps


def plot_antibiotic_counts(antibiotic_dict, organism, save_path):
    # This function plots barplots for each specie, showing the counts of resistant and susceptible genomes for each antibiotic
    # available in the dataset for that specie
    # Similar to Fig. 2A in the McHardy paper (https://doi.org/10.1093/bib/bbae206)

    # create a dataframe from the dictionary
    df_counts = pd.DataFrame.from_dict(antibiotic_dict, orient="index")
    df_counts.reset_index(inplace=True)
    df_counts.columns = ["Antibiotic", "Resistant", "Susceptible"]
    
    # melt the dataframe to long format
    df_counts_melted = df_counts.melt(
        id_vars=["Antibiotic"], 
        value_vars=["Resistant", "Susceptible"], 
        var_name="Resistance phenotype", 
        value_name="Count"
    )
    
    # change the color palette to be more distinct (red: resistant, green: susceptible)
    sns.set_palette(["#FF6347", "#32CD32"])
    # # draw a barplot for the antibiotic counts: x-axis: antibiotic name, y-axis: genome count, color: resistant/susceptible
    # plt.figure(figsize=(12, 7))
    # sns.barplot(data=df_counts_melted, x="Antibiotic", y="Count", hue="Resistance phenotype")
    # plt.title(f"Genome Counts of Resistant and Susceptible for {organism}")
    # plt.xticks(rotation=90)
    # plt.tight_layout()
    # plt.show()

    # draw another plot in log scale
    plt.figure(figsize=(13, 8))
    sns.barplot(data=df_counts_melted, x="Antibiotic", y="Count", hue="Resistance phenotype")
    plt.yscale("log")
    plt.ylim(10, 5*10**3)
    plt.ylabel("Genome Count (Log Scale)", fontsize=FONTSIZE+3)
    plt.xlabel("Antibiotic", fontsize=FONTSIZE+3)
    plt.yticks(fontsize=FONTSIZE)
    plt.xticks(rotation=90, fontsize=FONTSIZE)
    plt.title(f"Genome Counts of Resistant and Susceptible for {organism} (Log Scale)")
    plt.tight_layout()

    organism = organism.replace(" ", "_").lower()
    plt.savefig(f"{save_path}/{organism}__drug_counts.png", dpi=300, facecolor="w")
    # plt.show()
    plt.close()


# plot number of antibiotics per organism in a barplot
def plot_antibiotic_counts_per_organism(df_counts, save_path):

    # get the total
    combined = df_counts["Antibiotic Count"].sum()

    # draw the barplot
    plt.figure(figsize=(13, 8))
    sns.barplot(data=df_counts, x="Organism", y="Antibiotic Count", color="#FF6347")
    plt.ylabel("Antibiotic Count", fontsize=FONTSIZE+3)
    plt.xlabel("Organism", fontsize=FONTSIZE+3)
    plt.xticks(rotation=90, fontsize=FONTSIZE+3)
    plt.yticks(fontsize=FONTSIZE)
    plt.title(f"Antibiotic Counts per Species. Total: {combined} datasets", fontsize=FONTSIZE+3)
    # add the counts on top of the bars
    for i, v in enumerate(df_counts["Antibiotic Count"]):
        plt.text(i, v + 0.5, str(v), ha='center', fontsize=FONTSIZE)

    plt.ylim(0, df_counts["Antibiotic Count"].max() + 10)
    plt.tight_layout()
    plt.savefig(f"{save_path}/antibiotic_counts_per_organism.png", dpi=300, facecolor="w")
    plt.close()


def get_mic_breakpoints(df_bps, organism, drug):
    """
    Get the MIC breakpoints for a given organism and drug.
    """
    # # read the CSV file
    # df_bps = pd.read_csv(path_clsi_bps)
    
    # filter the dataframe for the given organism and drug
    df_filtered = df_bps[(df_bps['taxon'] == organism) & (df_bps['drug'] == drug)]
    
    if df_filtered.empty:
        return None, None
    
    # get the breakpoints
    mic_s = df_filtered['BP_S_CLSI (<=)'].values[0]
    mic_r = df_filtered['BP_R_CLSI (>=)'].values[0]
    
    return mic_s, mic_r


def get_drugs_for_organism_clsi(df, organism):
    """
    Get all drugs for a given organism.
    """
    # # read the CSV file
    # df = pd.read_csv(path_clsi_bps)

    # filter the dataframe for the given organism
    df_filtered = df[df['taxon'] == organism]
    
    if df_filtered.empty:
        return []
    
    # get the unique drugs
    drugs = df_filtered['drug'].unique().tolist()
    
    return drugs


# keep only those samples where 'resistant' is only '>' / '>=' / '=='
def is_resistant(row, mic_r):
    return row['Measurement sign'] in ['>', '>=', '=='] and row['MIC (mg/L)'] >= (mic_r//2)
    # NOTE: why mic_r//2? because phenotypic tests always tested for double dillutions. 
    # e.g. if the test says >32 than it means that the measured MIC value is >= 64, 
    # so with a breakpoint of 64 the category is resistant.
    # But, if the measured MIC value is >16 than you it is inconclusive.

# keep only those samples where 'susceptible' is only '<' / '<=' / '=='
def is_susceptible(row, mic_s):
    return row['Measurement sign'] in ['<', '<=', '=='] and row['MIC (mg/L)'] <= mic_s


def main():
    parser = get_parser()
    args = parser.parse_args()

    # create the output directory if it doesn't exist
    if not os.path.exists(args.path_out):
        os.makedirs(args.path_out, exist_ok=True)

    # load the data
    df = pd.read_csv(args.path_data, index_col=0)
    # make sure the 'Antibiotic' column is lowercase
    df['Antibiotic'] = df['Antibiotic'].str.lower()
    # replace space or '-' in drug names with underscore
    df['Antibiotic'] = df['Antibiotic'].str.replace(' ', '_', regex=False)
    df['Antibiotic'] = df['Antibiotic'].str.replace('-', '_', regex=False)

    # same for Organism group
    df['Organism group'] = df['Organism group'].str.lower()
    df['Organism group'] = df['Organism group'].str.replace(' ', '_', regex=False)

    # print number of resistant and susceptible samples (RAW DATASET)
    print(f"Number of samples (RAW DATA): {len(df)}")
    print(f"Number of resistant samples: {len(df[df['Resistance phenotype'] == 'resistant'])}")
    print(f"Number of susceptible samples: {len(df[df['Resistance phenotype'] == 'susceptible'])}")
    print(f"Number of unknown samples: {len(df[df['Resistance phenotype'] == 'not defined'])}")

    organism = os.path.basename(os.path.dirname(args.path_data))
    # get cleaned breakpoints df
    df_bps = clean_breakpoints_file(args.path_clsi_bps)

    # filter out to keep only the drugs that are in the breakpoints file
    drugs_in_bps = get_drugs_for_organism_clsi(df_bps, organism)
    df = df[df['Antibiotic'].isin(drugs_in_bps)]

    # add a check to ensure that list of drugs match
    if not set(df['Antibiotic'].unique()).issubset(set(drugs_in_bps)):
        raise ValueError("Some drugs in the dataset are not found in the breakpoints file.")

    # print number of resistant and susceptible samples (after 1st level of filtering -- CLSI drugs)
    print(f"Number of samples (AFTER FILTERING FOR CLSI DRUGS): {len(df)}")
    print(f"Number of resistant samples: {len(df[df['Resistance phenotype'] == 'resistant'])}")
    print(f"Number of susceptible samples: {len(df[df['Resistance phenotype'] == 'susceptible'])}")
    print(f"Number of unknown samples: {len(df[df['Resistance phenotype'] == 'not defined'])}")

    # get the list of all the unique organism groups
    # organism_groups = sorted(df["Organism group"].unique())
    organism_groups = [organism]

    print(f"Number of unique species: {len(organism_groups)}")
    print(f"Species: {organism_groups}")

    drugs = df['Antibiotic'].unique().tolist()


    dataset_counts = {}
    for organism in tqdm(organism_groups, desc="Organism groups", unit="group"):
        
        drug_count = {}
        for drug in drugs:
            # filter out for the current drug
            df_new = df[df['Antibiotic'] == drug]

            # get breakpoints for the current drug and organism
            mic_s, mic_r = get_mic_breakpoints(df_bps, organism, drug)
            if mic_s is None or mic_r is None:
                print(f"No breakpoints found for {organism} and {drug}. Skipping...")
                continue

            # get the number of 'not defined' resistance phenotype samples
            num_not_defined = len(df_new[df_new['Resistance phenotype'] == 'not defined'])
            
            # iterate through the not defined samples and classify them
            for index, row in tqdm(df_new[df_new['Resistance phenotype'] == 'not defined'].iterrows(), total=num_not_defined, desc=f"\nProcessing {drug}"):
                if is_resistant(row, mic_r):
                    df_new.at[index, 'Resistance phenotype'] = 'resistant'
                elif is_susceptible(row, mic_s):
                    df_new.at[index, 'Resistance phenotype'] = 'susceptible'
                else:
                    # drop the sample if it doesn't match either
                    print(f"Dropping sample {index} for {drug} as it doesn't match the MIC criteria.")
                    df_new.drop(index, inplace=True)
                    
            # check if there any 'not defined' samples left
            if len(df_new[df_new['Resistance phenotype'] == 'not defined']) > 0:
                print(f"Warning: There are still {len(df_new[df_new['Resistance phenotype'] == 'not defined'])} samples with 'not defined' resistance phenotype for {drug}.")

            # for the remaining samples, filter them
            df_filtered = df_new[(df_new.apply(lambda row: is_resistant(row, mic_r), axis=1)) | 
                                (df_new.apply(lambda row: is_susceptible(row, mic_s), axis=1))]

            num_resistant = len(df_filtered[df_filtered['Resistance phenotype'] == 'resistant'])
            num_susceptible = len(df_filtered[df_filtered['Resistance phenotype'] == 'susceptible'])
            num_not_defined = len(df_filtered[df_filtered['Resistance phenotype'] == 'not defined'])

            if num_resistant < 50 or num_susceptible < 50:
                print(f"Warning: Not enough samples for {drug}. Resistant: {num_resistant}, Susceptible: {num_susceptible}.")
                print("Not saving the filtered dataframe for this drug.")
                continue
            
            else:
                print(f"Number of resistant samples for {drug}: {num_resistant}")
                print(f"Number of susceptible samples for {drug}: {num_susceptible}")
                print(f"Number of not defined samples for {drug}: {num_not_defined}")

                # save the filtered dataframe
                output_path = os.path.join(args.path_out, organism, 'filtered', f"{organism}__{drug}_SR_filt.csv")
                if not os.path.exists(output_path):
                    # create the directory for saving the filtered data if it doesn't exist
                    os.makedirs(os.path.dirname(output_path), exist_ok=True)
                
                # save the filtered dataframe to a CSV file
                df_filtered.to_csv(output_path, index=False)

                # create the dictionary
                drug_count[drug] = {
                    "resistant": num_resistant,
                    "susceptible": num_susceptible
                }
        
                # # get the number of antibiotics per organism (i.e. total number of "datasets")
                # dataset_counts[organism] = dataset_counts.get(organism, 0) + 1

        if len(drug_count) == 0:
            print(f"No drugs with at least 50 S/R genomes found for {organism}")
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
