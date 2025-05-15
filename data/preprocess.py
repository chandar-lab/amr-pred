"""
Script for preprocessing the NCBI ASTs dataset. The script does the following:
1. Loads the NCBI ASTs dataset containing all species, phenotypic AST labels, and assemblies.
    1.1 Filters the dataset as per predefined thresholds for the number of samples per species and antibiotic.
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
    parser = argparse.ArgumentParser(description="Preprocess the NCBI resistance phenotypes ASTs dataset")
    parser.add_argument(
        "--path-data",
        type=str,
        default="20250505_ncbi_asts_filtered_all_organisms.tsv",
        help="Path to the filtered ncbi dataset (output of `download_dataset.py`) containing all species, "
                "phenotypics AST labels, and assemblies",
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
    plt.ylim(10, 10**4)
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


def main():
    parser = get_parser()
    args = parser.parse_args()

    # create the output directory if it doesn't exist
    if not os.path.exists(args.path_out):
        os.makedirs(args.path_out, exist_ok=True)

    # load the data
    df = pd.read_csv(args.path_data, sep="\t", low_memory=False)
    # print(len(df))
    # print(df.head(5))

    # drop rows where Assembly is nan
    df = df.dropna(subset=["Assembly"])
    # print("Number of rows after dropping Assembly nan:", df.shape[0])

    # NOTE: one of the preprocessing steps that McHardy paper did was to keep only the species with at least 500 samples
    # hence, we follow the same approach here (but 500 is an arbitrary number, can be changed to something else)
    # keep only the species that have at least 500 samples
    df = df[df["Organism group"].isin(df["Organism group"].value_counts()[df["Organism group"].value_counts() > 500].index)]
    # print(len(df))

    # get the list of all the unique organism groups
    organism_groups = sorted(df["Organism group"].unique())
    print(f"Number of unique species: {len(organism_groups)}")
    print(f"Species: {organism_groups}")

    dataset_counts = {}
    for organism in tqdm(organism_groups, desc="Organism groups", unit="group"):
        
        # get the list of antibiotics for the organism
        antibiotics = df[df["Organism group"] == organism]["Antibiotic"].unique()
        # print(antibiotics)
        antibiotics = sorted(antibiotics)

        # for each antibiotic, create a dictionary with antibiotic name, sub-dictionary with keys "resistant" and "susceptible" and their counts
        antibiotic_dict = {}
        for antibiotic in antibiotics:
            # get the data for the antibiotic
            data = df[(df["Organism group"] == organism) & (df["Antibiotic"] == antibiotic)]

            # NOTE: another preprocessing step that McHardy paper did was to ensure that for each specie, they have at least 100 resistant and susceptible phenotypes
            # so, we do the same here and drop the antibiotics that don't satisfy this condition

            # drop the antibiotics that don't have at least 100 resistant and susceptible phenotypes
            if len(data[data["Resistance phenotype"] == "resistant"]) < 100 or len(data[data["Resistance phenotype"] == "susceptible"]) < 100:
                continue
            
            # get the counts of resistant and susceptible
            resistant_count = len(data[data["Resistance phenotype"] == "resistant"])
            susceptible_count = len(data[data["Resistance phenotype"] == "susceptible"])
            
            # create the dictionary
            antibiotic_dict[antibiotic] = {
                "resistant": resistant_count,
                "susceptible": susceptible_count
            }

            org = organism.replace(" ", "_").lower()
            # save the data to a csv file
            data.to_csv(f"{args.path_out}/{org}__{antibiotic}.csv", index=False)
        
            # get the number of antibiotics per organism (i.e. total number of "datasets")
            dataset_counts[organism] = dataset_counts.get(organism, 0) + 1

        if len(antibiotic_dict) == 0:
            print(f"No antibiotics with at least 100 S/R genomes found for {organism}")
        else:
            if args.save_plots:
                save_path = os.path.join(args.path_out, "plots_single_species")
                # create the directory for saving the plots if it doesn't exist
                if not os.path.exists(save_path):
                    os.makedirs(save_path, exist_ok=True)
                # plot the antibiotic counts
                plot_antibiotic_counts(antibiotic_dict, organism, save_path)
        
                # create a dataframe from the dictionary
                df_counts = pd.DataFrame.from_dict(dataset_counts, orient="index")
                df_counts.reset_index(inplace=True)
                df_counts.columns = ["Organism", "Antibiotic Count"]
                # sort the dataframe by antibiotic count
                df_counts.sort_values(by="Antibiotic Count", ascending=False, inplace=True)

                # plot the antibiotic counts per organism
                plot_antibiotic_counts_per_organism(df_counts, save_path)


if __name__ == "__main__":
    main()
