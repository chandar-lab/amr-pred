"""
Script containing utility functions for data preprocessing and visualization
"""

import os
import glob
import logging
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# global parameters
FONTSIZE = 12
LABEL_MAPPING = {
    # NOTE: convention is to use 0 for susceptible and 1 for resistant
    "resistant": 1,
    "susceptible": 0
}

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
        temp_dict["assembly"] = assembly # ADDED FOR PHYLOGENY-BASED SPLITS
        dataset_list.append(temp_dict)
    
    return dataset_list


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
    # draw a barplot (log scale) for the antibiotic counts: x-axis: antibiotic name, y-axis: genome count, color: resistant/susceptible
    plt.figure(figsize=(13, 8))
    sns.barplot(data=df_counts_melted, x="Antibiotic", y="Count", hue="Resistance phenotype")
    plt.yscale("log")
    plt.ylim(10, 5*10**3) if organism != "escherichia_coli" else plt.ylim(10, 10**4)
    plt.ylabel("Genome Count (Log Scale)", fontsize=FONTSIZE+3)
    plt.xlabel("Antibiotic", fontsize=FONTSIZE+3)
    plt.yticks(fontsize=FONTSIZE)
    plt.xticks(rotation=90, fontsize=FONTSIZE)
    plt.title(f"Genome Counts for {organism} (Log Scale)")
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
    bp_s = df_filtered['BP_S_CLSI (<=)'].values[0]
    bp_r = df_filtered['BP_R_CLSI (>=)'].values[0]
    
    return bp_s, bp_r


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
