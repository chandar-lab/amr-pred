"""
Script containing utility functions for data preprocessing and visualization
"""
import os
import glob
import logging
import numpy as np
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
    ''' Returns a list of dicts with keys: 'seq' (path to .fna file), 'label' (0/1), 'assembly' (assembly id)'''

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


def get_folds_stats(folds, dist_table):
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

    return fold_stats, fold_distance_matrix



# ================================
# MIC reinterpretation with CLSI breakpoints
# Shared by data/reinterpret_data_with_mic.py (training datasets) and testing/utils.py (test datasets)
# ================================

# Non-dilution MIC values; these happen when the uploader to the NCBI forgot a decimal place
BAD_MIC_VALUES = {5.0, 25.0, 12.0}


def clean_ecoli_data(df):
    """
    Clean the E. coli data by removing Shigella samples and keeping only E. coli samples.
    """
    # lowercase the 'Scientific name' column for case-insensitive comparison
    df['Scientific name'] = df['Scientific name'].str.lower()

    # filter out Shigella samples
    df = df[~df['Scientific name'].str.contains('shigella', na=False)]

    return df


def normalize_ast_names(df):
    """
    Lowercase the 'Antibiotic' and 'Organism group' columns and replace spaces (and '-' in drug names)
    with underscores, so that they match the names in the breakpoints file.
    """
    df['Antibiotic'] = df['Antibiotic'].str.lower()
    df['Antibiotic'] = df['Antibiotic'].str.replace(' ', '_', regex=False)
    df['Antibiotic'] = df['Antibiotic'].str.replace('-', '_', regex=False)

    df['Organism group'] = df['Organism group'].str.lower()
    df['Organism group'] = df['Organism group'].str.replace(' ', '_', regex=False)
    return df


def add_revised_mic(df):
    """
    Add the 'MIC_revised' column: the MIC value corrected for the measurement sign.
    """
    df['MIC_revised'] = df['MIC (mg/L)'].copy()
    for index, row in df.iterrows():
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
    return df


def reinterpret_drug_with_mic(df_new, bp_s, bp_r, drug, logger=logging):
    """
    Label the samples of a single species x drug dataframe (with 'MIC_revised') using the CLSI breakpoints.
    Adds the 'Phenotype_revised' column, drops samples between the breakpoints (intermediate),
    handles the non-dilution MIC values, and deduplicates assemblies.
    Returns the filtered dataframe.
    """
    df_new = df_new.copy()
    # initialize the 'Phenotype_revised' column with NaN values
    df_new['Phenotype_revised'] = pd.Series(np.nan, index=df_new.index, dtype=object)

    # iterate through all the samples and classify them based on revised MIC values
    n_bad_mic_kept, n_bad_mic_dropped = 0, 0
    indices_to_drop = []
    for index, row in df_new.iterrows():
        if row['MIC (mg/L)'] in BAD_MIC_VALUES: # THIS HAPPENS WHEN THE UPLOADER TO THE NCBI FORGOT A DECIMAL PLACE!
            if row['Resistance phenotype'] in ['resistant', 'susceptible']:
                df_new.at[index, 'Phenotype_revised'] = row['Resistance phenotype']
                # Fix the MIC_revised value to be what we consider the correct value based on whether the sample is resistant or susceptible
                if row['Resistance phenotype'] == 'susceptible' and not row['MIC_revised'] <= bp_s:
                    df_new.at[index, 'MIC_revised'] /= 100 # e.g. 25 --> 0.25
                    if df_new.at[index, 'MIC_revised'] == 0.05: # Correct for the place value error with 5.0 --> 0.05, which should be 0.5
                        df_new.at[index, 'MIC_revised'] = 0.5
                n_bad_mic_kept += 1
            else:
                # If the original label is not defined, we drop the sample
                indices_to_drop.append(index)
                n_bad_mic_dropped += 1
            continue
        if row['MIC_revised'] >= bp_r:
            df_new.at[index, 'Phenotype_revised'] = 'resistant'
        elif row['MIC_revised'] <= bp_s:
            df_new.at[index, 'Phenotype_revised'] = 'susceptible'
        else:
            # between the breakpoints (intermediate): drop
            indices_to_drop.append(index)

    if n_bad_mic_kept or n_bad_mic_dropped:
        logger.info(
            f"{drug}: {n_bad_mic_kept + n_bad_mic_dropped} samples with non-dilution MIC "
            f"{sorted(BAD_MIC_VALUES)} (likely lost decimal point): {n_bad_mic_kept} kept with the "
            f"original lab label, {n_bad_mic_dropped} dropped (no S/R lab label)")

    # Drop samples outside the loop
    if indices_to_drop:
        df_new.drop(indices_to_drop, inplace=True)

    # check if there any nan values in the 'Phenotype_revised' column
    if df_new['Phenotype_revised'].isna().any():
        logger.info(f"Warning: There are {df_new['Phenotype_revised'].isna().sum()} samples with NaN phenotype for {drug}.")

    # Deduplicate assemblies: the same isolate can be tested on multiple AST platforms (e.g. Vitek, Microscan, Phoenix),
    # giving multiple rows per assembly. Assemblies with conflicting revised phenotypes are dropped,
    # otherwise only the first row is kept.
    num_phenotypes = df_new.groupby('Assembly')['Phenotype_revised'].transform('nunique')
    conflicting_assemblies = df_new.loc[num_phenotypes > 1, 'Assembly'].unique()
    num_rows_before_dedup = len(df_new)
    df_new = df_new[num_phenotypes == 1]
    df_new = df_new.drop_duplicates(subset=['Assembly'], keep='first')
    num_dropped = num_rows_before_dedup - len(df_new)
    if num_dropped > 0:
        logger.info(f"{drug}: removed {num_dropped} duplicate rows, including {len(conflicting_assemblies)} assemblies dropped for conflicting phenotypes: {list(conflicting_assemblies)}")

    return df_new


def reinterpret_test_data(df, organism, drug, path_clsi_bps, logger=logging):
    """
    Apply the same filtering and MIC reinterpretation as data/reinterpret_data_with_mic.py to a test dataframe
    for one species x drug. The dataframe needs the NCBI AST columns ('Organism group', 'Antibiotic',
    'Measurement sign', 'MIC (mg/L)', 'Resistance phenotype') plus 'Assembly'
    ('Scientific name' too for E. coli). Any existing 'MIC_revised'/'Phenotype_revised' columns are recomputed.
    """
    required = ['Organism group', 'Antibiotic', 'Measurement sign', 'MIC (mg/L)', 'Resistance phenotype', 'Assembly']
    if organism == "escherichia_coli":
        required.append('Scientific name')
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Test CSV is missing the columns {missing} needed for the MIC reinterpretation.")

    df_orig = df.copy()
    df = df.drop(columns=['MIC_revised', 'Phenotype_revised'], errors='ignore')
    logger.info(f"Reinterpreting the test data with CLSI breakpoints ({len(df)} rows)")

    if organism == "escherichia_coli":
        n_before = len(df)
        df = clean_ecoli_data(df)
        logger.info(f"Removed {n_before - len(df)} Shigella samples")
        df['Organism group'] = 'escherichia_coli'

    df = normalize_ast_names(df)
    df.loc[df['Organism group'] == 'acinetobacter_calcoaceticus/baumannii_complex', 'Organism group'] = 'acinetobacter_baumannii'

    # keep the rows for the model's species and drug
    n_before = len(df)
    df = df[(df['Organism group'] == organism) & (df['Antibiotic'] == drug)]
    if len(df) < n_before:
        logger.info(f"Removed {n_before - len(df)} rows for other species/drugs than {organism} x {drug}")

    # drop the samples with no assembly or no MIC value
    n_before = len(df)
    df = df.dropna(subset=['Assembly', 'MIC (mg/L)'])
    if len(df) < n_before:
        logger.info(f"Removed {n_before - len(df)} rows with no assembly or NaN MIC values")

    bp_s, bp_r = get_mic_breakpoints(clean_breakpoints_file(path_clsi_bps), organism, drug)
    if bp_s is None or bp_r is None:
        raise ValueError(f"No CLSI breakpoints found for {organism} x {drug} in {path_clsi_bps}.")

    df = add_revised_mic(df)
    n_before = len(df)
    df = reinterpret_drug_with_mic(df, bp_s, bp_r, drug, logger)
    logger.info(f"Removed {n_before - len(df)} rows with intermediate MIC, undefined non-dilution MIC, or duplicate/conflicting assemblies")

    if len(df) == 0:
        raise ValueError(f"No test samples left for {organism} x {drug} after the MIC reinterpretation.")
    logger.info(f"Test samples after reinterpretation (CLSI S <= {bp_s}, R >= {bp_r}): {len(df)} "
                f"({(df['Phenotype_revised'] == 'resistant').sum()} resistant, "
                f"{(df['Phenotype_revised'] == 'susceptible').sum()} susceptible)")

    # report how the labels compare to the ones given in the CSV
    if 'Phenotype_revised' in df_orig.columns:
        given = df_orig.loc[df.index, 'Phenotype_revised']
        n_diff = int((given != df['Phenotype_revised']).sum())
        if n_diff:
            logger.warning(f"{n_diff} kept samples had a different 'Phenotype_revised' in the input CSV; using the reinterpreted labels")
    n_flipped = int(df['Resistance phenotype'].isin(['resistant', 'susceptible']).sum()
                    - (df['Resistance phenotype'] == df['Phenotype_revised']).sum())
    logger.info(f"Samples whose lab S/R label differs from the reinterpreted label: {n_flipped}")

    return df
