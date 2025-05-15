import os
import glob
from collections import Counter
import pandas as pd
import numpy as np
from tqdm import tqdm
import scipy.sparse
import argparse

from Bio import SeqIO
from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import xgboost as xgb

# TODO: 
# 1. add other options to create the feature matrix
# 2. add options for other ML models


LABEL_MAPPING = {
    "resistant": 0,
    "susceptible": 1
}

def get_parser():
    
    parser = argparse.ArgumentParser(description="Train XGBoost model on k-mer features")
    parser.add_argument("--path-dataroot", type=str, required=True, help="Root path of the dataset genomic sequences for all species")
    parser.add_argument("--path-csv", type=str, required=True, help="Path to the folder containing CSV files for each specie-drug combination")
    parser.add_argument("--pathogen", type=str, required=True, help="Pathogen name (lowercase, separated by underscores) e.g. neisseria_gonorrhoeae")
    parser.add_argument("--antibiotic", type=str, required=True, help="Antibiotic name (lowercase) e.g. azithromycin")
    parser.add_argument("--k", type=int, default=8, help="Length of k-mers to extract (default: 8)")
    
    return parser


def load_data(path_data_csv, path_root_fna):

    df_data = pd.read_csv(path_data_csv)
    specie_drug_assemblies = df_data['Assembly'].tolist()
    # print(specie_drug_assemblies)
    # get the corresponding phenotypic labels for each assembly
    phenotypes = df_data['Resistance phenotype'].tolist()
    # print(phenotypes)

    dataset_list = []
    for assembly, phenotype in zip(specie_drug_assemblies, phenotypes):
        temp_dict = {}
        # if assembly not in dataset_dict:
        temp_dict["seq"] = glob.glob(os.path.join(path_root_fna, assembly, "ncbi_dataset", "data", assembly, "*.fna"))[0]
        temp_dict["label"] = LABEL_MAPPING[phenotype]
        dataset_list.append(temp_dict)
    print("Number of samples in the dataset: ", len(dataset_list))
    # print(dataset_list[:5])
    
    return dataset_list


def main():

    # Parse command line arguments
    parser = get_parser()
    args = parser.parse_args()
    path_root_fna = args.path_dataroot
    path_data_csv = args.path_csv

    path_root_fna = "/home/mila/n/naga-karthik.enamundram/scratch/dataset_ncbi_asts"
    # dataset = "neisseria_gonorrhoeae__azithromycin.csv"
    dataset = f"{args.pathogen}__{args.antibiotic}.csv"
    path_data_csv = os.path.join(args.path_csv, dataset)
    if not os.path.isfile(path_data_csv):
        raise FileNotFoundError(f"CSV file not found: {path_data_csv}. Please check the name of the pathogen and drug.")

    # NOTE: this is a list of dicts pointing for the paths to the genomic sequences (.fna files) and the corresponding S/R labels
    dataset_list = load_data(path_data_csv, path_root_fna)

    # Parameters
    k = args.k

    # Storage
    kmer_counts_list = []
    label_list = []

    print("Extracting k-mers...")
    # K-mer extraction loop
    for entry in tqdm(dataset_list, desc="Processing isolates"):
        fna_path = entry['seq']
        label = entry['label']

        if not os.path.isfile(fna_path):
            continue  # skip missing files

        # Method 1: Using k-mer counts as the feature matrix
        kmer_counter = Counter()
        for record in SeqIO.parse(fna_path, "fasta"):
            seq = str(record.seq).upper()
            for i in range(len(seq) - k + 1):
                kmer = seq[i:i + k]
                if "N" not in kmer:
                    kmer_counter[kmer] += 1

        kmer_counts_list.append(kmer_counter)
        label_list.append(label)

    # Vectorize into sparse feature matrix
    print("Vectorizing k-mers...")
    vectorizer = DictVectorizer(sparse=True)
    X = vectorizer.fit_transform(kmer_counts_list)
    y = np.array(label_list)

    print(f"Feature matrix shape: {X.shape}")


    # Train/test split (random)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    print(f"Train set shape: {X_train.shape}, Test set shape: {X_test.shape}")


    # Train XGBoost
    print("Training XGBoost model...")
    model = xgb.XGBClassifier(
        objective="binary:logistic",
        use_label_encoder=False,
        eval_metric="logloss",
        n_jobs=-1,
        verbosity=1
    )
    model.fit(X_train, y_train)

    # Evaluation
    print("Evaluating model...")
    y_pred = model.predict(X_test)
    print(y_pred)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))



if __name__ == "__main__":
    main()
