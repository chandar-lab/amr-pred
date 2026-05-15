''' Generate statistics on k-mer matrices and FASTA files. 

Usage follows other scripts in this repository:
Usage:
    PYTHONPATH=<path_to_repository> python testing/kmer_stats.py \
        --path-fna <path_to_fna_files> \
        --path-csv <path_to_test_csv> \
        --path-model <path_to_trained_model> \
        --path-out <output_directory>

'''

import argparse
import os
import logging
from Bio import SeqIO
from testing.utils import get_parser, load_model_data, PATHOGEN_NAME_MAPPER, load_data


def main():
    parser = get_parser()
    args = parser.parse_args()
    path_root_fna = args.path_fna
    path_data_csv = args.path_csv
    path_model = args.path_model
    path_out = args.path_out

    # Start logging
    file_handler = logging.FileHandler(os.path.join(args.path_out, f"inference.log"))
    console_handler = logging.StreamHandler()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[file_handler, console_handler]
    )

    # Load model and data
    _, X_test, y_test, model_info, _ = load_model_data(
        path_root_fna, path_data_csv, path_model, path_out
    )
    pathogen = model_info['pathogen']
    antibiotic = model_info['antibiotic']
    k = model_info['kmer_size']

    # Generate statistics:
    logging.info("STATISTICS FOR PATHOGEN: %s, ANTIBIOTIC: %s, K: %s", pathogen, antibiotic, k)
    logging.info("---------------------------------")
    logging.info("--- General dataset statistics ---")
    logging.info("Number of genomes: %d", X_test.shape[0])

    # Number of resistant and susceptible samples
    num_resistant = sum(y_test)
    num_susceptible = len(y_test) - num_resistant
    logging.info("Number of resistant samples: %d", num_resistant)
    logging.info("Number of susceptible samples: %d", num_susceptible)


    logging.info("--- FASTA file statistics ---")
    path_root_fna = os.path.join(path_root_fna, PATHOGEN_NAME_MAPPER[pathogen])
    test_datalist = load_data(path_data_csv, path_root_fna)
    # Calculate average genome length and contig values
    total_length = 0
    num_contigs = 0
    for fna_file in test_datalist:
        for record in SeqIO.parse(fna_file["seq"], "fasta"):
            total_length += len(record.seq)
            num_contigs += 1
    avg_genome_length = total_length / len(test_datalist) if len(test_datalist) > 0 else 0
    avg_contig_length = total_length / num_contigs if num_contigs > 0 else 0
    avg_num_contigs = num_contigs / len(test_datalist) if len(test_datalist) > 0 else 0
    logging.info("Average genome length (across all genomes): %.2f", avg_genome_length)
    logging.info("Average contig length (across all contigs): %.2f", avg_contig_length)
    logging.info("Average number of contigs per genome: %.2f", avg_num_contigs)


    logging.info("--- K-mer feature matrix statistics ---")
    # Calculate sparsity of the feature matrix
    num_nonzero = (X_test != 0).sum() # type: ignore
    total_elements = X_test.size
    sparsity = 1 - (num_nonzero / total_elements)
    logging.info("K-mer feature matrix shape: %s", X_test.shape)
    logging.info("Number of non-zero elements in feature matrix: %d", num_nonzero)
    logging.info("Sparsity of feature matrix: %.4f", sparsity)

    # Calculate the mean and std of k-mer frequencies across all samples
    mean_kmer_freq = X_test.mean()
    std_kmer_freq = X_test.std()
    logging.info("Mean k-mer frequency across all samples: %.4f", mean_kmer_freq)
    logging.info("Standard deviation of k-mer frequencies across all samples: %.4f", std_kmer_freq)

    # Calculate the mean and std of non-zero k-mer frequencies
    if num_nonzero > 0:
        mean_nonzero_kmer_freq = X_test[X_test != 0].mean()
        std_nonzero_kmer_freq = X_test[X_test != 0].std()
        logging.info("Mean non-zero k-mer frequency: %.4f", mean_nonzero_kmer_freq)
        logging.info("Standard deviation of non-zero k-mer frequencies: %.4f", std_nonzero_kmer_freq)

    # Print the maximum k-mer frequency observed
    max_kmer_freq = X_test.max()
    logging.info("Maximum k-mer frequency observed in feature matrix: %.4f", max_kmer_freq)

    # Print the datatype of the feature matrix
    logging.info("Data type of feature matrix: %s", X_test.dtype)

if __name__ == "__main__":
    main()





