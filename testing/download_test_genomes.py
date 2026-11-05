"""
Download the genomes of a test dataset from NCBI into the genome cache used for training and testing.

The test CSV is an export of the NCBI AST browser (https://www.ncbi.nlm.nih.gov/pathogens/ast) for one species,
possibly with several drugs. The assemblies are obtained as for training: Shigella isolates are removed from
E. coli, and the 'Isolate' column is merged with the NCBI isolates file (https://www.ncbi.nlm.nih.gov/pathogens/isolates).
The genomes of all the isolates with an assembly are downloaded (for all drugs), to
<path_out>/<species>/<assembly>/ncbi_dataset/data/<assembly>/*.fna. Genomes already in the cache are skipped.

Requires the NCBI datasets command line tool (https://www.ncbi.nlm.nih.gov/datasets/docs/v2/command-line-tools/download-and-install/).

Usage:
    PYTHONPATH=<path_to_repository> python testing/download_test_genomes.py \
        --path-csv <path_to_test_csv> \
        --path-isolates <path_to_isolates_tsv> \
        --path-out <path_to_genome_cache> \
        [--organism <species>] [--dry-run]
"""

import os
import logging
import argparse
import pandas as pd

from data.data_utils import clean_ecoli_data, add_assemblies, genome_fna_files, download_assemblies


def get_parser():
    parser = argparse.ArgumentParser(description="Download the genomes of a test dataset from NCBI.")
    parser.add_argument("--path-csv", type=str, required=True,
                        help="Path to the test CSV exported from the NCBI AST browser (one species, any drugs).")
    parser.add_argument("--path-isolates", type=str, required=True,
                        help="Path to the NCBI isolates file, used to get the assembly of each isolate.")
    parser.add_argument("--path-out", type=str, required=True,
                        help="Root of the genome cache; genomes go to <path_out>/<organism>/.")
    parser.add_argument("--organism", type=str, default=None,
                        help="Species, e.g. escherichia_coli. Defaults to the name of the CSV's parent directory.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Only report how many genomes are already in the cache and how many would be downloaded.")
    return parser


def main():
    args = get_parser().parse_args()
    organism = args.organism or os.path.basename(os.path.dirname(os.path.abspath(args.path_csv)))
    path_species = os.path.join(args.path_out, organism)

    os.makedirs(path_species, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(os.path.join(path_species, "download_test_genomes.log"))],
    )
    logging.info(f"Species: {organism}; test CSV: {args.path_csv}")

    df = pd.read_csv(args.path_csv, low_memory=False)
    n_isolates = df['Isolate'].nunique()
    if organism == "escherichia_coli":
        if 'Scientific name' not in df.columns:
            raise ValueError("The E. coli test CSV needs the 'Scientific name' column to remove the Shigella isolates.")
        df = clean_ecoli_data(df)
        logging.info(f"Removed {n_isolates - df['Isolate'].nunique()} Shigella isolates")
        n_isolates = df['Isolate'].nunique()

    df = add_assemblies(df, args.path_isolates, organism)
    assemblies = sorted(df['Assembly'].unique())
    logging.info(f"{df['Isolate'].nunique()}/{n_isolates} isolates have an assembly in {args.path_isolates}: "
                 f"{len(assemblies)} assemblies")
    if not assemblies:
        raise ValueError(f"No assemblies found for the isolates of {args.path_csv}; check --organism and --path-isolates.")

    if args.dry_run:
        n_cached = sum(bool(genome_fna_files(path_species, a)) for a in assemblies)
        logging.info(f"Dry run: {n_cached} genomes already in {path_species}, {len(assemblies) - n_cached} to download")
        return

    failed = download_assemblies(assemblies, path_species)
    logging.info(f"Done: {len(assemblies) - len(failed)}/{len(assemblies)} genomes in {path_species}")
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
