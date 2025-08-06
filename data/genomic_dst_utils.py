import subprocess
import os
import glob
from pathlib import Path
import logging

import numpy as np
from multiprocessing import Pool # Used for parallel processing of ani and mash distances
from tqdm import tqdm
from pathlib import Path
from pyani_plus.methods.anim import parse_delta # used to calculate ANI distance

MASH_DEFAULT_K = 21
DASHING_DEFAULT_K = 32
DEFAULT_SKETCH_SIZE = 10    # 2^10 = 1024


def get_assemblies_txt(assemblies_dir, force_regenerate=False):
    '''Get a .txt file containing the paths to all assemblies in the provided directory.
    The .txt file will be created if it does not exist.'''
    # Assemblies_dir is analogous to root_path_fna in the original code
    assemblies_txt = os.path.join(assemblies_dir, "all_assemblies.txt")
    if (not os.path.isfile(os.path.join(assemblies_txt))) or force_regenerate:
        logging.info(f"Generating list of assemblies at {assemblies_txt}...")
        # Get all assemblies from the assemblies_dir folder, and put them in a list
        assembly_files = glob.glob(os.path.join(assemblies_dir, "**", "ncbi_dataset", "data", "**", "*.fna"), recursive=True)

        # Save the list of assemblies to a file in the assemblies_dir folder
        with open(assemblies_txt, "w") as f:
            for assembly in assembly_files:
                f.write(f"{assembly}\n")
    else:
        logging.info(f"List of assemblies already exists at {assemblies_txt}, skipping regeneration.")

    return assemblies_txt


def _txt_based_distance(txt1, dist_func, k, sketch_size):
    '''Calculate the pairwise distance between all pairs of sequences in the provided text file.
    returns a matrix of distances, where the i-th row and j-th column is the distance between the i-th sequence and the j-th sequence in txt.'''

    # Read the sequences from the .txt files
    with open(txt1, 'r') as f:
        sequences1 = [line.strip() for line in f.readlines()]

    # Validate all files exist upfront
    missing_files = [seq for seq in sequences1 if not os.path.exists(seq)]
    if missing_files:
        raise FileNotFoundError(f"Assembly files do not exist: {missing_files[:5]}...")

    n = len(sequences1)
    
    # Use numpy to generate upper triangle indices more efficiently
    i_indices, j_indices = np.triu_indices(n, k=1)  # k=1 excludes diagonal
    pairs = [(i, j, sequences1[i], sequences1[j]) for i, j in zip(i_indices, j_indices)]
    # print(f"Generated {len(pairs)} pairs, now calculating distances...")
    
    # Calculate distances in parallel
    with Pool() as pool:
        results = list(tqdm(pool.imap(dist_func, [(assembly1, assembly2, k, sketch_size) for _, _, assembly1, assembly2 in pairs]), total=len(pairs)))
    
    # Initialize the distance matrix
    distance_matrix = np.zeros((n, n))
    
    # Vectorized assignment using numpy indexing
    distance_matrix[i_indices, j_indices] = results
    distance_matrix[j_indices, i_indices] = results  # Make symmetric

    return distance_matrix


def ani_distance_pair(fasta1, fasta2, verbose=False):
    '''Calculate the ANI distance between two FASTA files using the pyani tool.
    Returns the ANI distance as a float.
    Must first install pyani_plus, pip install pyani_plus'''
    # Only print terminal stdout if verbose flag is set, otherwise it will be suppressed
    stout = None
    sterr = None
    if not verbose:
        stout = subprocess.DEVNULL
        sterr = subprocess.DEVNULL
    
    # Generate the nucmer delta file for the two fasta files
    subprocess.run(["nucmer", "--prefix="+str(os.getpid()), fasta1, fasta2], stdout=stout, stderr=sterr)
    
    # Parse the output to get the ANI distance
    ani_distance = parse_delta(Path(str(os.getpid())+".delta"))
    if ani_distance is None:
        raise ValueError("ANI distance could not be calculated. Please check the input files.")
    # Delete the nucmer delta file
    os.remove(str(os.getpid())+".delta")

    return 1 - ani_distance[2]

def mash_distance_pair(fasta1, fasta2, k=MASH_DEFAULT_K, sketch_size=DEFAULT_SKETCH_SIZE):
    '''Calculate the Mash distance between two FASTA files using the mash tool.
    Returns the Mash distance as a float.
    Must first install mash, conda install -c bioconda mash'''
     # Check that the passed values are in the right range
    if k < 1 or k > 32:
        raise ValueError(f"k must be in the range [1, 32], got {k}")
    if sketch_size < 5 or sketch_size > 63:
        raise ValueError(f"sketch_size must be in the range [5, 63], got {sketch_size}")
    # Convert sketch size to a power of 2 to match dashing's functionality
    sketch_size = 2**sketch_size

    # Generate the mash sketches so they can be reused
    fasta1_sketch_name = os.path.join(Path(fasta1).parent, f"{Path(fasta1).stem}_size{sketch_size}_k{k}_sketch")
    fasta2_sketch_name = os.path.join(Path(fasta2).parent, f"{Path(fasta2).stem}_size{sketch_size}_k{k}_sketch")
    # Check if they already exist, if not, create them
    if not os.path.exists(fasta1_sketch_name + ".msh"):
        subprocess.run(["mash", "sketch", "-o", fasta1_sketch_name, "-s", str(sketch_size), "-k", str(k), fasta1], capture_output=True, text=True)
    if not os.path.exists(fasta2_sketch_name + ".msh"):
        subprocess.run(["mash", "sketch", "-o", fasta2_sketch_name, "-s", str(sketch_size), "-k", str(k), fasta2], capture_output=True, text=True)

    # Calculate the Mash distance using the sketches
    result = subprocess.run(["mash", "dist", fasta1_sketch_name + ".msh", fasta2_sketch_name + ".msh"], capture_output=True, text=True)
    line = result.stdout.strip().split('\n')[0]
    fields = line.split('\t')
    distance = float(fields[2])  # Mash distance
    
    return distance

def dashing_distance_pair(fasta1, fasta2, k=DASHING_DEFAULT_K, sketch_size=DEFAULT_SKETCH_SIZE):
    '''Calculate the Dashing distance between two FASTA files using the dashing tool.
    Returns the Dashing distance as a float.
    Must first install dashing, conda install -c bioconda dashing'''
    if k < 1 or k > 32:
        raise ValueError(f"k must be in the range [1, 32], got {k}")
    if sketch_size < 5 or sketch_size > 63:
        raise ValueError(f"sketch_size must be in the range [5, 63], got {sketch_size}")
    
    # Compute the dashing sketches
    fasta1_sketch_name = f"{fasta1}.w.{k}.spacing.{sketch_size}.hll"
    fasta2_sketch_name = f"{fasta2}.w.{k}.spacing.{sketch_size}.hll"
    if not os.path.exists(fasta1_sketch_name):
        subprocess.run(["dashing", "sketch", "-k", str(k), "-S", str(sketch_size), fasta1], capture_output=True, text=True)
    if not os.path.exists(fasta2_sketch_name):
        subprocess.run(["dashing", "sketch", "-k", str(k), "-S", str(sketch_size), fasta2], capture_output=True, text=True)

    # Use dashing to estimate the distance
    result = subprocess.run(["dashing", "dist", "--mash-dist", "--presketched", fasta1_sketch_name, fasta2_sketch_name], capture_output=True, text=True)

    # Parse the output to get the Dashing distance
    lines = result.stdout.strip().split('\n')
    if len(lines) < 1:
        raise ValueError("Dashing distance could not be calculated. Please check the input files.")
    
    # The first line contains the distance
    distance = float(lines[4].split('\t')[2])
    
    return distance

def _mash_distance_pair_wrapper(args):
    '''Wrapper function for mash_distance_pair to be used with multiprocessing.'''
    fasta1, fasta2, k, sketch_size = args
    return mash_distance_pair(fasta1, fasta2, k=k, sketch_size=sketch_size)

def _ani_distance_pair_wrapper(args):
    '''Wrapper function for ani_distance_pair to be used with multiprocessing.'''
    fasta1, fasta2 = args
    return ani_distance_pair(fasta1, fasta2)

def _dashing_distance_pair_wrapper(args):
    '''Wrapper function for dashing_distance_pair to be used with multiprocessing.'''
    fasta1, fasta2, k, sketch_size = args
    return dashing_distance_pair(fasta1, fasta2, k=k, sketch_size=sketch_size)

def mash_distance_txt(txt, k=MASH_DEFAULT_K, sketch_size=DEFAULT_SKETCH_SIZE):
    '''Calculate the pairwise Mash distance between all pairs of sequences listed in the provided .txt file.
    returns a matrix of Mash distances, where the i-th row and j-th column is the Mash distance between the i-th sequence in txt and the j-th sequence in txt.'''
    return _txt_based_distance(txt, _mash_distance_pair_wrapper, k, sketch_size)

def ani_distance_txt(txt):
    '''Calculate the pairwise ANI distance between all pairs of sequences listed in the passed .txt file.
    returns a matrix of ANI distances, where the i-th row and j-th column is the ANI distance between the i-th sequence in txt and the j-th sequence in txt.'''
    return _txt_based_distance(txt, _ani_distance_pair_wrapper)

def dashing_distance_txt(txt,k=DASHING_DEFAULT_K, sketch_size=DEFAULT_SKETCH_SIZE):
    '''Calculate the pairwise Dashing distance between all pairs of sequences listed in the passed .txt file.
    returns a matrix of Dashing distances, where the i-th row and j-th column is the Dashing distance between the i-th sequence in txt and the j-th sequence in txt.'''
    return _txt_based_distance(txt, _dashing_distance_pair_wrapper, k, sketch_size)



if __name__ == "__main__":
    # Example usage
    path_root_fna = "/home/mila/a/aselstya/scratch/amr_genomic_data/acinetobacter_baumannii/"
    baumannii1_name = "GCA_001441355.1"
    baumannii1 = glob.glob(os.path.join(path_root_fna, baumannii1_name, "ncbi_dataset", "data", baumannii1_name, "*.fna"))[0]
    baumannii2 = "/home/mila/a/aselstya/scratch/amr_genomic_data/acinetobacter_baumannii/GCA_000586655.1/ncbi_dataset/data/GCA_000586655.1/GCA_000586655.1_MRSN58_genomic.fna"
    aeruginosa1 = "/home/mila/a/aselstya/scratch/amr_genomic_data/pseudomonas_aeruginosa/GCA_000710625.1/ncbi_dataset/data/GCA_000710625.1/GCA_000710625.1_MRSN18971scaf_genomic.fna"
    all_assemblies_txt = "/home/mila/a/aselstya/scratch/amr_genomic_data/pseudomonas_aeruginosa/10_assemblies.txt"

    print("ANI distance between the two Acinetobacter baumannii assemblies:")
    print(ani_distance_pair(baumannii1, baumannii2))

    print("Mash distance between the two acinetobacter baumannii assemblies with sketch size 2^10:")
    print(mash_distance_pair(baumannii1, baumannii2, sketch_size=10))

    print("Dashing distance between the two acinetobacter baumannii assemblies with sketch size 10:")
    print(dashing_distance_pair(baumannii1, baumannii2, sketch_size=10))


    # start_time = time.time()
    # print("ANI distance between the two Acinetobacter baumannii assemblies:")
    # ani_dist = ani_distance_pair(baumannii1, baumannii2)
    # print(f"{ani_dist:.4f} (calculated in {time.time() - start_time:.2f} seconds)")

    # start_time = time.time()
    # print("Mash distance between Acinetobacter and a Pseudomonas aeruginosa assembly with sketch size 1000:")
    # mash_dist = mash_distance_pair(baumannii1, aeruginosa1, sketch_size=1000)
    # print(f"{mash_dist:.4f} (calculated in {time.time() - start_time:.2f} seconds)")

    # start_time = time.time()
    # print("Mash distance between Acinetobacter and a Pseudomonas aeruginosa assembly with sketch size 10:")
    # mash_dist = mash_distance_pair(baumannii1, aeruginosa1, sketch_size=10)
    # print(f"{mash_dist:.4f} (calculated in {time.time() - start_time:.2f} seconds)")
