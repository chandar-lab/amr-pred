import subprocess
import os
import glob
from pathlib import Path
import logging
import psutil

import numpy as np
from multiprocessing import Pool # Used for parallel processing of ani and mash distances
from tqdm import tqdm
from pathlib import Path
from pyani_plus.methods.anim import parse_delta # used to calculate ANI distance

MASH_DEFAULT_K = 21
DASHING_DEFAULT_K = 32
DEFAULT_SKETCH_SIZE = 10

NUM_PROC = len(psutil.Process().cpu_affinity())

def ani_distance_pair(fasta1, fasta2, k=None, sketch_size=None, verbose=False):
    '''Calculate the ANI distance between two FASTA files using the pyani tool.
    Returns the ANI distance as a float.
    k and sketch_size are not used in this function, but are included for compatibility with other distance functions.
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
    #sketch_size = 2**sketch_size

    # Generate the mash sketches so they can be reused
    fasta1_sketch_name = _sketch_assembly((fasta1, 'mash', sketch_size, k))
    fasta2_sketch_name = _sketch_assembly((fasta2, 'mash', sketch_size, k))

    # Calculate the Mash distance using the sketches
    result = subprocess.run(["mash", "dist", fasta1_sketch_name, fasta2_sketch_name], capture_output=True, text=True)
    line = result.stdout.strip().split('\n')[0]
    fields = line.split('\t')
    distance = float(fields[2])  # Mash distance
    
    return distance

def dashing_distance_pair(fasta1, fasta2, k=DASHING_DEFAULT_K, sketch_size=DEFAULT_SKETCH_SIZE):
    '''Calculate the Dashing distance between two FASTA files using the dashing tool.
    Returns the Dashing distance as a float.
    Must first install dashing, conda install -c bioconda dashing'''
    if k < 1 or k > 32:
        raise ValueError(f"k must be in the range [1, 31], got {k}")
    if sketch_size < 5 or sketch_size > 63:
        raise ValueError(f"sketch_size must be in the range [5, 63], got {sketch_size}")
    
    # Compute the dashing sketches, or use cached sketches if they exist
    fasta1_sketch_name = _sketch_assembly((fasta1, 'dashing', sketch_size, k))
    fasta2_sketch_name = _sketch_assembly((fasta2, 'dashing', sketch_size, k))

    # Use dashing to estimate the distance
    result = subprocess.run(["dashing", "dist", "--mash-dist", "--presketched", fasta1_sketch_name, fasta2_sketch_name], capture_output=True, text=True)

    # Parse the output to get the Dashing distance
    lines = result.stdout.strip().split('\n')
    if len(lines) < 1:
        raise ValueError("Dashing distance could not be calculated. Please check the input files.")
    
    # The first line contains the distance
    try:
        distance = float(lines[4].split('\t')[2])
    except (IndexError, ValueError) as e:
        print("error parsing Dashing output:", e)
        print("stdout was:", result.stdout)
        print("stderr was:", result.stderr)
        raise ValueError("Dashing distance could not be calculated. Please check the input files. fasta1: {}, fasta2: {}".format(fasta1_sketch_name, fasta2_sketch_name))


    return distance

def mash_distance_txt(txt, k=MASH_DEFAULT_K, sketch_size=DEFAULT_SKETCH_SIZE):
    '''Calculate the pairwise Mash distance between all pairs of sequences listed in the provided .txt file.
    returns a matrix of Mash distances, where the i-th row and j-th column is the Mash distance between the i-th sequence in txt and the j-th sequence in txt.'''
    # Pre-sketch the files
    with open(txt, 'r') as f:
        assemblies = f.readlines()
    assemblies = [assembly.strip() for assembly in assemblies if assembly.strip()]
    # Sketch the assemblies in parallel
    with Pool(NUM_PROC) as pool:
        sketches = list(tqdm(pool.imap(_sketch_assembly, [(assembly, 'mash', sketch_size, k) for assembly in assemblies]), total=len(assemblies)))
    # Now calculate the Mash distance using the sketches
    return _txt_based_distance(txt, _mash_distance_pair_wrapper, k, sketch_size)

def ani_distance_txt(txt):
    '''Calculate the pairwise ANI distance between all pairs of sequences listed in the passed .txt file.
    returns a matrix of ANI distances, where the i-th row and j-th column is the ANI distance between the i-th sequence in txt and the j-th sequence in txt.'''
    return _txt_based_distance(txt, _ani_distance_pair_wrapper, None, None)

def dashing_distance_txt(txt,k=DASHING_DEFAULT_K, sketch_size=DEFAULT_SKETCH_SIZE):
    '''Calculate the pairwise Dashing distance between all pairs of sequences listed in the passed .txt file.
    returns a matrix of Dashing distances, where the i-th row and j-th column is the Dashing distance between the i-th sequence in txt and the j-th sequence in txt.'''
    # Pre-sektch the files
    with open(txt, 'r') as f:
        assemblies = f.readlines()
    assemblies = [assembly.strip() for assembly in assemblies if assembly.strip()]
    # Sketch the assemblies in parallel
    with Pool(NUM_PROC) as pool:
        sketches = list(tqdm(pool.imap(_sketch_assembly, [(assembly, 'dashing', sketch_size, k) for assembly in assemblies]), total=len(assemblies)))
    # Now calculate the Dashing distance using the sketches
    return _txt_based_distance(txt, _dashing_distance_pair_wrapper, k, sketch_size)

def get_assemblies_txt(assemblies_dir, force_regenerate=False):
    '''Get a .txt file containing the paths to all assemblies in the provided directory.
    The .txt file will be created if it does not exist.'''
    # Assemblies_dir is analogous to root_path_fna in the original code
    assemblies_txt = os.path.join(assemblies_dir, "all_assemblies.txt")
    if (not os.path.isfile(os.path.join(assemblies_txt))) or force_regenerate:
        logging.info(f"Generating list of assemblies at {assemblies_txt}...")
        # Get all assemblies from the assemblies_dir folder, and put them in a list
        subfolders = os.listdir(assemblies_dir)
        assembly_files = [glob.glob(os.path.join(assemblies_dir, folder, "ncbi_dataset", "data", folder, "*.fna"))[0] for folder in subfolders if os.path.isdir(os.path.join(assemblies_dir, folder))]

        # Save the list of assemblies to a file in the assemblies_dir folder
        with open(assemblies_txt, "w") as f:
            for assembly in assembly_files:
                f.write(f"{assembly}\n")
    else:
        logging.info(f"List of assemblies already exists at {assemblies_txt}, skipping regeneration.")

    return assemblies_txt


def _mash_distance_pair_wrapper(args):
    '''Wrapper function for mash_distance_pair to be used with multiprocessing.'''
    fasta1, fasta2, k, sketch_size = args
    return mash_distance_pair(fasta1, fasta2, k=k, sketch_size=sketch_size)

def _ani_distance_pair_wrapper(args):
    '''Wrapper function for ani_distance_pair to be used with multiprocessing.'''
    fasta1, fasta2, k, sketch_size = args
    return ani_distance_pair(fasta1, fasta2)

def _dashing_distance_pair_wrapper(args):
    '''Wrapper function for dashing_distance_pair to be used with multiprocessing.'''
    fasta1, fasta2, k, sketch_size = args
    return dashing_distance_pair(fasta1, fasta2, k=k, sketch_size=sketch_size)

def _txt_based_distance(txt1, dist_func, k, sketch_size):
    '''Calculate the pairwise Mash distance between all pairs of sequences in the provided text file.
    returns a matrix of Mash distances, where the i-th row and j-th column is the Mash distance between the i-th sequence and the j-th sequence in txt.'''

    # Read the sequences from the .txt files
    with open(txt1, 'r') as f:
        sequences1 = f.readlines()

    # Create a list of all the pairs that need to be compared
    pairs = []
    for assembly1_idx in range(len(sequences1)):
        for assembly2_idx in range(len(sequences1)):
            if assembly2_idx <= assembly1_idx: # Skip the diagonal and lower triangle of the matrix.
                continue
            assembly1 = sequences1[assembly1_idx].strip()
            assembly2 = sequences1[assembly2_idx].strip()
            if not os.path.exists(assembly1):
                raise FileNotFoundError(f"Assembly file {assembly1} does not exist.")
            if not os.path.exists(assembly2):
                raise FileNotFoundError(f"Assembly file {assembly2} does not exist.")
            pairs.append((assembly1_idx, assembly2_idx, assembly1, assembly2))

    # Use multiprocessing to calculate the distances in parallel
    with Pool(NUM_PROC) as pool: #len(psutil.Process().cpu_affinity())
        results = list(tqdm(pool.imap(dist_func, [(assembly1, assembly2, k, sketch_size) for _, _, assembly1, assembly2 in pairs]), total=len(pairs)))

    # Calculate the distances in sequence
    # results = []
    # for assembly1_idx, assembly2_idx, assembly1, assembly2 in tqdm(pairs, desc="Calculating distances"):
    #     result = dist_func((assembly1, assembly2, k, sketch_size))
    #     results.append(result)

    # Initialize the distance matrix
    distance_matrix = np.zeros((len(sequences1), len(sequences1)))
    # Place the results in the distance matrix
    for (i, j, _, _) in pairs:
        value = results.pop(0)
        distance_matrix[i, j] = value
        distance_matrix[j, i] = value # Symmetric matrix because the distance is the same in both directions

    return distance_matrix

def _sketch_assembly(args):
    '''Wrapper to sketch an assembly file using the specified method and parameters.
    Needed to prevent race conditions when doing batch processing of assemblies.'''
    assembly, method, sketch_size, k = args
    sketch_file = ''
    if method == 'mash':
        # Generate the sketch file name
        sketch_file = os.path.join(Path(assembly).parent, f"{Path(assembly).stem}_size{sketch_size}_k{k}_sketch.msh")
        if not os.path.exists(sketch_file):
            # Convert sketch size to a power of 2 to match dashing's functionality
            sketch_size = 2**sketch_size
            subprocess.run(["mash", "sketch", "-o", sketch_file, "-s", str(sketch_size), "-k", str(k), assembly], capture_output=True, text=True)
   
    elif method == 'dashing':
        # Generate the sketch file if it doesn't already exist
        sketch_file = f"{assembly}.w.{k}.spacing.{sketch_size}.hll"
        if not os.path.exists(sketch_file):
            subprocess.run(["dashing", "sketch", "-k", str(k), "-S", str(sketch_size), assembly], capture_output=True, text=True)
    return sketch_file


def clear_cached_sketches(txt):
    '''Clear the cached sketches in the directories of the assemblies listed in the provided .txt file.'''
    with open(txt, 'r') as f:
        assemblies = f.readlines()
    assemblies = [assembly.strip() for assembly in assemblies if assembly.strip()]
    for assembly in assemblies:
        assembly_dir = os.path.dirname(assembly)
        # Remove all .msh files in the assembly directory
        for sketch_file in glob.glob(os.path.join(assembly_dir, "*.msh")):
            os.remove(sketch_file)
        # Remove all .hll files in the assembly directory
        for sketch_file in glob.glob(os.path.join(assembly_dir, "*.hll")):
            os.remove(sketch_file)
    logging.info(f"Cleared cached sketches in {len(assemblies)} assemblies.")


if __name__ == "__main__":
    # Example usage
    path_root_fna = "/home/mila/a/aselstya/scratch/amr_genomic_data/acinetobacter_baumannii/"
    baumannii1_name = "GCA_001441355.1"
    baumannii1 = glob.glob(os.path.join(path_root_fna, baumannii1_name, "ncbi_dataset", "data", baumannii1_name, "*.fna"))[0]
    baumannii2 = "/home/mila/a/aselstya/scratch/amr_genomic_data/acinetobacter_baumannii/GCA_000586655.1/ncbi_dataset/data/GCA_000586655.1/GCA_000586655.1_MRSN58_genomic.fna"
    aeruginosa1 = "/home/mila/a/aselstya/scratch/amr_genomic_data/pseudomonas_aeruginosa/GCA_000710625.1/ncbi_dataset/data/GCA_000710625.1/GCA_000710625.1_MRSN18971scaf_genomic.fna"
    all_assemblies_txt = "/home/mila/a/aselstya/scratch/amr_genomic_data/acinetobacter_baumannii/100_assemblies.txt"

    clear_cached_sketches("/home/mila/a/aselstya/scratch/amr_genomic_data/acinetobacter_baumannii-pseudomonas_aeruginosa/100_assemblies.txt")
    clear_cached_sketches(all_assemblies_txt)

    # Calculate the dashing distance between all_assemblies_txt
    # start_time = time.time()
    # dashing_distances = dashing_distance_txt(all_assemblies_txt, sketch_size=10)
    # print(f"Dashing distances calculated in {time.time() - start_time:.2f} seconds.")
    # print("Dashing distance matrix shape:", dashing_distances.shape)
    # Calculate the dashing distance between all_assemblies_txt




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
