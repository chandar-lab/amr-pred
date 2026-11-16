import subprocess
import os
import glob
import shutil
from pathlib import Path
import logging
import psutil

import numpy as np
from multiprocessing import Pool # Used for parallel processing of ani and mash distances
from tqdm import tqdm
from pyani_plus.methods.anim import parse_delta # used to calculate ANI distance

MASH_DEFAULT_K = 21
DASHING_DEFAULT_K = 32
DEFAULT_SKETCH_SIZE = 10    # 2^10 = 1024
NUM_PROC = len(psutil.Process().cpu_affinity())

# Install commands of the external tools used for the genomic distances (see the README)
TOOL_INSTALL = {
    "dashing": "conda install bioconda::dashing==0.4.0",
    "mash": "conda install bioconda::mash==2.1",
    "nucmer": "conda install bioconda::mummer (needed for ANI distances)",
}


def require_tool(name):
    '''Raise a clear error if an external command line tool is not on the PATH.'''
    if shutil.which(name) is None:
        raise RuntimeError(f"'{name}' was not found on the PATH; it is needed to compute the genomic distances "
                           f"for the phylogenetic folds. Install it with: {TOOL_INSTALL[name]}, "
                           f"or use random splits (use_genomic_splits=False).")


def _run_tool(cmd):
    '''Run an external tool, raising an error with its stderr if it fails. Returns the CompletedProcess.'''
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Command failed with exit code {result.returncode}: {' '.join(cmd)}\n{result.stderr.strip()}")
    return result


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
    
    # Calculate distances in parallel
    with Pool(NUM_PROC) as pool:
        results = list(tqdm(pool.imap(dist_func, [(assembly1, assembly2, k, sketch_size) for _, _, assembly1, assembly2 in pairs]), total=len(pairs)))
    
    # Initialize the distance matrix
    distance_matrix = np.zeros((n, n))
    
    # Vectorized assignment using numpy indexing
    distance_matrix[i_indices, j_indices] = results
    distance_matrix[j_indices, i_indices] = results  # Make symmetric

    return distance_matrix

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

    # Generate the mash sketches so they can be reused
    fasta1_sketch_name = _sketch_assembly((fasta1, 'mash', sketch_size, k))
    fasta2_sketch_name = _sketch_assembly((fasta2, 'mash', sketch_size, k))

    # Calculate the Mash distance using the sketches
    result = _run_tool(["mash", "dist", fasta1_sketch_name, fasta2_sketch_name])
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
    
    # Compute the dashing sketches, or use cached sketches if they exist
    fasta1_sketch_name = _sketch_assembly((fasta1, 'dashing', sketch_size, k))
    fasta2_sketch_name = _sketch_assembly((fasta2, 'dashing', sketch_size, k))

    # Use dashing to estimate the distance
    result = _run_tool(["dashing", "dist", "--mash-dist", "--presketched", fasta1_sketch_name, fasta2_sketch_name])

    # Parse the output to get the Dashing distance
    lines = result.stdout.strip().split('\n')
    if len(lines) < 5:
        raise ValueError(f"Unexpected dashing output for {fasta1} and {fasta2}:\n{result.stdout}")
    
    # The fifth line holds the distance
    distance = float(lines[4].split('\t')[2])

    return distance

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

def mash_distance_txt(txt, k=MASH_DEFAULT_K, sketch_size=DEFAULT_SKETCH_SIZE):
    '''Calculate the pairwise Mash distance between all pairs of sequences listed in the provided .txt file.
    returns a matrix of Mash distances, where the i-th row and j-th column is the Mash distance between the i-th sequence in txt and the j-th sequence in txt.'''
    require_tool("mash")
    # Pre-sketch the files
    with open(txt, 'r') as f:
        assemblies = f.readlines()
    assemblies = [assembly.strip() for assembly in assemblies if assembly.strip()]
    # Sketch the assemblies in parallel
    with Pool(NUM_PROC) as pool:
        list(tqdm(pool.imap(_sketch_assembly, [(assembly, 'mash', sketch_size, k) for assembly in assemblies]), total=len(assemblies)))
    # Now calculate the Mash distance using the sketches
    return _txt_based_distance(txt, _mash_distance_pair_wrapper, k, sketch_size)

def ani_distance_txt(txt):
    '''Calculate the pairwise ANI distance between all pairs of sequences listed in the passed .txt file.
    returns a matrix of ANI distances, where the i-th row and j-th column is the ANI distance between the i-th sequence in txt and the j-th sequence in txt.'''
    require_tool("nucmer")
    return _txt_based_distance(txt, _ani_distance_pair_wrapper, None, None)

def dashing_distance_txt(txt,k=DASHING_DEFAULT_K, sketch_size=DEFAULT_SKETCH_SIZE):
    '''Calculate the pairwise Dashing distance between all pairs of sequences listed in the passed .txt file.
    returns a matrix of Dashing distances, where the i-th row and j-th column is the Dashing distance between the i-th sequence in txt and the j-th sequence in txt.'''
    require_tool("dashing")
    # Pre-sketch the files
    with open(txt, 'r') as f:
        assemblies = f.readlines()
    assemblies = [assembly.strip() for assembly in assemblies if assembly.strip()]
    # Sketch the assemblies in parallel
    with Pool(NUM_PROC) as pool:
        list(tqdm(pool.imap(_sketch_assembly, [(assembly, 'dashing', sketch_size, k) for assembly in assemblies]), total=len(assemblies)))
    # Now calculate the Dashing distance using the sketches
    return _txt_based_distance(txt, _dashing_distance_pair_wrapper, k, sketch_size)

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
            _run_tool(["mash", "sketch", "-o", sketch_file, "-s", str(sketch_size), "-k", str(k), assembly])
   
    elif method == 'dashing':
        # Generate the sketch file if it doesn't already exist
        sketch_file = f"{assembly}.w.{k}.spacing.{sketch_size}.hll"
        if not os.path.exists(sketch_file):
            _run_tool(["dashing", "sketch", "-k", str(k), "-S", str(sketch_size), assembly])
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

