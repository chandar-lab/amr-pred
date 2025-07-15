import subprocess
import os
import time

import numpy as np
from multiprocessing import Pool # Used for parallel processing of ani and mash distances
from tqdm import tqdm
from pathlib import Path
from pyani_plus.methods.anim import parse_delta # used to calculate ANI distance

K = None # K-mer size for Mash and Dashing distance calculations
SKETCH_SIZE = None # Size of sketch used for Mash and Dashing

# def dnadiff_align(seq1, seq2, output_folder="mummerout", prefix=None, keep_report = False, verbose = False):
#     """
#     Run dnadiff to compare two genomic sequences and return the mean of the alignment percentage.
#     You must install MUMmer4 first using the tarball from their website. 
#     Make sure to add the folder containing the dnadiff binary to your PATH in .bashrc.
#     """
#     # Set the prefix to be the GCA ID of both sequences.
#     if prefix is None:
#         gca_id1 = '_'.join(os.path.basename(seq1).split('_')[0:2])
#         gca_id2 = '_'.join(os.path.basename(seq2).split('_')[0:2])
#         prefix = f"{output_folder}/{gca_id1}_{gca_id2}_dnadiff"

#     if verbose:
#         print(f"Using prefix: {prefix}")

#     # Check if the output folder already exists.
#     if not os.path.exists(output_folder):
#         os.makedirs(output_folder)
#         if verbose:
#             print(f"Created output folder: {output_folder}")
#     else:
#         output_folder = None
    
#     # Run dnadiff using subprocess.
#     if verbose:
#         subprocess.run(["dnadiff", "-p", prefix, seq1, seq2], check=True)
#     else:
#         subprocess.run(["dnadiff", "-p", prefix, seq1, seq2], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

#     # Extract the mean alignment percentage from the report file
#     report_file = f"{prefix}.report"
#     if os.path.exists(report_file):
#         with open(report_file, 'r') as file:
#             line = file.readlines()[6]
#             parts = line.split()
#             # There are two percentages in the report, one for each sequence.
#             # The returned value is the mean of the two.
#             seq1_pct = float(parts[1].split('(')[1].strip('%)'))
#             seq2_pct = float(parts[2].split('(')[1].strip('%)'))
#             mean_alignment_percentage = (seq1_pct + seq2_pct) / 2


#     # Delete the created files
#     if not keep_report:
#         extensions = ["1coords", "1delta", "delta", "mcoords", "mdelta", "qdiff", "rdiff", "report", "snps", "unqry", "unref"]
#         for ext in extensions:
#             file_path = f"{prefix}.{ext}"
#             if os.path.exists(file_path):
#                 os.remove(file_path)
#                 if verbose:
#                     print(f"Deleted {file_path}")
#             else:
#                 if verbose:
#                     print(f"{file_path} does not exist, skipping deletion.")
#         # Delete the output folder if it was newly created.
#         if output_folder and os.path.exists(output_folder):
#             os.rmdir(output_folder)
#             if verbose:
#                 print(f"Deleted output folder: {output_folder}")

#     return mean_alignment_percentage

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

def mash_distance_pair(fasta1, fasta2, k=K, sketch_size=SKETCH_SIZE, verbose=False):
    '''Calculate the Mash distance between two FASTA files using the mash tool.
    Returns the Mash distance as a float.
    Must first install mash, conda install -c bioconda mash'''
    
    # Load default values for k and sketch_size if not provided
    if k == None:
        k = 21 # Default k-mer size for Mash distance calculations
    if sketch_size == None:
        sketch_size = 1000 # Default sketch size for Mash distance calculations

    # generate the mash sketches for the two FASTA files
    # subprocess.run(["mash", "sketch", "-o", "tmp1", fasta1], stdout=stout, stderr=sterr)
    # subprocess.run(["mash", "sketch", "-o", "tmp2", fasta2], stdout=stout, stderr=sterr)
    # # Determine the mash distance between the two sketches
    # result = subprocess.run(["mash", "dist", "tmp1.msh", "tmp2.msh"],
    #                         capture_output=True, text=True)

    # New approach: Go straight to the mash distance calculation without creating intermediate files
    result = subprocess.run(["mash", "dist", fasta1, fasta2, "-k", str(k), "-s", str(sketch_size)], capture_output=True, text=True)
    line = result.stdout.strip().split('\n')[0]
    fields = line.split('\t')
    distance = float(fields[2])  # Mash distance
    
    return distance

def dashing_distance_pair(fasta1, fasta2, verbose=False):
    '''Calculate the Dashing distance between two FASTA files using the dashing2 tool.
    Returns the Dashing distance as a float.
    Must first install dashing2, conda install -c bioconda dashing2'''
    sketch_size = SKETCH_SIZE
    k = K

    if k == None:
        k = 32 # Default k-mer size for dashing distance calculations is 32
    if sketch_size == None:
        sketch_size = 10 # Default sketch size for dashing distance calculations is 1024

    # Run dashing2 to calculate the distance
    result = subprocess.run(["dashing", "dist", "--kmer-length", str(k), "-S", str(sketch_size), "--mash-dist", fasta1, fasta2], capture_output=True, text=True)

    # Parse the output to get the Dashing distance
    lines = result.stdout.strip().split('\n')
    if len(lines) < 1:
        raise ValueError("Dashing distance could not be calculated. Please check the input files.")
    
    # The first line contains the distance
    distance = float(lines[4].split('\t')[2])
    
    return distance

def mash_distance_txt(txt, k=None, sketch_size=None, verbose=False):
    '''Calculate the pairwise Mash distance between all pairs of sequences in the two assembly list .txt files.
    returns a matrix of Mash distances, where the i-th row and j-th column is the Mash distance between the i-th sequence in txt and the j-th sequence in txt.'''
    # Call mash_distance_pair using subprocess
    global K, SKETCH_SIZE
    if k != None:
        K = k
    if sketch_size != None:
        SKETCH_SIZE = 2**sketch_size

    return _txt_based_distance(txt, _mash_distance_pair_wrapper, verbose=verbose)

def ani_distance_txt(txt, verbose=False):
    '''Calculate the pairwise ANI distance between all pairs of sequences in the two assembly list .txt files.
    returns a matrix of ANI distances, where the i-th row and j-th column is the ANI distance between the i-th sequence in txt1 and the j-th sequence in txt2.'''
    # Call ani_distance_pair using subprocess
    return _txt_based_distance(txt, _ani_distance_pair_wrapper, verbose=verbose)

def dashing_distance_txt(txt, k=K, sketch_size=SKETCH_SIZE, verbose=False):
    '''Calculate the pairwise Dashing distance between all pairs of sequences in the two assembly list .txt files.
    returns a matrix of Dashing distances, where the i-th row and j-th column is the Dashing distance between the i-th sequence in txt1 and the j-th sequence in txt2.'''
    # To run dashing2, use:
    # dashing2 dist -F ../amr_genomic_data/acinetobacter_baumannii/all_assemblies.txt

    if k == None:
        k = 32 # Default k-mer size for dashing distance calculations
    if sketch_size == None:
        sketch_size = 10 # Default sketch size for dashing distance calculations, 2**10 = 1024
    
    result = subprocess.run(["dashing", "dist", "--kmer-length", str(k), "-S", str(sketch_size), "--mash-dist", "-F", txt], capture_output=True, text=True)
    result = result.stdout.strip().split('\n')
    for i in range(len(result)):
        if result[i].startswith("##Names"):
            result = result[i+1:]
            break
    # FOR DASHING2: result = result.stdout.split('\n')[3:-1] # Results start on line 3, and continues to the 2nd last list (-2)
    num_samples = len(result)
    
    # Initialize the distance matrix
    distance_matrix = np.zeros((num_samples, num_samples))
    # Parse the output to fill the distance matrix
    for row in range(len(result)):
        fields = result[row].split('\t')[1:] # The first column is just a header giving the assembly name
        for col in range(row+1, num_samples): # Skip the diagonal with row+1, since it's all distance zero anyway
            distance_matrix[row, col] = float(fields[col])
            distance_matrix[col, row] = float(fields[col]) # Symmetric

    return distance_matrix



def _mash_distance_pair_wrapper(args):
    '''Wrapper function for mash_distance_pair to be used with multiprocessing.'''
    fasta1, fasta2 = args
    return mash_distance_pair(fasta1, fasta2)

def _ani_distance_pair_wrapper(args):
    '''Wrapper function for ani_distance_pair to be used with multiprocessing.'''
    fasta1, fasta2 = args
    return ani_distance_pair(fasta1, fasta2)

def _txt_based_distance(txt1, dist_func, verbose=False):
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

    # Calculate the Mash distance for each pair of sequences
    with Pool() as pool:
        results = list(tqdm(pool.imap(dist_func, [(assembly1, assembly2) for _, _, assembly1, assembly2 in pairs]), total=len(pairs)))
    

    # Initialize the distance matrix
    distance_matrix = np.zeros((len(sequences1), len(sequences1)))
    # Place the results in the distance matrix
    for (i, j, _, _) in pairs:
        value = results.pop(0)
        distance_matrix[i, j] = value
        distance_matrix[j, i] = value # Symmetric matrix because the distance is the same in both directions

    return distance_matrix






if __name__ == "__main__":
    # Example usage
    baumannii1 = "/home/mila/a/aselstya/scratch/amr_genomic_data/acinetobacter_baumannii/GCA_000586635.1/ncbi_dataset/data/GCA_000586635.1/GCA_000586635.1_MRSN57_genomic.fna"
    baumannii2 = "/home/mila/a/aselstya/scratch/amr_genomic_data/acinetobacter_baumannii/GCA_000586655.1/ncbi_dataset/data/GCA_000586655.1/GCA_000586655.1_MRSN58_genomic.fna"
    aeruginosa1 = "/home/mila/a/aselstya/scratch/amr_genomic_data/pseudomonas_aeruginosa/GCA_000710625.1/ncbi_dataset/data/GCA_000710625.1/GCA_000710625.1_MRSN18971scaf_genomic.fna"
    all_assemblies_txt = "/home/mila/a/aselstya/scratch/amr_genomic_data/pseudomonas_aeruginosa/10_assemblies.txt"

    start_time = time.time()
    print("ANI distance between the two Acinetobacter baumannii assemblies:")
    ani_dist = ani_distance_pair(baumannii1, baumannii2)
    print(f"{ani_dist:.4f} (calculated in {time.time() - start_time:.2f} seconds)")

    start_time = time.time()
    print("Mash distance between the two Acinetobacter baumannii assemblies:")
    mash_dist = mash_distance_pair(baumannii1, baumannii2)
    print(f"{mash_dist:.4f} (calculated in {time.time() - start_time:.2f} seconds)")

    start_time = time.time()
    print("Dashing distance between the two Acinetobacter baumannii assemblies:")
    dashing_dist = dashing_distance_pair(baumannii1, baumannii2)
    print(f"{dashing_dist:.4f} (calculated in {time.time() - start_time:.2f} seconds)")

    print("\nCalculating pairwise dashing distances for all assemblies in the text file:")
    dashing_distance_matrix = dashing_distance_txt(all_assemblies_txt)
    print(dashing_distance_matrix)
