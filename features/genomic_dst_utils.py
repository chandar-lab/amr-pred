import subprocess
import os
import time

def dnadiff_align(seq1, seq2, output_folder="mummerout", prefix=None, keep_report = False, verbose = False):
    """
    Run dnadiff to compare two genomic sequences and return the mean of the alignment percentage.
    You must install MUMmer4 first using the tarball from their website. 
    Make sure to add the folder containing the dnadiff binary to your PATH in .bashrc.
    """
    # Set the prefix to be the GCA ID of both sequences.
    if prefix is None:
        gca_id1 = '_'.join(os.path.basename(seq1).split('_')[0:2])
        gca_id2 = '_'.join(os.path.basename(seq2).split('_')[0:2])
        prefix = f"{output_folder}/{gca_id1}_{gca_id2}_dnadiff"

    if verbose:
        print(f"Using prefix: {prefix}")

    # Check if the output folder already exists.
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        if verbose:
            print(f"Created output folder: {output_folder}")
    else:
        output_folder = None
    
    # Run dnadiff using subprocess.
    if verbose:
        subprocess.run(["dnadiff", "-p", prefix, seq1, seq2], check=True)
    else:
        subprocess.run(["dnadiff", "-p", prefix, seq1, seq2], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    # Extract the mean alignment percentage from the report file
    report_file = f"{prefix}.report"
    if os.path.exists(report_file):
        with open(report_file, 'r') as file:
            line = file.readlines()[6]
            parts = line.split()
            # There are two percentages in the report, one for each sequence.
            # The returned value is the mean of the two.
            seq1_pct = float(parts[1].split('(')[1].strip('%)'))
            seq2_pct = float(parts[2].split('(')[1].strip('%)'))
            mean_alignment_percentage = (seq1_pct + seq2_pct) / 2


    # Delete the created files
    if not keep_report:
        extensions = ["1coords", "1delta", "delta", "mcoords", "mdelta", "qdiff", "rdiff", "report", "snps", "unqry", "unref"]
        for ext in extensions:
            file_path = f"{prefix}.{ext}"
            if os.path.exists(file_path):
                os.remove(file_path)
                if verbose:
                    print(f"Deleted {file_path}")
            else:
                if verbose:
                    print(f"{file_path} does not exist, skipping deletion.")
        # Delete the output folder if it was newly created.
        if output_folder and os.path.exists(output_folder):
            os.rmdir(output_folder)
            if verbose:
                print(f"Deleted output folder: {output_folder}")

    return mean_alignment_percentage

def mash_distance(fasta1, fasta2, verbose=False):
    '''Calculate the Mash distance between two FASTA files using the mash tool.
    Returns the Mash distance as a float.
    Must first install mash, conda install -c bioconda mash'''
    # Only print terminal stdout if verbose flag is set, otherwise it will be suppressed
    stout = None
    sterr = None
    if not verbose:
        stout = subprocess.DEVNULL
        sterr = subprocess.DEVNULL
    
    # generate the mash sketches for the two FASTA files
    subprocess.run(["mash", "sketch", "-o", "tmp1", fasta1], stdout=stout, stderr=sterr)
    subprocess.run(["mash", "sketch", "-o", "tmp2", fasta2], stdout=stout, stderr=sterr)
    # Determine the mash distance between the two sketches
    result = subprocess.run(["mash", "dist", "tmp1.msh", "tmp2.msh"],
                            capture_output=True, text=True)
    line = result.stdout.strip().split('\n')[0]
    fields = line.split('\t')
    distance = float(fields[2])  # Mash distance
    # Delete the mash files
    os.remove("tmp1.msh")
    os.remove("tmp2.msh")
    
    return distance


if __name__ == "__main__":
    # Example usage
    baumannii1 = "/home/mila/a/aselstya/scratch/amr_genomic_data/acinetobacter_baumannii/GCA_000586635.1/ncbi_dataset/data/GCA_000586635.1/GCA_000586635.1_MRSN57_genomic.fna"
    baumannii2 = "/home/mila/a/aselstya/scratch/amr_genomic_data/acinetobacter_baumannii/GCA_000586655.1/ncbi_dataset/data/GCA_000586655.1/GCA_000586655.1_MRSN58_genomic.fna"
    aeruginosa1 = "/home/mila/a/aselstya/scratch/amr_genomic_data/pseudomonas_aeruginosa/GCA_000710625.1/ncbi_dataset/data/GCA_000710625.1/GCA_000710625.1_MRSN18971scaf_genomic.fna"
    output_folder = "mummerout"

    start_time = time.time()
    print("Mean alignment percentage for two Acinetobacter baumannii strains:")
    print(dnadiff_align(baumannii1, baumannii2, output_folder), "%")
    print("Mash: ", mash_distance(baumannii1, baumannii2))
    print("time taken (secs):", (time.time()-start_time))

    start_time = time.time()
    print("Mean alignment percentage for Acinetobacter baumannii and Pseudomonas aeruginosa:")
    print(dnadiff_align(baumannii1, aeruginosa1, output_folder), "%")
    print("Mash: ", mash_distance(baumannii1, aeruginosa1))
    print("time taken (secs):", (time.time()-start_time))

    start_time = time.time()
    print("Mash distance between two Acinetobacter baumannii strains:")
    print(mash_distance(baumannii1, baumannii2))
    print("time taken (secs):", (time.time()-start_time))
 