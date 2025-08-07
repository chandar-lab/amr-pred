''' Must move to the root of amr-pred before use.'''

import random
import numpy as np
import pickle
import time
import json
import os

from data.genomic_dst_utils import dashing_distance_txt, mash_distance_txt, ani_distance_txt
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

random.seed(1993) # For reproducibility

SAMPLE_SIZE = 10
PATHOGEN = "klebsiella_pneumoniae"  # Pathogen to use for the analysis
SKETCH_RANGE = list(range(5, 22))  # Sketch sizes to test

if __name__ == "__main__":
    # Check if the pathogen directory exists, if not create it
    if not os.path.exists(f"../amr_genomic_data/{PATHOGEN}"):
        os.makedirs(f"../amr_genomic_data/{PATHOGEN}")
    
    assemblies_txt_path = f"../amr_genomic_data/{PATHOGEN}/{SAMPLE_SIZE}_assemblies.txt"
    # Code to create a text file of 50 abaumannii assemblies and 50 aeruginosa assemblies
    if not os.path.exists(assemblies_txt_path):
        assemblies = []
        with open(f"../amr_genomic_data/acinetobacter_baumannii/all_assemblies.txt", "r") as file:
            baumannii_assemblies = file.readlines()
        assemblies.extend([assembly.strip() for assembly in random.sample(baumannii_assemblies, SAMPLE_SIZE // 2)])
        with open(f"../amr_genomic_data/pseudomonas_aeruginosa/all_assemblies.txt", "r") as file:
            aeruginosa_assemblies = file.readlines()
        assemblies.extend([assembly.strip() for assembly in random.sample(aeruginosa_assemblies, SAMPLE_SIZE // 2)])
        with open(assemblies_txt_path, "w") as file:
            for assembly in assemblies:
                file.write(f"{assembly}\n")

    # # Sample SAMPLE_SIZE examples from all_assemblies.txt and save them to {SAMPLE_SIZE}_assemblies.txt
    # if not os.path.exists(assemblies_txt_path):
    #     print(f"Sampling {SAMPLE_SIZE} assemblies from all_assemblies.txt...")
    #     with open("../amr_genomic_data/acinetobacter_baumannii/all_assemblies.txt") as file:
    #         assemblies = file.readlines()
    #     assemblies = [assembly.strip() for assembly in assemblies]
    #     sampled_assemblies = random.sample(assemblies, SAMPLE_SIZE)
    #     with open(f"../amr_genomic_data/acinetobacter_baumannii/{SAMPLE_SIZE}_assemblies.txt", "w") as file:
    #         for assembly in sampled_assemblies:
    #             file.write(f"{assembly}\n")
    
    # Generate ANI distance matrix for the sampled assemblies
    if not os.path.exists(f"../amr_genomic_data/{PATHOGEN}/{SAMPLE_SIZE}_assemblies_ani_distance_matrix.pkl"):
        print("Running ANI distance calculation...")
        start_time = time.time()
        # Generate ANI distance matrix
        ani_dist_matrix = ani_distance_txt(assemblies_txt_path)
        # Save the result as a pickle file
        ani_time = time.time() - start_time
        print(f"ANI distance matrix generated in {ani_time:.2f} seconds.")
        with open(f"../amr_genomic_data/{PATHOGEN}/{SAMPLE_SIZE}_assemblies_ani_distance_matrix.pkl", "wb") as file:
            pickle.dump(ani_dist_matrix, file)
    else:
        print("ANI distance matrix already exists. Loading from file...")
        with open(f"../amr_genomic_data/{PATHOGEN}/{SAMPLE_SIZE}_assemblies_ani_distance_matrix.pkl", "rb") as file:
            ani_dist_matrix = pickle.load(file)
    
    dashing_mean_errors = []
    dashing_stds = []
    mash_mean_errors = []
    mash_stds = []
    dashing_times = []
    mash_times = []

    # Run each library once quickly to allow warmup before timing
    print("Warming up Dashing and Mash libraries...")
    dashing_distance_txt(f"../amr_genomic_data/acinetobacter_baumannii/10_assemblies.txt")
    mash_distance_txt(f"../amr_genomic_data/acinetobacter_baumannii/10_assemblies.txt")

    for sketchsize in SKETCH_RANGE:
        # Check if a pickle already exists for the current sketch size
        if not os.path.exists(f"../amr_genomic_data/{PATHOGEN}/{SAMPLE_SIZE}_assemblies_dashing_distance_matrix_{sketchsize}.pkl"):
            print(f"\nRunning Dashing distance calculation with sketch size {sketchsize}...")
            # Generate the dashing distance matrix
            start_time = time.time()
            result = dashing_distance_txt(assemblies_txt_path, sketch_size=sketchsize)
            dashing_times.append(time.time() - start_time)
            print(f"Dashing distance matrix generated in {dashing_times[-1]:.2f} seconds.")
            # Dump the times to a file
            with open(f"../amr_genomic_data/{PATHOGEN}/{SAMPLE_SIZE}_dashing_times.csv", "a") as file:
                file.write(f"{sketchsize},{dashing_times[-1]:.2f}\n")
            # Save the result as a pickle file
            with open(f"../amr_genomic_data/{PATHOGEN}/{SAMPLE_SIZE}_assemblies_dashing_distance_matrix_{sketchsize}.pkl", "wb") as file:
                pickle.dump(result, file)
        else:
            print(f"\nDashing distance matrix for sketch size {sketchsize} already exists. Loading from file...")
            with open(f"../amr_genomic_data/{PATHOGEN}/{SAMPLE_SIZE}_assemblies_dashing_distance_matrix_{sketchsize}.pkl", "rb") as file:
                result = pickle.load(file)
            # Load the time from the times files
            with open(f"../amr_genomic_data/{PATHOGEN}/{SAMPLE_SIZE}_dashing_times.csv", "r") as file:
                for line in file:
                    if line.startswith(f"{sketchsize},"):
                        dashing_times.append(float(line.split(",")[1]))
                        break
                else:
                    print(f"Warning: No time found for sketch size {sketchsize} in dashing_times.csv")
        
        # Calculate the mean error between the dashing distance matrix and the ANI distance matrix
        mean_abs_error = np.mean(np.abs(result - ani_dist_matrix))
        std_abs_error = np.std(np.abs(result - ani_dist_matrix))
        print(f"Standard deviation of absolute error for sketch size {sketchsize}: {std_abs_error:.4f}")
        dashing_mean_errors.append(mean_abs_error)
        dashing_stds.append(std_abs_error)
        print(f"Mean absolute error for sketch size {sketchsize}: {mean_abs_error:.4f}")


        ### MASH ###
        # Check if a pickle already exists for the current sketch size
        if not os.path.exists(f"../amr_genomic_data/{PATHOGEN}/{SAMPLE_SIZE}_assemblies_mash_distance_matrix_{sketchsize}.pkl"):
            print(f"Running Mash distance for sketch size {sketchsize}...")
            # Generate the mash distance matrix
            start_time = time.time()
            result = mash_distance_txt(assemblies_txt_path, sketch_size=sketchsize)
            mash_times.append(time.time() - start_time)
            print(f"Mash distance matrix generated in {mash_times[-1]:.2f} seconds.")
            # Dump the times to a file
            with open(f"../amr_genomic_data/{PATHOGEN}/{SAMPLE_SIZE}_mash_times.csv", "a") as file:
                # sketch size followed by time
                file.write(f"{sketchsize},{mash_times[-1]:.2f}\n")
            # Save the result as a pickle file
            with open(f"../amr_genomic_data/{PATHOGEN}/{SAMPLE_SIZE}_assemblies_mash_distance_matrix_{sketchsize}.pkl", "wb") as file:
                pickle.dump(result, file)
        else:
            print(f"Mash distance matrix for sketch size {sketchsize} already exists. Loading from file...")
            with open(f"../amr_genomic_data/{PATHOGEN}/{SAMPLE_SIZE}_assemblies_mash_distance_matrix_{sketchsize}.pkl", "rb") as file:
                result = pickle.load(file)
            # Load the time from the times files
            with open(f"../amr_genomic_data/{PATHOGEN}/{SAMPLE_SIZE}_mash_times.csv", "r") as file:
                for line in file:
                    if line.startswith(f"{sketchsize},"):
                        mash_times.append(float(line.split(",")[1]))
                        break
                else:
                    print(f"Warning: No time found for sketch size {sketchsize} in mash_times.csv")

        # Calculate the mean error between the mash distance matrix and the ANI distance matrix
        mean_abs_error = np.mean(np.abs(result - ani_dist_matrix))
        std_abs_error = np.std(np.abs(result - ani_dist_matrix))
        mash_mean_errors.append(mean_abs_error)
        mash_stds.append(std_abs_error)
        print(f"Standard deviation of absolute error for sketch size {sketchsize}: {std_abs_error:.4f}")
        print(f"Mean absolute error for sketch size {sketchsize}: {mean_abs_error:.4f}")

    # Save the mean errors and times for dashing and mash to json files
    results = {
        "dashing_mean_errors": dashing_mean_errors,
        "mash_mean_errors": mash_mean_errors,
        "dashing_times": dashing_times,
        "mash_times": mash_times
    }
    with open(f"../amr_genomic_data/{PATHOGEN}/{SAMPLE_SIZE}_distance_results.json", "w") as file:
        json.dump(results, file)
    
    # ADD PLOTTING CODE HERE
    print("All calculations completed and results saved.")
    palette = {"ANI": "#8856a7", "Mash": "#3182bd", "Dashing": "#31a354"}

    # Plot the data
    fig, ax = plt.subplots(figsize=(10, 6))
    sketch_sizes = list(SKETCH_RANGE)

    # Convert lists to numpy arrays for easier calculation
    dashing_mean_errors_arr = np.array(dashing_mean_errors)
    dashing_stds_arr = np.array(dashing_stds)
    mash_mean_errors_arr = np.array(mash_mean_errors)
    mash_stds_arr = np.array(mash_stds)
    
    # Plot lines with shaded standard deviation regions
    ax.plot(sketch_sizes, dashing_mean_errors, marker='o', label='Dashing', color=palette["Dashing"])
    ax.fill_between(sketch_sizes, 
                    dashing_mean_errors_arr - dashing_stds_arr, 
                    dashing_mean_errors_arr + dashing_stds_arr, 
                    color=palette["Dashing"], alpha=0.2)
    
    # Add value labels for Dashing points
    for i, (x, y) in enumerate(zip(sketch_sizes, dashing_mean_errors)):
        ax.annotate(f'{y:.4f}', (x, y), textcoords="offset points", xytext=(0,10), ha='center', fontsize=8, color='black')
    
    ax.plot(sketch_sizes, mash_mean_errors, marker='o', label='Mash', color=palette["Mash"])
    ax.fill_between(sketch_sizes, 
                    mash_mean_errors_arr - mash_stds_arr, 
                    mash_mean_errors_arr + mash_stds_arr, 
                    color=palette["Mash"], alpha=0.2)
    
    # Add value labels for Mash points
    for i, (x, y) in enumerate(zip(sketch_sizes, mash_mean_errors)):
        ax.annotate(f'{y:.4f}', (x, y), textcoords="offset points", xytext=(0,-15), ha='center', fontsize=8, color='black')
    
    ax.axhline(y=0, color=palette["ANI"], linestyle='--', label='ANI')
    ax.set_xlabel('Sketch Size (log2 scale)')
    ax.set_ylabel('Mean Absolute Error')
    ax.set_title(f'Mean Absolute Error of Dashing and Mash Distances vs. ANI for {SAMPLE_SIZE} A. Baumannii Samples')

    # Set y-axis limits to at least 0.05, but extend if data requires it
    max_upper_bound = max(
        np.max(dashing_mean_errors_arr + dashing_stds_arr),
        np.max(mash_mean_errors_arr + mash_stds_arr)
    )
    y_max = max(0.05, max_upper_bound * 1.05)  # Add 5% padding above the highest point
    ax.set_ylim(bottom=0, top=y_max)
    
    ax.legend()
    plt.xticks(sketch_sizes)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"../amr_genomic_data/{PATHOGEN}/ani_distance_plot.png")
    print(f"Distance plot saved to ../amr_genomic_data/{PATHOGEN}/ani_distance_plot.png")

    # Plot the times
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(sketch_sizes, dashing_times, marker='o', label='Dashing', color=palette["Dashing"])
    ax.plot(sketch_sizes, mash_times, marker='o', label='Mash', color=palette["Mash"])
    
    # Add value labels for Dashing time points
    for i, (x, y) in enumerate(zip(sketch_sizes, dashing_times)):
        ax.annotate(f'{y:.2f}s', (x, y), textcoords="offset points", xytext=(0,10), ha='center', fontsize=8, color='black')
    
    # Add value labels for Mash time points
    for i, (x, y) in enumerate(zip(sketch_sizes, mash_times)):
        ax.annotate(f'{y:.2f}s', (x, y), textcoords="offset points", xytext=(0,-15), ha='center', fontsize=8, color='black')
    
    ax.set_xlabel('Sketch Size (log2 scale)')
    ax.set_ylabel('Mean time (seconds)')
    ax.set_title(f'Mean Time to Compute Dashing and Mash Distances for {SAMPLE_SIZE} A. Baumannii Samples')
    ax.legend()
    plt.xticks(sketch_sizes)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"../amr_genomic_data/{PATHOGEN}/ani_time_plot.png")
    print(f"Time plot saved to ../amr_genomic_data/{PATHOGEN}/ani_time_plot.png")



