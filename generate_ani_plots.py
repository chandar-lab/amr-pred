''' Plot the accuracy of various ANI methods and their time with respect to sketch size. '''
import time
import random
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from data.genomic_dst_utils import ani_distance_txt, mash_distance_txt, dashing_distance_txt, get_assemblies_txt

random.seed(1993)

# Constants to change
SKETCH_RANGE = list(range(5, 21))
PATHOGEN = "acinetobacter_baumannii"
NUM_SAMPLES = 100

def compute_distances(assemblies_file, method, sketch_size=10):
    function_map = {
        'mash': mash_distance_txt,
        'dashing': dashing_distance_txt
    }
    if method not in function_map:
        raise ValueError(f"Unknown method: {method}")
    dist_func = function_map[method]
    start_time = time.time()
    distance_matrix = dist_func(assemblies_file, sketch_size=sketch_size)
    end_time = time.time()
    elapsed_time = end_time - start_time
    return distance_matrix, elapsed_time

if __name__ == '__main__':
    # Define necessary files
    assemblies_dir = os.path.join("../amr_genomic_data", PATHOGEN)
    ani_file = os.path.join(assemblies_dir, f"{NUM_SAMPLES}_assemblies_ani_distance_matrix.pkl")
    
    sampled_assemblies_txt = os.path.join(assemblies_dir, f"{NUM_SAMPLES}_assemblies.txt")

    # Load the list of assemblies, sample it down, and then save back to new file
    if not os.path.isfile(sampled_assemblies_txt):
        all_assemblies_txt = get_assemblies_txt(assemblies_dir, force_regenerate=False)
        with open(all_assemblies_txt, 'r') as f:
            all_assemblies = f.readlines()
        all_assemblies = [assembly.strip() for assembly in all_assemblies if assembly.strip()]
        sampled_assemblies = random.sample(all_assemblies, NUM_SAMPLES)
        with open(sampled_assemblies_txt, 'w') as f:
            for assembly in sampled_assemblies:
                f.write(f"{assembly}\n")

    
    ### Compute ANI distance as ground truth ###
    if not os.path.isfile(ani_file):
        print("Computing ANI distance matrix...")
        start_time = time.time()
        ani_distance_matrix = ani_distance_txt(sampled_assemblies_txt)
        print(f"ANI distance matrix computed in {time.time() - start_time:.2f} seconds.")
        ani_distance_matrix.to_pickle(ani_file)
        print(f"Saved ANI distance matrix to {ani_file}")

    else:
        print("ANI distance matrix already exists, loading...")
        ani_distance_matrix = np.load(ani_file, allow_pickle=True)

    # Compute distances for each method and sketch size
    results = []
    # Check if a results dataframe is already on disk
    results_file = os.path.join(assemblies_dir, f"{NUM_SAMPLES}_assemblies_distance_results.pkl")
    if os.path.isfile(results_file):
        print("Loading existing distance results...")
        results_df = pd.read_pickle(results_file)
        results = results_df.to_dict('records')


    ### Loop for sketch sizes and methods ###
    for sketch_size in SKETCH_RANGE:
        for method in ['mash', 'dashing']:
            if any(result['sketch_size'] == sketch_size and result['method'] == method for result in results):
                print(f"Skipping {method} with sketch size {sketch_size}, already computed.")
                continue
            pkl_file = os.path.join(assemblies_dir, f"{NUM_SAMPLES}_assemblies_{method}_s{sketch_size}_distance_matrix.pkl")
            print(f"Computing {method} distance with sketch size {sketch_size}...")
            distance_matrix, elapsed_time = compute_distances(sampled_assemblies_txt, method, sketch_size)
            results.append({
                'sketch_size': sketch_size,
                'method': method,
                'distance_matrix': distance_matrix,
                'elapsed_time': elapsed_time
            })
            print(f"Computed {method} distance matrix with sketch size {sketch_size} in {elapsed_time:.2f} seconds.")
        # Save the results to a file
        results_df = pd.DataFrame(results)
        results_df.to_pickle(results_file)
        print(f"Saved results up to sketch size {sketch_size} to {results_file}")
    print("Distance calculations complete. Results saved.")

    # Now compute the mean absolute error of each distance matrix against ANI
    dashing_maes = {} # indexed by sketch size
    mash_maes = {}
    dashing_std = {}
    mash_std = {}
    dashing_times = {}
    mash_times = {}
    for result in results:
        if result['sketch_size'] not in SKETCH_RANGE:
            continue
        ani_distances = ani_distance_matrix.flatten()
        method_distances = result['distance_matrix'].flatten()
        mae = np.mean(np.abs(ani_distances - method_distances))
        # Compute std of the errors
        std_error = np.std(np.abs(ani_distances - method_distances))
        print(f"Mean Absolute Error for {result['method']} with sketch size {result['sketch_size']}: {mae}")
        if result['method'] == 'dashing':
            dashing_maes[result['sketch_size']] = mae
            dashing_std[result['sketch_size']] = std_error
            dashing_times[result['sketch_size']] = result['elapsed_time']
        elif result['method'] == 'mash':
            mash_maes[result['sketch_size']] = mae
            mash_std[result['sketch_size']] = std_error
            mash_times[result['sketch_size']] = result['elapsed_time']

    # First plot: MAE vs Sketch Size with one line for each method and shaded area for std
    plt.figure(figsize=(10, 6))
    plt.plot(list(dashing_maes.keys()), list(dashing_maes.values()), label='Dashing MAE', marker='o', color='green')
    plt.fill_between(list(dashing_maes.keys()),
                     np.array(list(dashing_maes.values())) - np.array(list(dashing_std.values())),
                     np.array(list(dashing_maes.values())) + np.array(list(dashing_std.values())),
                     alpha=0.2, color='green')
    
    # Add value labels for Dashing points
    for i, (x, y) in enumerate(zip(list(dashing_maes.keys()), list(dashing_maes.values()))):
        if i % 2 == 0:
            plt.annotate(f'{y:.4f}', (x, y), textcoords="offset points", xytext=(0, 15), ha='center', fontsize=8, color='darkgreen')
    
    plt.plot(list(mash_maes.keys()), list(mash_maes.values()), label='Mash MAE', marker='o', color='blue')
    plt.fill_between(list(mash_maes.keys()),
                     np.array(list(mash_maes.values())) - np.array(list(mash_std.values())),
                     np.array(list(mash_maes.values())) + np.array(list(mash_std.values())),
                     alpha=0.2, color='blue')
    
    # Add value labels for Mash points
    for i, (x, y) in enumerate(zip(list(mash_maes.keys()), list(mash_maes.values()))):
        if i % 2 == 0:
            plt.annotate(f'{y:.4f}', (x, y), textcoords="offset points", xytext=(0, -15), ha='center', fontsize=8, color='darkblue')

    # Add a dashed line for ANI MAE, 0's everywhere
    ani_mae = np.zeros(len(SKETCH_RANGE))
    plt.axhline(y=0, color="purple", linestyle='--', label='ANI')
    plt.xlabel('Sketch Size')
    plt.ylabel('Mean Absolute Error (MAE)')
    plt.title(f'Mean Absolute Error vs Sketch Size for {PATHOGEN} ({NUM_SAMPLES} samples)')

    # Set y-axis limits to at least 0.05, but extend if data requires it
    max_upper_bound = max(
        np.max(np.array(list(dashing_maes.values())) + np.array(list(dashing_std.values()))),
        np.max(np.array(list(mash_maes.values())) + np.array(list(mash_std.values())))
    )
    y_max = max(0.05, max_upper_bound * 1.05)  # Add 5% padding above the highest point
    plt.ylim(bottom=0, top=y_max)

    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(assemblies_dir, f"{NUM_SAMPLES}_assemblies_mae_vs_sketch_size.png"))
    print(f"Saved MAE plot to {os.path.join(assemblies_dir, f'{NUM_SAMPLES}_assemblies_mae_vs_sketch_size.png')}")

    # Now generate a second plot: Time vs Sketch Size with one line for each method
    plt.figure(figsize=(10, 6))
    plt.plot(list(dashing_times.keys()), list(dashing_times.values()), label='Dashing Time', marker='o', color='green')
    plt.plot(list(mash_times.keys()), list(mash_times.values()), label='Mash Time', marker='o', color='blue')
    plt.xlabel('Sketch Size')
    plt.ylabel('Time (seconds)')
    plt.title(f'Time vs Sketch Size for {PATHOGEN} ({NUM_SAMPLES} samples)')
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(assemblies_dir, f"{NUM_SAMPLES}_assemblies_time_vs_sketch_size.png"))
    print(f"Saved Time plot to {os.path.join(assemblies_dir, f'{NUM_SAMPLES}_assemblies_time_vs_sketch_size.png')}")

