## Antimicrobial Resistance Prediction

This is the official codebase for predicting antimicrobial resistance (AMR) using standard ML classifiers and genomic language models. This is a joint collaboration between Mila (Chandar Research Lab) and BioMérieux. 

### Getting Started

The repository is structured as follows:
* `config` directory contains the config files for different dataset combinations and ML models from scikit-learn. We use [`hydra`](https://hydra.cc/docs/intro/) for running experiments 
* `data` directory contains the code for downloading and preprocessing the data. Preprocessing includes reinterpreting the MIC values associated with public phenotypic data according to CLSI-defined breakpoints for a given list of bacterial species. 
  * In addition to random splits, we also create phylogeny-based splits based on intra-species genomic distances.  
* `features` directory contains the code for creating the feature matrix to train classical ML models. In this folder, `kmer.py` is the most important script that extracts k-mers input genomic sequences and outputs a feature matrix.
* `training` directory contains utility scripts used during training
* `testing` directory contains the script used to run inference using the best model.

### How to Use

#### Step 0: Setting up the Environment

* Clone the repository
```bash
git clone https://github.com/chandar-lab/amr-pred.git
git checkout dev
```

<details>

<summary> Installing dependencies with uv </summary>

* Install [uv](https://docs.astral.sh/uv/getting-started/installation/)
* Run `uv sync`. This will install all dependencies and create a virtual environment `amr-pred` within the repository.
* Activate the environment using `source <path_to_repo>/.venv/bin/activate`
* To deactivate the environment, use `deactivate`

</details>


<details>

<summary> Installing dependencies without uv </summary>

Without `uv`, you can create a conda environment and install from pyproject.toml using `pip install -e .`
```bash
conda create -n amr-pred python=3.12 -y
conda activate amr-pred
pip install -e .
```

</details>


#### Step 1: Preprocessing

* Download the isolates database from the NCBI website (https://www.ncbi.nlm.nih.gov/pathogens/isolates/#) in `.tsv` format.
* From NCBI's [AST browser](https://www.ncbi.nlm.nih.gov/pathogens/ast#), download the dataset for each species. Click "Filters" -> "Organism group" (e.g. staphylococcus aureus) -> "Resistance phenotype" (check-box SUSCEPTIBLE, RESISTANT, NOT DEFINED). Download the dataset in `.csv` format. 
  * This step downloads the data for all antibiotics available for the given organism group. We will use this `.csv` to reinterpret the data according to MIC values in the next step.
* Run the `data/reinterpret_data_with_mic.py` script. This script creates individual specie x drug datasets based on the available MIC breakpoints in the breakpoints file. 
* Run `data/download_dataset.py` to download the raw genomic sequences from the NCBI database. Before running this script, make sure that `ncbi_datasets` python package is installed. See installation instructions [here](https://www.ncbi.nlm.nih.gov/datasets/docs/v2/command-line-tools/download-and-install/).


#### Step 2: Running Inference

* Run `testing/test_model.py` for running model inference. Do NOT change the format of the model name when running inference. This script takes preprocessed csv file (dataset) as input, runs the model and saves: (i) all unique kmers, (ii) feature matrix, (iii) individual predictions and labels (`y_true` and `y_pred`), and (iv) classification report (sklearn). 