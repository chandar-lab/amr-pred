## Antimicrobial Resistance Prediction

This is the official codebase for predicting antimicrobial resistance (AMR) using standard ML classifiers. This is a joint collaboration between the Chandar Research Lab at Mila and BioMérieux. 

### Getting Started

The repository is structured as follows:
* `config` directory contains the config files for different dataset combinations and ML models from scikit-learn. We use [`hydra`](https://hydra.cc/docs/intro/) for running experiments 
* `data` directory contains the code for downloading and preprocessing the data. Preprocessing includes reinterpreting the MIC values associated with public phenotypic data according to CLSI-defined breakpoints for a given list of bacterial species. 
  * In addition to random splits, we also create phylogeny-based splits based on intra-species genomic distances.  
* `features` directory contains the code for creating the feature matrix to train classical ML models. In this folder, `kmer.py` is the most important script that extracts k-mers input genomic sequences and outputs a feature matrix.
* `training` directory contains utility scripts used during training
* `testing` directory contains the script used to run inference using saved models.

### How to Use

#### Step 0: Setting up the Environment

* Clone the repository
```bash
git clone https://github.com/chandar-lab/amr-pred.git
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

<details>
<summary>Generating phylogenetic folds</summary>

In order to generate phylogenetic folds, you will have to download at least one of the following genomic distance tools. They are most easily installed using anaconda, so the recommended method is to create an `amr-utils` conda environment that is just used for genomic distance tools:
* dashing: `conda install bioconda::dashing==0.4.0`
* mash: `conda install bioconda::mash==2.1`
</details>


#### Step 1: Preprocessing

* Download the isolates database from the NCBI website (https://www.ncbi.nlm.nih.gov/pathogens/isolates/#) in `.tsv` format.
* From NCBI's [AST browser](https://www.ncbi.nlm.nih.gov/pathogens/ast#), download the dataset for each species. Click "Filters" -> "Organism group" (e.g. staphylococcus aureus) -> "Resistance phenotype" (check-box SUSCEPTIBLE, RESISTANT, NOT DEFINED). Download the dataset in `.csv` format. 
  * This step downloads the data for all antibiotics available for the given organism group. We will use this `.csv` to reinterpret the data according to MIC values in the next step.
* Run the `data/reinterpret_data_with_mic.py` script. This script creates individual specie x drug datasets based on the available MIC breakpoints in the breakpoints file. 
  * This script requires three command line arguments to run, as well as hydra config information from `config/main.yaml`. It processes one species per run, so the following arguments must be passed: `--path-clsi-bps` (the location of the CLSI breakpoints file), `--path-data` (the path to the .csv file downloaded from the NCBI for the target species), and `--path-isolates`, the path to the .tsv file downloaded from the NCBI. `path_datasets_reinterpreted` from the hydra config is used as the output of the script.
* Run `data/download_dataset.py` to download the raw genomic sequences from the NCBI database. Before running this script, make sure that `ncbi_datasets` python package is installed. See installation instructions [here](https://www.ncbi.nlm.nih.gov/datasets/docs/v2/command-line-tools/download-and-install/).
  * Although this script takes a species-antibiotic combination as input via the hydra config, it will download all sequences for the given species. 
  * To change the target species, use `--run dataset=<difficulty>/<dataset_name>` when running the script, using on the files in the `config/dataset` directory.


#### Step 2: Training a Model

* Models are trained using the `train_model.py` script in the root directory, and configuration is specified using hydra overrides.
* The primary override necessary is `dataset=<difficulty>/<dataset_name>`, which is used to specify the species-antibiotic combination that will be used for training.
* An example line that will launch a training job for a_baumannii__ciprofloxacin using 1 million features, a k-mer length of 17, random evaluation folds and no sweeps is: 

```bash
python ./amr-pred/train_model.py --run dataset=easy/a_baumannii__ciprofloxacin matrix.k=17 matrix.max_features=1_000_000 use_genomic_splits=False use_sweeps=False
```
* Additional settings can be found by inspecting the config files in `config/`. Hyperparameter tuning ranges are also contained therein.
* To train a model on a species-drug combination that is not one of the nine included in our study, simply copy the format used by one of the given combinations into a new YAML file and change the relevant fields.


#### Step 3: Running Inference

* Run `testing/test_model.py` for running model inference. Do NOT change the format of the model name when running inference. This script takes preprocessed csv file (dataset) as input, runs the model and saves: (i) all unique kmers, (ii) feature matrix, (iii) individual predictions and labels (`y_true` and `y_pred`), and (iv) classification report (sklearn). 
* Additional files `explain_model.py` and `kmer_stats.py` are mirror the format of `test_model.py`, but are used for interpretability k-mer extraction and dataset analysis, respectively. 