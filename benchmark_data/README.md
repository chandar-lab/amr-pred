# Benchmark data

The datasets used in our study, for the 9 species x antibiotic combinations in `config/dataset/`. The default
paths in `config/main.yaml` point here, so training uses these files without any setup apart from downloading the
genomes (`data/download_dataset.py`). The trained models are released separately on Zenodo (`python testing/download_models.py`).

```
benchmark_data/
├── clsi_breakpoints_20250527.csv      # CLSI MIC breakpoints used to label the samples
├── train/<species>/filtered/<species>__<drug>_SR_filt.csv
├── folds/<species>/<species>__<drug>__dashing__phyloFolds_hierarchical.csv
└── test/<species>/<species>__<drug>_SR_filt.csv
```

## Training datasets (`train/`)

Phenotypes exported from the [NCBI AST browser](https://www.ncbi.nlm.nih.gov/pathogens/ast) on 2025-07-22
(records created up to 2025-07-19), matched to their genome assemblies with the
[NCBI isolates browser](https://www.ncbi.nlm.nih.gov/pathogens/isolates) export of the same date, and processed with
`data/reinterpret_data_with_mic.py`:

- the samples are labelled from their MIC values with the CLSI breakpoints (`Phenotype_revised`: `resistant` if
  MIC >= the R breakpoint, `susceptible` if MIC <= the S breakpoint); samples between the breakpoints are dropped.
  `MIC_revised` is the MIC corrected for the measurement sign (doubled for `>`, halved for `<`);
- isolates without an assembly, and Shigella isolates (for E. coli), are removed;
- each assembly appears once: assemblies tested several times with conflicting labels are dropped.

The columns are those of the NCBI AST export, plus `Assembly`, `MIC_revised` and `Phenotype_revised` (the label).

## Phylogenetic folds (`folds/`)

The assignment of each training assembly to one of the 5 cross-validation folds (`assembly`, `fold`), from Ward
hierarchical clustering of the pairwise [dashing](https://github.com/dnbaker/dashing) distances between the genomes
(`data/generate_phylogeny_folds.py`). Related genomes are kept in the same fold, so each test fold holds clades unseen
in training. They are used when `use_genomic_splits=True` with the default `split_method` (`dashing`),
`clustering_method` (`hierarchical`) and `num_folds` (5); other settings compute new folds.

## Test datasets (`test/`)

A temporal hold-out set: NCBI AST browser records created between 2025-07-28 and 2026-09-22 (exported on
2026-11-05), processed in the same way as the training data. No isolate, BioSample or assembly is shared with the
training data. The testing scripts (`testing/test_model.py`) take these files as `--path-csv`, and recompute the
labels from the MIC values with the breakpoints file.

## CLSI breakpoints

`clsi_breakpoints_20250527.csv` (`;`-separated) holds the CLSI susceptible (`BP_S_CLSI (<=)`) and resistant
(`BP_R_CLSI (>=)`) MIC breakpoints (mg/L) per species and drug, from:

> Lewis II J. CLSI M100, 35th edition. Clinical and Laboratory Standards Institute, 2025.
