# General pipeline:
# - Load the data
# - Process with Evo
# - Pass as the X matrix to fitmodel_nestedcv from run_baseline_models.py

### MODEL: https://huggingface.co/togethercomputer/evo-1-131k-base ###

import glob
import os
from Bio import SeqIO

from transformers import (
    AutoModel,
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoConfig,
)

model_name = 'togethercomputer/evo-1-8k-base'
config = AutoConfig.from_pretrained(model_name, trust_remote_code=True, revision="1.1_fix")
config.use_cache = True

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    config=config,
    trust_remote_code=True,
    revision="1.1_fix"
)



tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    trust_remote_code=True,
    revision="1.1_fix"
)

# Load a sample genome
path_root_fna = "/home/mila/a/aselstya/scratch/amr_genomic_data/acinetobacter_baumannii/"
baumannii1_name = "GCA_000586635.1"
baumannii1 = glob.glob(os.path.join(path_root_fna, baumannii1_name, "ncbi_dataset", "data", baumannii1_name, "*.fna"))[0]
print(f"Loaded genome: {baumannii1}")

# Tokenize the first contig
tokenized_seq = []
for record in list(SeqIO.parse(baumannii1, "fasta"))[-2:-1]: 
    seq = str(record.seq).upper()
    inputs = tokenizer(seq, return_tensors="pt")
    tokenized_seq.append(inputs)

print(tokenized_seq[0])

