import numpy as np
import pandas as pd
from Bio import Seq, SeqIO
import tracemalloc, argparse, os

tracemalloc.start()

parser = argparse.ArgumentParser()

# Add a required string argument for the paths file
parser.add_argument("-i", type=str, dest='INPUT_FASTA', help='FASTA file of genome SNP concatenate', required=True)
parser.add_argument('-o', type=str, dest='OUT_FASTA', help='Output FASTA file for the sites with SNPs', required=True)

cmd_line_args = parser.parse_args()

# required arguments
INPUT_FASTA = cmd_line_args.INPUT_FASTA
OUT_FASTA = cmd_line_args.OUT_FASTA

# Read all sequences into memory once (dict: sample_id -> string) and exclude H37Rv if it is there
seqs = {rec.id: str(rec.seq) for rec in SeqIO.parse(INPUT_FASTA, "fasta") if 'h37rv' not in rec.id.lower()}

samples = list(seqs.keys())
length = len(next(iter(seqs.values())))  # sequence length

# Create per-sample lists for filtered sequences
filtered = {s: [] for s in samples}

# Iterate site-by-site
for pos, bases in enumerate(zip(*seqs.values()), start=1):
    unique = set(bases)
    
    # Drop constant sites and Drop sites with '-' or 'N'
    if len(unique) != 1 and '-' not in unique and 'N' not in unique:

        # Keep site: append base to each sample's filtered sequence
        for s, b in zip(samples, bases):
            filtered[s].append(b)

print(f"Kept {len(filtered[samples[0]])}/{length} sites")

# Write output FASTA
with open(OUT_FASTA, "w") as out:
    for s in samples:
        seq_str = ''.join(filtered[s])
        out.write(f">{s}\n{seq_str}\n")

# returns a tuple: current, peak memory in bytes 
script_memory = tracemalloc.get_traced_memory()[1] / 1e9
tracemalloc.stop()
print(f"    {script_memory} GB\n")