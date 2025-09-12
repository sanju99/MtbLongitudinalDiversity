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

OUT_CSV = OUT_FASTA.replace('.fasta', '.csv')

# exclude H37Rv if it is there
fasta_file = pd.DataFrame([(seq.id, seq.seq) for seq in SeqIO.parse(INPUT_FASTA, "fasta") if 'h37rv' not in seq.id.lower()])
    
genome_matrix = []

for i, row in fasta_file.iterrows():

    genome_matrix.append(np.array(row[1]))

    if i % 100 == 0:
        print(f"Adding {i}: genome {row[0]} to matrix")

genome_matrix = np.array(genome_matrix)

# convert to dataframe
genome_matrix = pd.DataFrame(genome_matrix)
genome_matrix.index = fasta_file[0].values

# 1-index the columns so that they are the actual position in the H37Rv coordinate system
genome_matrix.columns = genome_matrix.columns + 1

num_unique_nucs_per_site = pd.DataFrame(genome_matrix.nunique(axis=0)).reset_index()
num_unique_nucs_per_site.columns = ['site', 'num_unique']
keep_sites = num_unique_nucs_per_site.query("num_unique > 1").site.values

print(genome_matrix.shape)
print(f"Keeping {len(keep_sites)}/{genome_matrix.shape[1]} sites that are not constant")

print(keep_sites)

# keep the header to know which sites were kept
genome_matrix[keep_sites].to_csv(OUT_CSV)


##########################################################################################################################################


genome_matrix = pd.read_csv(OUT_CSV, index_col=[0])

# remove deletions because we only want to consider SNPs
drop_cols_deletions = genome_matrix.columns[np.unique(np.where(genome_matrix == '-')[1])]

# remove ambiguous calls because they are so rare that they are causing some ambiguously constant sites. Not correct to encode them as something else, so just drop those sites?
drop_cols_missing = genome_matrix.columns[np.unique(np.where(genome_matrix == 'N')[1])]

combined_drop_cols = list(set(drop_cols_deletions).union(drop_cols_missing))
print(f"Dropping {len(combined_drop_cols)} sites with missing calls or deletions in any sample")

genome_matrix_keep = genome_matrix.drop(combined_drop_cols, axis=1)
del genome_matrix

keep_cols = genome_matrix_keep.columns

# overwrite the old one. That's just a placeholder because the top step takes longest, so to save space if you need to rerun
genome_matrix_keep.to_csv(OUT_CSV)

print(f"Writing {genome_matrix_keep.shape[0]} samples with {genome_matrix_keep.shape[1]} SNPs")

with open(OUT_FASTA, "w+") as file:

    for sample in genome_matrix_keep.index.values:
    
        # combine all the nucleotides into a single string
        snp_concatenate = ''.join(genome_matrix_keep.loc[sample, :])

        file.write(f">{sample}\n")
        file.write(f"{snp_concatenate}\n")

# returns a tuple: current, peak memory in bytes 
script_memory = tracemalloc.get_traced_memory()[1] / 1e9
tracemalloc.stop()
print(f"    {script_memory} GB\n")