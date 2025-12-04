import pysam
import argparse
import pandas as pd
import os

parser = argparse.ArgumentParser()

# Add a required string argument for the paths file
parser.add_argument("-i", type=str, dest='INPUT_BAM', help='BAM file annotated with haplotype groups (HP tag)', required=True)

cmd_line_args = parser.parse_args()
INPUT_BAM = cmd_line_args.INPUT_BAM
OUT_DIR = os.path.dirname(INPUT_BAM)

# Open BAM for reading
bam_in = pysam.AlignmentFile(INPUT_BAM, "rb")

haplotypes_reads = {}
no_haplotype_reads = []

# Loop through all reads
for read in bam_in:

    if read.has_tag("HP"):
        hp = str(read.get_tag("HP"))
        
        if hp not in haplotypes_reads.keys():
            haplotypes_reads[hp] = [read.query_name]
        else:
            haplotypes_reads[hp] += [read.query_name]
    else:
        no_haplotype_reads.append(read.query_name)
        
for hp, reads_lst in haplotypes_reads.items():
    print(f"Haplotype {hp}: {len(reads_lst)} reads")
    pd.Series(reads_lst).to_csv(f"{OUT_DIR}/haplotype_{hp}_read_names.txt", sep='\t', header=None, index=False)
    
print(f"{len(no_haplotype_reads)} reads without a haplotype")
pd.Series(no_haplotype_reads).to_csv(f"{OUT_DIR}/no_haplotype_read_names.txt", sep='\t', header=None, index=False)