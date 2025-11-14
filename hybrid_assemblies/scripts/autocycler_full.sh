#!/usr/bin/env bash

# This script is a wrapper for running a fully-automated Autocycler assembly.
# Usage:
#   autocycler_full.sh <read_fastq> <threads> <jobs> [read_type] [output_dir]
#
# Example:
#   autocycler_full.sh reads.fastq 32 4 ont_r10 results/
#
# Copyright 2025 Ryan Wick (rrwick@gmail.com)
# Licensed under the GNU General Public License v3.

set -e  # exit on error

# -------------------------------
# Get command-line arguments
# -------------------------------
reads=$1                 # input reads FASTQ
threads=$2               # threads per job
jobs=$3                  # number of simultaneous jobs
read_type=$4  # optional, defaults to ont_r10
output_dir=$5  # optional, defaults to autocycler_outdir

# -------------------------------
# Validate inputs
# -------------------------------
if [[ -z "$reads" || -z "$threads" || -z "$jobs" ]]; then
    echo "Usage: $0 <read_fastq> <threads> <jobs> [read_type] [output_dir]" 1>&2
    exit 1
fi
if [[ ! -f "$reads" ]]; then
    echo "Error: Input file '$reads' does not exist." 1>&2
    exit 1
fi
if (( threads > 128 )); then threads=128; fi
case $read_type in
    ont_r9|ont_r10|pacbio_clr|pacbio_hifi) ;;
    *) echo "Error: read_type must be one of: ont_r9, ont_r10, pacbio_clr, pacbio_hifi" 1>&2; exit 1 ;;
esac

# Create output directory and subfolders
mkdir -p "$output_dir"
cd "$output_dir"

# -------------------------------
# Step 0: Estimate genome size
# -------------------------------
genome_size=$(autocycler helper genome_size --reads "$reads" --threads "$threads")

# -------------------------------
# Step 1: Subsample reads
# -------------------------------
mkdir -p subsampled_reads
autocycler subsample \
    --reads "$reads" \
    --out_dir subsampled_reads \
    --genome_size "$genome_size" \
    2>> autocycler.stderr

# -------------------------------
# Step 2: Assemble subsampled files
# -------------------------------
mkdir -p assemblies
rm -f assemblies/jobs.txt

for assembler in raven myloasm miniasm flye metamdbg necat nextdenovo plassembler canu; do
    for i in 01 02 03 04; do
        echo "autocycler helper $assembler \
            --reads subsampled_reads/sample_$i.fastq \
            --out_prefix assemblies/${assembler}_$i \
            --threads $threads \
            --genome_size $genome_size \
            --read_type $read_type \
            --min_depth_rel 0.1" >> assemblies/jobs.txt
    done
done

set +e
nice -n 19 parallel \
    --jobs "$jobs" \
    --joblog assemblies/joblog.tsv \
    --results assemblies/logs \
    --timeout 8h < assemblies/jobs.txt
set -e

# -------------------------------
# Adjust weights for clustering/consensus
# -------------------------------
shopt -s nullglob
for f in assemblies/plassembler*.fasta; do
    sed -i 's/circular=True/circular=True Autocycler_cluster_weight=3/' "$f"
done
for f in assemblies/canu*.fasta assemblies/flye*.fasta; do
    sed -i 's/^>.*$/& Autocycler_consensus_weight=2/' "$f"
done
shopt -u nullglob

# Remove subsampled reads to save space
rm -f subsampled_reads/*.fastq

# -------------------------------
# Step 3–7: Run core Autocycler pipeline
# -------------------------------
autocycler compress -i assemblies -a autocycler_out 2>> autocycler.stderr
autocycler cluster -a autocycler_out 2>> autocycler.stderr

for c in autocycler_out/clustering/qc_pass/cluster_*; do
    autocycler trim -c "$c" 2>> autocycler.stderr
    autocycler resolve -c "$c" 2>> autocycler.stderr
done

autocycler combine \
    -a autocycler_out \
    -i autocycler_out/clustering/qc_pass/cluster_*/5_final.gfa \
    2>> autocycler.stderr