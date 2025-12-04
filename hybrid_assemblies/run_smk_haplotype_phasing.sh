#!/bin/bash 
#SBATCH -c 4
#SBATCH -t 0-11:59
#SBATCH -p short
#SBATCH --mem=50G
#SBATCH -o /home/sak0914/Errors/zerrors_%j.out 
#SBATCH -e /home/sak0914/Errors/zerrors_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=skulkarni@g.harvard.edu

source activate /home/sak0914/anaconda3/envs/snakemake

snakemake --snakefile rules/haplotype_phasing.smk \
          --configfile  config_haplotype_phasing.yaml \
          --cores 8 \
          --use-conda \
          --conda-frontend mamba \
          --resources mem_mb=50000 \
          --rerun-incomplete --keep-going \
          --directory /home/sak0914/MtbLongitudinalDiversity/hybrid_assemblies \
          --conda-prefix /home/sak0914/MtbLongitudinalDiversity/hybrid_assemblies/.snakemake/conda \
          --unlock

snakemake --snakefile rules/haplotype_phasing.smk \
          --configfile  config_haplotype_phasing.yaml \
          --cores 8 \
          --use-conda \
          --conda-frontend mamba \
          --resources mem_mb=50000 \
          --rerun-incomplete --keep-going \
          --directory /home/sak0914/MtbLongitudinalDiversity/hybrid_assemblies \
          --conda-prefix /home/sak0914/MtbLongitudinalDiversity/hybrid_assemblies/.snakemake/conda # --dry-run