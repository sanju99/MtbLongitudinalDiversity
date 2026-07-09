#!/bin/bash 
#SBATCH -c 8
#SBATCH -t 0-11:59
#SBATCH -p short
#SBATCH --mem=100G
#SBATCH -o /home/sak0914/Errors/zerrors_%j.out 
#SBATCH -e /home/sak0914/Errors/zerrors_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=skulkarni@g.harvard.edu

source activate snakemake

snakemake --snakefile rules/NTM_hybridASM.smk \
          --configfile  config_ASM_NTM.yaml \
          --cores 8 \
          --use-conda \
          --conda-frontend mamba \
          --resources mem_mb=100000 \
          --rerun-incomplete --keep-going \
          --directory /home/sak0914/MtbLongitudinalDiversity/hybrid_assemblies \
          --conda-prefix /home/sak0914/MtbLongitudinalDiversity/hybrid_assemblies/.snakemake/conda \
          --unlock

snakemake --snakefile rules/NTM_hybridASM.smk \
          --configfile  config_ASM_NTM.yaml \
          --cores 8 \
          --use-conda \
          --conda-frontend mamba \
          --resources mem_mb=100000 \
          --rerun-incomplete --keep-going \
          --directory /home/sak0914/MtbLongitudinalDiversity/hybrid_assemblies \
          --conda-prefix /home/sak0914/MtbLongitudinalDiversity/hybrid_assemblies/.snakemake/conda #--dry-run