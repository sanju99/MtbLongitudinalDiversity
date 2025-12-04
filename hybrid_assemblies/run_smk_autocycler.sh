#!/bin/bash 
#SBATCH -c 16
#SBATCH -t 4-23:59
#SBATCH -p medium
#SBATCH --mem=100G
#SBATCH -o /home/sak0914/Errors/zerrors_%j.out 
#SBATCH -e /home/sak0914/Errors/zerrors_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=skulkarni@g.harvard.edu

# source activate snakemake

# snakemake --snakefile snakefile \
#           --use-conda --conda-frontend conda --conda-prefix /home/sak0914/Mtb_Megapipe/.snakemake/conda \
#           --configfile config.yaml \
#           --directory /home/sak0914/MtbLongitudinalDiversity/hybrid_assemblies \
#           --cores 8 --resources mem_mb=100000 \
#           --rerun-incomplete --keep-going \
#           --unlock


# snakemake --snakefile snakefile \
#           --use-conda --conda-frontend conda --conda-prefix /home/sak0914/Mtb_Megapipe/.snakemake/conda \
#           --configfile config.yaml \
#           --directory /home/sak0914/MtbLongitudinalDiversity/hybrid_assemblies \
#           --cores 8 --resources mem_mb=100000 \
#           --rerun-incomplete --keep-going --dry-run

# mamba env create -f /home/sak0914/MtbLongitudinalDiversity/hybrid_assemblies/envs/autocycler.yaml

source activate snakemake

snakemake --snakefile rules/run_autocycler.smk \
          --configfile config_hybridASM_autocycler.yaml \
          --cores 16 \
          --use-conda \
          --conda-frontend mamba \
          --resources mem_mb=100000 \
          --rerun-incomplete --keep-going \
          --directory /home/sak0914/MtbLongitudinalDiversity/hybrid_assemblies \
          --unlock
          # --conda-prefix /home/sak0914/MtbLongitudinalDiversity/hybrid_assemblies/.snakemake/conda \


snakemake --snakefile rules/run_autocycler.smk \
          --configfile config_hybridASM_autocycler.yaml \
          --cores 16 \
          --use-conda \
          --conda-frontend mamba \
          --resources mem_mb=100000 \
          --rerun-incomplete --keep-going \
          --directory /home/sak0914/MtbLongitudinalDiversity/hybrid_assemblies #--dry-run
          # --conda-create-envs-only #--conda-prefix /home/sak0914/MtbLongitudinalDiversity/hybrid_assemblies/.snakemake/conda #--dry-run