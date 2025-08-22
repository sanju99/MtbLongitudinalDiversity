#!/bin/bash 
#SBATCH -c 8
#SBATCH -t 4-23:59
#SBATCH -p medium
#SBATCH --mem=100G
#SBATCH -o /home/sak0914/Errors/zerrors_%j.out 
#SBATCH -e /home/sak0914/Errors/zerrors_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=skulkarni@g.harvard.edu

source activate snakemake

snakemake --snakefile snakefile \
          --use-conda --conda-frontend conda --conda-prefix /home/sak0914/Mtb_Megapipe/.snakemake/conda \
          --configfile config.yaml \
          --directory /home/sak0914/longitudinal_changes \
          --cores 8 --resources mem_mb=100000 \
          --rerun-incomplete --keep-going \
          --unlock


snakemake --snakefile snakefile \
          --use-conda --conda-frontend conda --conda-prefix /home/sak0914/Mtb_Megapipe/.snakemake/conda \
          --configfile config.yaml \
          --directory /home/sak0914/longitudinal_changes \
          --cores 8 --resources mem_mb=100000 \
          --rerun-incomplete --keep-going #--dry-run