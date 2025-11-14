#!/bin/bash 
#SBATCH -c 4
#SBATCH -t 0-11:59
#SBATCH -p short
#SBATCH --mem=50G
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

source activate snakemake

snakemake --snakefile rules/Marin_generatehybridASM_PacBio.smk \
          --configfile  config_hybridASM.yaml \
          --cores 8 \
          --use-conda \
          --conda-frontend mamba \
          --resources mem_mb=50000 \
          --rerun-incomplete --keep-going \
          --directory /home/sak0914/MtbLongitudinalDiversity/hybrid_assemblies \
          --conda-prefix /home/sak0914/MtbLongitudinalDiversity/hybrid_assemblies/.snakemake/conda \
          --unlock

snakemake --snakefile rules/Marin_generatehybridASM_PacBio.smk \
          --configfile  config_hybridASM.yaml \
          --cores 8 \
          --use-conda \
          --conda-frontend mamba \
          --resources mem_mb=50000 \
          --rerun-incomplete --keep-going \
          --directory /home/sak0914/MtbLongitudinalDiversity/hybrid_assemblies \
          --conda-prefix /home/sak0914/MtbLongitudinalDiversity/hybrid_assemblies/.snakemake/conda # --dry-run