#!/bin/bash 
#SBATCH -c 8
#SBATCH -t 4-23:59
#SBATCH -p medium
#SBATCH --mem=20G
#SBATCH -o /home/sak0914/Errors/zerrors_%j.out 
#SBATCH -e /home/sak0914/Errors/zerrors_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=skulkarni@g.harvard.edu

source activate /home/sak0914/anaconda3/envs/snakemake

snakemake --snakefile snakefile \
          --configfile config.yaml \
          --config isolates_to_run="/home/sak0914/MtbLongitudinalDiversity/lowAF_variant_calling/data/TRUST_SR_samples.tsv" \
          --use-conda --conda-frontend conda --conda-prefix /home/sak0914/Mtb_Megapipe/.snakemake/conda \
          --directory /home/sak0914/MtbLongitudinalDiversity/lowAF_variant_calling \
          --cores 8 --resources mem_mb=20000 \
          --rerun-incomplete --keep-going \
          --unlock


snakemake --snakefile snakefile \
          --configfile config.yaml \
          --use-conda --conda-frontend conda --conda-prefix /home/sak0914/Mtb_Megapipe/.snakemake/conda \
          --config isolates_to_run="/home/sak0914/MtbLongitudinalDiversity/lowAF_variant_calling/data/TRUST_SR_samples.tsv" \
          --directory /home/sak0914/MtbLongitudinalDiversity/lowAF_variant_calling \
          --cores 8 --resources mem_mb=20000 \
          --rerun-incomplete --keep-going #--dry-run #--allowed-rules write_lowAF_SNPs #--dry-run ##freebayes_VCF_normalization_decomposition excludeLowConf_regions_freebayes_VCF write_lowAF_SNPs #--dry-run