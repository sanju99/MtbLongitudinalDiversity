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
          --configfile config.yaml \
          --use-conda --conda-frontend conda --conda-prefix /home/sak0914/Mtb_Megapipe/.snakemake/conda \
          --config isolates_to_run="/home/sak0914/MtbLongitudinalDiversity/lowAF_variant_calling/data/personal_assemblies_samples.tsv" \
          output_dir="/n/data1/hms/dbmi/farhat/Sanjana/TRUST_aln_personal_assembly" \
          --directory /home/sak0914/MtbLongitudinalDiversity/lowAF_variant_calling \
          --cores 8 --resources mem_mb=100000 \
          --rerun-incomplete --keep-going \
          --unlock


snakemake --snakefile snakefile \
          --configfile config.yaml \
          --use-conda --conda-frontend conda --conda-prefix /home/sak0914/Mtb_Megapipe/.snakemake/conda \
          --config isolates_to_run="/home/sak0914/MtbLongitudinalDiversity/lowAF_variant_calling/data/personal_assemblies_samples.tsv" \
          output_dir="/n/data1/hms/dbmi/farhat/Sanjana/TRUST_aln_personal_assembly" \
          --directory /home/sak0914/MtbLongitudinalDiversity/lowAF_variant_calling \
          --cores 8 --resources mem_mb=100000 \
          --rerun-incomplete --keep-going #--dry-run #--allowed-rules align_reads_mark_duplicates freebayes_variant_calling liftoff_genes_from_H37Rv filter_high_quality_lowAF_variants liftover_variants_from_personal_genome_coords_to_H37Rv_coords #--dry-run