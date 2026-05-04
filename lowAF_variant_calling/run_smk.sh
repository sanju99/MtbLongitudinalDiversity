#!/bin/bash 
#SBATCH -c 1
#SBATCH -t 0-11:59
#SBATCH -p short
#SBATCH --mem=50G
#SBATCH -o /home/sak0914/Errors/zerrors_%j.out 
#SBATCH -e /home/sak0914/Errors/zerrors_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=skulkarni@g.harvard.edu

source activate /home/sak0914/anaconda3/envs/snakemake

snakemake --snakefile snakefile \
          --configfile config.yaml \
          --config isolates_to_run="/home/sak0914/MtbLongitudinalDiversity/lowAF_variant_calling/data/Roger_samples_to_run.tsv" output_dir="/n/data1/hms/dbmi/farhat/Sanjana/Vargas_et_al_lowFreq_samples" \
          --use-conda --conda-frontend conda --conda-prefix /home/sak0914/Mtb_Megapipe/.snakemake/conda \
          --directory /home/sak0914/MtbLongitudinalDiversity/lowAF_variant_calling \
          --cores 8 --resources mem_mb=25000 \
          --rerun-incomplete --keep-going \
          --unlock


snakemake --snakefile snakefile \
          --configfile config.yaml \
          --use-conda --conda-frontend conda --conda-prefix /home/sak0914/Mtb_Megapipe/.snakemake/conda \
          --config isolates_to_run="/home/sak0914/MtbLongitudinalDiversity/lowAF_variant_calling/data/Roger_samples_to_run.tsv" output_dir="/n/data1/hms/dbmi/farhat/Sanjana/Vargas_et_al_lowFreq_samples" \
          --directory /home/sak0914/MtbLongitudinalDiversity/lowAF_variant_calling \
          --cores 8 --resources mem_mb=25000 \
          --rerun-incomplete --keep-going --allowed-rules get_pilon_SNPs_indels exclude_low_conf_regions_pilon convert_BAM_to_CRAM filter_pilon_high_quality_lowAF_SNVs bgzip_tabix_vcf_file add_aln_stats_to_pilon_SNPs add_aln_stats_to_freebayes_SNVs #--dry-run  #filter_pilon_high_quality_lowAF_SNVs bgzip_tabix_vcf_file add_aln_stats_to_pilon_SNPs #--dry-run