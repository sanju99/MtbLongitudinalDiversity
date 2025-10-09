#!/bin/bash 
#SBATCH -c 2
#SBATCH -t 0-11:59
#SBATCH -p short
#SBATCH --mem=50G
#SBATCH -o /home/sak0914/Errors/zerrors_%j.out 
#SBATCH -e /home/sak0914/Errors/zerrors_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=skulkarni@g.harvard.edu

source activate snakemake

fName_dir="/home/sak0914/MtbLongitudinalDiversity/lowAF_variant_calling/variantDetector"

fNames=("SR_samples_align_3,3.1.1_MFS-135_ASM.txt" 
        "SR_samples_align_4.1.1.1_MFS-136_ASM.txt"
        "SR_samples_align_4.1.1.3_MFS-56_ASM.txt"
        "SR_samples_align_4.1.2_MFS-308_ASM.txt"
        "SR_samples_align_4.3.2.1_MFS-160_ASM.txt"
        )

for i in "${!fNames[@]}"; do
    
    fName="${fNames[$i]}"
    sample=$(grep -o 'MFS-[0-9]\+' <<< "$fName")

    echo "Aligning samples in $fName to $sample"

    snakemake --snakefile snakefile \
              --configfile config.yaml \
              --use-conda --conda-frontend conda --conda-prefix /home/sak0914/Mtb_Megapipe/.snakemake/conda \
              --config isolates_to_run="$fName_dir/$fName" \
              output_dir="/n/data1/hms/dbmi/farhat/Sanjana/TRUST_lowAF" assembly_sample=$sample \
              --directory /home/sak0914/MtbLongitudinalDiversity/lowAF_variant_calling \
              --cores 8 --resources mem_mb=25000 \
              --rerun-incomplete --keep-going \
              --unlock


    snakemake --snakefile snakefile \
              --configfile config.yaml \
              --use-conda --conda-frontend conda --conda-prefix /home/sak0914/Mtb_Megapipe/.snakemake/conda \
               --config isolates_to_run="$fName_dir/$fName" \
              output_dir="/n/data1/hms/dbmi/farhat/Sanjana/TRUST_lowAF" assembly_sample=$sample \
              --directory /home/sak0914/MtbLongitudinalDiversity/lowAF_variant_calling \
              --cores 8 --resources mem_mb=25000 \
              --rerun-incomplete --keep-going #--dry-run # --allowed-rules filter_high_quality_lowAF_variants liftover_variants_between_two_personal_ref_genome_coords exclude_regions_from_other_personal_ref_genome --dry-run

          
done