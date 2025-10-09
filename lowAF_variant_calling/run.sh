#!/bin/bash 
#SBATCH -c 1
#SBATCH -t 0-11:59
#SBATCH -p short
#SBATCH --mem=10G
#SBATCH -o /home/sak0914/Errors/zerrors_%j.out 
#SBATCH -e /home/sak0914/Errors/zerrors_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=skulkarni@g.harvard.edu

source activate bayesian_modeling

# personal_ref_dir="/n/data1/hms/dbmi/farhat/Sanjana/TRUST_aln_personal_assembly"

# for sample_dir in "$personal_ref_dir"/*; do
#     sample=$(basename "$sample_dir")
        
#     python3 -u ~/MtbLongitudinalDiversity/lowAF_variant_calling/variantDetector/02_combine_lowAF_variants.py \
#             -s $sample \
#             -bed1 "$personal_ref_dir/$sample/freebayes/lowAF_variants.bed" \
#             -bed2 "$personal_ref_dir/$sample/freebayes/lowAF_variants.H37Rv.excludeLowConf.bed" \
#             -tsv /n/data1/hms/dbmi/farhat/Sanjana/TRUST_lowAF/$sample/freebayes/$sample.excludeLowConf.tsv
            
#     echo "Finished $sample"
                
# done

H37Rv_ref_dir="/n/data1/hms/dbmi/farhat/Sanjana/TRUST_lowAF"

for sample_dir in "$H37Rv_ref_dir"/*; do

    sample=$(basename "$sample_dir")
    
    python3 -u ~/MtbLongitudinalDiversity/lowAF_variant_calling/variantDetector/03_get_alignment_stats.py \
            -s $sample \
            -b $H37Rv_ref_dir/$sample/bam/$sample.dedup.bam \
            -v $H37Rv_ref_dir/$sample/freebayes/$sample.excludeLowConf.tsv
    
    echo "Finished $sample"
    # exit
    
done