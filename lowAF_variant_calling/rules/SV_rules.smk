import os, glob
import numpy as np
import pandas as pd

# define some paths to make the path names more readable
sample_out_dir = f"{output_dir}/{{sample_ID}}"

scripts_dir = config["scripts_dir"]
references_dir = config["references_dir"]

primary_directory = "/home/sak0914/MtbLongitudinalDiversity/lowAF_variant_calling"
   
        
rule call_SVs_delly:
    input:
        cram_file = lambda wildcards: sample_CRAM_dict[wildcards.sample_ID],
    output:
        SV_calls = f"{sample_out_dir}/delly/{{sample_ID}}.vcf",
    threads:
        4
    shell:
        """
        source activate /home/sak0914/anaconda3/envs/snakemake/envs/delly
        
        if [ ! -f {input.cram_file}.crai ]; then
            samtools index {input.cram_file}
        fi
        
        # determine the appropriate FASTA file by getting the name of the reference chromosome used for alignment
        chrom_name=$(samtools idxstats {input.cram_file} | cut -f1 | head -1)

        if [ "$chrom_name" = "Chromosome" ]; then
            ref_fasta="/home/sak0914/Mtb_Megapipe/references/ref_genome/H37Rv_NC_000962.3.fna"
        elif [ "$chrom_name" = "NC_000962.3" ]; then
            ref_fasta="/home/sak0914/Mtb_Megapipe/references/ref_genome/refseq.fna"
        else
            echo "CHROM name $chrom_name is invalid"
        fi
                
        delly call -g $ref_fasta {input.cram_file} > {output.SV_calls} -h {threads}
        """