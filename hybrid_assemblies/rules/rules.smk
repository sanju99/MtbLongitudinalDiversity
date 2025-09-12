import os, glob
import numpy as np
import pandas as pd

# define some paths to make the path names more readable
sample_out_dir = f"{output_dir}/{{sample_ID}}"

scripts_dir = config["scripts_dir"]
references_dir = config["references_dir"]

conda_directory = config['conda_dir']
primary_directory = "/home/sak0914/MtbLongitudinalDiversity/hybrid_assemblies"



rule get_Illumina_FASTQ_files:
    output:
        fastq1 = f"{sample_out_dir}/FASTQ/SR.R1.fastq.gz",
        fastq2 = f"{sample_out_dir}/FASTQ/SR.R2.fastq.gz",
    params:
        SR_runID = lambda w: sample_SRrunID_dict[w.sample_ID],
        sample_out_dir = lambda w: f"{output_dir}/{w.sample_ID}/FASTQ",
        download_script = f"{primary_directory}/scripts/download_FASTQ.sh"
    shell:
        """
        module load sratoolkit/3.2.0
        
        bash {params.download_script} {params.sample_out_dir} {params.SR_runID}
        """

            
            
rule get_long_read_FASTQ_file:
    output:
        fastq = f"{sample_out_dir}/FASTQ/LR.fastq.gz",
    params:
        LR_runID = lambda w: sample_LRrunID_dict[w.sample_ID],
        sample_out_dir = lambda w: f"{output_dir}/{w.sample_ID}/FASTQ"
    shell:
        """
        module load sratoolkit/3.2.0
        
        fastq-dump --outdir {params.sample_out_dir} {params.LR_runID}
        
        # rename to LR.fastq
        mv "{params.sample_out_dir}/{params.LR_runID}.fastq" "{params.sample_out_dir}/LR.fastq"
        
        gzip "{params.sample_out_dir}/LR.fastq"
        """

            

rule subsample_and_assemble_reads:
    group: 
        "sequential"
    input:
        fastq = f"{sample_out_dir}/FASTQ/LR.fastq.gz",
    output:
        subsampled_reads = [temp(f"{sample_out_dir}/autocycler/subsampled_reads/sample_0{num}.fastq") for num in [1, 2, 3, 4]],
        assemblies_dir = directory(f"{sample_out_dir}/autocycler/assemblies"),
    params:
        subsampled_reads_dir = f"{sample_out_dir}/autocycler/subsampled_reads",
    threads:
        8
    conda:
        f"{conda_directory}/envs/autocycler.yaml"
    shell:
        """
        genome_size=$(autocycler helper genome_size --reads {input.fastq} --threads {threads})
        
        echo "Estimated genome size: $genome_size"

        # Step 1: subsample the long-read set into multiple files
        autocycler subsample --reads {input.fastq} --out_dir {params.subsampled_reads_dir} --genome_size "$genome_size"

        # Step 2: assemble each subsampled file
        for assembler in canu flye metamdbg miniasm necat nextdenovo plassembler raven; do
            echo "Assembling with $assembler"
            
            for i in 01 02 03 04; do
                autocycler helper "$assembler" --reads "{params.subsampled_reads_dir}/sample_$i.fastq" \
                                               --out_prefix "{output.assemblies_dir}/${{assembler}}_$i" \
                                               --threads {threads} \
                                               --genome_size "$genome_size"
            done
        done
        """
        
        
    
rule generate_consensus_assembly:
    group: 
        "sequential"
    input:
        fastq = f"{sample_out_dir}/FASTQ/LR.fastq.gz",
    output:
        autocycler_output = f"{sample_out_dir}/autocycler/autocycler_out",
    params:
        assemblies_dir = directory(f"{sample_out_dir}/autocycler/assemblies"),
    threads:
        8
    conda:
        f"{conda_directory}/envs/autocycler.yaml"
    shell:
        """
        # Step 3: compress the input assemblies into a unitig graph
        autocycler compress -i {params.assemblies_dir} -a {output.autocycler_output}

        # Step 4: cluster the input contigs into putative genomic sequences
        autocycler cluster -a {output.autocycler_output}

        # Steps 5 and 6: trim and resolve each QC-pass cluster
        for c in autocycler_out/clustering/qc_pass/cluster_*; do
            autocycler trim -c "$c"
            autocycler resolve -c "$c"
         done

        # Step 7: combine resolved clusters into a final assembly
        # autocycler combine -a autocycler_out -i autocycler_out/clustering/qc_pass/cluster_*/5_final.gfa
        """