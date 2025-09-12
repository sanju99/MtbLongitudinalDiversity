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

            

rule hifiasm_assembly:
    group: 
        "sequential"
    input:
        fastq = f"{sample_out_dir}/FASTQ/LR.fastq.gz",
    output:
        assembly = f"{sample_out_dir}/hifiasm/p_ctg.fasta",
    params:
        hifiasm_dir = directory(f"{sample_out_dir}/hifiasm")
    threads:
        8
    conda:
        f"{conda_directory}/envs/hifiasm.yaml"
    shell:
        """
        # -l0 for haploid genomes. -f0 for small genomes.
        hifiasm -o {output.hifiasm_dir}/ -t {threads} {input.fastq} -l0 -f0
        
        # convert the graph assembly file to a FASTA file
        awk '/^S/{print ">"$2;print $3}' "{params.hifiasm_dir}/p_ctg.gfa" > {output.assembly}
        
        # do the same if there are any alternative assemblies
        if [ -f "{output.hifiasm_dir}/a_ctg.gfa" ]; then
            awk '/^S/{print ">"$2;print $3}' "{params.hifiasm_dir}/a_ctg.gfa" > "{params.hifiasm_dir}/a_ctg.fasta"
        
            # align alternative assembly to primary assemblies
            minimap2 -o alternate_to_primary_aln.paf {output.assembly} "{params.hifiasm_dir}/a_ctg.fasta"
        fi
        """