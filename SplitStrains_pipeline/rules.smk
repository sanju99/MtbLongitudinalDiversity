import os, glob
import numpy as np
import pandas as pd

# define some paths to make the path names more readable
sample_out_dir = f"{output_dir}/{{sample_ID}}"

primary_directory = os.getcwd()


rule run_split_strains:
    output:
        result_file = f"{sample_out_dir}/{{sample}}.result.txt"
    params:
        sample_out_dir = sample_out_dir,
        splitStrains_python_script = os.path.join(os.path.dirname(primary_directory), "SplitStrains", "splitStrains.py"),
    conda:
        "/home/sak0914/anaconda3/envs/SplitStrains"
    shell:
        """
        python3 -u {params.splitStrains_python_script} \
            "{params.sample_out_dir}/full.dedup.bam" \
            -o {params.sample_out_dir} \
            -s 1 -e 4411532 \
            --classify > {output.result_file}
        """


rule separate_BAM_files:
    input:
        split_strain_reads_output = f"{sample_out_dir}/{{percent}}_strain.reads",
    output:
        bam_file = f"{sample_out_dir}/{{percent}}.removed.bam",
    params:
        sample_out_dir = sample_out_dir,
        rmreads_python_script = os.path.join(os.path.dirname(primary_directory), "SplitStrains", "rmreads.py"),
    conda:
        "/home/sak0914/anaconda3/envs/SplitStrains"
    shell:
        """
        python3 -u {params.rmreads_python_script} {input.split_strain_reads_output} "{params.sample_out_dir}/full.dedup.bam" {output.bam_file}
        """


rule index_and_depth_BAM_files:
    input:
        bam_file = f"{sample_out_dir}/{{percent}}.removed.bam",
    output:
        bam_index_file = f"{sample_out_dir}/{{percent}}.removed.bam.bai",
        bam_depth_file = f"{sample_out_dir}/{{percent}}.removed.depth.tsv.gz",
    conda:
        "/home/sak0914/Mtb_Megapipe/envs/read_processing_aln.yaml",
    shell:
        """
        samtools index {input.bam_file}

        # get depths at all positions (-a). Use minimum mapping quality of at least 1 for reads to contribute
        samtools depth -a -Q 1 {input.bam_file} | gzip -c > {output.bam_depth_file}
        """