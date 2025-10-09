import os, glob
import numpy as np
import pandas as pd

# define some paths to make the path names more readable
sample_out_dir = f"{output_dir}/{{sample_ID}}"

scripts_dir = config["scripts_dir"]
references_dir = config["references_dir"]

conda_directory = config['conda_dir']
primary_directory = os.getcwd()



rule kraken_classification:
    input:
        fastq = lambda wildcards: sample_FQ_dict[wildcards.sample_ID],
    output:
        kraken_report = f"{sample_out_dir}/PacBio/kraken/kraken_report_standard_DB.txt",
        kraken_classifications = f"{sample_out_dir}/PacBio/kraken/kraken_classifications_standard_DB",
    conda:
        f"{conda_directory}/envs/read_processing_aln_bwa.yaml"
    params:
        kraken_db = config['kraken_db'],
        output_dir = output_dir,
    threads:
        8
    shell:
        """
        # --confidence is the minimum fraction of k-mers in a read that must match a given taxon for that read to be assigned to that taxon
        kraken2 --db {params.kraken_db} \
                --threads {threads} \
                --confidence 0 \
                --gzip-compressed \
                --report {output.kraken_report} \
                --output {output.kraken_classifications} \
                --memory-mapping \
                {input.fastq}
        """
        
        
        
rule extract_kraken_read_names_long_reads:
    input:
        kraken_classifications = f"{sample_out_dir}/PacBio/kraken/kraken_classifications_standard_DB",
    output:  
        kraken_classifications_gzipped = f"{sample_out_dir}/PacBio/kraken/kraken_classifications_standard_DB.csv.gz", # gets gzipped by the python script. Did this to add headers
        keep_read_names = f"{sample_out_dir}/PacBio/kraken/keep_read_names.txt"
    params:
        kraken_db = config['kraken_db'],
        extract_kraken_reads_script = os.path.join(primary_directory, scripts_dir, "extract_kraken_read_names.py"),
        taxid = config['taxid'],
    shell:
        """
        python3 -u {params.extract_kraken_reads_script} \
                -t {params.taxid} \
                -d {params.kraken_db} \
                -i {input.kraken_classifications} \
                -o {output.keep_read_names} \
                --include-children \
                --include-parents
                
        rm {input.kraken_classifications}
        """



rule extract_kraken_classified_long_reads:
    input:
        fastq = lambda wildcards: sample_FQ_dict[wildcards.sample_ID],
        keep_read_names = f"{sample_out_dir}/PacBio/kraken/keep_read_names.txt"
    output:
        fastq_classified = f"{sample_out_dir}/PacBio/kraken/{{sample_ID}}.kraken.filtered.fastq.gz",
    conda:
        f"{conda_directory}/envs/read_processing_aln_bwa.yaml"
    shell:
        """
        # seqtk will write outputs to unzipped files, even if the input was compressed
        seqtk subseq {input.fastq} {input.keep_read_names} | gzip -c > {output.fastq_classified}
        
        gzip {input.keep_read_names}
        """
        


rule align_reads_to_IS6110_region:
    input:
        fastq_classified = f"{sample_out_dir}/PacBio/kraken/{{sample_ID}}.kraken.filtered.fastq.gz",
    output:
        sam_file = temp(f"{sample_out_dir}/PacBio/IS6110/{{sample_ID}}.sam"),
        bam_file = f"{sample_out_dir}/PacBio/IS6110/{{sample_ID}}.bam",
        bam_index_file = f"{sample_out_dir}/PacBio/IS6110/{{sample_ID}}.bam.bai",
        IS6110_reads_file_unzipped = temp(f"{sample_out_dir}/PacBio/IS6110/{{sample_ID}}.fastq"),
        IS6110_reads_file = f"{sample_out_dir}/PacBio/IS6110/{{sample_ID}}.fastq.gz",
    params:
        output_dir = output_dir,
        IS6110_H37Rv_sequence = os.path.join(primary_directory, references_dir, "ref_genome", "IS6110.H37Rv.fasta"),
    conda:
        f"{conda_directory}/envs/long_read_aln.yaml"
    threads:
        8
    shell:
        """
        minimap2 -ax map-hifi {params.IS6110_H37Rv_sequence} {input.fastq_classified} > {output.sam_file}

        # sort alignment and convert to bam file
        samtools view -b {output.sam_file} | samtools sort > {output.bam_file}

        # index the BAM file with samtools
        samtools index {output.bam_file}
        
        # extract reads that map to the IS6110 sequence and save
        bedtools bamtofastq -i {output.bam_file} -fq {output.IS6110_reads_file_unzipped}
        
        # gzip
        gzip -c {output.IS6110_reads_file_unzipped} > {output.IS6110_reads_file}
        """
        
        
rule align_IS6110_reads_to_H37Rv:
    input:
        IS6110_reads_file = f"{sample_out_dir}/PacBio/IS6110/{{sample_ID}}.fastq.gz",
    output:
        sam_file = temp(f"{sample_out_dir}/PacBio/IS6110/{{sample_ID}}.H37Rv.sam"),
        bam_file = f"{sample_out_dir}/PacBio/IS6110/{{sample_ID}}.H37Rv.bam",
        bam_index_file = f"{sample_out_dir}/PacBio/IS6110/{{sample_ID}}.H37Rv.bam.bai",
    params:
        output_dir = output_dir,
        personal_ref_genome = lambda w: sample_asm_dict[w.sample_ID],
    conda:
        f"{conda_directory}/envs/long_read_aln.yaml"
    threads:
        8
    shell:
        """
        minimap2 -ax map-hifi {params.personal_ref_genome} {input.IS6110_reads_file} > {output.sam_file}

        # sort alignment and convert to bam file
        samtools view -b {output.sam_file} | samtools sort > {output.bam_file}

        # index the BAM file with samtools
        samtools index {output.bam_file}
        """
        
        
rule align_IS6110_to_personal_genome:
    input:
        personal_ref_genome = lambda w: sample_asm_dict[w.sample_ID],
    output:
        unsorted_sam_file = temp(f"{sample_out_dir}/IS6110/{{sample_ID}}.unsorted.sam"),
        sam_file = f"{sample_out_dir}/IS6110/{{sample_ID}}.sam",
        bam_file = f"{sample_out_dir}/IS6110/{{sample_ID}}.bam",
        bam_index_file = f"{sample_out_dir}/IS6110/{{sample_ID}}.bam.bai",
    params:
        IS6110_H37Rv_sequence = os.path.join(primary_directory, references_dir, "ref_genome", "IS6110.H37Rv.fasta"),
    conda:
        f"{conda_directory}/envs/long_read_aln.yaml"
    threads:
        8
    shell:
        """
        # target, then query
        minimap2 -ax asm5 -X {input.personal_ref_genome} {params.IS6110_H37Rv_sequence} > {output.unsorted_sam_file}

        # sort alignment and convert to bam file
        samtools view -b {output.unsorted_sam_file} | samtools sort > {output.bam_file}

        # index the BAM file with samtools
        samtools index {output.bam_file}
        
        # save a sorted SAM file
        samtools view {output.bam_file} > {output.sam_file}
        """
        