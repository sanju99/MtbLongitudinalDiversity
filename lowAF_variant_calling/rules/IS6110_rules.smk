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
        


# extract reads that map to the IS6110 sequence and save
# bedtools bamtofastq -i {output.bam_file} -fq {output.IS6110_reads_file_unzipped}
        
        
rule align_IS6110_to_personal_genome:
    input:
        personal_ref_genome = lambda w: sample_asm_dict[w.sample_ID],
    output:
        unsorted_sam_file = temp(f"{sample_out_dir}/PacBio/IS6110/{{sample_ID}}.unsorted.sam"),
        sam_file = f"{sample_out_dir}/PacBio/IS6110/{{sample_ID}}.sam",
        bam_file = f"{sample_out_dir}/PacBio/IS6110/{{sample_ID}}.bam",
        bam_index_file = f"{sample_out_dir}/PacBio/IS6110/{{sample_ID}}.bam.bai",
    params:
        IS6110_H37Rv_sequence = "/home/sak0914/MtbLongitudinalDiversity/lowAF_variant_calling/references/ref_genome/IS6110.H37Rv.fasta",
    conda:
        f"{conda_directory}/envs/long_read_aln.yaml"
    threads:
        8
    shell:
        """
        # target, then query
        minimap2 -ax asm5 {input.personal_ref_genome} {params.IS6110_H37Rv_sequence} > {output.unsorted_sam_file}

        # sort alignment and convert to bam file
        samtools view -b {output.unsorted_sam_file} | samtools sort > {output.bam_file}

        # index the BAM file with samtools
        samtools index {output.bam_file}
        
        # save a sorted SAM file
        samtools view {output.bam_file} > {output.sam_file}
        """
        
        
        
rule convert_IS6110_coordinates_to_H37Rv:
    input:
        bam_file = f"{sample_out_dir}/PacBio/IS6110/{{sample_ID}}.bam",
        paf_file = f"{sample_out_dir}/assembly/{{sample_ID}}.H37Rv.paf",
    output:
        bed_file = f"{sample_out_dir}/PacBio/IS6110/{{sample_ID}}.bed",
        bed_file_H37Rv = f"{sample_out_dir}/PacBio/IS6110/{{sample_ID}}.H37Rv.bed",
    shell:    
       """
       source activate /home/sak0914/MtbLongitudinalDiversity/hybrid_assemblies/.snakemake/conda/8b57fc236dc2bbc64134703f48ce44cb_
       
       bedtools bamtobed  -i {input.bam_file} > {output.bed_file}
       
       paftools.js liftover {input.paf_file} {output.bed_file} | bedtools sort > {output.bed_file_H37Rv}
       """