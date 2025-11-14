import os, glob
import numpy as np
import pandas as pd

# define some paths to make the path names more readable
sample_out_dir = f"{output_dir}/{{sample_ID}}"

scripts_dir = config["scripts_dir"]
references_dir = config["references_dir"]

conda_directory = config['conda_dir']
primary_directory = os.getcwd()

sample_H37Rv_ref_dir = f"/n/data1/hms/dbmi/farhat/Sanjana/TRUST_lowAF/{{sample_ID}}"


rule get_input_FASTQ_files:
    group: 
        "sequential"
    output:
        fastq1 = f"{sample_out_dir}/{{sample_ID}}_R1.fastq.gz",
        fastq2 = f"{sample_out_dir}/{{sample_ID}}_R2.fastq.gz",

        fastq1_unzipped = temp(f"{sample_out_dir}/{{sample_ID}}_1.fastq"),
        fastq2_unzipped = temp(f"{sample_out_dir}/{{sample_ID}}_2.fastq"),
    params:
        sample_out_dir = sample_out_dir,
        fastq_dir = config["fastq_dir"],
        download_script = f"{primary_directory}/scripts/download_FASTQ.sh",
    shell:        
        """
        # copy the FASTQ files from the directory specified in the config file to the sample directory
        # they will be deleted in the next rule after performing adapter trimming, so they won't be doubly stored
        cp {params.fastq_dir}/{wildcards.sample_ID}/{wildcards.sample_ID}_R1.fastq.gz {output.fastq1}
        cp {params.fastq_dir}/{wildcards.sample_ID}/{wildcards.sample_ID}_R2.fastq.gz {output.fastq2}

        gunzip -c {output.fastq1} > {output.fastq1_unzipped}
        gunzip -c {output.fastq2} > {output.fastq2_unzipped}

        # first check that the original FASTQ files have the same numbers of lines
        FQ1_line_count=$(wc -l {output.fastq1_unzipped} | awk '{{print $1}}')
        FQ2_line_count=$(wc -l {output.fastq2_unzipped} | awk '{{print $1}}')

        # check that neither FASTQ file has no reads
        if [ $FQ1_line_count -eq 0 ] || [ $FQ2_line_count -eq 0 ]; then
            echo "Error: At least one of the FASTQ files for $sample_ID/$sample_ID has no reads"
            exit 1
        # Compare the counts and raise an error if they are not equal 
        elif [ "$FQ1_line_count" -ne "$FQ2_line_count" ]; then
            echo "Error: FASTQ files for $sample_ID/$sample_ID have different line counts: $FQ1_line_count and $FQ2_line_count"
            exit 1
        fi

        # compare paired end read files. If they are the same, then add to error list. Suppress output with -s tag, so it doesn't print out the differences
        # If the files are identical, the exit status is 0, and the condition is considered true, so an error will be returned.
        if cmp -s {output.fastq1_unzipped} {output.fastq2_unzipped}; then
           echo "Error: {output.fastq1_unzipped} and {output.fastq2_unzipped} are duplicates"
           exit 1
        fi
        """


rule trim_adapters:
    input:
        fastq1 = f"{sample_out_dir}/{{sample_ID}}_R1.fastq.gz",
        fastq2 = f"{sample_out_dir}/{{sample_ID}}_R2.fastq.gz",
    output:
        fastq1_trimmed = f"{sample_out_dir}/fastp/{{sample_ID}}.R1.trimmed.fastq.gz",
        fastq2_trimmed = f"{sample_out_dir}/fastp/{{sample_ID}}.R2.trimmed.fastq.gz",
        fastp_html = f"{sample_out_dir}/fastp/fastp.html",
        fastp_json = f"{sample_out_dir}/fastp/fastp.json"
    conda:
        f"{conda_directory}/envs/read_processing_aln_bwa.yaml"
        # "/home/sak0914/Mtb_Megapipe/.snakemake/conda/73c414a0fdfb349af0d394f0508ea848_"
    params:
        min_read_length = config["min_read_length"]
    threads:
        8
    shell:
        """
        fastp -i {input.fastq1} -I {input.fastq2} \
              -o {output.fastq1_trimmed} -O {output.fastq2_trimmed} \
              -h {output.fastp_html} \
              -j {output.fastp_json} \
              --length_required {params.min_read_length} \
              --dedup \
              --thread {threads}

        rm {input.fastq1} {input.fastq2}
        """


rule kraken_classification:
    input:
        fastq1_trimmed = f"{sample_out_dir}/fastp/{{sample_ID}}.R1.trimmed.fastq.gz",
        fastq2_trimmed = f"{sample_out_dir}/fastp/{{sample_ID}}.R2.trimmed.fastq.gz",
    output:
        kraken_report = f"{sample_out_dir}/kraken/kraken_report_standard_DB.txt",
        kraken_classifications = f"{sample_out_dir}/kraken/kraken_classifications_standard_DB",
    conda:
        f"{conda_directory}/envs/read_processing_aln_bwa.yaml"
        # "/home/sak0914/Mtb_Megapipe/.snakemake/conda/73c414a0fdfb349af0d394f0508ea848_"
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
                --paired {input.fastq1_trimmed} {input.fastq2_trimmed} \
                --gzip-compressed \
                --report {output.kraken_report} \
                --output {output.kraken_classifications} \
                --memory-mapping
        """
        
        
        
rule extract_kraken_read_names:
    input:
        kraken_classifications = f"{sample_out_dir}/kraken/kraken_classifications_standard_DB",
    output:  
        kraken_classifications_gzipped = f"{sample_out_dir}/kraken/kraken_classifications_standard_DB.csv.gz", # gets gzipped by the python script. Did this to add headers
        keep_read_names = f"{sample_out_dir}/kraken/keep_read_names.txt"
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



rule extract_kraken_reads:
    input:
        fastq1_trimmed = f"{sample_out_dir}/fastp/{{sample_ID}}.R1.trimmed.fastq.gz",
        fastq2_trimmed = f"{sample_out_dir}/fastp/{{sample_ID}}.R2.trimmed.fastq.gz",
        keep_read_names = f"{sample_out_dir}/kraken/keep_read_names.txt"
    output:
        fastq1_trimmed_classified = f"{sample_out_dir}/kraken/{{sample_ID}}.R1.kraken.filtered.fastq.gz",
        fastq2_trimmed_classified = f"{sample_out_dir}/kraken/{{sample_ID}}.R2.kraken.filtered.fastq.gz",    
    conda:
        f"{conda_directory}/envs/read_processing_aln_bwa.yaml"
        # "/home/sak0914/Mtb_Megapipe/.snakemake/conda/73c414a0fdfb349af0d394f0508ea848_"
    shell:
        """
        # seqtk will write outputs to unzipped files, even if the input was compressed
        seqtk subseq {input.fastq1_trimmed} {input.keep_read_names} | gzip -c > {output.fastq1_trimmed_classified} 
        seqtk subseq {input.fastq2_trimmed} {input.keep_read_names} | gzip -c > {output.fastq2_trimmed_classified} 
        
        rm {input.fastq1_trimmed} {input.fastq2_trimmed}
        
        gzip {input.keep_read_names}
        """
        


rule align_reads_mark_duplicates:
    input:
        fastq1_trimmed_classified = f"{sample_H37Rv_ref_dir}/{{sample_ID}}/kraken/{{sample_ID}}.R1.kraken.filtered.fastq.gz",
        fastq2_trimmed_classified = f"{sample_H37Rv_ref_dir}/{{sample_ID}}/kraken/{{sample_ID}}.R2.kraken.filtered.fastq.gz",  
    output:
        sam_file = temp(f"{sample_out_dir}/bam/{{sample_ID}}.sam"),
        bam_file = temp(f"{sample_out_dir}/bam/{{sample_ID}}.bam"),
        bam_index_file = temp(f"{sample_out_dir}/bam/{{sample_ID}}.bam.bai"),
        bam_file_markdup = f"{sample_out_dir}/bam/{{sample_ID}}.markDup.bam",
        bam_file_markdup_metrics = f"{sample_out_dir}/bam/{{sample_ID}}.markDup.bam.metrics",
        bam_index_file_markdup = f"{sample_out_dir}/bam/{{sample_ID}}.markDup.bam.bai",
    params:
        output_dir = output_dir,
        ref_genome = lambda w: sample_asm_dict[w.sample_ID],
        bwa_mem_seed_length = config['bwa_mem_seed_length']
    conda:
        f"{conda_directory}/envs/read_processing_aln_bwa.yaml"
        # "/home/sak0914/Mtb_Megapipe/.snakemake/conda/73c414a0fdfb349af0d394f0508ea848_"
    threads:
        8
    shell:
        """
        # index reference genome (which is required before aligning reads)
        bwa index {params.ref_genome}
        
        # align reads
        bwa mem -M -R "@RG\\tID:{wildcards.sample_ID}\\tSM:{wildcards.sample_ID}" \
                    -k {params.bwa_mem_seed_length} \
                    -t {threads} \
                    {params.ref_genome} \
                    {input.fastq1_trimmed_classified} {input.fastq2_trimmed_classified} \
                    > {output.sam_file}

        # sort alignment and convert to bam file
        samtools view -b {output.sam_file} | samtools sort > {output.bam_file}

        # index alignment, which creates a .bai index file
        samtools index {output.bam_file}

        # -Xmx6g specifies to allocate 6 GB
        picard -Xmx10g MarkDuplicates I={output.bam_file} O={output.bam_file_markdup} REMOVE_DUPLICATES=false M={output.bam_file_markdup_metrics} ASSUME_SORT_ORDER=coordinate READ_NAME_REGEX='(?:.*.)?([0-9]+)[^.]*.([0-9]+)[^.]*.([0-9]+)[^.]*$'

        # index the deduplicated alignment with samtools, which will create a dedup_bam_file.bai file
        samtools index {output.bam_file_markdup}
        """

        


rule get_BAM_file_depths:
    input:
        bam_file_markdup = f"{sample_out_dir}/bam/{{sample_ID}}.markDup.bam",
    params:
        ref_genome = lambda w: sample_asm_dict[w.sample_ID],
        sample_out_dir = sample_out_dir,
    output:
        depth_file = temp(f"{sample_out_dir}/bam/{{sample_ID}}.depth.tsv.gz"),
    conda:
        f"{conda_directory}/envs/read_processing_aln.yaml"
    shell:
        """
        # get all runs associated with this sample_ID and compute depth
        # -a computes depth at all positions, not just those with non-zero depth
        # -Q is for minimum mapping quality: use 1, so that multiply mapped reads aren't counted. These have mapping quality of 0
        samtools depth -a -Q 1 {input.bam_file_markdup} | gzip -c > {output.depth_file}
        """




rule pilon_variant_calling:
    input:
        bam_file_markdup = f"{sample_out_dir}/bam/{{sample_ID}}.markDup.bam",
    output:
        vcf_file = temp(f"{sample_out_dir}/pilon/{{sample_ID}}.vcf"),
        vcf_file_gzip = f"{sample_out_dir}/pilon/{{sample_ID}}_full.vcf.gz",
        fasta_file = temp(f"{sample_out_dir}/pilon/{{sample_ID}}.fasta"),
        variants_vcf_file = f"{sample_out_dir}/pilon/{{sample_ID}}_variants.vcf",
    params:
        ref_genome = lambda w: sample_asm_dict[w.sample_ID],
        sample_pilon_dir = f"{sample_out_dir}/pilon",
    conda:
        f"{conda_directory}/envs/variant_calling.yaml"
    shell:
        """
        pilon -Xmx10g --minmq 1 --genome {params.ref_genome} --bam {input.bam_file_markdup} --output {wildcards.sample_ID} --outdir {params.sample_pilon_dir} --variant
            
        # left-align indels and drop duplicates, then gzip the full VCF file 
        # this affects those cases where the position of the indel is ambiguous
        # however, because of the shifting positions, the position of the indel can change, so need to sort it
        bcftools norm --rm-dup none --fasta-ref {params.ref_genome} {output.vcf_file} | bcftools sort | gzip -c > {output.vcf_file_gzip}
        
        # save a VCF file of the variants
        # bcftools view --types snps,mnps,indels,other {output.vcf_file_gzip} > {output.variants_vcf_file}
        
        # this requires that BC ≥ 5 for a non-REF allele. The order of values in BC is A,C,G,T
        bcftools filter -i '((REF=="A" && ((BC[1]>=5) || (BC[2]>=5) || (BC[3]>=5))) || (REF=="C" && ((BC[0]>=5) || (BC[2]>=5) || (BC[3]>=5))) || (REF=="G" && ((BC[0]>=5) || (BC[1]>=5) || (BC[3]>=5))) || (REF=="T" && ((BC[0]>=5) || (BC[1]>=5) || (BC[2]>=5))))' {output.vcf_file_gzip} > {output.variants_vcf_file}
        """

        


rule freebayes_variant_calling:
    input:
        merged_bam_file = f"{sample_out_dir}/bam/{{sample_ID}}.markDup.bam",
    output:
        vcf_file_init = f"{sample_out_dir}/freebayes/{{sample_ID}}.init.vcf",
    params:
        ref_genome = lambda w: sample_asm_dict[w.sample_ID],
    conda:
        f"{conda_directory}/envs/variant_calling.yaml"
    threads:
        1
    shell:
        """
        # -p is ploidy
        # freebayes says it automatically does left-alignment of indels, but there was an issue with that in the WHO catalog, so do it as well
        # so left-align indels and drop duplicate records
        # leave --min-alternate-count at the default of 2
        # the minimum AF we're going down to is 1%, so set --min-alternate-fraction to 0.01
        freebayes -f {params.ref_genome} \
                  -p 1 \
                  --min-alternate-count 2 \
                  --min-alternate-fraction 0.01 \
                  --min-mapping-quality 40 \
                  --min-base-quality 30 \
                  -b {input.merged_bam_file} \
                  -v {output.vcf_file_init}
        """
        
        
        
        
rule freebayes_VCF_normalization_decomposition:
    input:
        vcf_file_init = f"{sample_out_dir}/freebayes/{{sample_ID}}.init.vcf",
    output:
        vcf_file_split = temp(f"{sample_out_dir}/freebayes/{{sample_ID}}.split.vcf"),
        vcf_file = f"{sample_out_dir}/freebayes/{{sample_ID}}.vcf",
    params:
        ref_genome = lambda w: sample_asm_dict[w.sample_ID],
    conda:
        f"{conda_directory}/envs/variant_normalization.yaml"
    shell:
        """
        # left-align indels and split multiallelics after decomposing variants above
        bcftools norm --multiallelics -both {input.vcf_file_init} --fasta-ref {params.ref_genome} > {output.vcf_file_split}
        
        vcfwave {output.vcf_file_split} | bcftools sort > {output.vcf_file}
        """
    
        
        
        
rule save_TSV_file_of_VCF_file:
    input:
        vcf_file = f"{sample_out_dir}/freebayes/{{sample_ID}}.vcf",
    output:
        tsv_file = f"{sample_out_dir}/freebayes/{{sample_ID}}.tsv",        
        field_names = temp(f"{sample_out_dir}/freebayes/field_names.txt"),
    conda:
        f"{conda_directory}/envs/variant_annotation.yaml"
    shell:
        """
        # make a TSV file version of each of the VCF subsets made above (and the full one after excluding low confidence sites)
        # get all field names because not sure which ones we will need for low AF variant detection
        echo -e "POS\nREF\nALT\nQUAL\nFILTER" > {output.field_names}
        grep "^##INFO=<ID=" {input.vcf_file} | cut -d'=' -f3 | cut -d',' -f1 >> {output.field_names}
                
        # write TSV files of both
        SnpSift extractFields {input.vcf_file} $(paste -sd " " {output.field_names}) > {output.tsv_file}
        """
        
        
        
        
rule filter_high_quality_lowAF_variants:
    input:
        depth_file_gzip = f"{sample_out_dir}/bam/{{sample_ID}}.depth.tsv.gz",
        freebayes_tsv_file = f"{sample_out_dir}/freebayes/{{sample_ID}}.tsv",
    output:
        lowAF_variants_bed_file = f"{sample_out_dir}/freebayes/lowAF_variants.bed",
    params:
        ref_genome = lambda w: sample_asm_dict[w.sample_ID],
        output_dir = sample_out_dir,
    # conda:
    #     f"{conda_directory}/envs/python_bioinformatics_utils.yaml"
    shell:
        """
        source activate /home/sak0914/anaconda3/envs/python_bioinformatics_utils 
        
        # this scripts writes the lowAF variants BED file. The variants are in the coordinates of the personal ref genome
        # the next step will be to run paftools liftover to transfer the coordinates to H37Rv for easy comparison
        python3 -u ~/MtbLongitudinalDiversity/lowAF_variant_calling/variantDetector/01_write_lowAF_BED.py \
                -d {input.depth_file_gzip} \
                -i {input.freebayes_tsv_file} \
                -o {output.lowAF_variants_bed_file} \
                -g {params.ref_genome}
        """

        
        
        
rule transfer_lowAF_variants_to_H37Rv_coordinates:
    input:
        lowAF_variants_bed_file = f"{sample_out_dir}/freebayes/lowAF_variants.bed",
    output:
        personal_H37Rv_paf_file = f"{sample_out_dir}/assembly/{{sample_ID}}.H37Rv.paf",
        lowAF_variants_H37Rv_bed_file = f"{sample_out_dir}/freebayes/lowAF_variants.H37Rv.bed",
        lowAF_variants_H37Rv_excludeLowConf_bed_file = f"{sample_out_dir}/freebayes/lowAF_variants.H37Rv.excludeLowConf.bed",
    conda:
        f"{conda_directory}/envs/long_read_aln.yaml"
    params:
        ref_genome = lambda w: sample_asm_dict[w.sample_ID],
        H37Rv_genome = os.path.join(primary_directory, "references", "ref_genome", "H37Rv_NC_000962.3.fna"),
        exclude_regions = os.path.join(primary_directory, references_dir, config['exclude_regions_file']),
    shell:
        """
        # generate paf file. Target = H37Rv first, then query = personal genome second
        minimap2 -x asm5 -c --cs {params.H37Rv_genome} {params.ref_genome} --secondary=no > {output.personal_H37Rv_paf_file}
        
        # transfer lowAF variants from personal to H37Rv coordinates
        paftools.js liftover {output.personal_H37Rv_paf_file} {input.lowAF_variants_bed_file} > {output.lowAF_variants_H37Rv_bed_file}
        
        # finally remove the low confidence variants now that we have the variants in H37Rv coordinates
        bedtools subtract -a {output.lowAF_variants_H37Rv_bed_file} -b {params.exclude_regions} -header > {output.lowAF_variants_H37Rv_excludeLowConf_bed_file}
        """
        
        
        
        

rule merge_results_H37Rv_personal_genome_calling:
    input:
        lowAF_variants_bed_file = f"{sample_out_dir}/freebayes/lowAF_variants.bed",
        lowAF_variants_transferred_to_H37Rv_excludeLowConf_bed_file = f"{sample_out_dir}/freebayes/lowAF_variants.H37Rv.excludeLowConf.bed",
        lowAF_variants_H37Rv_excludeLowConf_tsv_file = f"{sample_H37Rv_ref_dir}/freebayes/{{sample_ID}}.excludeLowConf.tsv",
    output:
        f"{sample_out_dir}/lowAF_comparison/ground_truth.csv",
        f"{sample_out_dir}/lowAF_comparison/H37Rv_detected.csv",
        f"{sample_out_dir}/lowAF_comparison/confusion_matrix.csv",
    params:
        sample_ID = f"{{sample_ID}}",
    # conda:
    #    f"{conda_directory}/envs/python_bioinformatics_utils.yaml"
    shell:
        """ 
        source activate /home/sak0914/anaconda3/envs/python_bioinformatics_utils
        
        python3 -u ~/MtbLongitudinalDiversity/lowAF_variant_calling/variantDetector/02_combine_lowAF_variants.py \
                -s {params.sample_ID} \
                -bed1 {input.lowAF_variants_bed_file} \
                -bed2 {input.lowAF_variants_transferred_to_H37Rv_excludeLowConf_bed_file} \
                -tsv {input.lowAF_variants_H37Rv_excludeLowConf_tsv_file}
        """
             
        
rule liftoff_genes_from_H37Rv:
    input:
        ref_genome = lambda w: sample_asm_dict[w.sample_ID],
    output:
        liftoff_gff_file = f"{sample_out_dir}/assembly/H37Rv.liftoff.gff",
        liftoff_intermediate_dir = temp(directory(f"{sample_out_dir}/assembly/intermediate_files")),
        polished_liftoff_gff_file = f"{sample_out_dir}/assembly/H37Rv.liftoff.gff_polished",
    params:
        H37Rv_genome = os.path.join(primary_directory, "references", "ref_genome", "H37Rv_NC_000962.3.fna"),
        H37Rv_gff_file = os.path.join(primary_directory, "references", "ref_genome", "H37Rv.NCBI.gff3"),
    threads:
        1
    # conda:
    #     f"{conda_directory}/envs/liftover.yaml"
    shell:
        """        
        source activate liftoff
        
        liftoff -g {params.H37Rv_gff_file} \
                -o {output.liftoff_gff_file} \
                -copies -polish \
                -dir {output.liftoff_intermediate_dir} \
                {input.ref_genome} {params.H37Rv_genome}
        """