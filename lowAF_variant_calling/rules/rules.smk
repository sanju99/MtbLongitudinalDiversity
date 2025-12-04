import os, glob
import numpy as np
import pandas as pd


# define some paths to make the path names more readable
sample_out_dir = f"{output_dir}/{{sample_ID}}"
run_out_dir = f"{output_dir}/{{sample_ID}}/{{run_ID}}"

scripts_dir = config["scripts_dir"]
references_dir = config["references_dir"]

conda_directory = config['conda_dir']
primary_directory = "/home/sak0914/MtbLongitudinalDiversity/lowAF_variant_calling"



rule get_input_FASTQ_files:
    group: 
        "sequential"
    output:
        fastq1 = f"{run_out_dir}/{{run_ID}}_R1.fastq.gz",
        fastq2 = f"{run_out_dir}/{{run_ID}}_R2.fastq.gz",

        fastq1_unzipped = temp(f"{run_out_dir}/{{run_ID}}_1.fastq"),
        fastq2_unzipped = temp(f"{run_out_dir}/{{run_ID}}_2.fastq"),
    params:
        sample_out_dir = sample_out_dir,
        fastq_dir = config["fastq_dir"],
        download_script = f"{primary_directory}/scripts/download_FASTQ.sh",
    run:        
        if download_public_FASTQ_dict[wildcards.sample_ID] == 1:
            shell("""
                module load sratoolkit/2.10.7

                # the script performs the same QC as in the next block
                bash {params.download_script} {params.sample_out_dir} {wildcards.run_ID}
            """)
        elif download_public_FASTQ_dict[wildcards.sample_ID] == 0:
            shell("""
                # copy the FASTQ files from the directory specified in the config file to the sample directory
                # they will be deleted in the next rule after performing adapter trimming, so they won't be doubly stored
                cp {params.fastq_dir}/{wildcards.run_ID}/{wildcards.run_ID}_R1.fastq.gz {output.fastq1}
                cp {params.fastq_dir}/{wildcards.run_ID}/{wildcards.run_ID}_R2.fastq.gz {output.fastq2}

                gunzip -c {output.fastq1} > {output.fastq1_unzipped}
                gunzip -c {output.fastq2} > {output.fastq2_unzipped}

                # first check that the original FASTQ files have the same numbers of lines
                FQ1_line_count=$(wc -l {output.fastq1_unzipped} | awk '{{print $1}}')
                FQ2_line_count=$(wc -l {output.fastq2_unzipped} | awk '{{print $1}}')
                
                # check that neither FASTQ file has no reads
                if [ $FQ1_line_count -eq 0 ] || [ $FQ2_line_count -eq 0 ]; then
                    echo "Error: At least one of the FASTQ files for $sample_ID/$run_ID has no reads"
                    exit 1
                # Compare the counts and raise an error if they are not equal 
                elif [ "$FQ1_line_count" -ne "$FQ2_line_count" ]; then
                    echo "Error: FASTQ files for $sample_ID/$run_ID have different line counts: $FQ1_line_count and $FQ2_line_count"
                    exit 1
                fi
                
                # compare paired end read files. If they are the same, then add to error list. Suppress output with -s tag, so it doesn't print out the differences
                # If the files are identical, the exit status is 0, and the condition is considered true, so an error will be returned.
                if cmp -s {output.fastq1_unzipped} {output.fastq2_unzipped}; then
                   echo "Error: {output.fastq1_unzipped} and {output.fastq2_unzipped} are duplicates"
                   exit 1
                fi
            """)


rule trim_adapters:
    input:
        fastq1 = f"{run_out_dir}/{{run_ID}}_R1.fastq.gz",
        fastq2 = f"{run_out_dir}/{{run_ID}}_R2.fastq.gz",
    output:
        fastq1_trimmed = f"{run_out_dir}/fastp/{{run_ID}}.R1.trimmed.fastq.gz",
        fastq2_trimmed = f"{run_out_dir}/fastp/{{run_ID}}.R2.trimmed.fastq.gz",
        fastp_html = f"{run_out_dir}/fastp/fastp.html",
        fastp_json = f"{run_out_dir}/fastp/fastp.json"
    conda:
        f"{conda_directory}/envs/read_processing_aln.yaml"
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
        fastq1_trimmed = f"{run_out_dir}/fastp/{{run_ID}}.R1.trimmed.fastq.gz",
        fastq2_trimmed = f"{run_out_dir}/fastp/{{run_ID}}.R2.trimmed.fastq.gz",
    output:
        kraken_report = f"{run_out_dir}/kraken/kraken_report_standard_DB.txt",
        kraken_classifications = f"{run_out_dir}/kraken/kraken_classifications_standard_DB",
    conda:
        f"{conda_directory}/envs/read_processing_aln.yaml"
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
        kraken_classifications = f"{run_out_dir}/kraken/kraken_classifications_standard_DB",
    output:  
        kraken_classifications_gzipped = f"{run_out_dir}/kraken/kraken_classifications_standard_DB.csv.gz", # gets gzipped by the python script. Did this to add headers
        keep_read_names = f"{run_out_dir}/kraken/keep_read_names.txt"
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
        fastq1_trimmed = f"{run_out_dir}/fastp/{{run_ID}}.R1.trimmed.fastq.gz",
        fastq2_trimmed = f"{run_out_dir}/fastp/{{run_ID}}.R2.trimmed.fastq.gz",
        keep_read_names = f"{run_out_dir}/kraken/keep_read_names.txt"
    output:
        fastq1_trimmed_classified = f"{run_out_dir}/kraken/{{run_ID}}.R1.kraken.filtered.fastq.gz",
        fastq2_trimmed_classified = f"{run_out_dir}/kraken/{{run_ID}}.R2.kraken.filtered.fastq.gz",    
    conda:
        f"{conda_directory}/envs/read_processing_aln.yaml"
    shell:
        """
        # seqtk will write outputs to unzipped files, even if the input was compressed
        seqtk subseq {input.fastq1_trimmed} {input.keep_read_names} | gzip -c > {output.fastq1_trimmed_classified} 
        seqtk subseq {input.fastq2_trimmed} {input.keep_read_names} | gzip -c > {output.fastq2_trimmed_classified} 
        
        rm {input.fastq1_trimmed} {input.fastq2_trimmed}
        
        gzip {input.keep_read_names}
        """
        
        
        
rule fastlin_typing:
    input:
        fastq1_trimmed_classified = f"{run_out_dir}/kraken/{{run_ID}}.R1.kraken.filtered.fastq.gz",
        fastq2_trimmed_classified = f"{run_out_dir}/kraken/{{run_ID}}.R2.kraken.filtered.fastq.gz",
    output:
        fastq1_trimmed_classified_gzipped = temp(f"{run_out_dir}/fastlin/{{run_ID}}_1.fastq.gz"),
        fastq2_trimmed_classified_gzipped = temp(f"{run_out_dir}/fastlin/{{run_ID}}_2.fastq.gz"),
        fastlin_dir = directory(f"{run_out_dir}/fastlin"),
        fastlin_output = f"{run_out_dir}/fastlin/output.txt"
    conda:
        f"{primary_directory}/envs/read_processing_aln.yaml"
    params:
        fastlin_barcodes = os.path.join(primary_directory, references_dir, "phylogeny", "MTBC_barcodes.tsv"),
    shell:
        """
        cp {input.fastq1_trimmed_classified} {output.fastq1_trimmed_classified_gzipped}
        cp {input.fastq2_trimmed_classified} {output.fastq2_trimmed_classified_gzipped}
        
        # there's one TRUST sample with coverage > 900
        fastlin -d {output.fastlin_dir} -b {params.fastlin_barcodes} -o {output.fastlin_output} -x 1000
        """



rule align_reads_mark_duplicates:
    input:
        fastq1_trimmed_classified = f"{run_out_dir}/kraken/{{run_ID}}.R1.kraken.filtered.fastq",
        fastq2_trimmed_classified = f"{run_out_dir}/kraken/{{run_ID}}.R2.kraken.filtered.fastq",  
    output:
        sam_file = temp(f"{run_out_dir}/bam/{{run_ID}}.sam"),
        bam_file = temp(f"{run_out_dir}/bam/{{run_ID}}.bam"),
        bam_index_file = temp(f"{run_out_dir}/bam/{{run_ID}}.bam.bai"),
        bam_file_markdup = f"{run_out_dir}/bam/{{run_ID}}.markDup.bam",
        bam_file_markdup_metrics = f"{run_out_dir}/bam/{{run_ID}}.markDup.bam.metrics",
        bam_index_file_dedup = f"{run_out_dir}/bam/{{run_ID}}.markDup.bam.bai",
    params:
        output_dir = output_dir,
        ref_genome = os.path.join(primary_directory, references_dir, "ref_genome", "H37Rv_NC_000962.3.fna"),
        bwa_mem_seed_length = config['bwa_mem_seed_length']
    conda:
        f"{conda_directory}/envs/read_processing_aln_bwa.yaml"
    threads:
        8
    shell:
        """
        bwa mem -M -R "@RG\\tID:{wildcards.run_ID}\\tSM:{wildcards.sample_ID}" \
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
        bam_file_markdup = lambda wildcards: [f"{output_dir}/{wildcards.sample_ID}/{run_ID}/bam/{run_ID}.markDup.bam" for run_ID in sample_run_dict[wildcards.sample_ID]],
    params:
        ref_genome = os.path.join(primary_directory, references_dir, "ref_genome", "H37Rv_NC_000962.3.fna"),
        sample_out_dir = sample_out_dir,
    output:
        depth_file = temp(f"{sample_out_dir}/bam/{{sample_ID}}.depth.tsv"),
        depth_file_gzip = f"{sample_out_dir}/bam/{{sample_ID}}.depth.tsv.gz",
    conda:
        f"{conda_directory}/envs/read_processing_aln.yaml"
    shell:
        """
        # get all runs associated with this sample_ID and compute depth
        # -a computes depth at all positions, not just those with non-zero depth
        # -Q is for minimum mapping quality: use 1, so that multiply mapped reads aren't counted. These have mapping quality of 0
        samtools depth -a -Q 1 {input.bam_file_markdup} > {output.depth_file}

        # get the length of the reference genome
        genome_length=$(tail -n +2 {params.ref_genome} | tr -d '\n' | wc -c) # remove first line (FASTA header) and newline characters, then count characters to get ref genome length

        # when there are multiple bam files, each one is its own column in the depth file.
        num_sites_H37Rv=$(wc -l {output.depth_file} | awk '{{print $1}}')
    
        if [ ! "$num_sites_H37Rv" -eq "$genome_length" ]; then
            echo "Check that all $genome_length sites in the H37Rv reference genome are in {output.depth_file}, which currently has $num_sites_H37Rv sites"
            exit 1
        fi

        gzip -c {output.depth_file} > {output.depth_file_gzip}
        """




rule get_BAMs_passing_QC_thresholds:
    input:
        depth_file_gzip = f"{sample_out_dir}/bam/{{sample_ID}}.depth.tsv.gz", # contains depths for all BAM files for all WGS runs
        bam_file_markdup = lambda wildcards: [f"{output_dir}/{wildcards.sample_ID}/{run_ID}/bam/{run_ID}.markDup.bam" for run_ID in sample_run_dict[wildcards.sample_ID]],
    output:
        pass_BAMs_file = f"{sample_out_dir}/bam/pass_BAMs.txt",
    params:
        sample_out_dir = sample_out_dir,
        BAM_depth_QC_script = os.path.join(primary_directory, scripts_dir, "BAM_depth_QC.py"),
        median_depth = config["median_depth"],
        min_cov = config["min_cov"],
        genome_cov_prop = config["genome_cov_prop"],
    shell:
        """
        # run the script to determine which runs pass the BAM depth criteria
        python3 -u {params.BAM_depth_QC_script} -i {input.depth_file_gzip} -b {input.bam_file_markdup} --median-depth {params.median_depth} --min-cov {params.min_cov} --genome-cov-prop {params.genome_cov_prop}
        """
        
        

rule merge_BAMs:
    input:
        pass_BAMs_file = f"{sample_out_dir}/bam/pass_BAMs.txt",
    output:
        merged_bam_file = f"{sample_out_dir}/bam/{{sample_ID}}.markDup.bam",
        merged_bam_index_file = f"{sample_out_dir}/bam/{{sample_ID}}.markDup.bam.bai",
    conda:
        f"{conda_directory}/envs/read_processing_aln.yaml"
    params:
        ref_genome = os.path.join(primary_directory, references_dir, "ref_genome", "H37Rv_NC_000962.3.fna"),
        sample_out_dir = sample_out_dir,
        median_depth = config["median_depth"],
        min_cov = config["min_cov"],
        genome_cov_prop = config["genome_cov_prop"],
    shell:
        """
        num_runs_passed=$(wc -l {input.pass_BAMs_file} | awk '{{print $1}}')

        # stop processing samples that don't pass the BAM coverage requirements
        if [ $num_runs_passed -eq 0 ]; then
            echo "No BAM files for {wildcards.sample_ID} passed the minimum coverage requirements. Halting pipeline for this sample"
            exit
            
        else 
            echo "$num_runs_passed WGS runs for {wildcards.sample_ID} have median depth ≥ {params.median_depth} and at least {params.genome_cov_prop} of sites with {params.min_cov}x coverage"

            # merge them using samtools. works because the original bam files were sorted prior to running picard and dropping duplicates (after which they remain sorted)
            samtools merge -b {input.pass_BAMs_file} -o {output.merged_bam_file}

            if [ $num_runs_passed -eq 1 ]; then

                # if only one BAM file passed, delete the original BAM file to reduce disk space usage because it's a duplicate of the merged BAM file
                for file_path in $(cat {input.pass_BAMs_file}); do
                    rm "$file_path" "$file_path.bai"
                done

            fi

            # index the merged BAM file for variant calling
            samtools index {output.merged_bam_file}

        fi
        """
        
        
        
rule freebayes_variant_calling:
    input:
        merged_bam_file = f"{sample_out_dir}/bam/{{sample_ID}}.dedup.bam",
    output:
        vcf_file_init = f"{sample_out_dir}/freebayes/{{sample_ID}}.init.vcf",
    params:
        ref_genome = os.path.join(primary_directory, references_dir, "ref_genome", "H37Rv_NC_000962.3.fna"),
    conda:
        f"{conda_directory}/envs/variant_calling.yaml"
    shell:
        """
        # -p is ploidy
        # freebayes says it automatically does left-alignment of indels, but there was an issue with that in the WHO catalog, so do it as well
        # so left-align indels and drop duplicate records
        # leave --min-alternate-count at the default of 2
        # the minimum AF we're going down to is 1%, so set --min-alternate-fraction to 0.01
        # --max-complex-gap
        freebayes -f {params.ref_genome} \
                  -p 1 \
                  --min-alternate-count 2 \
                  --min-alternate-fraction 0.01 \
                  --min-mapping-quality 30 \
                  --min-base-quality 30 \
                  -b {input.merged_bam_file} \
                  -v {output.vcf_file_init}
        """
        
        
        
rule freebayes_variant_calling_all_haplotypes:
    input:
        merged_bam_file = f"{sample_out_dir}/bam/{{sample_ID}}.dedup.bam",
    output:
        vcf_file_init = f"{sample_out_dir}/freebayes/{{sample_ID}}.init.allHaps.vcf",
    params:
        ref_genome = os.path.join(primary_directory, references_dir, "ref_genome", "H37Rv_NC_000962.3.fna"),
    conda:
        f"{conda_directory}/envs/variant_calling.yaml"
    shell:
        """
        # -p is ploidy
        # freebayes says it automatically does left-alignment of indels, but there was an issue with that in the WHO catalog, so do it as well
        # so left-align indels and drop duplicate records
        # leave --min-alternate-count at the default of 2
        # the minimum AF we're going down to is 1%, so set --min-alternate-fraction to 0.01
        # --max-complex-gap
        freebayes -f {params.ref_genome} \
                  -p 1 \
                  --min-alternate-count 2 \
                  --min-alternate-fraction 0.01 \
                  --min-mapping-quality 30 \
                  --min-base-quality 30 \
                  -b {input.merged_bam_file} \
                  -v {output.vcf_file_init} \
                  --report-all-haplotype-alleles
        """
        





rule freebayes_split_haplotypes_separate_records:
    input:
        vcf_file_init = f"{sample_out_dir}/freebayes/{{sample_ID}}.init.allHaps.vcf",
    output:
        vcf_file_vcfwave = temp(f"{sample_out_dir}/freebayes/{{sample_ID}}.excludeLowConf.vcfwave.vcf"),
        vcf_file_split = f"{sample_out_dir}/freebayes/{{sample_ID}}.excludeLowConf.split.vcf",
    params:
        exclude_regions = os.path.join(primary_directory, references_dir, config['exclude_regions_file']),
    conda:
        f"{conda_directory}/envs/variant_normalization.yaml"
    shell:
        """
        # vcfwave properly splits them, but misses splitting the AO field
        vcfwave {input.vcf_file_init} | bedtools subtract -a '-' -b {params.exclude_regions} -header | bcftools sort > {output.vcf_file_vcfwave}
        
        # vcfwave won't split multiallelic SNPs, so do that here
        bcftools norm --multiallelics -snps {output.vcf_file_vcfwave} > {output.vcf_file_split}
        """
        
        


rule fix_AO_field_split_haplotypes:
    input:
        vcf_file_split = f"{sample_out_dir}/freebayes/{{sample_ID}}.excludeLowConf.split.vcf",
    output:
        vcf_file_split_fixedAO = f"{sample_out_dir}/freebayes/{{sample_ID}}.excludeLowConf.split.fixedAO.vcf",
    shell:
        """
        set +u
        eval "$(conda shell.bash hook)"
        conda activate base
        set -u
        
        python3 -u ~/MtbLongitudinalDiversity/lowAF_variant_calling/rules/split_AO_haplotypes.py \
                -i {input.vcf_file_split} \
                -o {output.vcf_file_split_fixedAO}
        """



rule extract_TSV_of_final_freebayes_VCF:
    input:
        vcf_file_split_fixedAO = f"{sample_out_dir}/freebayes/{{sample_ID}}.excludeLowConf.split.fixedAO.vcf",
    output:
        vcf_file_final = f"{sample_out_dir}/freebayes/{{sample_ID}}.excludeLowConf.split.fixedAO.final.vcf",
        field_names = temp(f"{sample_out_dir}/freebayes/field_names.txt"),
        tsv_file_final = f"{sample_out_dir}/freebayes/{{sample_ID}}.excludeLowConf.split.fixedAO.final.tsv",
    conda:
        f"{conda_directory}/envs/variant_annotation.yaml"
    params:
        ref_genome = os.path.join(primary_directory, references_dir, "ref_genome", "H37Rv_NC_000962.3.fna"),
    shell:
        """
        # left-normalize indels. Don't do anything to multiallelics
        bcftools norm {input.vcf_file_split_fixedAO} --fasta-ref {params.ref_genome} > {output.vcf_file_final}

        echo -e "POS\nREF\nALT\nQUAL\nFILTER" > {output.field_names}
        grep "^##INFO=<ID=" {output.vcf_file_final} | cut -d'=' -f3 | cut -d',' -f1 >> {output.field_names}
        
        SnpSift extractFields {output.vcf_file_final} $(paste -sd " " {output.field_names}) > {output.tsv_file_final}
        """



rule freebayes_VCF_normalization_decomposition:
    input:
        vcf_file_init = f"{sample_out_dir}/freebayes/{{sample_ID}}.init.vcf",
    output:
        vcf_file_split = temp(f"{sample_out_dir}/freebayes/{{sample_ID}}.split.vcf"),
        vcf_file = f"{sample_out_dir}/freebayes/{{sample_ID}}.vcf",
    params:
        ref_genome = os.path.join(primary_directory, references_dir, "ref_genome", "H37Rv_NC_000962.3.fna"),
    conda:
        f"{conda_directory}/envs/variant_normalization.yaml"
    shell:
        """
        # left-align indels and split multiallelics after decomposing variants above
        bcftools norm --multiallelics -both {input.vcf_file_init} --fasta-ref {params.ref_genome} > {output.vcf_file_split}
        
        # vcfwave won't split MNPs into consecutive SNPs, which is good in this case so that snpEff annotates them properly in the next step
        vcfwave {output.vcf_file_split} | bcftools sort > {output.vcf_file}
        """



rule excludeLowConf_regions_freebayes_VCF:
    input:
        vcf_file = f"{sample_out_dir}/freebayes/{{sample_ID}}.vcf",
    output:
        vcf_file_annot = f"{sample_out_dir}/freebayes/{{sample_ID}}.eff.vcf",
        tsv_file = f"{sample_out_dir}/freebayes/{{sample_ID}}.tsv",
        
        vcf_exclude_low_conf = f"{sample_out_dir}/freebayes/{{sample_ID}}.excludeLowConf.eff.vcf",
        tsv_exclude_low_conf = f"{sample_out_dir}/freebayes/{{sample_ID}}.excludeLowConf.tsv",
        
        field_names = temp(f"{sample_out_dir}/freebayes/field_names.txt"),
    conda:
        f"{conda_directory}/envs/variant_annotation.yaml"
    params:
        snpEff_db = config['snpEff_db'],
        exclude_regions = os.path.join(primary_directory, references_dir, config['exclude_regions_file']),
    shell:
        """
        # annotate with snpeff
        snpEff eff {params.snpEff_db} -noStats -no-downstream -no-upstream -lof {input.vcf_file} > {output.vcf_file_annot}

        # make a TSV file version of each of the VCF subsets made above (and the full one after excluding low confidence sites)
        # get all field names because not sure which ones we will need for low AF variant detection
        echo -e "POS\nREF\nALT\nQUAL\nFILTER\nANN[0].GENE\nANN[0].HGVS_C\nANN[0].HGVS_P" > {output.field_names}
        grep "^##INFO=<ID=" {output.vcf_file_annot} | cut -d'=' -f3 | cut -d',' -f1 >> {output.field_names}
        
        # full TSV file, WITH the excluded regions          
        SnpSift extractFields {output.vcf_file_annot} $(paste -sd " " {output.field_names}) > {output.tsv_file}
        
        # this BED file is /home/sak0914/Mtb_Megapipe/references/ref_genome/RLC_Regions.Plus.LowPmapK50E4.H37Rv.bed with extensions of 50 bp on each side of each region
        bedtools subtract -a {output.vcf_file_annot} -b {params.exclude_regions} -header > {output.vcf_exclude_low_conf}
        SnpSift extractFields {output.vcf_exclude_low_conf} $(paste -sd " " {output.field_names}) > {output.tsv_exclude_low_conf}        
        """
        


rule write_lowAF_SNPs:
    input:
        merged_bam_file = f"{sample_out_dir}/bam/{{sample_ID}}.dedup.bam",
        tsv_exclude_low_conf = f"{sample_out_dir}/freebayes/{{sample_ID}}.excludeLowConf.tsv",
    output:
        lowAF_SNPs_file = f"{sample_out_dir}/freebayes/lowAF_SNPs.csv",
    params:
        sample_ID = f"{{sample_ID}}",
    shell:
        """
        source activate /home/sak0914/anaconda3/envs/python_bioinformatics_utils
        
        python3 -u ~/MtbLongitudinalDiversity/lowAF_variant_calling/variantDetector/03_get_H37Rv_alignment_stats.py \
                -s {params.sample_ID} \
                -b {input.merged_bam_file} \
                -v {input.tsv_exclude_low_conf}
        """
        
              
        
        
rule create_VCF_subsets_ROI:
    input:
        vcf_file_annot = f"{sample_out_dir}/freebayes/{{sample_ID}}.eff.vcf",
    output:
        vcf_SNPs_only = f"{sample_out_dir}/freebayes/{{sample_ID}}.SNPs.vcf",
        vcf_phase_variation_ROI = f"{sample_out_dir}/freebayes/{{sample_ID}}.excludeLowConf.regionsOfInterest.vcf",
        
        field_names = temp(f"{sample_out_dir}/freebayes/field_names.txt"),
        tsv_WHO_catalog_regions = f"{sample_out_dir}/freebayes/{{sample_ID}}.WHOcatalog.tsv"
    conda:
        f"{conda_directory}/envs/variant_annotation.yaml"
    params:
        phase_variation_ROI = os.path.join(primary_directory, references_dir, config['phase_variation_ROI_file']),
        WHO_catalog_regions = os.path.join(primary_directory, references_dir, config['WHO_catalog_regions_file']),
    shell:
        """
        # get only SNPs and MNPs and save to another file. DO NOT exclude Max's low-quality regions (strictest exclusion criteria)
        # because we use the full VCF to get the proportion of low-frequency SNPs (for the SNP density threshold flagging)
        bcftools view --types snps,mnps {input.vcf_file_annot} > {output.vcf_SNPs_only}
        
        # make a TSV file version of each of the VCF subsets made above (and the full one after excluding low confidence sites)
        # get all field names because not sure which ones we will need for low AF variant detection
        echo -e "POS\nREF\nALT\nQUAL\nFILTER\nANN[*].GENE\nANN[*].HGVS_C\nANN[*].HGVS_P\nANN[*].EFFECT" > {output.field_names}
        grep "^##INFO=<ID=" {input.vcf_file_annot} | cut -d'=' -f3 | cut -d',' -f1 >> {output.field_names}
        grep "^##FORMAT=<ID=" {input.vcf_file_annot} | cut -d'=' -f3 | cut -d',' -f1 >> {output.field_names}

        # phase variation regions of interest
        bedtools intersect -a {output.vcf_exclude_low_conf} -b {params.phase_variation_ROI} -header > {output.vcf_phase_variation_ROI}
                
        # 2023 WHO catalog regions. extract variants to a TSV file. Keep all fields
        bedtools intersect -a {output.vcf_exclude_low_conf} -b {params.WHO_catalog_regions} -header | SnpSift extractFields '-' $(paste -sd " " {output.field_names}) -e "" > {output.tsv_WHO_catalog_regions}
        """



rule process_WHO_catalog_variants:
    input:
        tsv_WHO_catalog_regions = f"{sample_out_dir}/freebayes/{{sample_ID}}.WHOcatalog.tsv"
    output:
        tsv_WHO_catalog_regions_annot = f"{sample_out_dir}/freebayes/{{sample_ID}}.WHOcatalog.annot.tsv"
    params:
        process_variants_for_WHO_catalog_script = os.path.join(primary_directory, "scripts", "process_variants_for_WHO_catalog.py")
    shell:
        """
        python3 {params.process_variants_for_WHO_catalog_script} -i {input.tsv_WHO_catalog_regions} -o {output.tsv_WHO_catalog_regions_annot}
        
        rm {input.tsv_WHO_catalog_regions}
        """
        
        
        
rule pilon_variant_calling:
    input:
        merged_bam_file = f"{sample_out_dir}/bam/{{sample_ID}}.dedup.bam",
    output:
        vcf_file = temp(f"{sample_out_dir}/pilon/{{sample_ID}}.vcf"),
        vcf_file_gzip = f"{sample_out_dir}/pilon/{{sample_ID}}_full.vcf.gz",
        fasta_file = temp(f"{sample_out_dir}/pilon/{{sample_ID}}.fasta"),        
    params:
        ref_genome = os.path.join(primary_directory, references_dir, "ref_genome", "H37Rv_NC_000962.3.fna"),
        sample_pilon_dir = f"{sample_out_dir}/pilon",
    conda:
        f"{primary_directory}/envs/variant_calling.yaml"
    shell:
        """        
        pilon -Xmx10g --minmq 1 --genome {params.ref_genome} --bam {input.merged_bam_file} --output {wildcards.sample_ID} --outdir {params.sample_pilon_dir} --variant
            
        # left-align indels and split multi-allelics, then gzip the full VCF file 
        # this affects those cases where the position of the indel is ambiguous
        bcftools norm --multiallelics -both --fasta-ref {params.ref_genome} {output.vcf_file} | bcftools sort | gzip -c > {output.vcf_file_gzip}
        """


rule get_pilon_SNPs_indels:
    input:
        vcf_file_gzip = f"{sample_out_dir}/pilon/{{sample_ID}}_full.vcf.gz",
    output:
        vcf_SNPs = f"{sample_out_dir}/pilon/{{sample_ID}}_SNPs.vcf",
        vcf_indels = f"{sample_out_dir}/pilon/{{sample_ID}}_indels.vcf",
    conda:
        f"{primary_directory}/envs/variant_calling.yaml"
    shell:
        """
        # save a VCF file of the SNP variants. Anything with BC >= 5 for at least 2 alleles       
        bcftools filter -i '(strlen(REF) == strlen(ALT)) & ((REF=="A" && ((BC[1]>=5) || (BC[2]>=5) || (BC[3]>=5))) || (REF=="C" && ((BC[0]>=5) || (BC[2]>=5) || (BC[3]>=5))) || (REF=="G" && ((BC[0]>=5) || (BC[1]>=5) || (BC[3]>=5))) || (REF=="T" && ((BC[0]>=5) || (BC[1]>=5) || (BC[2]>=5))))' {input.vcf_file_gzip} > {output.vcf_SNPs}
        
        # indels won't be included, so save a separate file of indels, where IC >= 5 or DC >= 5
        bcftools filter -i "(strlen(REF) != strlen(ALT)) & (IC >= 5 | DC >= 5)" {input.vcf_file_gzip} > {output.vcf_indels}
        """



rule get_PASS_Amb_VCFs:
    input:
        vcf_file_gzip = f"{sample_out_dir}/pilon/{{sample_ID}}_full.vcf.gz",
    output:
        vcf_file_bgzip = f"{sample_out_dir}/pilon/{{sample_ID}}.PASSorAmb.vcf.bgz",
        vcf_file_bgzip_index = f"{sample_out_dir}/pilon/{{sample_ID}}.PASSorAmb.vcf.bgz.tbi",
    conda:
        "/home/sak0914/anaconda3/envs/MtbQuantCNN"
    shell:
        """
        zcat {input.vcf_file_gzip} | \
        awk 'BEGIN{{FS="\\t";OFS="\\t"}} /^#/ {{print; next}} $7=="PASS" || $7=="Amb" {{print}}' | \
        bgzip > {output.vcf_file_bgzip} && \
        
        tabix -p vcf {output.vcf_file_bgzip} -f
        """




rule annotate_exclude_low_confidence_regions_pilon_VCF:
    input:
        vcf_SNPs = f"{sample_out_dir}/pilon/{{sample_ID}}_SNPs.vcf",
        vcf_indels = f"{sample_out_dir}/pilon/{{sample_ID}}_indels.vcf",
    output:
        field_names = temp(f"{sample_out_dir}/pilon/field_names.txt"),
        
        vcf_SNPs_annot = f"{sample_out_dir}/pilon/{{sample_ID}}_SNPs.eff.vcf",
        vcf_indels_annot = f"{sample_out_dir}/pilon/{{sample_ID}}_indels.eff.vcf",
                
        vcf_exclude_low_conf_SNPs = f"{sample_out_dir}/pilon/{{sample_ID}}_SNPs.excludeLowConf.eff.vcf",
        tsv_exclude_low_conf_SNPs = f"{sample_out_dir}/pilon/{{sample_ID}}_SNPs.excludeLowConf.tsv",
        
        vcf_exclude_low_conf_indels = f"{sample_out_dir}/pilon/{{sample_ID}}_indels.excludeLowConf.eff.vcf",
        tsv_exclude_low_conf_indels = f"{sample_out_dir}/pilon/{{sample_ID}}_indels.excludeLowConf.tsv",
    conda:
        f"{conda_directory}/envs/variant_annotation.yaml"
    params:
        snpEff_db = config['snpEff_db'],
        exclude_regions = os.path.join(primary_directory, references_dir, config['exclude_regions_file']),
    shell:
        """
        # annotate with snpeff
        snpEff eff {params.snpEff_db} -noStats -no-downstream -no-upstream -lof {input.vcf_SNPs} > {output.vcf_SNPs_annot}
        snpEff eff {params.snpEff_db} -noStats -no-downstream -no-upstream -lof {input.vcf_indels} > {output.vcf_indels_annot}

        # make a TSV file version of each of the VCF subsets made above (and the full one after excluding low confidence sites)
        # get all field names because not sure which ones we will need for low AF variant detection
        echo -e "POS\nREF\nALT\nQUAL\nFILTER\nANN[0].GENE\nANN[0].HGVS_C\nANN[0].HGVS_P" > {output.field_names}
        grep "^##INFO=<ID=" {output.vcf_SNPs_annot} | cut -d'=' -f3 | cut -d',' -f1 >> {output.field_names}

        # exclude low confidence regions and save
        bedtools subtract -a {output.vcf_SNPs_annot} -b {params.exclude_regions} -header > {output.vcf_exclude_low_conf_SNPs}
        SnpSift extractFields {output.vcf_exclude_low_conf_SNPs} $(paste -sd " " {output.field_names}) > {output.tsv_exclude_low_conf_SNPs}   
        
        bedtools subtract -a {output.vcf_indels_annot} -b {params.exclude_regions} -header > {output.vcf_exclude_low_conf_indels}
        SnpSift extractFields {output.vcf_exclude_low_conf_indels} $(paste -sd " " {output.field_names}) > {output.tsv_exclude_low_conf_indels}
        """
        


rule create_lineage_helper_files:
    input:
        vcf_file_gzip = f"{sample_out_dir}/pilon/{{sample_ID}}_full.vcf.gz",
    params:
        lineage_pos_for_F2 = os.path.join(primary_directory, references_dir, "phylogeny", "Coll2014_positions_all.txt"),
        output_dir = output_dir,
    output:
        bcf_file = f"{sample_out_dir}/lineage/{{sample_ID}}.bcf",
        bcf_index_file = f"{sample_out_dir}/lineage/{{sample_ID}}.bcf.csi",
        vcf_lineage_positions = f"{sample_out_dir}/lineage/{{sample_ID}}_lineage_positions.vcf",
    conda:
        f"{primary_directory}/envs/variant_calling.yaml"
    shell:
        """
        # convert the full VCF file to a BCF fileto get only the lineage-defining positions according to the Coll 2014 scheme
        bcftools view {input.vcf_file_gzip} -O b -o {output.bcf_file}

        # index bcf file
        bcftools index {output.bcf_file}

        # create VCF file of just the lineage positions, which will be used by the F2 metric script. Per the documentation, if --regions-file is a tab-delimited file, then it needs two columns (CHROM and POS), and POS is 1-indexed and inclusive
        # THIS IS DIFFERENT BEHAVIOR FROM IF IT WAS A BED FILE OR IF YOU USE BEDTOOLS. IN BOTH OF THOSE CASES, YOU NEED THREE COLUMNS (CHROM, BEG, AND END), AND THEY ARE 0-INDEXED WITH END BEING EXCLUSIVE (I.E. HALF-OPEN)
        bcftools view {output.bcf_file} --regions-file {params.lineage_pos_for_F2} -O v -o {output.vcf_lineage_positions}   
        """


rule calculate_strain_mixing:
    input:
        bcf_file = f"{sample_out_dir}/lineage/{{sample_ID}}.bcf",
        bcf_index_file = f"{sample_out_dir}/lineage/{{sample_ID}}.bcf.csi",
        vcf_lineage_positions = f"{sample_out_dir}/lineage/{{sample_ID}}_lineage_positions.vcf",
        vcf_file_gzip = f"{sample_out_dir}/pilon/{{sample_ID}}_full.vcf.gz",
    params:
        lineage_SNP_info = os.path.join(primary_directory, references_dir, "phylogeny", "Coll2014_SNPs_all.csv"),
        FN_metric_script = os.path.join(primary_directory, scripts_dir, "calculate_strain_mixing_VCF.py"),
        output_dir = output_dir,        
    output:
        strain_mixing_table = f"{sample_out_dir}/lineage/mixing_scores_Coll2014.csv",
        minor_allele_fractions_output = f"{sample_out_dir}/lineage/minor_allele_fractions.csv",        
    shell:
        """
        python3 -u {params.FN_metric_script} -i {params.output_dir}/{wildcards.sample_ID} -o {output.minor_allele_fractions_output} --lineage-file {params.lineage_SNP_info} --num_max_mixed_lineages 3
        
        rm {input.bcf_file} {input.bcf_index_file} {input.vcf_lineage_positions}        
        """
        
        
        
rule lineage_typing:
    input:
        vcf_file_gzip = f"{sample_out_dir}/pilon/{{sample_ID}}_full.vcf.gz",
    output:
        vcf_file_gunzip = temp(f"{sample_out_dir}/lineage/{{sample_ID}}_full.vcf"),
        fast_lineage_caller_output = f"{sample_out_dir}/lineage/fast_lineage_caller_output.txt",
    params:
        lineage_SNP_info = os.path.join(primary_directory, references_dir, "phylogeny", "Coll2014_SNPs_all.csv"),
        F2_metric_script = os.path.join(primary_directory, scripts_dir, "calculate_strain_mixing_VCF.py"),
        output_dir = output_dir,   
    shell:
        """
        # fast-lineage-caller won't work on gzipped files, so need to unzip it first. 
        # It doesn't even error when you pass in a gzipped file. It just returns nothing, making debugging difficult
        gunzip -c {input.vcf_file_gzip} > {output.vcf_file_gunzip}
        
        fast-lineage-caller {output.vcf_file_gunzip} --pass --out {output.fast_lineage_caller_output}
        """
        
        
        
rule generate_VCF_for_TBtypeR:
    input:
        merged_bam_file = f"{sample_out_dir}/bam/{{sample_ID}}.dedup.bam",
    output:
        tbtypeR_VCF_file_init = temp(f"{sample_out_dir}/TBtypeR/{{sample_ID}}.vcf"),
        tbtypeR_VCF_file = f"{sample_out_dir}/TBtypeR/{{sample_ID}}.vcf.gz",
    conda:
        f"{primary_directory}/envs/variant_calling.yaml"
    params:
        ref_genome = os.path.join(primary_directory, references_dir, "ref_genome", "H37Rv_NC_000962.3.fna"),
        tbtypeR_targets = "/home/sak0914/MtbLongitudinalDiversity/direct_sputum/tbtyper_targets_Chromosome.tsv"
    threads:
        8
    shell:
        """
        bcftools mpileup {input.merged_bam_file} -q 1 -Q 10 -d 1000 -f {params.ref_genome} -a FMT/AD -Ou --threads {threads} \
          | bcftools call --ploidy 1 -A -m --prior 1e-2 -C alleles -T {params.tbtypeR_targets} -Ou --threads {threads} \
          | bcftools annotate -x INFO,^FORMAT/GT,^FORMAT/AD -Ov -o {output.tbtypeR_VCF_file_init} --threads {threads}
          
        # rename Chromosome to NC_000962.3 for compatibility
        sed 's/Chromosome/NC_000962.3/g' {output.tbtypeR_VCF_file_init} | gzip -c > {output.tbtypeR_VCF_file}
        """
        
        
        
rule remove_low_quality_sites_TBtypeR_VCF:
    input:
        tbtypeR_VCF_file = f"{sample_out_dir}/TBtypeR/{{sample_ID}}.vcf.gz",
    output:
        tbtypeR_VCF_file_filtered = f"{sample_out_dir}/TBtypeR/{{sample_ID}}.filtered.vcf.gz",
    conda:
        f"{primary_directory}/envs/variant_calling.yaml"
    params:
        exclude_low_mappability_regions = "/home/sak0914/MtbLongitudinalDiversity/lowAF_variant_calling/references/BED_files/exclude_regions_50bp.EBRlessThan95Percent_refseq.bed",
        exclude_low_quality_unfixed_SNV_sites = "/home/sak0914/MtbLongitudinalDiversity/lowAF_variant_calling/references/BED_files/exclude_unfixed_SNV_sites.bed",
    shell:
        """
        bedtools subtract -a {input.tbtypeR_VCF_file} -b {params.exclude_low_mappability_regions} -header | bedtools subtract -a '-' -b {params.exclude_low_quality_unfixed_SNV_sites} -header | bcftools sort | gzip -c > {output.tbtypeR_VCF_file_filtered}
        """