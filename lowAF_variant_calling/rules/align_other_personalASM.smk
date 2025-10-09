import os, glob
import numpy as np
import pandas as pd

# define some paths to make the path names more readable
sample_out_dir = f"{output_dir}/{{sample_ID}}/personal_ref_genomes/{assembly_sample}"

ref_genome = f"{assembly_dir}/{assembly_sample}/assembly/{assembly_sample}.fasta"

scripts_dir = config["scripts_dir"]
references_dir = config["references_dir"]

conda_directory = config['conda_dir']
primary_directory = os.getcwd()


rule align_reads_mark_duplicates:
    input:
        fastq1_trimmed_classified = f"{output_dir}/{{sample_ID}}/{{sample_ID}}/kraken/{{sample_ID}}.R1.kraken.filtered.fastq.gz",
        fastq2_trimmed_classified = f"{output_dir}/{{sample_ID}}/{{sample_ID}}/kraken/{{sample_ID}}.R2.kraken.filtered.fastq.gz",
    output:
        sam_file = temp(f"{sample_out_dir}/bam/{{sample_ID}}.sam"),
        bam_file = temp(f"{sample_out_dir}/bam/{{sample_ID}}.bam"),
        bam_index_file = temp(f"{sample_out_dir}/bam/{{sample_ID}}.bam.bai"),
        bam_file_markdup = f"{sample_out_dir}/bam/{{sample_ID}}.markDup.bam",
        bam_file_markdup_metrics = f"{sample_out_dir}/bam/{{sample_ID}}.markDup.bam.metrics",
        bam_index_file_markdup = f"{sample_out_dir}/bam/{{sample_ID}}.markDup.bam.bai",
    params:
        output_dir = output_dir,
        personal_ref_genome = ref_genome,
        bwa_mem_seed_length = config['bwa_mem_seed_length'],
    conda:
        f"{conda_directory}/envs/read_processing_aln_bwa.yaml"
    threads:
        8
    shell:
        """
        # index reference genome (which is required before aligning reads)
        bwa index {params.personal_ref_genome}
        
        # align reads
        bwa mem -M -R "@RG\\tID:{wildcards.sample_ID}\\tSM:{wildcards.sample_ID}" \
                    -k {params.bwa_mem_seed_length} \
                    -t {threads} \
                    {params.personal_ref_genome} \
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
        personal_ref_genome = ref_genome,
        sample_out_dir = sample_out_dir,
    output:
        depth_file = f"{sample_out_dir}/bam/{{sample_ID}}.depth.tsv.gz",
        depth_file_properly_paired_only = f"{sample_out_dir}/bam/{{sample_ID}}.properlyPaired.depth.tsv.gz",
    conda:
        f"{conda_directory}/envs/read_processing_aln.yaml"
    shell:
        """
        # get all runs associated with this sample_ID and compute depth
        # -a computes depth at all positions, not just those with non-zero depth
        # -Q is for minimum mapping quality: use 1, so that multiply mapped reads aren't counted. These have mapping quality of 0
        samtools depth -a --min-MQ 1 {input.bam_file_markdup} | gzip -c > {output.depth_file}
        
        # include only properly paired reads in the coverage count
        samtools depth -a --incl-flags 0x2 --min-MQ 1 {input.bam_file_markdup} | gzip -c > {output.depth_file_properly_paired_only}
        """




rule freebayes_variant_calling:
    input:
        merged_bam_file = f"{sample_out_dir}/bam/{{sample_ID}}.markDup.bam",
    output:
        vcf_file_init = temp(f"{sample_out_dir}/freebayes/{{sample_ID}}.init.vcf"),
        vcf_file_norm = temp(f"{sample_out_dir}/freebayes/{{sample_ID}}.norm.vcf"),
        vcf_file = f"{sample_out_dir}/freebayes/{{sample_ID}}.vcf",
    params:
        personal_ref_genome = ref_genome,
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
        freebayes -f {params.personal_ref_genome} \
                  -p 1 \
                  --min-alternate-count 2 \
                  --min-alternate-fraction 0.01 \
                  --min-mapping-quality 40 \
                  --min-base-quality 30 \
                  -b {input.merged_bam_file} \
                  -v {output.vcf_file_init}

        # left-align and deduplicate variants with the same POS, REF, and ALT in the full VCF file
        bcftools norm --rm-dup none --fasta-ref {params.personal_ref_genome} {output.vcf_file_init} | bcftools sort > {output.vcf_file_norm}

        # Split (split = '-' before any, join = '+') multi-allelic sites for easier parsing of the variants. But bcftools norm will not do --rm-dup and --multiallelics in the same step
        # to be safe, sort again before saving
        bcftools norm --multiallelics -any {output.vcf_file_norm} | bcftools sort > {output.vcf_file}
        """
        
        


rule save_TSV_files_of_VCF_files:
    input:
        vcf_file = f"{sample_out_dir}/freebayes/{{sample_ID}}.vcf",
    output:
        freebayes_tsv_file = f"{sample_out_dir}/freebayes/{{sample_ID}}.excludeLowConf.tsv",
    conda:
        f"{conda_directory}/envs/variant_annotation.yaml"
    params:
        exclude_regions_file = "~/MtbLongitudinalDiversity/lowAF_variant_calling/references/BED_files/exclude_regions_50bp.EBRlessThan95Percent.bed",
    shell:
        """
        # save a CSV file of the freebayes variants. For freebayes, AF is 0 or 1 in haploid organisms
        SnpSift extractFields {input.vcf_file} CHROM POS REF ALT QUAL FILTER DP DPB RO AO MQM MQMR SRF SRR SAF SAR SRP SAP RPP RPPR RPL RPR > {output.freebayes_tsv_file}
        """
        
        
        
        
rule filter_high_quality_lowAF_variants:
    input:
        depth_file_properly_paired_only = f"{sample_out_dir}/bam/{{sample_ID}}.properlyPaired.depth.tsv.gz",
        freebayes_tsv_file = f"{sample_out_dir}/freebayes/{{sample_ID}}.excludeLowConf.tsv",
    output:
        lowAF_variants_bed_file = f"{sample_out_dir}/freebayes/lowAF_variants.bed",
    params:
        personal_ref_genome = ref_genome,
        output_dir = sample_out_dir,
    shell:
        """
        # this scripts writes the lowAF variants BED file. The variants are in the coordinates of the personal ref genome

        set +u
        eval "$(conda shell.bash hook)"
        conda activate bayesian_modeling
        set -u
                
        python3 -u ~/MtbLongitudinalDiversity/lowAF_variant_calling/variantDetector/01_write_lowAF_BED.py \
                -d {input.depth_file_properly_paired_only} \
                -i {input.freebayes_tsv_file} \
                -o {output.lowAF_variants_bed_file} \
                -g {params.personal_ref_genome}
        """

   
   
   
rule liftover_variants_between_two_personal_ref_genome_coords:
    input:
        personal_ref_genome = lambda wildcards: sample_asm_dict[wildcards.sample_ID],
        lowAF_variants_bed_file = f"{sample_out_dir}/freebayes/lowAF_variants.bed",
    output:
        paf_file = f"{sample_out_dir}/assembly/{{sample_ID}}.{assembly_sample}.paf",
        lowAF_variants_bed_file = f"{sample_out_dir}/freebayes/lowAF_variants.{{sample_ID}}.bed",
    params:
        other_personal_ref_genome = ref_genome,
    conda:
        f"{conda_directory}/envs/long_read_aln.yaml"
    threads:
        8
    shell:
        """
        # generate paf file. Target = first, query = second
        minimap2 -x asm5 -c --cs {input.personal_ref_genome} {params.other_personal_ref_genome} > {output.paf_file}
        
        # convert the coordinates from the closest personal genome to the actual personal reference genome. This only works if there is one for the exact sample
        paftools.js liftover {output.paf_file} {input.lowAF_variants_bed_file} > {output.lowAF_variants_bed_file}
        """
     
        
        
        
rule exclude_regions_from_other_personal_ref_genome:
    input:
        exclude_regions_BED_file = f"{assembly_dir}/{assembly_sample}/assembly/exclude_regions_50bp.EBRlessThan95Percent.bed",
        lowAF_variants_bed_file = f"{sample_out_dir}/freebayes/lowAF_variants.bed",
        two_personal_ref_genomes_paf_file = f"{sample_out_dir}/assembly/{{sample_ID}}.{assembly_sample}.paf",
    output:
        lowAF_variants_excludeLowConf_bed_file = f"{sample_out_dir}/freebayes/lowAF_variants.excludeLowConf.bed",
        lowAF_variants_excludeLowConf_personalCoords_bed_file = f"{sample_out_dir}/freebayes/lowAF_variants.excludeLowConf.{{sample_ID}}.bed",
    params:
        personal_ref_genome = ref_genome,
    conda:
        f"{conda_directory}/envs/long_read_aln.yaml"
    threads:
        8
    shell:
        """
        # remove the low confidence regions in the coordinates of the other personal ref genome
        bedtools subtract -a {input.lowAF_variants_bed_file} -b {input.exclude_regions_BED_file} -header > {output.lowAF_variants_excludeLowConf_bed_file}
        
        # then transform to the coordinates of the actual sample
        paftools.js liftover {input.two_personal_ref_genomes_paf_file} {output.lowAF_variants_excludeLowConf_bed_file} > {output.lowAF_variants_excludeLowConf_personalCoords_bed_file}
        """
        
        
        
        
rule liftover_variants_from_personal_genome_coords_to_H37Rv_coords:
    input:
        lowAF_variants_bed_file = f"{sample_out_dir}/freebayes/lowAF_variants.bed",
    output:
        paf_file = f"{sample_out_dir}/assembly/H37Rv.{assembly_sample}.paf",
        lowAF_variants_H37Rv_bed_file = f"{sample_out_dir}/freebayes/lowAF_variants.H37Rv.bed",
        lowAF_variants_H37Rv_excludeLowConf_bed_file = f"{sample_out_dir}/freebayes/lowAF_variants.H37Rv.excludeLowConf.bed",
    params:
        personal_ref_genome = ref_genome,
        H37Rv_genome = "/home/sak0914/MtbLongitudinalDiversity/lowAF_variant_calling/references/ref_genome/H37Rv_NC_000962.3.fna", #os.path.join(primary_directory, "references", "ref_genome", "H37Rv_NC_000962.3.fna"),
        exclude_H37Rv_regions_BED_file = "~/MtbLongitudinalDiversity/lowAF_variant_calling/references/BED_files/exclude_regions_50bp.EBRlessThan95Percent.bed",
    conda:
        f"{conda_directory}/envs/long_read_aln.yaml"
    threads:
        8
    shell:
        """
        # generate paf file. Target = H37Rv first, then query = personal genome
        minimap2 -x asm5 -c --cs {params.H37Rv_genome} {params.personal_ref_genome} > {output.paf_file}
        
        # convert the coordinates from personal genome to H37Rv
        paftools.js liftover {output.paf_file} {input.lowAF_variants_bed_file} > {output.lowAF_variants_H37Rv_bed_file}
        
        # We removed them from the short reads aligned to H37Rv, so have to remove them here too to avoid artifactual false negatives
        bedtools subtract -a {output.lowAF_variants_H37Rv_bed_file} -b {params.exclude_H37Rv_regions_BED_file} -header > {output.lowAF_variants_H37Rv_excludeLowConf_bed_file}
        """
        
        
rule generate_exclude_regions_personal_ref_genome_coordinates:
    params:
        personal_ref_genome = lambda w: sample_asm_dict[w.sample_ID],
        H37Rv_genome = "/home/sak0914/MtbLongitudinalDiversity/lowAF_variant_calling/references/ref_genome/H37Rv_NC_000962.3.fna",
        exclude_H37Rv_regions_BED_file = "~/MtbLongitudinalDiversity/lowAF_variant_calling/references/BED_files/exclude_regions_50bp.EBRlessThan95Percent.bed",
    output:
        exclude_regions_BED_file = f"{sample_out_dir}/assembly/exclude_regions_50bp.EBRlessThan95Percent.bed",
        H37Rv_other_personal_paf_file = f"{sample_out_dir}/assembly/H37Rv.{{sample_ID}}.paf",
    conda:
        f"{conda_directory}/envs/long_read_aln.yaml"
    shell:
        """
        # generate paf file. Target = personal genome first, then query = H37Rv second
        minimap2 -x asm5 -c --cs {params.personal_ref_genome} {params.H37Rv_genome} > {output.H37Rv_other_personal_paf_file}
        
        # convert the exclude coordinates from H37Rv to personal ref genome
        paftools.js liftover {output.H37Rv_other_personal_paf_file} {params.exclude_H37Rv_regions_BED_file} > {output.exclude_regions_BED_file}
        """