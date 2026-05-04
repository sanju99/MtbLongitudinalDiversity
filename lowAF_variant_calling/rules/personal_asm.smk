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


rule align_reads_mark_duplicates:
    input:
        fastq1_trimmed_classified = f"{sample_H37Rv_ref_dir}/{{sample_ID}}/kraken/{{sample_ID}}.R1.kraken.filtered.fastq.gz",
        fastq2_trimmed_classified = f"{sample_H37Rv_ref_dir}/{{sample_ID}}/kraken/{{sample_ID}}.R2.kraken.filtered.fastq.gz",  
    output:
        sam_file = temp(f"{sample_out_dir}/bam/{{sample_ID}}.sam"),
        bam_file = temp(f"{sample_out_dir}/bam/{{sample_ID}}.bam"),
        bam_index_file = temp(f"{sample_out_dir}/bam/{{sample_ID}}.bam.bai"),
        bam_file_markDup = f"{sample_out_dir}/bam/{{sample_ID}}.markDup.bam",
        bam_file_markDup_metrics = f"{sample_out_dir}/bam/{{sample_ID}}.markDup.bam.metrics",
        bam_index_file_markDup = f"{sample_out_dir}/bam/{{sample_ID}}.markDup.bam.bai",
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
        picard -Xmx10g markDuplicates I={output.bam_file} O={output.bam_file_markDup} REMOVE_DUPLICATES=false M={output.bam_file_markDup_metrics} ASSUME_SORT_ORDER=coordinate READ_NAME_REGEX='(?:.*.)?([0-9]+)[^.]*.([0-9]+)[^.]*.([0-9]+)[^.]*$'

        # index the markDuplicated alignment with samtools, which will create a markDup_bam_file.bai file
        samtools index {output.bam_file_markDup}
        """

        


rule get_BAM_file_depths:
    input:
        bam_file_markDup = f"{sample_out_dir}/bam/{{sample_ID}}.markDup.bam",
    params:
        ref_genome = lambda w: sample_asm_dict[w.sample_ID],
        sample_out_dir = sample_out_dir,
    output:
        depth_file = f"{sample_out_dir}/bam/{{sample_ID}}.depth.tsv.gz",
    conda:
        f"{conda_directory}/envs/read_processing_aln.yaml"
    shell:
        """
        # get all runs associated with this sample_ID and compute depth
        # -a computes depth at all positions, not just those with non-zero depth
        # -Q is for minimum mapping quality: use 1, so that multiply mapped reads aren't counted. These have mapping quality of 0
        samtools depth -a -Q 1 {input.bam_file_markDup} | gzip -c > {output.depth_file}
        """




rule pilon_variant_calling:
    input:
        merged_bam_file = f"{sample_out_dir}/bam/{{sample_ID}}.markDup.bam",
    output:
        vcf_file = temp(f"{sample_out_dir}/pilon/{{sample_ID}}.vcf"),
        vcf_file_gzip = f"{sample_out_dir}/pilon/{{sample_ID}}_full.vcf.gz",
        fasta_file = temp(f"{sample_out_dir}/pilon/{{sample_ID}}.fasta"),        
    params:
        ref_genome = lambda w: sample_asm_dict[w.sample_ID],
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
        # need to include ALT == '.' because if ALT = '.' (meaning REF and ALT are the same), it does not treat . as length 1. It's probably length 0. So you will miss low AF things.
        bcftools filter -i '(strlen(REF) == strlen(ALT) || ALT == ".") & ((REF=="A" && ((BC[1]>=5) || (BC[2]>=5) || (BC[3]>=5))) || (REF=="C" && ((BC[0]>=5) || (BC[2]>=5) || (BC[3]>=5))) || (REF=="G" && ((BC[0]>=5) || (BC[1]>=5) || (BC[3]>=5))) || (REF=="T" && ((BC[0]>=5) || (BC[1]>=5) || (BC[2]>=5))))' {input.vcf_file_gzip} > {output.vcf_SNPs}

        # indels won't be included, so save a separate file of indels, where IC >= 5 or DC >= 5
        bcftools filter -i "(strlen(REF) != strlen(ALT) || ALT == '.') & (IC >= 5 | DC >= 5)" {input.vcf_file_gzip} > {output.vcf_indels}
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
        # left-align indels and decompose complex variants above
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
        
        
        
        
rule filter_pilon_high_quality_lowAF_SNVs:
    input:
        bam_file = f"{sample_out_dir}/bam/{{sample_ID}}.markDup.bam",
        depth_file_gzip = f"{sample_out_dir}/bam/{{sample_ID}}.depth.tsv.gz",
        pilon_VCF_file = f"{sample_out_dir}/pilon/{{sample_ID}}_SNPs.vcf",
    output:
        lowAF_SNVs_bed_file = f"{sample_out_dir}/pilon/lowAF_SNVs.bed",
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
        python3 -u ~/MtbLongitudinalDiversity/lowAF_variant_calling/pilonVariantDetector/01_write_lowAF_BED.py \
                -v {input.pilon_VCF_file} \
                -b {input.bam_file} \
                -d {input.depth_file_gzip} \
                -o {output.lowAF_SNVs_bed_file} \
                -g {params.ref_genome} \
                --BED
        """

        
        
rule generate_paf_file:
    output:
        personal_H37Rv_paf_file = f"{sample_out_dir}/assembly/{{sample_ID}}.H37Rv.paf",
    conda:
        f"{conda_directory}/envs/long_read_aln.yaml"
    params:
        ref_genome = lambda w: sample_asm_dict[w.sample_ID],
        H37Rv_genome = os.path.join(primary_directory, "references", "ref_genome", "H37Rv_NC_000962.3.fna"),
    shell:
        """
        # generate paf file. Target = H37Rv first, then query = personal genome second
        minimap2 -x asm5 -c --cs {params.H37Rv_genome} {params.ref_genome} --secondary=no > {output.personal_H37Rv_paf_file}
        """
       
        
        
rule transfer_lowAF_variants_to_H37Rv_coordinates:
    input:
        lowAF_variants_bed_file = f"{sample_out_dir}/freebayes/lowAF_variants.bed",
        personal_H37Rv_paf_file = f"{sample_out_dir}/assembly/{{sample_ID}}.H37Rv.paf",
    output:
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
        # transfer lowAF variants from personal to H37Rv coordinates
        paftools.js liftover {input.personal_H37Rv_paf_file} {input.lowAF_variants_bed_file} > {output.lowAF_variants_H37Rv_bed_file}
        
        # finally remove the low confidence variants now that we have the variants in H37Rv coordinates
        bedtools subtract -a {output.lowAF_variants_H37Rv_bed_file} -b {params.exclude_regions} -header > {output.lowAF_variants_H37Rv_excludeLowConf_bed_file}
        """
        
        
        
rule transfer_pilon_lowAF_SNVs_to_H37Rv_coordinates:
    input:
        lowAF_SNVs_bed_file = f"{sample_out_dir}/pilon/lowAF_SNVs.bed",
    output:
        personal_H37Rv_paf_file = f"{sample_out_dir}/assembly/{{sample_ID}}.H37Rv.paf",
        lowAF_SNVs_H37Rv_bed_file = f"{sample_out_dir}/pilon/lowAF_SNVs.H37Rv.bed",
        lowAF_SNVs_H37Rv_excludeLowConf_bed_file = f"{sample_out_dir}/pilon/lowAF_SNVs.H37Rv.excludeLowConf.bed",
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
        paftools.js liftover {output.personal_H37Rv_paf_file} {input.lowAF_SNVs_bed_file} > {output.lowAF_SNVs_H37Rv_bed_file}
        
        # finally remove the low confidence variants now that we have the variants in H37Rv coordinates
        bedtools subtract -a {output.lowAF_SNVs_H37Rv_bed_file} -b {params.exclude_regions} -header > {output.lowAF_SNVs_H37Rv_excludeLowConf_bed_file}
        """
        
        

rule merge_results_H37Rv_personal_genome_calling:
    input:
        lowAF_variants_bed_file = f"{sample_out_dir}/freebayes/lowAF_variants.bed",
        lowAF_variants_transferred_to_H37Rv_excludeLowConf_bed_file = f"{sample_out_dir}/freebayes/lowAF_variants.H37Rv.excludeLowConf.bed",
        lowAF_variants_H37Rv_excludeLowConf_tsv_file = f"{sample_H37Rv_ref_dir}/freebayes/{{sample_ID}}.cleaned.excludeLowConf.fixedAO.tsv",
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
        
        
        
        
rule merge_results_H37Rv_personal_genome_calling_pilon:
    input:
        lowAF_SNVs_bed_file = f"{sample_out_dir}/pilon/lowAF_SNVs.bed",
        lowAF_SNVs_transferred_to_H37Rv_excludeLowConf_bed_file = f"{sample_out_dir}/pilon/lowAF_SNVs.H37Rv.excludeLowConf.bed",
        lowAF_SNVs_H37Rv_excludeLowConf_tsv_file = f"{sample_H37Rv_ref_dir}/pilon/lowAF_SNVs.csv",
    output:
        f"{sample_out_dir}/pilon/ground_truth.csv",
        f"{sample_out_dir}/pilon/H37Rv_detected.csv",
        f"{sample_out_dir}/pilon/confusion_matrix.csv",
    params:
        sample_ID = f"{{sample_ID}}",
    # conda:
    #    f"{conda_directory}/envs/python_bioinformatics_utils.yaml"
    shell:
        """ 
        source activate /home/sak0914/anaconda3/envs/python_bioinformatics_utils
        
        python3 -u ~/MtbLongitudinalDiversity/lowAF_variant_calling/pilonVariantDetector/02_combine_lowAF_SNVs.py \
                -s {params.sample_ID} \
                -bed1 {input.lowAF_SNVs_bed_file} \
                -bed2 {input.lowAF_SNVs_transferred_to_H37Rv_excludeLowConf_bed_file} \
                -csv {input.lowAF_SNVs_H37Rv_excludeLowConf_tsv_file}
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
        source activate /home/sak0914/anaconda3/envs/liftoff
        
        liftoff -g {params.H37Rv_gff_file} \
                -o {output.liftoff_gff_file} \
                -copies -polish \
                -dir {output.liftoff_intermediate_dir} \
                {input.ref_genome} {params.H37Rv_genome}
        """
        
        
        
rule align_PB_reads_to_H37Rv:
    input:
        PB_reads_fq = lambda wildcards: sample_PB_dict[wildcards.sample_ID]
    output:
        BAM_file = f"{sample_out_dir}/SV_detection/{{sample_ID}}.bam",
        BAM_index_file = f"{sample_out_dir}/SV_detection/{{sample_ID}}.bam.bai",
    params:
        H37Rv_genome = os.path.join(primary_directory, "references", "ref_genome", "H37Rv_NC_000962.3.fna"),
    shell:
        """
        source activate /home/sak0914/anaconda3/envs/SV_detection
        
        minimap2 -ax map-hifi {params.H37Rv_genome} {input.PB_reads_fq} | samtools sort > {output.BAM_file}
        
        samtools index {output.BAM_file}
        """



rule call_SVs_sniffles:
    input:
        BAM_file = f"{sample_out_dir}/SV_detection/{{sample_ID}}.bam",
        BAM_index_file = f"{sample_out_dir}/SV_detection/{{sample_ID}}.bam.bai",
    output:
        SV_VCF = f"{sample_out_dir}/SV_detection/{{sample_ID}}.noQC.vcf",
    params:
        H37Rv_genome = os.path.join(primary_directory, "references", "ref_genome", "H37Rv_NC_000962.3.fna"),
    shell:
        """
        source activate /home/sak0914/anaconda3/envs/SV_detection
        
        sniffles -i {input.BAM_file} -v {output.SV_VCF} --reference {params.H37Rv_genome} --allow-overwrite --no-qc
        """
        
        
rule normalize_SVs_write_TSV:
    input:
        SV_VCF = f"{sample_out_dir}/SV_detection/{{sample_ID}}.noQC.vcf",
    output:
        norm_SV_VCF = f"{sample_out_dir}/SV_detection/{{sample_ID}}.noQC.norm.vcf",
        norm_SV_TSV = f"{sample_out_dir}/SV_detection/{{sample_ID}}.noQC.norm.tsv",
    params:
        H37Rv_genome = os.path.join(primary_directory, "references", "ref_genome", "H37Rv_NC_000962.3.fna"),
    shell:
        """
        source activate /home/sak0914/Mtb_Megapipe/.snakemake/conda/a127fe87b4a6b9a6a5f6a81ead3133a8_
        
        bcftools norm --rm-dup none --fasta-ref {params.H37Rv_genome} {input.SV_VCF} | bcftools sort > {output.norm_SV_VCF}
        
        SnpSift extractFields {output.norm_SV_VCF} POS REF ALT QUAL FILTER SVTYPE SVLEN END COVERAGE VAF -e "" > {output.norm_SV_TSV}
        """