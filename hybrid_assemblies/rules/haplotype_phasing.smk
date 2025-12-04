import numpy as np
import pandas as pd

configfile: "config.yaml"

### Define PATHs to files defined in thoe config file ###
refGenome_FA_PATH = config["RefGenome_FA_PATH"]

# Define PATH of main OUTPUT directory
output_dir = config["output_dir"]

primary_directory = "/home/sak0914/MtbLongitudinalDiversity/hybrid_assemblies"

# Read in data regarding input 
input_DataInfo_DF = pd.read_csv(config["inputSampleData"])

# drop samples that don't have an original ID (SNNN-NN type ID)
input_DataInfo_DF = input_DataInfo_DF.dropna(subset=['Original_ID', 'PacBio_FQ_PATH', 'Illumina_ID']).reset_index(drop=True)

input_All_SampleIDs = list( input_DataInfo_DF["Original_ID"].values )

SampleID_To_PB_FQ_Dict = dict(input_DataInfo_DF[["Original_ID", "PacBio_FQ_PATH"]].values)

rule all:
    input:
        [f"{output_dir}/{sample}/TBtypeR/haplotype_{num}.vcf.gz" for sample in input_All_SampleIDs for num in [1, 2]],
        # [f"{output_dir}/{sample}/variants/{sample}.phased.SNPs.excludeLowConf.vcf" for sample in input_All_SampleIDs]
        
sample_out_dir = f"{output_dir}/{{sample_ID}}"


rule align_PB_reads_H37Rv:
    input:
        PacBio_reads = lambda wildcards: SampleID_To_PB_FQ_Dict[wildcards.sample_ID]
    output:
        bam_file = f"{sample_out_dir}/bam/{{sample_ID}}.bam",
    params:
        ref_genome = refGenome_FA_PATH,
    conda:
        f"{primary_directory}/envs/haplotype_phasing.yaml",
    threads:
        8
    shell:
        """
        minimap2 -ax map-hifi {params.ref_genome} {input.PacBio_reads} -t {threads} | samtools view -b | samtools sort -o {output.bam_file}
        
        samtools index {output.bam_file}
        """
        
        
rule run_longcallD:
    input:
        bam_file = f"{sample_out_dir}/bam/{{sample_ID}}.bam",
    output:
        phased_bam_file = f"{sample_out_dir}/bam/{{sample_ID}}.phased.bam",
        phased_vcf_file = f"{sample_out_dir}/variants/{{sample_ID}}.phased.vcf",
        phased_SNPs_vcf_file = f"{sample_out_dir}/variants/{{sample_ID}}.phased.SNPs.excludeLowConf.vcf",
    params:
        ref_genome = refGenome_FA_PATH,
        exclude_regions_file = "/home/sak0914/MtbLongitudinalDiversity/lowAF_variant_calling/references/BED_files/exclude_regions_50bp.EBRlessThan95Percent.bed",
    conda:
        f"{primary_directory}/envs/haplotype_phasing.yaml",
    threads:
        8
    shell:
        """
        longcallD call -t{threads} {params.ref_genome} {input.bam_file} --hifi -b {output.phased_bam_file} > {output.phased_vcf_file}
        
        samtools index {output.phased_bam_file}
        
        bcftools view --types snps,mnps {output.phased_vcf_file} | bedtools subtract -a '-' -b {params.exclude_regions_file} -header > {output.phased_SNPs_vcf_file}
        """
        
        
rule split_haplotype_reads:
    input:
        phased_bam_file = f"{sample_out_dir}/bam/{{sample_ID}}.phased.bam",
    output:
        no_haplotype_reads = f"{sample_out_dir}/bam/no_haplotype_read_names.txt",
    params:
        split_haplotype_reads_script = f"{primary_directory}/scripts/split_haplotype_reads.py",
    shell:
        """
        source activate /home/sak0914/anaconda3/envs/python_bioinformatics_utils
        
        python3 -u {params.split_haplotype_reads_script} -i {input.phased_bam_file}
        """
        
        
rule split_haplotype_BAMs:
    input:
        phased_bam_file = f"{sample_out_dir}/bam/{{sample_ID}}.phased.bam",
        haplotype_1_reads = f"{sample_out_dir}/bam/haplotype_1_read_names.txt",
        haplotype_2_reads = f"{sample_out_dir}/bam/haplotype_2_read_names.txt",
    output:
        haplotype_1_bam_file = f"{sample_out_dir}/bam/haplotype_1.bam",
        haplotype_2_bam_file = f"{sample_out_dir}/bam/haplotype_2.bam",
    conda:
        f"{primary_directory}/envs/haplotype_phasing.yaml",
    threads:
        8
    shell:
        """        
        samtools view -b -h -N {input.haplotype_1_reads} {input.phased_bam_file} > {output.haplotype_1_bam_file}
        samtools index {output.haplotype_1_bam_file}
        
        samtools view -b -h -N {input.haplotype_2_reads} {input.phased_bam_file} > {output.haplotype_2_bam_file}
        samtools index {output.haplotype_2_bam_file}
        """
        
        
rule run_bcftools_variant_calling:
    input:
        haplotype_1_bam_file = f"{sample_out_dir}/bam/haplotype_1.bam",
        haplotype_2_bam_file = f"{sample_out_dir}/bam/haplotype_2.bam",
    output:
        haplotype_1_vcf_file = f"{sample_out_dir}/variants/haplotype_1.vcf",
        haplotype_2_vcf_file = f"{sample_out_dir}/variants/haplotype_2.vcf",
        haplotype_1_variants_vcf_file = f"{sample_out_dir}/variants/haplotype_1.variants.vcf",
        haplotype_2_variants_vcf_file = f"{sample_out_dir}/variants/haplotype_2.variants.vcf",
    conda:
        f"{primary_directory}/envs/haplotype_phasing.yaml",
    params:
        ref_genome = refGenome_FA_PATH,
        exclude_regions_file = "/home/sak0914/MtbLongitudinalDiversity/lowAF_variant_calling/references/BED_files/exclude_regions_50bp.EBRlessThan95Percent.bed",
    threads:
        8
    shell:
        """     
        # -q 1 = min mapping quality
        # -Q 20 = min base quality
        # -d 200 = maximum depth to consider
        bcftools mpileup {input.haplotype_1_bam_file} -q 1 -Q 20 -d 200 -f {params.ref_genome} -a FMT/AD,FMT/DP,FMT/ADF,FMT/ADR,FMT/SCR -Ou --threads {threads} | bcftools call --ploidy 1 -A -m --prior 1e-2 -Ov --threads {threads} > {output.haplotype_1_vcf_file}
        
        bcftools mpileup {input.haplotype_2_bam_file} -q 1 -Q 20 -d 200 -f {params.ref_genome} -a FMT/AD,FMT/DP,FMT/ADF,FMT/ADR,FMT/SCR -Ou --threads {threads} | bcftools call --ploidy 1 -A -m --prior 1e-2 -Ov --threads {threads} > {output.haplotype_2_vcf_file}
        
        bcftools filter -e "ALT == '.'" {output.haplotype_1_vcf_file} | bedtools subtract -a '-' -b {params.exclude_regions_file} -header | bcftools sort > {output.haplotype_1_variants_vcf_file}
        bcftools filter -e "ALT == '.'" {output.haplotype_2_vcf_file} | bedtools subtract -a '-' -b {params.exclude_regions_file} -header | bcftools sort > {output.haplotype_2_variants_vcf_file}
        """
        
        
        
rule create_VCF_for_tbtyper:
    input:
        haplotype_1_bam_file = f"{sample_out_dir}/bam/haplotype_1.bam",
        haplotype_2_bam_file = f"{sample_out_dir}/bam/haplotype_2.bam",
    output:
        haplotype_1_init_vcf = temp(f"{sample_out_dir}/TBtypeR/haplotype_1.init.vcf"),
        haplotype_2_init_vcf = temp(f"{sample_out_dir}/TBtypeR/haplotype_2.init.vcf"),
        haplotype_1_vcf_file = f"{sample_out_dir}/TBtypeR/haplotype_1.vcf.gz",
        haplotype_2_vcf_file = f"{sample_out_dir}/TBtypeR/haplotype_2.vcf.gz",
    params:
        tbtypeR_targets = "/home/sak0914/MtbLongitudinalDiversity/direct_sputum/tbtyper_targets_Chromosome.tsv",
        ref_genome = refGenome_FA_PATH,
    conda:
        f"{primary_directory}/envs/haplotype_phasing.yaml",
    shell:
        """
        bcftools mpileup {input.haplotype_1_bam_file} -q 1 -Q 20 -d 200 -f {params.ref_genome} -a FMT/AD -Ou --threads {threads} | bcftools call --ploidy 1 -A -m --prior 1e-2 -C alleles -T {params.tbtypeR_targets} -Ou --threads {threads} | bcftools annotate -x INFO,^FORMAT/GT,^FORMAT/AD -Ov -o {output.haplotype_1_init_vcf} --threads {threads}
        
        bcftools mpileup {input.haplotype_2_bam_file} -q 1 -Q 20 -d 200 -f {params.ref_genome} -a FMT/AD -Ou --threads {threads} | bcftools call --ploidy 1 -A -m --prior 1e-2 -C alleles -T {params.tbtypeR_targets} -Ou --threads {threads} | bcftools annotate -x INFO,^FORMAT/GT,^FORMAT/AD -Ov -o {output.haplotype_2_init_vcf} --threads {threads}
        
        # rename Chromosome to NC_000962.3 for compatibility
        sed 's/Chromosome/NC_000962.3/g' {output.haplotype_1_init_vcf} | gzip -c > {output.haplotype_1_vcf_file}
        sed 's/Chromosome/NC_000962.3/g' {output.haplotype_2_init_vcf} | gzip -c > {output.haplotype_2_vcf_file}
        """