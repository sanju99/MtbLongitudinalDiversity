import numpy as np
import pandas as pd

configfile: "config.yaml"

### Define PATHs to files defined in thoe config file ###
refGenome_FA_PATH = config["RefGenome_FA_PATH"]
refGenome_GFF_PATH = config["RefGenome_GFF_PATH"]
Illumina_FQ_DIR = config["Illumina_FQ_DIR"]

H37rv_DnaA_FA_PATH = config["H37rv_DnaA_FA_PATH"]

# Define PATH of main OUTPUT directory
output_dir = config["output_dir"]

primary_directory = "/home/sak0914/MtbLongitudinalDiversity/hybrid_assemblies"

# Read in data regarding input 
input_DataInfo_DF = pd.read_csv( config["inputSampleData"])

# drop samples that don't have an original ID (SNNN-NN type ID)
input_DataInfo_DF = input_DataInfo_DF.dropna(subset=['Original_ID', 'PacBio_FQ_PATH', 'Illumina_ID']).reset_index(drop=True)

input_All_SampleIDs = list( input_DataInfo_DF["Original_ID"].values )

SampleID_To_PB_FQ_Dict = dict(input_DataInfo_DF[["Original_ID", "PacBio_FQ_PATH"]].values)
Illumina_MFS_to_Original_ID_Dict = dict(input_DataInfo_DF[["Original_ID", "Illumina_ID"]].values)

rule all:
    input:
        [f"{output_dir}/{sample}/autocycler_metrics.tsv" for sample in input_All_SampleIDs],
        # [f"{output_dir}/{sample}/autocycler_out/circlator/consensus_assembly.fasta" for sample in input_All_SampleIDs],
        # [f"{output_dir}/{sample}/PB/Flye_Assembly/assembly.fasta" for sample in input_All_SampleIDs],
        # [f"{output_dir}/{sample}/PB/Flye_Assembly_RenamedAndLengthFiltered/{sample}.flyeassembly.I3.Renamed.100Kb.fasta" for sample in input_All_SampleIDs],
        # [f"{output_dir}/{sample}/IlluminaWGS/Kraken2/{sample}.R{num}.kraken.filtered.fastq.gz" for sample in input_All_SampleIDs for num in [1, 2]],
        # [f"{output_dir}/{sample}/FlyeAssembly_I3_PilonPolishing/pilon_IllPE_Polishing_I3_Asm_ChangeSNPsINDELsOnly/{sample}.Flye.I3Asm.PilonPolished.fasta" for sample in input_All_SampleIDs],
        # [f"{output_dir}/{sample}/LineageCalling/LineageCall_FlyeI3AsmPP/{sample}.AsmToRef.FlyeI3AsmPP.lineage_call.tsv" for sample in input_All_SampleIDs],
        # [f"{output_dir}/{sample}/LineageCalling/LineageCall_FlyeI3Asm/{sample}.AsmToRef.FlyeI3Asm.lineage_call.tsv" for sample in input_All_SampleIDs]
        

sample_out_dir = f"{output_dir}/{{sample_ID}}"

rule autocycler_assembly:
    input:
        PacBio_reads = lambda wildcards: SampleID_To_PB_FQ_Dict[wildcards.sample_ID]
    output:
        assembly_out = f"{sample_out_dir}/autocycler_out/consensus_assembly.fasta",
    # conda:
    #     f"{primary_directory}/envs/autocycler.yaml"
    threads: 
        16
    params:
        autocycler_script = f"{primary_directory}/scripts/autocycler_full.sh",
        output_dir = sample_out_dir,
    shell:
        """
        source activate /home/sak0914/anaconda3/envs/snakemake/envs/autocycler
        
        bash {params.autocycler_script} {input.PacBio_reads} {threads} 4 pacbio_hifi {params.output_dir}
        """



rule compute_autocycler_assembly_metrics:
    input:
        assembly_out = f"{sample_out_dir}/autocycler_out/consensus_assembly.fasta",
    output:
        assembly_metrics_file = f"{sample_out_dir}/autocycler_metrics.tsv",
    params:
        sample_ID = f"{{sample_ID}}",
        output_dir = sample_out_dir,
    shell:
        """
        source activate /home/sak0914/anaconda3/envs/snakemake/envs/autocycler
        
        autocycler table -a {params.output_dir} -n {params.sample_ID} > {output.assembly_metrics_file}
        """


###################################################################################
######### CIRCLATOR for setting start at DnaA (Assuming Circular genome) ##########
###################################################################################

rule circlator_FixStart_DnaA:
    input:
        DnaA_Seq_fa = H37rv_DnaA_FA_PATH,
        consensus_assembly = f"{sample_out_dir}/autocycler_out/consensus_assembly.fasta",
    output:
        consensus_assembly_fixStart = f"{sample_out_dir}/autocycler_out/circlator/consensus_assembly.fasta",
    # conda:
    #     f"{primary_directory}/envs/circlator.yaml"
    threads: 
        1
    params:
        circlator_out_prefix = f"{sample_out_dir}/autocycler_out/circlator/consensus_assembly"
    shell:
        """
        source activate /home/sak0914/anaconda3/envs/circlator
        
        circlator fixstart --genes_fa {input.DnaA_Seq_fa} {input.consensus_assembly} {params.circlator_out_prefix}
        """

###################################################################################



### Filter long read assembly (Flye 3X polished) for only contigs greater than 100kb

rule filterByLength_100kbCPBigs_FlyeAssembly_I3:
    input:
        PacBio_Flye_Assembly_fa = f"{sample_out_dir}/PB/Flye_Assembly_RenamedAndLengthFiltered/{{sample_ID}}.flyeassembly.I3.fixstart.fasta",
    output:
        PacBio_Flye_Assembly_Renamed_100KbCPBigs_FA = f"{sample_out_dir}/PB/Flye_Assembly_RenamedAndLengthFiltered/{{sample_ID}}.flyeassembly.I3.Renamed.100Kb.fasta",
        PacBio_Flye_Assembly_Renamed_100KbCPBigs_FAI = f"{sample_out_dir}/PB/Flye_Assembly_RenamedAndLengthFiltered/{{sample_ID}}.flyeassembly.I3.Renamed.100Kb.fasta.fai"
    threads: 1
    conda:
        f"{primary_directory}/envs/bioinfo_util_env_V1.yaml"
    shell:
        " bioawk -c fastx '{{ print \">{wildcards.sample_ID}_\"$name \"\\n\" $seq }}' {input.PacBio_Flye_Assembly_fa} "
        " | "
        " bioawk -c fastx '{{ if(length($seq) > 100000) {{ print \">\"$name; print $seq }}}}' > {output.PacBio_Flye_Assembly_Renamed_100KbCPBigs_FA} \n"
        " samtools faidx {output.PacBio_Flye_Assembly_Renamed_100KbCPBigs_FA} "




rule align_flye_assembly_to_H37Rv:
    input:
        FlyeAsm_I3_FA = f"{sample_out_dir}/PB/Flye_Assembly_RenamedAndLengthFiltered/{{sample_ID}}.flyeassembly.I3.Renamed.100Kb.fasta",
    output:
        sam_file = f"{sample_out_dir}/LineageCalling/LineageCall_FlyeI3Asm/{{sample_ID}}.Flye.I3Asm.H37Rv.sam",
    params:
        refGenome_FA_PATH = refGenome_FA_PATH,
    threads: 
        8
    shell:
        """
        source activate /home/sak0914/anaconda3/envs/liftoff
        
        minimap2 -ax asm5 {params.refGenome_FA_PATH} {input} -t {threads} -o {output.sam_file}
        """
        
        
        
rule flye_assembly_variants_relative_to_H37Rv:
    input:
        sam_file = f"{sample_out_dir}/LineageCalling/LineageCall_FlyeI3Asm/{{sample_ID}}.Flye.I3Asm.H37Rv.sam",
    output:
        bam_file = f"{sample_out_dir}/LineageCalling/LineageCall_FlyeI3Asm/{{sample_ID}}.Flye.I3Asm.H37Rv.bam",
        vcf_file = f"{sample_out_dir}/LineageCalling/LineageCall_FlyeI3Asm/{{sample_ID}}.vcf",
        fasta_file = temp(f"{sample_out_dir}/LineageCalling/LineageCall_FlyeI3Asm/{{sample_ID}}.fasta"),
    # conda:
    #     f"{primary_directory}/envs/IlluminaPE_Processing.yaml"
    params:
        refGenome_FA_PATH = refGenome_FA_PATH,
        Pilon_OutputDir_PATH = f"{sample_out_dir}/LineageCalling/LineageCall_FlyeI3Asm/"
    shell:
        """
        source activate /home/sak0914/anaconda3/envs/IlluminaPE_Processing
             
        samtools sort {input.sam_file} -o {output.bam_file} 
        samtools index {output.bam_file}
                 
        pilon -Xmx14g --genome {params.refGenome_FA_PATH} \
                       --bam {output.bam_file} \
                       --output {wildcards.sample_ID} \
                       --outdir {params.Pilon_OutputDir_PATH} \
                       --variant
                       
        rm {input.sam_file}
        """
        


rule fast_lineage_caller_flye_assembly_H37Rv_aligned_VCF:
    input:
        vcf_file = f"{sample_out_dir}/LineageCalling/LineageCall_FlyeI3Asm/{{sample_ID}}.vcf"
    output:
        lineage_file = f"{sample_out_dir}/LineageCalling/LineageCall_FlyeI3Asm/{{sample_ID}}.AsmToRef.FlyeI3Asm.lineage_call.tsv",
        vcf_file = f"{sample_out_dir}/LineageCalling/LineageCall_FlyeI3Asm/{{sample_ID}}.vcf.gz",
    shell:
        """
        fast-lineage-caller --pass --out {output.lineage_file} {input.vcf_file}
        
        # gzip the vcf
        gzip {input.vcf_file}
        """



########## Combined Illumina + PacBio Analysis Steps ##########



rule bwa_map_IllPE_AlignTo_I3_Assembly:
    input:
        FlyeAsm_I3_FA = f"{sample_out_dir}/PB/Flye_Assembly_RenamedAndLengthFiltered/{{sample_ID}}.flyeassembly.I3.Renamed.100Kb.fasta",
        fastq1_trimmed_classified = f"{Illumina_FQ_DIR}/{{sample_ID}}/{{sample_ID}}/kraken/{{sample_ID}}.R1.kraken.filtered.fastq.gz",
        fastq2_trimmed_classified = f"{Illumina_FQ_DIR}/{{sample_ID}}/{{sample_ID}}/kraken/{{sample_ID}}.R2.kraken.filtered.fastq.gz",
    output:
        temp(f"{sample_out_dir}/FlyeAssembly_I3_PilonPolishing/Ill_AlnTo_FlyeAsm_I3/{{sample_ID}}.IllPE.AlnTo.I3Asm.sam")
    # conda:
    #     f"{primary_directory}/envs/IlluminaPE_Processing.yaml"
    params:
        rg=r"@RG\tID:{sample_ID}\tSM:{sample_ID}"
    threads: 
        8
    shell:
        """
        source activate /home/sak0914/anaconda3/envs/IlluminaPE_Processing
        
        bwa index {input.FlyeAsm_I3_FA}
        
        bwa mem -M -R '{params.rg}' -t {threads} {input.FlyeAsm_I3_FA} {input.fastq1_trimmed_classified} {input.fastq2_trimmed_classified} > {output}
        """


rule samtools_ViewAndSort_IllPE_AlignTo_I3_Assembly:
    input:
        f"{sample_out_dir}/FlyeAssembly_I3_PilonPolishing/Ill_AlnTo_FlyeAsm_I3/{{sample_ID}}.IllPE.AlnTo.I3Asm.sam"
    output:
        bam = f"{sample_out_dir}/FlyeAssembly_I3_PilonPolishing/Ill_AlnTo_FlyeAsm_I3/{{sample_ID}}.IllPE.AlnTo.I3Asm.bam",
        bai = f"{sample_out_dir}/FlyeAssembly_I3_PilonPolishing/Ill_AlnTo_FlyeAsm_I3/{{sample_ID}}.IllPE.AlnTo.I3Asm.bam.bai",
    # conda:
    #     f"{primary_directory}/envs/IlluminaPE_Processing.yaml"
    shell:
        """
        source activate /home/sak0914/anaconda3/envs/IlluminaPE_Processing
        
        samtools view -bS {input} | samtools sort - > {output.bam}
        
        samtools index {output.bam}
        """



rule samtools_Depth_IllPE_AlignTo_I3_Assembly:
    input:
        f"{sample_out_dir}/FlyeAssembly_I3_PilonPolishing/Ill_AlnTo_FlyeAsm_I3/{{sample_ID}}.IllPE.AlnTo.I3Asm.bam",
    output:
        f"{sample_out_dir}/FlyeAssembly_I3_PilonPolishing/Ill_AlnTo_FlyeAsm_I3/{{sample_ID}}.IllPE.AlnTo.I3Asm.bam.depth.txt"
    # conda:
    #     f"{primary_directory}/envs/IlluminaPE_Processing.yaml"
    shell:
        """
        source activate /home/sak0914/anaconda3/envs/IlluminaPE_Processing
        
        samtools depth -a {input} > {output}
        """



rule samtools_Depth_AverageAll_IllPE_AlignTo_I3_Assembly:
    input:
        f"{sample_out_dir}/FlyeAssembly_I3_PilonPolishing/Ill_AlnTo_FlyeAsm_I3/{{sample_ID}}.IllPE.AlnTo.I3Asm.bam.depth.txt"
    output:
        f"{sample_out_dir}/FlyeAssembly_I3_PilonPolishing/Ill_AlnTo_FlyeAsm_I3/{{sample_ID}}.IllPE.AlnTo.I3Asm.bam.depth.averaged.txt"
    shell:
        "awk '{{sum+=$3}} END {{ print \"Average = \",sum/NR}}' {input} > {output}"







#####################################
#### PICARD (remove duplicates) #####
#####################################

rule picard_RemoveDup_IllPE_AlnTo_I3_Asm:
    input:
        IllPE_BwaMEM_BAM = f"{sample_out_dir}/FlyeAssembly_I3_PilonPolishing/Ill_AlnTo_FlyeAsm_I3/{{sample_ID}}.IllPE.AlnTo.I3Asm.bam",
    output:
        IllPE_BwaMEM_BAM = f"{sample_out_dir}/FlyeAssembly_I3_PilonPolishing/Ill_AlnTo_FlyeAsm_I3/{{sample_ID}}.IllPE.AlnTo.I3Asm.markDup.bam",
        IllPE_BwaMEM_BAI = f"{sample_out_dir}/FlyeAssembly_I3_PilonPolishing/Ill_AlnTo_FlyeAsm_I3/{{sample_ID}}.IllPE.AlnTo.I3Asm.markDup.bam.bai",
        IllPE_BwaMEM_METRICS = f"{sample_out_dir}/FlyeAssembly_I3_PilonPolishing/Ill_AlnTo_FlyeAsm_I3/{{sample_ID}}.IllPE.AlnTo.I3Asm.markDup.bam.metrics",
    # conda:
    #     f"{primary_directory}/envs/IlluminaPE_Processing.yaml"
    shell:
        """
        source activate /home/sak0914/anaconda3/envs/IlluminaPE_Processing
        
        picard -Xmx10g MarkDuplicates I={input.IllPE_BwaMEM_BAM} O={output.IllPE_BwaMEM_BAM} REMOVE_DUPLICATES=false M={output.IllPE_BwaMEM_METRICS} ASSUME_SORT_ORDER=coordinate
        
        samtools index {output.IllPE_BwaMEM_BAM}
        """


rule pilon_IllPE_Polishing_I3_Asm:
    input:
        FlyeAsm_I3_FA = f"{sample_out_dir}/PB/Flye_Assembly_RenamedAndLengthFiltered/{{sample_ID}}.flyeassembly.I3.Renamed.100Kb.fasta",
        IllPE_BwaMEM_BAM = f"{sample_out_dir}/FlyeAssembly_I3_PilonPolishing/Ill_AlnTo_FlyeAsm_I3/{{sample_ID}}.IllPE.AlnTo.I3Asm.markDup.bam",
    output:
        pilon_VCF = temp(f"{sample_out_dir}/FlyeAssembly_I3_PilonPolishing/pilon_IllPE_Polishing_I3_Asm_ChangeSNPsINDELsOnly/{{sample_ID}}.Flye.I3Asm.PilonPolished.vcf"),
        pilon_VCF_gzipped = f"{sample_out_dir}/FlyeAssembly_I3_PilonPolishing/pilon_IllPE_Polishing_I3_Asm_ChangeSNPsINDELsOnly/{{sample_ID}}.Flye.I3Asm.PilonPolished.vcf.gz",
        
        I3M_Asm_PilonPolished_FA = f"{sample_out_dir}/FlyeAssembly_I3_PilonPolishing/pilon_IllPE_Polishing_I3_Asm_ChangeSNPsINDELsOnly/{{sample_ID}}.Flye.I3Asm.PilonPolished.fasta",
        I3M_Asm_PilonPolished_ChangesFile = f"{sample_out_dir}/FlyeAssembly_I3_PilonPolishing/pilon_IllPE_Polishing_I3_Asm_ChangeSNPsINDELsOnly/{{sample_ID}}.Flye.I3Asm.PilonPolished.changes"
    params:
        Pilon_OutputDir_PATH = f"{sample_out_dir}/FlyeAssembly_I3_PilonPolishing/pilon_IllPE_Polishing_I3_Asm_ChangeSNPsINDELsOnly/"
    # conda:
    #     f"{primary_directory}/envs/IlluminaPE_Processing.yaml"
    shell:
        """
        source activate /home/sak0914/anaconda3/envs/IlluminaPE_Processing
        
        pilon -Xmx14g --fix snps,indels \
                       --genome {input.FlyeAsm_I3_FA} \
                       --bam {input.IllPE_BwaMEM_BAM} \
                       --output {wildcards.sample_ID}.Flye.I3Asm.PilonPolished \
                       --outdir {params.Pilon_OutputDir_PATH} \
                       --variant \
                       --changes
                       
        gzip -c {output.pilon_VCF} > {output.pilon_VCF_gzipped}
        """
                
        
        
rule align_pilon_polished_assembly_to_H37Rv:
    input:
        f"{sample_out_dir}/FlyeAssembly_I3_PilonPolishing/pilon_IllPE_Polishing_I3_Asm_ChangeSNPsINDELsOnly/{{sample_ID}}.Flye.I3Asm.PilonPolished.fasta",
    output:
        sam_file = f"{sample_out_dir}/LineageCalling/LineageCall_FlyeI3AsmPP/{{sample_ID}}.Flye.I3Asm.PilonPolished.H37Rv.sam",
    params:
        refGenome_FA_PATH = refGenome_FA_PATH,
    threads: 
        8
    shell:
        """
        source activate /home/sak0914/anaconda3/envs/liftoff
        
        minimap2 -ax asm5 {params.refGenome_FA_PATH} {input} -t {threads} -o {output.sam_file}
        """
        
        
        
rule call_pilon_polished_assembly_variants_relative_to_H37Rv:
    input:
        sam_file = f"{sample_out_dir}/LineageCalling/LineageCall_FlyeI3AsmPP/{{sample_ID}}.Flye.I3Asm.PilonPolished.H37Rv.sam",
    output:
        bam_file = f"{sample_out_dir}/LineageCalling/LineageCall_FlyeI3AsmPP/{{sample_ID}}.Flye.I3Asm.PilonPolished.H37Rv.bam",
        vcf_file = f"{sample_out_dir}/LineageCalling/LineageCall_FlyeI3AsmPP/{{sample_ID}}.vcf",
        fasta_file = temp(f"{sample_out_dir}/LineageCalling/LineageCall_FlyeI3AsmPP/{{sample_ID}}.fasta"),
    # conda:
    #     f"{primary_directory}/envs/IlluminaPE_Processing.yaml"
    params:
        refGenome_FA_PATH = refGenome_FA_PATH,
        Pilon_OutputDir_PATH = f"{sample_out_dir}/LineageCalling/LineageCall_FlyeI3AsmPP/"
    shell:
        """
        source activate /home/sak0914/anaconda3/envs/IlluminaPE_Processing
             
        samtools sort {input.sam_file} -o {output.bam_file} 
        samtools index {output.bam_file}
        
        rm {input.sam_file}
         
        pilon -Xmx14g --genome {params.refGenome_FA_PATH} \
                       --bam {output.bam_file} \
                       --output {wildcards.sample_ID} \
                       --outdir {params.Pilon_OutputDir_PATH} \
                       --variant
        """
        


rule fast_lineage_caller_pilon_polished_H37Rv_aligned_VCF:
    input:
        vcf_file = f"{sample_out_dir}/LineageCalling/LineageCall_FlyeI3AsmPP/{{sample_ID}}.vcf"
    output:
        lineage_file = f"{sample_out_dir}/LineageCalling/LineageCall_FlyeI3AsmPP/{{sample_ID}}.AsmToRef.FlyeI3AsmPP.lineage_call.tsv",
        vcf_file = f"{sample_out_dir}/LineageCalling/LineageCall_FlyeI3AsmPP/{{sample_ID}}.vcf.gz",
    shell:
        """
        fast-lineage-caller --pass --out {output.lineage_file} {input.vcf_file}
        
        # gzip the vcf
        gzip {input.vcf_file}
        """


###############################################################