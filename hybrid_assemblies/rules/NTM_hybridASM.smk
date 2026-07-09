# 2.Mtb.Generate.HybridAsm.PacBioHiFi.smk
### Snakemake - Run pipeline for PacBio HiFi (Sequel II) de novo assembly (+ short-read polishing of base-level errors)
### Maximillian Marin (mgmarin@g.harvard.edu)

### Import Statements ###
import pandas as pd

### Define PATHs to files defined in thoe config file ###
refGenome_FA_PATH = config["RefGenome_FA_PATH"]
Illumina_FQ_DIR = config["Illumina_FQ_DIR"]

DnaA_FA_PATH = config["DnaA_FA_PATH"]

# Define PATH of main OUTPUT directory
output_dir = config["output_dir"]

primary_directory = "/home/sak0914/MtbLongitudinalDiversity/hybrid_assemblies"

sample_out_dir = f"{output_dir}/{{sample_ID}}"

# Read in data regarding input 
input_DataInfo_DF = pd.read_csv( config["inputSampleData_TSV"], sep='\t')

# drop samples that don't have an original ID (SNNN-NN type ID)
input_DataInfo_DF = input_DataInfo_DF.dropna(subset=['Original_ID', 'PacBio_FQ_PATH']).reset_index(drop=True)

Illumina_samples = os.listdir(Illumina_FQ_DIR)

input_DataInfo_DF = input_DataInfo_DF.query("Illumina_ID in @Illumina_samples")

input_All_SampleIDs = list( input_DataInfo_DF["Original_ID"].values )

SampleID_To_PB_FQ_Dict = dict(input_DataInfo_DF[["Original_ID", "PacBio_FQ_PATH"]].values)
Illumina_MFS_to_Original_ID_Dict = dict(input_DataInfo_DF[["Original_ID", "Illumina_ID"]].values)

rule all:
    input:
        [f"{output_dir}/{sample}/IlluminaWGS/FASTQs_Trimmomatic_Trimming_V2/{sample}_{num}_trimmed.fastq.gz" for sample in input_All_SampleIDs for num in [1, 2]],
        # [f"{output_dir}/{sample}/PB/Flye_Assembly/assembly.fasta" for sample in input_All_SampleIDs],
        # [f"{output_dir}/{sample}/IlluminaWGS/Kraken2/{sample}.R{num}.kraken.filtered.fastq.gz" for sample in input_All_SampleIDs for num in [1, 2]],
        # [f"{output_dir}/{sample}/FlyeAssembly_I3_PilonPolishing/pilon_IllPE_Polishing_I3_Asm_ChangeSNPsINDELsOnly/{sample}.Flye.I3Asm.PilonPolished.fasta" for sample in input_All_SampleIDs],
        # [f"{output_dir}/{sample}/LineageCalling/LineageCall_FlyeI3AsmPP/{sample}.AsmToRef.FlyeI3AsmPP.lineage_call.tsv" for sample in input_All_SampleIDs],
        # [f"{output_dir}/{sample}/LineageCalling/LineageCall_FlyeI3Asm/{sample}.AsmToRef.FlyeI3Asm.lineage_call.tsv" for sample in input_All_SampleIDs],
        # [f"{output_dir}/{sample}/PB/kraken/kraken_report_standard_DB.txt" for sample in input_All_SampleIDs]
        # [f"{output_dir}/{sample}/PB/NTM.reads.fastq.gz" for sample in input_All_SampleIDs],
        # [f"{output_dir}/{sample}/FlyeAssembly_I3_PilonPolishing/pilon_IllPE_Polishing_I3_Asm_ChangeSNPsINDELsOnly/BUSCO/short_summary.specific.actinobacteria_class_odb10.BUSCO.txt" for sample in input_All_SampleIDs],
        # [f"{output_dir}/{sample}/PB/Flye_Assembly_RenamedAndLengthFiltered/{sample}.flyeassembly.I3.fixstart.fasta" for sample in input_All_SampleIDs],



rule PacBio_kraken_classification:
    input:
        PB_reads_fq = lambda wildcards: SampleID_To_PB_FQ_Dict[wildcards.sample_ID]
    output:
        kraken_report = f"{sample_out_dir}/PB/kraken/kraken_report_standard_DB.txt",
        kraken_classifications = f"{sample_out_dir}/PB/kraken/kraken_classifications_standard_DB",
    params:
        kraken_db = config['Kraken2_DB_PATH'],
    threads:
        8
    shell:
        """
        source activate /home/sak0914/MtbLongitudinalDiversity/hybrid_assemblies/.snakemake/conda/f37b23e402e32aeb54fd9892801493ef_
        
        # --confidence is the minimum fraction of k-mers in a read that must match a given taxon for that read to be assigned to that taxon
        kraken2 --db {params.kraken_db} \
                --threads {threads} \
                --confidence 0 \
                --gzip-compressed \
                --report {output.kraken_report} \
                --output {output.kraken_classifications} \
                {input.PB_reads_fq}
        """




rule PacBio_extract_kraken_read_names:
    input:
        kraken_classifications = f"{sample_out_dir}/PB/kraken/kraken_classifications_standard_DB",
    output:  
        keep_read_names = f"{sample_out_dir}/PB/keep_read_names.txt",
    params:
        kraken_db = config['Kraken2_DB_PATH'],
        extract_kraken_reads_script = os.path.join(primary_directory, "scripts", "extract_kraken_read_names.py"),
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
        """



rule PacBio_extract_kraken_reads:
    input:
        PB_reads_fq = lambda wildcards: SampleID_To_PB_FQ_Dict[wildcards.sample_ID],
        keep_read_names = f"{sample_out_dir}/PB/keep_read_names.txt",
    output:
        PB_reads_fq_classified = f"{sample_out_dir}/PB/NTM.reads.fastq.gz",
    conda:
        "/home/sak0914/Mtb_Megapipe/envs/read_processing_aln.yaml"
    shell:
        """
        # seqtk will write outputs to unzipped files, even if the input was compressed
        seqtk subseq {input.PB_reads_fq} {input.keep_read_names} | gzip -c > {output.PB_reads_fq_classified} 
                
        gzip {input.keep_read_names}
        """
   
   
   
rule flye_Assemble_PB_CCS_kraken_filtered: # Flye v2.9.2 w/ asmCov = 200
    input:
        PB_reads_fq_classified = f"{sample_out_dir}/PB/NTM.reads.fastq.gz",
    output:
        assembly_fa = f"{sample_out_dir}/PB/Flye_Assembly/assembly.fasta",
        assembly_info_txt = f"{sample_out_dir}/PB/Flye_Assembly/assembly_info.txt"
    conda:
        f"{primary_directory}/envs/flye.yaml"
    threads: 
        8
    params:
        Flye_OutputDir_PATH = f"{sample_out_dir}/PB/Flye_Assembly/"
    shell:
        """
        flye --pacbio-hifi {input.PB_reads_fq_classified} \
             --out-dir {params.Flye_OutputDir_PATH} \
             --genome-size 5.5m \
             --threads {threads} \
             --asm-coverage 200 \
             --iterations 3
        """
        



rule run_busco_on_draft_assembly: # Flye v2.9.2 w/ asmCov = 200
    input:
        assembly_fa = f"{sample_out_dir}/PB/Flye_Assembly/assembly.fasta",
    output:
        busco_output = f"{sample_out_dir}/PB/Flye_Assembly/BUSCO/short_summary.specific.actinobacteria_class_odb10.BUSCO.txt",
    # conda:
    #    f"{primary_directory}/envs/busco.yaml"
    threads: 
        1
    params:
        output_path = f"{sample_out_dir}/PB/Flye_Assembly/BUSCO"
    shell:
        """
        source activate /home/sak0914/anaconda3/envs/snakemake/envs/busco
        
        # the Actinobacteria class is the deepest taxonomic level that of BUSCOv.4 lineages available:
        # https://busco.ezlab.org/list_of_lineages.html
        # -f forces creation of a new directory, which is necessary to overwrite an existing one
        busco -m genome -i {input.assembly_fa} -o {params.output_path} -l actinobacteria_class_odb10 -f
        """
        
        

###################################################################################
######### CIRCLATOR for setting start at DnaA (Assuming Circular genome) ##########
###################################################################################

rule circlator_FixStart_DnaA:
    input:
        DnaA_Seq_fa = DnaA_FA_PATH,
        flye_assembly_fa = f"{sample_out_dir}/PB/Flye_Assembly/assembly.fasta"
    output:
        flye_assembly_FixStart_assembly = f"{sample_out_dir}/PB/Flye_Assembly_RenamedAndLengthFiltered/{{sample_ID}}.flyeassembly.I3.fixstart.fasta",
    # conda:
    #     f"{primary_directory}/envs/circlator.yaml"
    threads: 
        1
    params:
        circlator_out_prefix = f"{sample_out_dir}/PB/Flye_Assembly_RenamedAndLengthFiltered/{{sample_ID}}.flyeassembly.I3.fixstart"
    shell:
        """
        source activate /home/sak0914/anaconda3/envs/circlator
        
        circlator fixstart --genes_fa {input.DnaA_Seq_fa} {input.flye_assembly_fa} {params.circlator_out_prefix}
        """


###################################################################################


# adapter list from: https://github.com/stephenturner/adapters/blob/master/adapters_combined_256_unique.fasta
# Also adapter list combined w/ ftp://ftp.ncbi.nlm.nih.gov/pub/kitts/adaptors_for_screening_proks.fa
# Adapter list path: references/CustomTrimmomatic_IlluminaWGS_AdapterList.WiProkAdaptersNCBI.fasta

rule trimmomatic_Illumina_PE_Trimming_V2:
    input:
        r1 = lambda wildcards: f"{Illumina_FQ_DIR}/{Illumina_MFS_to_Original_ID_Dict[wildcards.sample_ID]}/{Illumina_MFS_to_Original_ID_Dict[wildcards.sample_ID]}_R1.fastq.gz",
        r2 = lambda wildcards: f"{Illumina_FQ_DIR}/{Illumina_MFS_to_Original_ID_Dict[wildcards.sample_ID]}/{Illumina_MFS_to_Original_ID_Dict[wildcards.sample_ID]}_R2.fastq.gz"
    output:
        r1 = f"{sample_out_dir}/IlluminaWGS/FASTQs_Trimmomatic_Trimming_V2/{{sample_ID}}_1_trimmed.fastq.gz",
        r2 = f"{sample_out_dir}/IlluminaWGS/FASTQs_Trimmomatic_Trimming_V2/{{sample_ID}}_2_trimmed.fastq.gz",
        # reads where trimming entirely removed the mate
        r1_unpaired = f"{sample_out_dir}/IlluminaWGS/FASTQs_Trimmomatic_Trimming_V2/{{sample_ID}}_1_trimmed.unpaired.fastq.gz",
        r2_unpaired = f"{sample_out_dir}/IlluminaWGS/FASTQs_Trimmomatic_Trimming_V2/{{sample_ID}}_2_trimmed.unpaired.fastq.gz",
    conda:
        f"{primary_directory}/envs/trimmomatic.yaml"
    params:
        adapters_fasta = f'{primary_directory}/references/CustomTrimmomatic_IlluminaWGS_AdapterList.WiProkAdaptersNCBI.fasta',
    threads:
        8
    shell:
        """
        trimmomatic PE -threads {threads} \
                    {input.r1} {input.r2} \
                    {output.r1} {output.r1_unpaired} \
                    {output.r2} {output.r2_unpaired} \
                    ILLUMINACLIP:{params.adapters_fasta}:2:30:10:2:true \
                    SLIDINGWINDOW:4:20 \
                    MINLEN:75
        """
        

rule kraken_classification:
    input:
        r1 = f"{sample_out_dir}/IlluminaWGS/FASTQs_Trimmomatic_Trimming_V2/{{sample_ID}}_1_trimmed.fastq.gz",
        r2 = f"{sample_out_dir}/IlluminaWGS/FASTQs_Trimmomatic_Trimming_V2/{{sample_ID}}_2_trimmed.fastq.gz",
    output:
        kraken_report = f"{sample_out_dir}/IlluminaWGS/Kraken2/kraken_report_standard_DB.txt",
        kraken_classifications = f"{sample_out_dir}/IlluminaWGS/Kraken2/kraken_classifications_standard_DB",
    params:
        kraken_db = config["Kraken2_DB_PATH"]
    threads:
        8
    conda:
        "/home/sak0914/Mtb_Megapipe/envs/read_processing_aln.yaml"
    shell:
        """
        # --confidence is the minimum fraction of k-mers in a read that must match a given taxon for that read to be assigned to that taxon
        kraken2 --db {params.kraken_db} \
                --threads {threads} \
                --confidence 0 \
                --paired {input.r1} {input.r2} \
                --gzip-compressed \
                --report {output.kraken_report} \
                --output {output.kraken_classifications}
        """



rule extract_kraken_read_names:
    input:
        kraken_classifications = f"{sample_out_dir}/IlluminaWGS/Kraken2/kraken_classifications_standard_DB",
    output:  
        kraken_classifications_gzipped = f"{sample_out_dir}/IlluminaWGS/Kraken2/kraken_classifications_standard_DB.csv.gz",
        keep_read_names = f"{sample_out_dir}/IlluminaWGS/Kraken2/NTM_read_names.txt",
    params:
        kraken_db = config['Kraken2_DB_PATH'],
        extract_kraken_reads_script = os.path.join(primary_directory, "scripts", "extract_kraken_read_names.py"),
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
        """



rule extract_kraken_reads:
    input:
        r1 = f"{sample_out_dir}/IlluminaWGS/FASTQs_Trimmomatic_Trimming_V2/{{sample_ID}}_1_trimmed.fastq.gz",
        r2 = f"{sample_out_dir}/IlluminaWGS/FASTQs_Trimmomatic_Trimming_V2/{{sample_ID}}_2_trimmed.fastq.gz",
        keep_read_names = f"{sample_out_dir}/IlluminaWGS/Kraken2/NTM_read_names.txt",
    output:
        fastq1_trimmed_classified = f"{sample_out_dir}/IlluminaWGS/Kraken2/{{sample_ID}}.R1.kraken.filtered.fastq.gz",
        fastq2_trimmed_classified = f"{sample_out_dir}/IlluminaWGS/Kraken2/{{sample_ID}}.R2.kraken.filtered.fastq.gz",    
    conda:
        "/home/sak0914/Mtb_Megapipe/envs/read_processing_aln.yaml"
    shell:
        """
        # seqtk will write outputs to unzipped files, even if the input was compressed
        seqtk subseq {input.r1} {input.keep_read_names} | gzip -c > {output.fastq1_trimmed_classified} 
        seqtk subseq {input.r2} {input.keep_read_names} | gzip -c > {output.fastq2_trimmed_classified} 
                
        gzip {input.keep_read_names}
        """
        
        


########## Combined Illumina + PacBio Analysis Steps ##########



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
        
        
        
rule bwa_map_IllPE_AlignTo_I3_Assembly:
    input:
        FlyeAsm_I3_FA = f"{sample_out_dir}/PB/Flye_Assembly_RenamedAndLengthFiltered/{{sample_ID}}.flyeassembly.I3.Renamed.100Kb.fasta",
        fastq1_trimmed_classified = f"{sample_out_dir}/IlluminaWGS/Kraken2/{{sample_ID}}.R1.kraken.filtered.fastq.gz",
        fastq2_trimmed_classified = f"{sample_out_dir}/IlluminaWGS/Kraken2/{{sample_ID}}.R2.kraken.filtered.fastq.gz",    
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
        
        bwa mem -k 80 -M -R '{params.rg}' -t {threads} {input.FlyeAsm_I3_FA} {input.fastq1_trimmed_classified} {input.fastq2_trimmed_classified} > {output}
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
                


rule run_busco_on_polished_assembly: # Flye v2.9.2 w/ asmCov = 200
    input:
        I3M_Asm_PilonPolished_FA = f"{sample_out_dir}/FlyeAssembly_I3_PilonPolishing/pilon_IllPE_Polishing_I3_Asm_ChangeSNPsINDELsOnly/{{sample_ID}}.Flye.I3Asm.PilonPolished.fasta",
    output:
        busco_output = f"{sample_out_dir}/FlyeAssembly_I3_PilonPolishing/pilon_IllPE_Polishing_I3_Asm_ChangeSNPsINDELsOnly/BUSCO/short_summary.specific.actinobacteria_class_odb10.BUSCO.txt",
    # conda:
    #    f"{primary_directory}/envs/busco.yaml"
    threads: 
        1
    params:
        output_path = f"{sample_out_dir}/FlyeAssembly_I3_PilonPolishing/pilon_IllPE_Polishing_I3_Asm_ChangeSNPsINDELsOnly/BUSCO"
    shell:
        """
        source activate /home/sak0914/anaconda3/envs/snakemake/envs/busco
        
        # the Actinobacteria class is the deepest taxonomic level that of BUSCOv.4 lineages available:
        # https://busco.ezlab.org/list_of_lineages.html
        # -f forces creation of a new directory, which is necessary to overwrite an existing one
        busco -m genome -i {input.I3M_Asm_PilonPolished_FA} -o {params.output_path} -l actinobacteria_class_odb10 -f
        """
        
        
        
###############################################################