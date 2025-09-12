import os, glob
import numpy as np
import pandas as pd

# define some paths to make the path names more readable
sample_out_dir = f"{output_dir}/{{sample_ID}}"

scripts_dir = config["scripts_dir"]
references_dir = config["references_dir"]

conda_directory = config['conda_dir']
primary_directory = os.getcwd()



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
        fastq1_trimmed_classified = f"{sample_out_dir}/kraken/{{sample_ID}}.R1.kraken.filtered.fastq",
        fastq2_trimmed_classified = f"{sample_out_dir}/kraken/{{sample_ID}}.R2.kraken.filtered.fastq",    
    conda:
        f"{conda_directory}/envs/read_processing_aln_bwa.yaml"
        # "/home/sak0914/Mtb_Megapipe/.snakemake/conda/73c414a0fdfb349af0d394f0508ea848_"
    shell:
        """
        # seqtk will write outputs to unzipped files, even if the input was compressed
        seqtk subseq {input.fastq1_trimmed} {input.keep_read_names} > {output.fastq1_trimmed_classified} 
        seqtk subseq {input.fastq2_trimmed} {input.keep_read_names} > {output.fastq2_trimmed_classified} 
        
        rm {input.fastq1_trimmed} {input.fastq2_trimmed}
        """
        


rule align_reads_mark_duplicates:
    input:
        fastq1_trimmed_classified = f"{sample_out_dir}/kraken/{{sample_ID}}.R1.kraken.filtered.fastq",
        fastq2_trimmed_classified = f"{sample_out_dir}/kraken/{{sample_ID}}.R2.kraken.filtered.fastq",  
    output:
        sam_file = temp(f"{sample_out_dir}/bam/{{sample_ID}}.sam"),
        bam_file = temp(f"{sample_out_dir}/bam/{{sample_ID}}.bam"),
        bam_index_file = temp(f"{sample_out_dir}/bam/{{sample_ID}}.bam.bai"),
        bam_file_markdup = f"{sample_out_dir}/bam/{{sample_ID}}.markDup.bam",
        bam_file_markdup_metrics = f"{sample_out_dir}/bam/{{sample_ID}}.markDup.bam.metrics",
        bam_index_file_dedup = f"{sample_out_dir}/bam/{{sample_ID}}.markDup.bam.bai",
    params:
        output_dir = output_dir,
        personal_ref_genome = lambda w: sample_asm_dict[w.sample_ID],
        bwa_mem_seed_length = config['bwa_mem_seed_length']
    conda:
        f"{conda_directory}/envs/read_processing_aln_bwa.yaml"
        # "/home/sak0914/Mtb_Megapipe/.snakemake/conda/73c414a0fdfb349af0d394f0508ea848_"
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
        
        # uncomment if you don't want to keep them for space
        # rm {input.fastq1_trimmed_classified} {input.fastq2_trimmed_classified}
        """

        

rule freebayes_variant_calling:
    input:
        merged_bam_file = f"{sample_out_dir}/bam/{{sample_ID}}.markDup.bam",
    output:
        vcf_file_init = temp(f"{sample_out_dir}/freebayes/{{sample_ID}}.init.vcf"),
        vcf_file_norm = temp(f"{sample_out_dir}/freebayes/{{sample_ID}}.norm.vcf"),
        vcf_file = f"{sample_out_dir}/freebayes/{{sample_ID}}.vcf",
    params:
        personal_ref_genome = lambda w: sample_asm_dict[w.sample_ID],
    conda:
        f"{conda_directory}/envs/variant_calling.yaml"
        # "/home/sak0914/Mtb_Megapipe/.snakemake/conda/4629a2c28c4f2581b5861cb80e5d0fcd_"
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
        
        
        
rule liftoff_genes_from_H37Rv:
    input:
        personal_ref_genome = lambda w: sample_asm_dict[w.sample_ID],
    output:
        liftoff_gff_file = f"{sample_out_dir}/assembly/H37Rv.liftoff.gff",
        liftoff_intermediate_dir = temp(directory(f"{sample_out_dir}/assembly/intermediate_files")),
        polished_liftoff_gff_file = f"{sample_out_dir}/assembly/H37Rv.liftoff.gff_polished",
    params:
        H37Rv_genome = os.path.join(primary_directory, "references", "ref_genome", "H37Rv_NC_000962.3.fna"),
        H37Rv_gff_file = os.path.join(primary_directory, "references", "ref_genome", "H37Rv.NCBI.gff3"),
    conda:
        f"{conda_directory}/envs/liftover.yaml"
        # "/home/sak0914/anaconda3/envs/liftoff",
    threads:
        1
    shell:
        """
        liftoff -g {params.H37Rv_gff_file} \
                -o {output.liftoff_gff_file} \
                -copies -polish \
                -dir {output.liftoff_intermediate_dir} \
                {input.personal_ref_genome} {params.H37Rv_genome}
        """
        
        
        
rule filter_high_quality_lowAF_variants:
    input:
        vcf_file = f"{sample_out_dir}/freebayes/{{sample_ID}}.vcf",
        polished_liftoff_gff_file = f"{sample_out_dir}/assembly/H37Rv.liftoff.gff_polished",
    output:
        lowAF_variants_bed_file = f"{sample_out_dir}/freebayes/lowAF_variants.bed",
    run:
        # get a dataframe of variants from the freebayes VCF
        df_variants = convert_freebayes_VCF_records_to_CSV(input.vcf_file)

        # apply QC filters to keep only the high quality ones
        df_lowAF_variants = apply_lowAF_QCfilters(df_variants)
        
        # write a BED-formatted file. Keep all the other fields in the CSV to easily get variant information
        # first get the chromosome name, which is the name of the contig from the hybrid assembly
        liftoff_gff = pd.read_csv(input.polished_liftoff_gff_file,
                           sep='\t',
                           header=None,
                           comment='#',
                           names = ['Chromosome', 'Source', 'FeatureType', 'Start', 'End', 'Score', 'Sense', 'Phase', 'Attributes']
                          )
                          
        chrom_name = liftoff_gff.Chromosome.unique()[0]
        del liftoff_gff

        # add chromosome name to the dataframe
        df_lowAF_variants['CHROM'] = chrom_name

        # 0-based half-open intervals
        df_lowAF_variants['BEG'] = df_lowAF_variants['POS'] - 1

        # interval should cover the full variant, so take the length difference between REF and ALT, absolute value, then add 1 so that it includes the full region
        # this works because all variants have been left-aligned
        df_lowAF_variants['END'] = df_lowAF_variants['BEG'] + (np.abs(df_lowAF_variants['ALT'].str.len() - df_lowAF_variants['REF'].str.len()) + 1)
        
        # save BED file
        df_lowAF_variants[['CHROM', 'BEG', 'END', 'POS', 'REF', 'ALT', 'QUAL', 'FILTER', 'DP', 'RO', 'AO', 'AF', 'MQM', 'MQMR', 'SRF', 'SRR', 'SAF', 'SAR', 'SRP', 'SAP', 'RPP', 'RPPR', 'RPL', 'RPR', 'SAP_prob', 'SRP_prob']].to_csv(output.lowAF_variants_bed_file, sep='\t', index=False)

        
    

rule liftover_variants_from_personal_genome_coords_to_H37Rv_coords:
    input:
        personal_ref_genome = lambda w: sample_asm_dict[w.sample_ID],
        lowAF_variants_bed_file = f"{sample_out_dir}/freebayes/lowAF_variants.bed",
    output:
        paf_file = f"{sample_out_dir}/assembly/{{sample_ID}}.H37Rv.paf",
        lowAF_variants_H37Rv_bed_file = f"{sample_out_dir}/freebayes/lowAF_variants.H37Rv.bed",
    params:
        H37Rv_genome = os.path.join(primary_directory, "references", "ref_genome", "H37Rv_NC_000962.3.fna"),
    conda:
        f"{conda_directory}/envs/liftover.yaml"
        # "/home/sak0914/anaconda3/envs/liftoff",
        # f"{conda_directory}/.snakemake/conda/liftover"
    threads:
        8
    shell:
        """
        # generate paf file
        minimap2 -x asm5 -c --cs -t {threads} {params.H37Rv_genome} {input.personal_ref_genome} > {output.paf_file}
        
        paftools.js liftover {output.paf_file} {input.lowAF_variants_bed_file} > {output.lowAF_variants_H37Rv_bed_file}
        """