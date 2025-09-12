


source activate liftoff

while IFS=$'\t' read -r sample assembly; do
    
    dirpath=$(dirname "$assembly")
        
    if [ ! -f $dirpath/H37Rv.liftoff.gff_polished ]; then
    
        liftoff -g ~/MtbLongitudinalDiversity/lowAF_variant_calling/references/ref_genome/H37Rv.NCBI.gff3 \
                    -o $dirpath/H37Rv.liftoff.gff \
                    -copies -polish \
                    -dir $dirpath \
                    $assembly ~/MtbLongitudinalDiversity/lowAF_variant_calling/references/ref_genome/H37Rv_NC_000962.3.fna
                    
    fi

done < "/home/sak0914/MtbLongitudinalDiversity/lowAF_variant_calling/data/personal_assemblies_samples.tsv"

        
source deactivate

while IFS=$'\t' read -r sample assembly; do
    
    dirpath=$(dirname "$assembly")
    freebayes_VCF_fName="$dirpath/$sample/freebayes/$sample.vcf"
        
    if [ -f $freebayes_VCF_fName ]; then
    
        liftoff -g ~/MtbLongitudinalDiversity/lowAF_variant_calling/references/ref_genome/H37Rv.NCBI.gff3 \
                    -o $dirpath/H37Rv.liftoff.gff \
                    -copies -polish \
                    -dir $dirpath \
                    $assembly ~/MtbLongitudinalDiversity/lowAF_variant_calling/references/ref_genome/H37Rv_NC_000962.3.fna
                    
    fi

done < "/home/sak0914/MtbLongitudinalDiversity/lowAF_variant_calling/data/personal_assemblies_samples.tsv"

        
        
        
rule filter_high_quality_lowAF_variants:
    input:
        vcf_file = f"{sample_out_dir}/freebayes/{{sample_ID}}.vcf",
        polished_liftoff_gff_file = f"{sample_out_dir}/assembly/H37Rv.liftoff.gff_polished",
    output:
        lowAF_variants_bed_file = f"{sample_out_dir}/freebayes/lowAF_variants.bed",
    run:


        
    

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