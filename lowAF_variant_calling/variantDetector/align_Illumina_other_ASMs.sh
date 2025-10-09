

assembly_sample="MFS-1"
assembly_dir="/n/data1/hms/dbmi/farhat/Sanjana/TRUST_aln_personal_assembly/$assembly_sample/assembly"

H37Rv_genome="/home/sak0914/MtbLongitudinalDiversity/lowAF_variant_calling/references/ref_genome/H37Rv_NC_000962.3.fna"

SR_sample="MFS-158"
output_dir="/n/data1/hms/dbmi/farhat/Sanjana/TRUST_lowAF/$SR_sample/freebayes"

# generate a chain mapping coordinates between the personal reference genome and H37Rv
minimap2 -x asm5 -c --cs $assembly_dir/$assembly_sample.fasta $H37Rv_genome > $assembly_dir/H37Rv.$sample.paf

# lift over the variants from H37Rv coordinates to 
paftools.js liftover "$assembly_dir/$sample.H37Rv.paf" $output_dir/lowAF_variants.H37Rv.excludeLowConf.bed > $output_dir/lowAF_variants.$sample.excludeLowConf.bed


SNP_dist = pd.read_csv("../../data/freebayes_excludeLowConf_SNP_dist.csv", header=None, names=['Sample1', 'Sample2', 'Dist'])
SNP_dist['Sample1'] = SNP_dist['Sample1'].str.split('.').str[0]
SNP_dist['Sample2'] = SNP_dist['Sample2'].str.split('.').str[0]

SNP_dist = SNP_dist.merge(df_trust_patients[['SampleID', 'pid', 'Coll2014']].rename(columns={'SampleID': 'Sample1', 'pid': 'pid1', 'Coll2014': 'Coll2014_1'}), how='left')
SNP_dist = SNP_dist.merge(df_trust_patients[['SampleID', 'pid', 'Coll2014']].rename(columns={'SampleID': 'Sample2', 'pid': 'pid2', 'Coll2014': 'Coll2014_2'}), how='left')


def compare_variants_personal_ref_vs_close_personal_ref(sample, assembly_sample):
    
    # variants called from the personal ref genome of the sample
    lowAF_variants = pd.read_csv(f"{personal_ref_dir}/{sample}/freebayes/lowAF_variants.H37Rv.excludeLowConf.bed",
                                 sep='\t',
                                 header=None,
                                 usecols=[0, 1, 2],
                                 names=['CHROM', 'BEG', 'END']
                                )
    
    # variants called from the genome assembly of a closely related sample
    lowAF_variants_close_sample = pd.read_csv(f"{personal_ref_dir}/{sample}/freebayes/lowAF_variants.{assembly_sample}.excludeLowConf.bed",
                                              sep='\t',
                                              header=None,
                                             )
    
    lowAF_variants_close_sample_H37Rv_coordinates = lowAF_variants_close_sample[3].str.split('_', expand=True)
    lowAF_variants_close_sample_H37Rv_coordinates.columns = ['CHROM', 'BEG', 'END']
    lowAF_variants_close_sample_H37Rv_coordinates[['BEG', 'END']] = lowAF_variants_close_sample_H37Rv_coordinates[['BEG', 'END']].astype(int)

    lowAF_variants_found = lowAF_variants.merge(lowAF_variants_close_sample_H37Rv_coordinates, on=['CHROM', 'BEG', 'END'], how='left')
    
    print(f"Found {len(lowAF_variants_found)}/{len(lowAF_variants_found)} {sample} variants using the {assembly_sample} assembly")

    return lowAF_variants_found