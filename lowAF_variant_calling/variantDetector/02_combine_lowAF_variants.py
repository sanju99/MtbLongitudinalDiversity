import numpy as np
import pandas as pd
import glob, os, argparse, time, vcf, warnings
warnings.filterwarnings('ignore')
from utils import *

parser = argparse.ArgumentParser()

parser.add_argument("-s", "--sample", dest='sample', type=str, required=True, help='Sample name')
parser.add_argument("-bed1", dest='lowAF_variants_personal_genome_BED_fName', type=str, help='BED file of low frequency variants called using the personal reference genome')
parser.add_argument("-bed2", dest="lowAF_variants_H37Rv_BED_fName", type=str, help='BED file of low frequency variants called using the personal reference genome, then transferred to the H37Rv reference genome, after excluding low confidence sites.')
parser.add_argument("-tsv", dest="lowAF_variants_H37Rv_TSV_fName", type=str, help='TSV file of low frequency variants called using the H37Rv reference genome, after excluding low confidence sites. TSV should be extracted using SnpSift')

cmd_line_args = parser.parse_args()
sample = cmd_line_args.sample
lowAF_variants_personal_genome_BED_fName = cmd_line_args.lowAF_variants_personal_genome_BED_fName
lowAF_variants_H37Rv_BED_fName = cmd_line_args.lowAF_variants_H37Rv_BED_fName
lowAF_variants_H37Rv_TSV_fName = cmd_line_args.lowAF_variants_H37Rv_TSV_fName

save_dir = os.path.dirname(os.path.dirname(lowAF_variants_personal_genome_BED_fName))


############################## STEP 1: CREATE GROUND TRUTH VARIANTS CSV FILE IN H37Rv COORDINATES ##############################


# QC information about the variants in personal coords is in here. Will merge with the file below in H37Rv coordinates
# df_lowAF_variants_personal_asm_personal_coords = pd.read_csv(f"{personal_ref_dir}/{sample}/freebayes/lowAF_variants.bed", sep='\t')
# we already ran split_MNPs_into_SNPs in 01_write_lowAF_BED.py before writing the BED file
df_lowAF_variants_personal_asm_personal_coords = pd.read_csv(lowAF_variants_personal_genome_BED_fName, sep='\t')

keep_cols = ['REF', 'ALT', 'QUAL', 'FILTER', 'DP', 'RO', 'AO', 'AF', 'MQM', 'MQMR', 'SRF', 'SRR', 'SAF', 'SAR', 'SRP', 'SAP', 'RPP', 'RPPR', 'RPL', 'RPR']

# there may not be any low AF variants
if len(df_lowAF_variants_personal_asm_personal_coords) > 0:

    # df_lowAF_variants_personal_asm_transferred = pd.read_csv(f"{personal_ref_dir}/{sample}/freebayes/lowAF_variants.H37Rv.excludeLowConf.bed", 
    #                                                          sep='\t', 
    #                                                          header=None,
    #                                                          usecols=[0, 1, 2, 3],
    #                                                          names=['H37Rv_CHROM', 'H37Rv_BEG', 'H37Rv_END', 'Combined_Personal']
    #                                             )

    df_lowAF_variants_personal_asm_transferred = pd.read_csv(lowAF_variants_H37Rv_BED_fName, 
                                                             sep='\t', 
                                                             header=None,
                                                             usecols=[0, 1, 2, 3],
                                                             names=['H37Rv_CHROM', 'H37Rv_BEG', 'H37Rv_END', 'Combined_Personal']
                                                )
    
    # this means that low-AF variants were found, but they weren't transferred over to H37Rv coordinates
    # just write an empty dataframe, treating it like ground truth = 0
    if len(df_lowAF_variants_personal_asm_transferred) == 0:
        
        keep_cols = ['POS'] + keep_cols
        df_ground_truth = pd.DataFrame(columns = keep_cols)
    else:
        
        # when paftools.js lifts over variants, it combines the CHROM, BEG, and END fields, split by underscores. So unsplit them using rsplit to split from the end
        df_lowAF_variants_personal_asm_transferred[['CHROM', 'BEG', 'END']] = df_lowAF_variants_personal_asm_transferred['Combined_Personal'].str.rsplit('_', n=2, expand=True)

        del df_lowAF_variants_personal_asm_transferred['Combined_Personal']

        df_lowAF_variants_personal_asm_transferred[['BEG', 'END']] = df_lowAF_variants_personal_asm_transferred[['BEG', 'END']].astype(int)
        df_lowAF_variants_personal_asm_personal_coords[['BEG', 'END']] = df_lowAF_variants_personal_asm_personal_coords[['BEG', 'END']].astype(int)

        df_lowAF_variants_personal_asm_transferred = df_lowAF_variants_personal_asm_personal_coords.merge(df_lowAF_variants_personal_asm_transferred, on=['CHROM', 'BEG', 'END'], how='inner')

        # add 1 to BEG to get POS in H37Rv coordinates
        df_lowAF_variants_personal_asm_transferred['H37Rv_POS'] = df_lowAF_variants_personal_asm_transferred['H37Rv_BEG'] + 1

        df_ground_truth = df_lowAF_variants_personal_asm_transferred[['H37Rv_POS'] + keep_cols]
        df_ground_truth.rename(columns={'H37Rv_POS': 'POS'}, inplace=True)
        df_ground_truth['Result'] = 'TP'

        keep_cols = ['POS'] + keep_cols
        df_ground_truth = df_ground_truth[keep_cols]
    
    
# if there are no low AF variants, make an empty dataframe
else:
    keep_cols = ['POS'] + keep_cols
    df_ground_truth = pd.DataFrame(columns = keep_cols)

df_ground_truth.to_csv(f"{save_dir}/lowAF_comparison/ground_truth.csv", index=False)


############################## STEP 2: ANNOTATE THE H37Rv-DERIVED LOW AF VARIANTS USING THE COVERAGE AND SNP DENSITY VALUES (THESE ARE THE PARAMETERS TO TUNE) ##############################


# get the dataframe of variants from the freebayes VCF of Illumina reads aligned to H37Rv
df_lowAF_variants_H37Rv_asm = pd.read_csv(lowAF_variants_H37Rv_TSV_fName, sep='\t')
df_lowAF_variants_H37Rv_asm['SampleID'] = sample

# need to split MNVs into SNVs
df_lowAF_variants_H37Rv_asm_split_MNVs = split_MNPs_into_SNPs(df_lowAF_variants_H37Rv_asm)

# recombine with indels because df_lowAF_variants_H37Rv_asm_split_MNVs will only contain SNVs and MNVs (now split)
df_lowAF_variants_H37Rv_asm = pd.concat([df_lowAF_variants_H37Rv_asm.query("REF.str.len() != ALT.str.len()"),
                                         df_lowAF_variants_H37Rv_asm_split_MNVs
                                        ]).reset_index(drop=True)

# add AF column
df_lowAF_variants_H37Rv_asm['AF'] = df_lowAF_variants_H37Rv_asm['AO'] / df_lowAF_variants_H37Rv_asm['DP']

# for those where the minor AF is not the listed AF, switch REF and ALT. This is because of the difference between H37Rv and the personal ref genome
# i.e. if H37Rv = G (15%) and the alternate allele = C (85%), then in the H37Rv VCF it will say REF = G, ALT = C with AF = 0.85
# but that means that in the personal genome VCF, the major allele is C, so the VCF entry will be REF = C, ALT = G with AF = 0.15
# these are the same information, so need to switch things around because when using the apply_freebayes_lowAF_QCfilters function, it will be screened out due to the AF max of 0.75
df_lowAF_variants_H37Rv_asm['minor_AF'] = np.min([df_lowAF_variants_H37Rv_asm['AF'], 1 - df_lowAF_variants_H37Rv_asm['AF']], axis=0)

df_lowAF_variants_H37Rv_asm.rename(columns={'REF': 'orig_REF', 'ALT': 'orig_ALT'}, inplace=True)

# switch REF and ALT for those where the listed AF in the VCF file is not the minor allele
df_lowAF_variants_H37Rv_asm.loc[df_lowAF_variants_H37Rv_asm['AF'] != df_lowAF_variants_H37Rv_asm['minor_AF'], 'REF'] = df_lowAF_variants_H37Rv_asm['orig_ALT']
df_lowAF_variants_H37Rv_asm.loc[df_lowAF_variants_H37Rv_asm['AF'] != df_lowAF_variants_H37Rv_asm['minor_AF'], 'ALT'] = df_lowAF_variants_H37Rv_asm['orig_REF']

# same for the others though
df_lowAF_variants_H37Rv_asm.loc[df_lowAF_variants_H37Rv_asm['AF'] == df_lowAF_variants_H37Rv_asm['minor_AF'], 'REF'] = df_lowAF_variants_H37Rv_asm['orig_REF']
df_lowAF_variants_H37Rv_asm.loc[df_lowAF_variants_H37Rv_asm['AF'] == df_lowAF_variants_H37Rv_asm['minor_AF'], 'ALT'] = df_lowAF_variants_H37Rv_asm['orig_ALT']

# then rename the AF columns so that we use the minor AF in the apply_freebayes_lowAF_QCfilters function
df_lowAF_variants_H37Rv_asm.rename(columns={'AF': 'orig_AF'}, inplace=True)
df_lowAF_variants_H37Rv_asm.rename(columns={'minor_AF': 'AF'}, inplace=True)

# then after the switch above, apply QC filters to keep only the high quality ones and remove those with fixed allele frequencies
df_lowAF_variants_H37Rv_asm = apply_freebayes_lowAF_QCfilters(df_lowAF_variants_H37Rv_asm)

df_lowAF_variants_H37Rv_asm[keep_cols].to_csv(f"{save_dir}/lowAF_comparison/H37Rv_detected.csv", index=False)



def get_confusion_matrix_lowAF_detection(fName_personal_asm_variants, fName_H37Rv_variants):
    
    merge_cols = ['POS', 'REF', 'ALT']
    
    df_personal_asm_variants = pd.read_csv(fName_personal_asm_variants)[merge_cols]
    df_H37Rv_variants = pd.read_csv(fName_H37Rv_variants)[merge_cols]
    
    for col in df_H37Rv_variants.columns:
        df_H37Rv_variants.rename(columns={col: f"H37Rv_{col}"}, inplace=True)
        
    # Only merge on POS because the REF and ALT may be different because the REF for H37Rv will not always be the REF for the personal assembly
    # df_combined = df_personal_asm_variants.merge(df_H37Rv_variants, left_on=merge_cols, right_on=[f"H37Rv_{col}" for col in merge_cols], how='outer')
    df_combined = df_personal_asm_variants.merge(df_H37Rv_variants, left_on='POS', right_on='H37Rv_POS', how='outer')
    
    # possible if there are no low-AF variants and no detected low-AF variants from the H37Rv alignment
    # both dataframes above would be empty in this case
    if len(df_combined) == 0:
        print(f"No low-AF variants detected or real for {sample}")
                
    else:
        # TP = not NA in both
        df_combined.loc[(~pd.isnull(df_combined['POS'])) & (~pd.isnull(df_combined['H37Rv_POS'])), 'Result'] = 'TP'

        # FP = NA in POS and not NA in H37Rv
        df_combined.loc[(pd.isnull(df_combined['POS'])) & (~pd.isnull(df_combined['H37Rv_POS'])), 'Result'] = 'FP'

        # FN = not NA in POS and NA in H37Rv
        df_combined.loc[(~pd.isnull(df_combined['POS'])) & (pd.isnull(df_combined['H37Rv_POS'])), 'Result'] = 'FN'

    # save the dataframe, even if it's empty
    df_combined.to_csv(f"{os.path.dirname(fName_personal_asm_variants)}/confusion_matrix.csv", index=False)


############################## STEP 3: WRITE A TABLE OF ALL DETECTED VARIANTS USING BOTH METHODS WITH CONFUSION MATRIX ANNOTATIONS ##############################

    
get_confusion_matrix_lowAF_detection(f"{save_dir}/lowAF_comparison/ground_truth.csv", 
                                     f"{save_dir}/lowAF_comparison/H37Rv_detected.csv"
                                    )