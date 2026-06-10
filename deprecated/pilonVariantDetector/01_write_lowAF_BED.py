import numpy as np
import pandas as pd
import glob, os, argparse, time, vcf, sys
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')
sys.path.append(
    str(Path(__file__).resolve().parent.parent / "variantDetector")
)
from utils import *

parser = argparse.ArgumentParser()

parser.add_argument("-v", dest='vcf_file', type=str, required=True, help='VCF file to further filter')
parser.add_argument("-b", dest='bam_file', type=str, required=True, help='BAM file.')
parser.add_argument("-d", dest='depth_file', type=str, required=True, help='Depth file to use to filter out low quality low frequency variants.')
parser.add_argument("-o", "--output", dest='output_file', type=str, required=True, help='Output file of round 1-filtered variants')
parser.add_argument("-g", "--genome", dest='genome', type=str, help='Full path to genome assembly')
parser.add_argument("--BED", dest='save_BED', action="store_true", help="Save BED file instead of the default CSV")

cmd_line_args = parser.parse_args()

vcf_file = cmd_line_args.vcf_file
bam_file = cmd_line_args.bam_file
depth_file = cmd_line_args.depth_file
output_file = cmd_line_args.output_file
genome = cmd_line_args.genome
save_BED = cmd_line_args.save_BED

dir_name = os.path.dirname(os.path.dirname(vcf_file))


def filter_high_quality_lowAF_variants(vcf_file, bam_file, depth_file, output_file, save_BED=False):

    # VCF file
    df_variants = read_pilon_vcf(vcf_file)
    
    int_cols = ['DP', 'IC', 'DC', 'BQ', 'MQ', 'TD']

    for col in int_cols:
        df_variants[col] = df_variants[col].astype(int)

    df_variants['AF'] = df_variants['AF'].astype(float)
    
    # exclude fixed variants to reduce the number that have to be processed below. These will have proper AFs. It's low AF variants that will have AF = 0
    df_fixed_variants = df_variants.query("ALT!='.'")
    df_unfixed_variants = pd.concat([df_variants.query("ALT=='.'"), df_variants.query("ALT!='.' & AF <= 0.95")])
    
    if len(df_unfixed_variants) == 0:
        df_unfixed_variants.to_csv(output_file, index=False)
        exit()

    # need AO columns for both groups
    df_fixed_variants[['ALT', 'AO']] = df_fixed_variants.apply(lambda row: get_alt_allele(row["BC"], row["REF"]), axis=1, result_type="expand")
    df_unfixed_variants[['ALT', 'AO']] = df_unfixed_variants.apply(lambda row: get_alt_allele(row["BC"], row["REF"]), axis=1, result_type="expand")

    # recompute new AF only for the variants with NA in the ALT and AF columns. Make use of pilon's output AF because it is weighted by base and mapping qualities
    df_unfixed_variants['AF'] = df_unfixed_variants['AO'] / df_unfixed_variants['DP']
    
    # recombine
    df_variants = pd.concat([df_fixed_variants, df_unfixed_variants]).sort_values("POS").reset_index(drop=True)
    
    # save the fixed variants. Require base quality to be at least 30
    df_fixed_variants.query("BQ >= 30 & MQ >= 40 & DP >= 10 & AF > 0.95").to_csv(f"{os.path.dirname(vcf_file)}/fixed_SNVs.csv", index=False)

    # apply QC filters
    df_lowAF_variants = apply_pilon_lowAF_QCfilters(df_variants).reset_index(drop=True)
    
    if len(df_lowAF_variants) == 0:
        # save empty file
        df_lowAF_variants.to_csv(output_file, index=False)
        exit()

    chrom = df_lowAF_variants.CHROM.unique()[0]
    bam = pysam.AlignmentFile(bam_file, "rb")
        
    # the base quality (BQ) column in pilon is the average BQ of all reads at the site. Need BQ of reads that support the variant
    for i, row in df_lowAF_variants.iterrows():
        mean_base_qual, SAF, SAR = compute_mean_base_quality_strand_support_of_variant(bam, row['POS'], row['ALT'], chrom=chrom)
        df_lowAF_variants.loc[i, ['Mean_BQ_ALT_allele', 'SAF', 'SAR']] = [mean_base_qual, SAF, SAR]
        
    assert sum(pd.isnull(df_lowAF_variants['Mean_BQ_ALT_allele'])) == 0
    
    # require at least 2 forward and 2 reverse reads for each variant
    df_lowAF_variants = df_lowAF_variants.query("Mean_BQ_ALT_allele >= 20 & SAF >= 2 & SAR >= 2")

    # require that all low-AF variants occur in areas where the depth is at least half the median depth
    df_depth = pd.read_csv(depth_file, compression='gzip', header=None, sep='\t', names=['CHROM','POS', 'DEPTH'])

    depth_min = df_depth['DEPTH'].median() / 2

    low_cov_sites = df_depth.query("DEPTH < @depth_min").POS.values
    # if save_BED:
    # df_lowAF_variants = df_lowAF_variants.query("DP >= @depth_min")
    df_lowAF_variants = df_lowAF_variants.query("POS not in @low_cov_sites")

    # no 'RO' anymore
    save_cols = ['DP', 'TD', 'BQ', 'MQ', 'BC', 'IC', 'DC', 'XC', 'AF', 'AO', 'Mean_BQ_ALT_allele', 'SAF', 'SAR']
    
    # the save_BED switch is True for personal genome-aligned files and False for H37Rv-aligned files
    if save_BED:
        # add chromosome name to the dataframe
        df_lowAF_variants['CHROM'] = chrom

        # 0-based half-open intervals
        df_lowAF_variants['BEG'] = df_lowAF_variants['POS'] - 1

        # interval should cover the full variant, so take the length difference between REF and ALT, absolute value, then add 1 so that it includes the full region
        # this works because all variants have been left-aligned
        df_lowAF_variants['END'] = df_lowAF_variants['BEG'] + (np.abs(df_lowAF_variants['ALT'].str.len() - df_lowAF_variants['REF'].str.len()) + 1)

        # save
        df_lowAF_variants[['CHROM', 'BEG', 'END', 'POS', 'REF', 'ALT', 'QUAL', 'FILTER'] + save_cols].to_csv(output_file, sep='\t', index=False)
    else:
        df_lowAF_variants[['CHROM', 'POS', 'REF', 'ALT', 'QUAL', 'FILTER'] + save_cols].to_csv(output_file, index=False)
    
    
if not os.path.isfile(genome):
    raise ValueError(f"{genome} does not exist")
    

filter_high_quality_lowAF_variants(vcf_file,
                                   bam_file,
                                   depth_file,
                                   output_file,
                                   save_BED=save_BED
                                  )