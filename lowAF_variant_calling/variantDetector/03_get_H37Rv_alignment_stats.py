import pysam, os
import numpy as np
import pandas as pd
import argparse, vcf, warnings
from utils import *
warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser()

parser.add_argument("-s", "--sample", dest='sample', type=str, required=True, help='Sample name used to prefix output files')
parser.add_argument("-b", "--bam", dest='bam_file', type=str, required=True, help='BAM file to get alignment statistics for.')
parser.add_argument("-v", "--variants", dest='variants_file', type=str, required=True, help='TSV file of candidate low AF variants to get statistics for. This was generated from a VCF file')
parser.add_argument("-g", "--genome", dest='genome_file', type=str, default="/home/sak0914/MtbLongitudinalDiversity/lowAF_variant_calling/references/ref_genome/H37Rv_NC_000962.3.fna", help='Reference genome FASTA file')

cmd_line_args = parser.parse_args()

sample = cmd_line_args.sample
bam_file = cmd_line_args.bam_file
variants_file = cmd_line_args.variants_file
genome_file = cmd_line_args.genome_file

out_dir = f"{os.path.dirname(os.path.dirname(bam_file))}/freebayes"

# if os.path.isfile(f"{out_dir}/lowAF_SNPs.csv"):
#     exit()


############################ GOAL: TO GET 5 ALIGNMENT STATISTICS FROM THE BAM FILE ############################


############################ STEP 0: READ IN THE TEXT FILE OF CANDIDATE LOW-AF VARIANTS ############################


df_variants = pd.read_csv(variants_file, sep='\t')

# keep only low frequency variants. Extract fixed variants separately
df_variants = apply_freebayes_lowAF_QCfilters(df_variants, AF_min=0.05, AF_max=0.98)

# split SNPs and indels in close proximity to each other
# df_variants = split_SNPs_indels_same_haplotype(df_variants)

# only SNPs for right now. Indels are more complicated, i.e. don't have a base quality. Probably have to run only some of the steps on the indels
df_variants = df_variants.query("REF.str.len() == ALT.str.len()").reset_index(drop=True)

# split MNPs into SNPs
# df_variants = split_MNPs_into_SNPs(df_variants)


############################ STEP 1: ROLLING AVERAGE OF COVERAGE FROM LEFT AND RIGHT SIDES ############################


depth_file = os.path.join(os.path.dirname(bam_file), f"{sample}.depth.tsv.gz")
df_depth = pd.read_csv(depth_file, compression='gzip', sep='\t', header=None, names=['CHROM', 'POS', 'COV'])

# compute rolling average of coverage. Smaller than 100 is too small, not enough smoothing
window_size = 100
df_depth['COV_LEFT_ROLLING_AVG'] = df_depth['COV'].rolling(window=window_size, min_periods=1, closed='right').mean()

# for this one, compute the rolling average as you would from the left, but reverse the values beforehand. This is the easiest way. Then reverse them again
df_depth['COV_RIGHT_ROLLING_AVG'] = df_depth['COV'][::-1].rolling(window=window_size, min_periods=1, closed='right').mean()[::-1]

# take the maximum at each site
df_depth['COV_MAX_ROLLING_AVG'] = np.max([df_depth['COV_LEFT_ROLLING_AVG'], df_depth['COV_RIGHT_ROLLING_AVG']], axis=0)


############################ STEP 2: AVERAGE QUALITY OF BASES SUPPORTING EACH VARIANT ############################


# make empty column first. If df_variants is empty, then Mean_BQ_ALT_allele won't be added, which will cause problems later when trying to access the column
df_variants['Mean_BQ_ALT_allele'] = np.nan

for i, row in df_variants.iterrows():
    mean_base_qual = compute_mean_base_quality_of_variant_support(bam_file, row['POS'], row['ALT'])
    df_variants.loc[i, 'Mean_BQ_ALT_allele'] = mean_base_qual

    
############################ STEP 3: NUMBER OF SOFT-CLIPPED BASES ADJACENT TO OR AT EACH VARIANT SITE ############################


# generate the file of soft-clipped bases (analogous to the depth file, with number of soft-clipped bases at each site
soft_clipped_bases_file = os.path.join(os.path.dirname(bam_file), f"{sample}.softClips.tsv.gz")

if not os.path.isfile(soft_clipped_bases_file):
    save_table_of_soft_clips(sample, bam_file, genome_file)

df_soft_clips = pd.read_csv(soft_clipped_bases_file, compression='gzip', sep='\t', header=None, names=['CHROM', 'POS', 'CLIPPED_BASES'])


############################ STEP 4: NUMBER OF DISCORDANTLY PAIRED READS AT EACH SITE ############################


# generate the file of soft-clipped bases (analogous to the depth file, with number of soft-clipped bases at each site
discordant_alns_file = os.path.join(os.path.dirname(bam_file), f"{sample}.DiscordantReads.tsv.gz")

if not os.path.isfile(discordant_alns_file):
    save_table_of_discordant_reads(sample, bam_file, genome_file)

df_discordant_alns = pd.read_csv(discordant_alns_file, compression='gzip', sep='\t', header=None, names=['CHROM', 'POS', 'DISCORDANT_READS'])


# ############################ STEP 5: LOCAL SNP DENSITY ############################

# SNPs_VCF_file = os.path.join(os.path.dirname(variants_file), f"{sample}.SNPs.vcf")

# sites_with_high_SNP_density = compute_sites_with_high_lowAF_SNP_density(SNPs_VCF_file, genome_file)

# df_variants['High_SNP_Density'] = df_variants['POS'].isin(sites_with_high_SNP_density).astype(int)


############################ STEP 6: DISTRIBUTION OF START AND END SITES OF READS COVERING THE VARIANT ############################

# if the start and end sites of reads supporting the VARIANT are very different from the start and end sites of reads covering the SITE, then it's not a low-AF variant. It's probably fixed, and this is just a mapping issue


############################ STEP 7: PUT IT ALL TOGETHER AND SAVE TO PASS INTO THE BAYESIAN MODEL ############################


# merge in the coverage and soft clipped bases information
df_variants = df_variants.merge(df_depth[['POS', 'COV', 'COV_MAX_ROLLING_AVG']]).merge(df_soft_clips[['POS', 'CLIPPED_BASES']]).merge(df_discordant_alns[['POS', 'DISCORDANT_READS']])

# compute the ratio
df_variants['COV_RATIO'] = df_variants['COV'] / df_variants['COV_MAX_ROLLING_AVG']

# normalize to coverage so that differences in coverage between samples doesn't swamp the signal
df_variants['DISCORDANT_READS_RATIO'] = df_variants['DISCORDANT_READS'] / df_variants['COV']
df_variants['CLIPPED_BASES_RATIO'] = df_variants['CLIPPED_BASES'] / df_variants['COV']

os.makedirs(out_dir, exist_ok=True)

df_variants.to_csv(f"{out_dir}/lowAF_SNPs.csv", index=False)