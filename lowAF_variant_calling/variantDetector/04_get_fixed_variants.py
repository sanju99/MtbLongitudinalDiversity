import pysam, os
import numpy as np
import pandas as pd
import argparse, vcf, warnings, pysam
from Bio import Seq, SeqIO
from utils import *
warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser()

parser.add_argument("-d", dest='out_dir', type=str, required=True, help='Top-level directory where sample outputs are')

cmd_line_args = parser.parse_args()

out_dir = cmd_line_args.out_dir

h37Rv_path = "/n/data1/hms/dbmi/farhat/Sanjana/H37Rv"
h37Rv_regions = pd.read_csv(os.path.join(h37Rv_path, "mycobrowser_h37rv_v4.csv"))
h37Rv_seq = SeqIO.read(os.path.join(h37Rv_path, "GCF_000195955.2_ASM19595v2_genomic.gbff"), "genbank")
h37Rv_end_length = len(h37Rv_seq) - 100

# remove rRNAs, which are highly conserved. rrs, rrl, and rrf
rRNA_pos = []

for i, row in h37Rv_regions.query("Functional_Category=='stable RNAs' & Feature=='rRNA'").iterrows():
    # print(row['Name'])
    rRNA_pos += list(np.arange(row['Start'], row['Stop'] + 1))
    
    
def get_deletions_at_variant_sites(sample, df_variants):
    
    bamfile = pysam.AlignmentFile(f"{out_dir}/{sample}/bam/{sample}.dedup.bam", "rb")
    
    variant_pos = df_variants.POS.unique()
    
    positions = []
    del_counts = []

    for pileupcolumn in bamfile.pileup(truncate=True):
        pos = pileupcolumn.pos + 1  # 0-based reference position

        if pos in variant_pos:
            del_count = sum(pr.is_del for pr in pileupcolumn.pileups)

            positions.append(pos)
            del_counts.append(del_count)

    df_deletions = pd.DataFrame({"POS": positions, "DEL": del_counts})

    return df_deletions



def read_in_process_fixed_variants(sample, DP_thresh=10, num_support_each_direction=5):
    
    df_fixed_variants = pd.read_csv(f"{out_dir}/{sample}/freebayes/{sample}.excludeLowConf.tsv", sep='\t').rename(columns={'ANN[0].GENE': 'GENE',
                                                                                                 'ANN.0..GENE': 'GENE',
                                                                                                 'ANN[0].HGVS_C': 'HGVS_C',
                                                                                                 'ANN.0..HGVS_C': 'HGVS_C',
                                                                                                 'ANN[0].HGVS_P': 'HGVS_P',
                                                                                                 'ANN.0..HGVS_P': 'HGVS_P'
                                                                                                })
    
    df_fixed_variants['AF'] = df_fixed_variants['AO'] / df_fixed_variants['DP']
    
    df_fixed_variants = df_fixed_variants.query("DP >= @DP_thresh & AF > 0.90 & MQM >= 40 & SAR >= @num_support_each_direction & SAF >= @num_support_each_direction & REF.str.len() == ALT.str.len()")
    
#     # require that all low-AF variants occur in areas where the depth is at least half the median depth
#     df_depth = pd.read_csv(f"{out_dir}/{sample}/bam/{sample}.depth.tsv.gz", compression='gzip', header=None, sep='\t', names=['CHROM','POS', 'DEPTH'])

#     depth_min = df_depth['DEPTH'].median() / 2
    
#     # depth will be artificially low near the ends, so exclude the first and last 100 bp
#     df_fixed_variants = df_fixed_variants.query("DP >= @depth_min | (POS <= 100 | POS >= @h37Rv_end_length)")
    
    # exclude genes Rv2081c and Rv2082. Very problematic for low frequency variant calling. But keep for fixed variants
    df_fixed_variants = df_fixed_variants.query("~(POS >= 2337869 & POS <= 2340874) & POS not in @rRNA_pos")
    
    # exlude SNPs that overlap deletions
    df_deletions = get_deletions_at_variant_sites(sample, df_fixed_variants)
    
    # exclude positions with more than 2 reads supporting a deletion
    exclude_pos = df_deletions.query("DEL > 2").POS.unique()
    print(f"    Excluding {len(exclude_pos)} variants with more than 2 reads supporting a deletion")
    
    df_fixed_variants = df_fixed_variants.query("POS not in @exclude_pos")
    
    # split MNPs into SNPs
    df_fixed_variants = split_MNPs_into_SNPs(df_fixed_variants)
    
    print(f"Saving {len(df_fixed_variants)} fixed SNPs for {sample}")
    
    df_fixed_variants.to_csv(f"{out_dir}/{sample}/freebayes/fixed_SNPs.csv", index=False)
    
    
samples_lst = os.listdir(out_dir)
print(f"Getting fixed SNPs for {len(samples_lst)} samples")
    
for sample in samples_lst:
    if os.path.isfile(f"{out_dir}/{sample}/freebayes/{sample}.excludeLowConf.tsv"):
        read_in_process_fixed_variants(sample)