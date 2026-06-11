import numpy as np
import pandas as pd
import os, glob, warnings, pysam
warnings.filterwarnings('ignore')
from Bio import Seq, SeqIO
import scipy.stats as st
from utils import *

h37Rv_path = "~/MtbLongitudinalDiversity/H37Rv"
h37Rv_seq = SeqIO.read(os.path.join(h37Rv_path, "GCF_000195955.2_ASM19595v2_genomic.gbff"), "genbank")
h37Rv_genes = pd.read_csv(os.path.join(h37Rv_path, "mycobrowser_h37rv_genes_v4.csv"))
h37Rv_regions = pd.read_csv(os.path.join(h37Rv_path, "mycobrowser_h37rv_v4.csv"))

# remove rRNAs, which are highly conserved. rrs, rrl, and rrf
rRNA_pos = []

for i, row in h37Rv_regions.query("Functional_Category=='stable RNAs' & Feature=='rRNA'").iterrows():
    rRNA_pos += list(np.arange(row['Start'], row['Stop'] + 1))
    
    
# exclude any indel within 100 bp of an insertion seq / phage and also within those regions
insertion_seqs_phages_pos = []

for i, row in h37Rv_regions.query("Functional_Category=='insertion seqs and phages'").iterrows():
    insertion_seqs_phages_pos += list(np.arange(row['Start'] - 100, row['Stop'] + 100 + 1))
    
insertion_seqs_phages_pos = np.unique(insertion_seqs_phages_pos)


def keep_middle_percentage_of_reads(df, n_prop):
    
    # should be a proportion, not a percentage
    if n_prop > 1:
        n_prop /= 100

    # Calculate how many rows to keep
    n_rows = len(df)
    keep = int(n_rows * n_prop)

    # Calculate start and end indices
    start = (n_rows - keep) // 2
    end = start + keep

    # Slice the middle N%
    return df.iloc[start:end].reset_index(drop=True)


def compute_adjusted_AF_indels(sample, df_indels, chrom, bam_file, n_prop=0.50, dist_from_indel=10):
    
    df_indels = df_indels.reset_index(drop=True)
    
    if bam_file[-4:] == '.bam':
        bam = pysam.AlignmentFile(bam_file, "rb")
    elif bam_file[-4:] == 'cram':
        bam = pysam.AlignmentFile(bam_file, "rc")
    else:
        raise ValueError(f"{bam_file} name is not a BAM or CRAM file")

    for i, row in df_indels.iterrows():
        
        pos = row['POS']
        ref = row['REF']
        alt = row['ALT']
        
        indel_length = abs(len(ref) - len(alt))
        
        # cigar encodings
        if len(alt) > len(ref):
            search_string = 'I'
            search_val = 1
        else:
            search_string = 'D'
            search_val = 2
            
        reads_overlapping = []
        
        for read in bam.fetch(chrom, pos - 1, pos):  # pysam uses 0-based, half-open

            # exclude reads with supplementary alignments, secondary alignments, or are discordantly paired
            if not read.is_supplementary and not read.is_secondary and read.is_proper_pair:
            
                if read.cigarstring is not None:

                    indel_cigar_length = sum(length for (op, length) in (read.cigartuples or []) if op == search_val and length == indel_length)

                    # get number of bases soft clipped from this read
                    num_soft_clipped_bases = sum(length for (op, length) in (read.cigartuples or []) if op == 4)

                    reads_overlapping.append({
                        "read_name": read.query_name,
                        "ref_start": read.reference_start + 1,  # make it 1-based
                        "ref_end": read.reference_end,
                        "num_indel": indel_cigar_length,
                        "num_soft_clipped": num_soft_clipped_bases,
                        "cigar": read.cigarstring,
                        "cigartuples": read.cigartuples,
                    })

        df_overlapping_reads = pd.DataFrame(reads_overlapping).sort_values(['ref_start', 'ref_end'])
        
        if len(df_overlapping_reads) > 0:
            
            # take the middle N% of reads? No, this is hard to tune because it needs to be low enough to work for some fixed indels, 
            # but high enough for very rare indels (like AF < 10%) so that we still detect them
            # df_reads_close_to_indel = keep_middle_percentage_of_reads(df_overlapping_reads, n_prop)
    
            # include a read only if neither its start nor end are within 10 base pairs of the indel site
            df_reads_close_to_indel = df_overlapping_reads.loc[(abs(df_overlapping_reads['ref_start'] - pos) >= dist_from_indel) & 
                                                               (abs(df_overlapping_reads['ref_end'] - pos) >= dist_from_indel)
                                                              ]

            # also exclude reads that start or end in the middle of the indel            
            # this position is inclusive. So the read must not have a start or end within this
            indel_end = pos + indel_length

            df_reads_close_to_indel = df_reads_close_to_indel.loc[~((df_reads_close_to_indel['ref_start'] >= pos) & (df_reads_close_to_indel['ref_start'] <= indel_end))]

            df_reads_close_to_indel = df_reads_close_to_indel.loc[~((df_reads_close_to_indel['ref_end'] >= pos) & (df_reads_close_to_indel['ref_end'] <= indel_end))]

            # AND include a read only if neither its start nor end are within 10 base pairs of the deletion END
            df_reads_close_to_indel = df_reads_close_to_indel.loc[(abs(df_reads_close_to_indel['ref_start'] - indel_end) >= dist_from_indel) & 
                                                                  (abs(df_reads_close_to_indel['ref_end'] - indel_end) >= dist_from_indel)
                                                                 ]


            # AND exclude reads with ANY soft clipping from contributing to the depth
            df_reads_close_to_indel = df_reads_close_to_indel.query("num_soft_clipped == 0")
            
            # add a check to see if the start and end of reads that support and don't support the indel are significantly different. If they are, then it's probably a fixed indel.
            # KS test is very sensitive, using the median (Mann-Whitney U test) is more robust to outliers
            df_reads_close_to_indel['Indel'] = (df_reads_close_to_indel['num_indel'] > 0).astype(int)
            
            if df_reads_close_to_indel.Indel.nunique() == 2:

                start_pval = st.mannwhitneyu(df_reads_close_to_indel.query("Indel==1").ref_start,
                                            df_reads_close_to_indel.query("Indel==0").ref_start
                                           ).pvalue

                end_pval = st.mannwhitneyu(df_reads_close_to_indel.query("Indel==1").ref_end,
                                            df_reads_close_to_indel.query("Indel==0").ref_end
                                           ).pvalue

                # set the AF as 1
                if start_pval < 0.01 and end_pval < 0.01:
                    # subtract the soft-clipped reads from the total reads. These will not support the indel, which doesn't mean that the indel is present at low frequency. It's artifactual
                    df_indels.loc[i, ['Indel_Support', 'Total_Reads', 'AF_Adj']] = [len(df_reads_close_to_indel), 
                                                                                    len(df_reads_close_to_indel),# - row['CLIPPED_BASES'],
                                                                                    1
                                                                                   ]
                    
                else:
                    num_reads_close_supporting_indel = len(df_reads_close_to_indel.query("num_indel > 0"))

                    # these are very low quality indels. If all reads have been excluded, then it was probably an area of tons of discordant alignments, meaning there's no low frequency indel. 
                    # It's reference bias due to SVs
                    if len(df_reads_close_to_indel) == 0:
                        adj_AF = 0
                        print(sample, pos)
                    else:
                        adj_AF = num_reads_close_supporting_indel / (len(df_reads_close_to_indel))# - row['CLIPPED_BASES'])

                    # subtract the soft-clipped reads from the total reads. These will not support the indel, which doesn't mean that the indel is present at low frequency. It's artifactual
                    df_indels.loc[i, ['Indel_Support', 'Total_Reads', 'AF_Adj']] = [num_reads_close_supporting_indel, 
                                                                                    len(df_reads_close_to_indel),# - row['CLIPPED_BASES'],
                                                                                    adj_AF
                                                                                   ]

            else:
                num_reads_close_supporting_indel = len(df_reads_close_to_indel.query("num_indel > 0"))

                # these are very low quality indels. If all reads have been excluded, then it was probably an area of tons of discordant alignments, meaning there's no low frequency indel. 
                # It's reference bias due to SVs
                if len(df_reads_close_to_indel) == 0:
                    adj_AF = 0
                    print(sample, pos)
                else:
                    adj_AF = num_reads_close_supporting_indel / (len(df_reads_close_to_indel))# - row['CLIPPED_BASES'])

                # subtract the soft-clipped reads from the total reads. These will not support the indel, which doesn't mean that the indel is present at low frequency. It's artifactual
                df_indels.loc[i, ['Indel_Support', 'Total_Reads', 'AF_Adj']] = [num_reads_close_supporting_indel, 
                                                                                len(df_reads_close_to_indel),# - row['CLIPPED_BASES'],
                                                                                adj_AF
                                                                               ]
            
        else:
            # there should always be overlapping reads. If not, it means there's an SV or something with few to no reads in the pileup
            # but if that's the case, then there is def no indel, so remove these from the unfixed indels table
            df_indels.loc[i, ['Indel_Support', 'Total_Reads', 'AF_Adj']] = [np.nan, np.nan, np.nan]

    bam.close()

    return df_indels.dropna(subset='Indel_Support')



dist_from_indel = 10

df_indels_results = []

###### FILL IN ######
# samples_lst = 
# sample_dir =
# out_file =

for i, sample in enumerate(samples_lst):
        
    df_variants = pd.read_csv(f"{sample_dir}/{sample}/freebayes/....tsv", sep='\t').query("POS not in @rRNA_pos & POS not in @insertion_seqs_phages_pos")
        
    df_indels = df_variants.query("REF.str.len() != ALT.str.len()")
    
    # keep only low frequency variants. Extract fixed variants separately
    df_indels = apply_freebayes_lowAF_QCfilters(df_indels)
    
    if len(df_indels) > 0:
    
        df_indels = df_indels[['POS', 'REF', 'ALT', 'AF', 'AO', 'DP']].sort_values('POS')
        
        # require that the depth be at least the global median, so that we don't get low AF indels called in places with huge reference bias
        df_depth = pd.read_csv(f"{sample_dir}/{sample}/bam/{sample}.depth.tsv.gz", sep='\t', compression='gzip', names=['CHROM', 'POS', 'COV'])
        
        # median / 2 was a bit too strict, so dropped it to 1/4
        min_depth = df_depth.COV.median() / 4
                
        # df_indels = df_indels.query("POS not in @low_depth_sites")
        df_indels = df_indels.query("DP >= @min_depth")
        
        if len(df_indels) > 0:
            
            df_indels_adjusted_AF = compute_adjusted_AF_indels(sample,
                                                               df_indels, 
                                                               'Chromosome', 
                                                               # f"{sample_dir}/{sample}/bam/{sample}.dedup.bam", # REPLACE WITH THE FORMAT OF YOUR BAM OR CRm FILE
                                                               dist_from_indel=dist_from_indel,
                                                              )
            
            df_indels_adjusted_AF['SampleID'] = sample

            df_indels_results.append(df_indels_adjusted_AF)
            
            # print progress and save intermediate results
            if i % 100 == 0:
                if len(df_indels_results) > 0:
                    pd.concat(df_indels_results).to_csv(out_file, index=False)
                print(sample)
    
    
if len(df_indels_results) > 0:
    df_indels_results = pd.concat(df_indels_results)

    df_indels_results.to_csv(out_file, index=False)
else:
    print(f"No indels across {len(samples_lst)} samples")