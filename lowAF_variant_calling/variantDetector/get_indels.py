import numpy as np
import pandas as pd
import sklearn.model_selection
import argparse, os, glob, re, warnings, shutil, pysam
import matplotlib.pyplot as plt
import seaborn as sns
warnings.filterwarnings('ignore')
import statsmodels.api as sm
from Bio import Seq, SeqIO
import scipy.stats as st

out_file = f"indels_called_from_H37Rv.csv"

personal_ref_dir = "/n/data1/hms/dbmi/farhat/Sanjana/TRUST_aln_personal_assembly"
H37Rv_ref_dir = "/n/data1/hms/dbmi/farhat/Sanjana/TRUST_lowAF"

df_LR_assembly_metadata = pd.read_csv("../../data/TRUST.PBAsm.ZA_106CI_AssemblySummary.csv")
df_LR_assembly_metadata['Lineage'] = df_LR_assembly_metadata['Lineage_Asm'].str.replace('lineage', '')

# this keeps only those with NumContigs = 1 and numContigs_Complete = 1 in df_LR_assembly_metadata, and the lineages (Coll2014 exactly) of the LR and SR sequences must match
df_personal_assemblies = pd.read_csv("../data/personal_assemblies_samples.tsv", sep='\t', header=None, names = ['Sample', 'Assembly', 'PacBio'])
samples_with_assemblies = df_personal_assemblies['Sample'].values

df_trust_patients = pd.read_csv("~/TRUST_data_processing/processed_data/20250904_cleaned_patient_outcomes_data.csv")

for i, row in df_trust_patients.iterrows():
    if not row['Original_ID'].startswith('S'):
        
        length_numerical = len(row['Original_ID'].split('-')[0])
            
        newID = 'S' + '0' * (4 - length_numerical) + row['Original_ID']
        df_trust_patients.loc[i, 'Original_ID'] = newID
        
    if not pd.isnull(row['Lineage']):
        if type(row['Lineage']) == float:
            df_trust_patients.loc[i, 'Lineage'] = str(int(row['Lineage']))
        else:
            df_trust_patients.loc[i, 'Lineage'] = str(row['Lineage'])
     
print(df_trust_patients.Lineage.value_counts())
            
F2_thresh = 0.03

mixed_lineage_samples = df_trust_patients.query("F2 > @F2_thresh").SampleID.unique()
unmixed_lineage_samples = df_trust_patients.query("F2 <= @F2_thresh").SampleID.unique()

h37Rv_path = "/n/data1/hms/dbmi/farhat/Sanjana/H37Rv"
h37Rv_seq = SeqIO.read(os.path.join(h37Rv_path, "GCF_000195955.2_ASM19595v2_genomic.gbff"), "genbank")
h37Rv_genes = pd.read_csv(os.path.join(h37Rv_path, "mycobrowser_h37rv_genes_v4.csv"))
h37Rv_regions = pd.read_csv(os.path.join(h37Rv_path, "mycobrowser_h37rv_v4.csv"))

h37Rv_coords = pd.read_csv(os.path.join(h37Rv_path, "h37Rv_coords_to_gene.csv"))
h37Rv_coords_dict = dict(zip(h37Rv_coords["pos"].values, h37Rv_coords["region"].values))

# remove rRNAs, which are highly conserved. rrs, rrl, and rrf
rRNA_pos = []

for i, row in h37Rv_regions.query("Functional_Category=='stable RNAs' & Feature=='rRNA'").iterrows():
    print(row['Name'])
    rRNA_pos += list(np.arange(row['Start'], row['Stop'] + 1))
    
    
# exclude any indel within 100 bp of an insertion seq / phage and also within those regions
insertion_seqs_phages_pos = []

for i, row in h37Rv_regions.query("Functional_Category=='insertion seqs and phages'").iterrows():
    
    insertion_seqs_phages_pos += list(np.arange(row['Start'] - 100, row['Stop'] + 100 + 1))
    
insertion_seqs_phages_pos = np.unique(insertion_seqs_phages_pos)
len(insertion_seqs_phages_pos)


def apply_freebayes_lowAF_QCfilters(df_variants, DP=True, AF_min=0.05, AF_max=0.98, MQ_thresh=40, num_support_each_direction=2):
    '''
    Use 0.98 when doing the validaiton using the personal ref genomes because if there's a variant present at 100% relative to H37Rv, there won't be a variant when called against the
    personal reference genome because it's purely the nucleotide in the assembly. 
    
    But for calling variants against H37Rv, later on, keep everything
    '''
    
    # add AF column
    df_variants['AF'] = df_variants['AO'] / df_variants['DP']
    
    if DP:
        df_lowAF_variants = df_variants.query("DP >= 5 & AF >= @AF_min & AF <= @AF_max & MQM >= @MQ_thresh")
    else:
        df_lowAF_variants = df_variants.query("AF >= @AF_min & AF <= @AF_max & MQM >= @MQ_thresh")

    df_lowAF_variants = pd.concat([df_lowAF_variants.query("(REF.str.len() - ALT.str.len() > 10)"),
                                   df_lowAF_variants.query("~(REF.str.len() - ALT.str.len() > 10) & SAF >= @num_support_each_direction & SAR >= @num_support_each_direction")
                                  ])    
    
    return df_lowAF_variants.reset_index(drop=True)


def keep_middle_percentage_of_reads(df, n_prop):
    
    # should be a proportion, not a percentage
    if n_prop > 1:
        n_prop /= 1

    # Calculate how many rows to keep
    n_rows = len(df)
    keep = int(n_rows * n_prop)

    # Calculate start and end indices
    start = (n_rows - keep) // 2
    end = start + keep

    # Slice the middle N%
    return df.iloc[start:end].reset_index(drop=True)


def compute_adjusted_AF_indels(df_indels, chrom, bam_file, n_prop=0.50, dist_from_indel=10):
    
    df_indels = df_indels.reset_index(drop=True)
    
    bam = pysam.AlignmentFile(bam_file, "rb")

    for i, row in df_indels.iterrows():
        
        pos = row['POS']
        ref = row['REF']
        alt = row['ALT']
        
        indel_length = abs(len(ref) - len(alt))
        
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
            df_indels = pd.DataFrame()

    bam.close()

    return df_indels


samples_lst = os.listdir(H37Rv_ref_dir)
# samples_lst = ['MFS-1']
print(f"Working on {len(samples_lst)} samples")


AF_min = 0.05
AF_max = 100

dist_from_indel = 10

df_indels_results = []

for i, sample in enumerate(samples_lst):
        
    # df_variants = pd.read_csv(f"{H37Rv_ref_dir}/{sample}/freebayes/{sample}.excludeLowConf.tsv", sep='\t').query("POS not in @rRNA_pos & POS not in @insertion_seqs_phages_pos")
    df_variants = pd.read_csv(f"{H37Rv_ref_dir}/{sample}/freebayes/{sample}.excludeLowConf.split.fixedAO.final.tsv", sep='\t').query("POS not in @rRNA_pos & POS not in @insertion_seqs_phages_pos")
        
    df_indels = df_variants.query("REF.str.len() != ALT.str.len()")
    
    # keep only low frequency variants. Extract fixed variants separately
    df_indels = apply_freebayes_lowAF_QCfilters(df_indels, AF_min=AF_min, AF_max=AF_max)
    
    if len(df_indels) > 0:
    
        # df_soft_clips = pd.read_csv(f"{H37Rv_ref_dir}/{sample}/bam/{sample}.softClips.tsv.gz", sep='\t', header=None, names=['CHROM', 'POS', 'CLIPPED_BASES'])

        df_candidate_unfixed_indels = df_indels[['POS', 'REF', 'ALT', 'AF', 'AO', 'DP']].sort_values('POS')#.merge(df_soft_clips, on='POS')
        
#         # require that the depth be at least the global median, so that we don't get low AF indels called in places with huge reference bias
#         df_depth = pd.read_csv(f"{H37Rv_ref_dir}/{sample}/bam/{sample}.depth.tsv.gz", sep='\t', compression='gzip', names=['CHROM', 'POS', 'COV'])
        
#         min_depth = df_depth.COV.median() / 2
        
#         low_depth_sites = df_depth.query("COV < @min_depth").POS.values
        
#         df_candidate_unfixed_indels = df_candidate_unfixed_indels.query("POS not in @low_depth_sites")
        
        if len(df_candidate_unfixed_indels) > 0:
            
            df_candidate_unfixed_indels_adjusted_AF = compute_adjusted_AF_indels(df_candidate_unfixed_indels, 
                                                                                 'Chromosome', 
                                                                                 f"{H37Rv_ref_dir}/{sample}/bam/{sample}.dedup.bam", 
                                                                                 dist_from_indel=dist_from_indel,
                                                                                )
            
            df_candidate_unfixed_indels_adjusted_AF['SampleID'] = sample

            df_indels_results.append(df_candidate_unfixed_indels_adjusted_AF)
            
            if i % 100 == 0:
                pd.concat(df_indels_results).to_csv(out_file, index=False)
                print(sample)
    
df_indels_results = pd.concat(df_indels_results)

df_indels_results = df_indels_results.merge(df_trust_patients[['SampleID', 'pid']], how='left')
# assert sum(pd.isnull(df_benchmarking['pid'])) == 0

df_indels_results.to_csv(out_file, index=False)