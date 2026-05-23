import numpy as np
import pandas as pd
import os, glob, warnings, pysam, argparse
warnings.filterwarnings('ignore')
from Bio import Seq, SeqIO
import scipy.stats as st
from utils import *

personal_ref_dir = "/n/data1/hms/dbmi/farhat/Sanjana/TRUST_aln_personal_assembly"
H37Rv_ref_dir = "/n/data1/hms/dbmi/farhat/Sanjana/TRUST_lowAF"

parser = argparse.ArgumentParser()

parser.add_argument("--AF_min",  type=float, default=0.01)
parser.add_argument("--AF_max",  type=float, default=0.99)
parser.add_argument("-o",  dest='out_file', type=str, required=True)

cmd_line_args = parser.parse_args()

AF_min = cmd_line_args.AF_min
AF_max = cmd_line_args.AF_max
out_file = cmd_line_args.out_file

# df_LR_assembly_metadata = pd.read_csv("../../data/TRUST.PBAsm.ZA_106CI_AssemblySummary.csv")
# df_LR_assembly_metadata['Lineage'] = df_LR_assembly_metadata['Lineage_Asm'].str.replace('lineage', '')

# # this keeps only those with NumContigs = 1 and numContigs_Complete = 1 in df_LR_assembly_metadata, and the lineages (Coll2014 exactly) of the LR and SR sequences must match
# df_personal_assemblies = pd.read_csv("../data/personal_assemblies_samples.tsv", sep='\t', header=None, names = ['Sample', 'Assembly', 'PacBio'])
# samples_with_assemblies = df_personal_assemblies['Sample'].values

df_trust_patients = pd.read_csv("~/TRUST_data_processing/processed_data/combined_patient_WGS_data.csv")

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

# remove rRNAs, which are highly conserved. rrs, rrl, and rrf
rRNA_pos = []

for i, row in h37Rv_regions.query("Functional_Category=='stable RNAs' & Feature=='rRNA'").iterrows():
    rRNA_pos += list(np.arange(row['Start'], row['Stop'] + 1))
    
    
# exclude any indel within 100 bp of an insertion seq / phage and also within those regions
insertion_seqs_phages_pos = []

for i, row in h37Rv_regions.query("Functional_Category=='insertion seqs and phages'").iterrows():
    insertion_seqs_phages_pos += list(np.arange(row['Start'] - 100, row['Stop'] + 100 + 1))
    
insertion_seqs_phages_pos = np.unique(insertion_seqs_phages_pos)
# len(insertion_seqs_phages_pos)


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



def has_indel_at_pos(read, pos_1based, indel_type):
    """
    indel_type: "I" or "D"
    """
    if read.cigartuples is None:
        return False

    ref_pos = read.reference_start + 1  # convert to 1-based

    for op, length in read.cigartuples:

        # Match / mismatch
        if op == 0:
            ref_pos += length

        # Insertion
        elif op == 1:
            # insertion is anchored *after* previous reference base
            if indel_type == "I" and ref_pos - 1 == pos_1based:
                return True

        # Deletion
        elif op == 2:
            if indel_type == "D" and (ref_pos <= pos_1based < ref_pos + length or ref_pos - 1 == pos_1based):
                return True
            ref_pos += length

        # Reference skip
        elif op == 3:
            ref_pos += length

        # Soft clip (does not affect reference)
        elif op == 4:
            continue

    return False



def compute_adjusted_AF_indels(sample, df_indels, chrom, aln_file, n_prop=0.50, dist_from_indel=10):
    
    df_indels = df_indels.reset_index(drop=True)
    
    if aln_file[-4:] == '.bam':
        bam = pysam.AlignmentFile(aln_file, "rb")
    elif aln_file[-5:] == '.cram':
        bam = pysam.AlignmentFile(aln_file, "rc")
    else:
        raise ValueError(f"{aln_file} is not valid")

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
                    # need this because sometimes you can have multiple indels at the same site. If that's the case, all of them should be represented
                    # so you should have multiple non-zero values of indel_length_from_cigar in df_overlapping_reads
                    # if has_indel_at_pos returns False BUT there is an indel of the correct length in that read, then that means the indel is shifted in the read
                    # it's in the same position as the indel we think, but it's in some repetitive region with a whole repeat unit inserted or deleted
                    # in that case, the indel has been left-aligned in most reads, but in some reads it's later on, depending on where the start and end of that read are
                    # this read SHOULD contribute to the total count of reads supporting the variant                    
                    # compute the total number of I or D values in the cigar string. So if the cigar is 1M10I4X15I, and search_string = 'I', it returns 25
                    # if there are multiple indels at the same site, they will be reflected by different values of the num_indel column and will split the support accordingly
                    indel_length_from_cigar = sum(length for (op, length) in (read.cigartuples or []) if op == search_val and length > 0)
                        
                    # get number of bases soft clipped from this read
                    num_soft_clipped_bases = sum(length for (op, length) in (read.cigartuples or []) if op == 4)

                    reads_overlapping.append({
                        "read_name": read.query_name,
                        "ref_start": read.reference_start + 1,  # make it 1-based
                        "ref_end": read.reference_end,
                        "num_indel": indel_length_from_cigar,
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
            
            # these are very low quality indels. If all reads have been excluded, then it was probably an area of tons of discordant alignments, meaning there's no low frequency indel. 
            # It's reference bias due to SVs
            if len(df_reads_close_to_indel) == 0:
                df_indels.loc[i, ['Indel_Support', 'Total_Reads', 'AF_Adj']] = [0, 0, 0]
                print(sample, pos)
            else:
                num_reads_close_supporting_indel = len(df_reads_close_to_indel.query("num_indel == @indel_length"))

                adj_AF = num_reads_close_supporting_indel / len(df_reads_close_to_indel)

                # first: there are reads supporting the indel (num_indel.max() > 0) and reads not supporting it (num_indel.min() == 0)
                # need to make separate cases for if there are multiple indels at the site, which happens in repetitive regions
                # where there may be one or multiple copies of the repeat unit inserted or deleted
                # also need to check if indel_length is in df_reads_close_to_indel.num_indel.values because the reads supporting it may be too close to the indel and therefore removed above
                if df_reads_close_to_indel.num_indel.min() == 0 and df_reads_close_to_indel.num_indel.max() > 0 and indel_length in df_reads_close_to_indel.num_indel.values:
                   
                    # add a check to see if the start and end of reads that support and don't support the indel are significantly different. If they are, then it's probably a fixed indel.
                    # KS test is very sensitive, using the median (Mann-Whitney U test) is more robust to outliers
                    start_pval = st.mannwhitneyu(df_reads_close_to_indel.query("num_indel==@indel_length").ref_start,
                                                df_reads_close_to_indel.query("num_indel==0").ref_start
                                               ).pvalue

                    end_pval = st.mannwhitneyu(df_reads_close_to_indel.query("num_indel==@indel_length").ref_end,
                                                df_reads_close_to_indel.query("num_indel==0").ref_end
                                               ).pvalue

                    # set the AF as 1
                    if start_pval < 0.01 and end_pval < 0.01:

                        # the reads that don't support the indel just don't fully span the indel. They look like they don't support the indel but it's artifactual
                        # so remove them and recalculate the AF. These reads shouldn't contribute to the depth
                        df_reads_close_to_indel = df_reads_close_to_indel.query("num_indel > 0")

                        #  These will not support the indel, which doesn't mean that the indel is present at low frequency. It's artifactual
                        df_indels.loc[i, ['Indel_Support', 'Total_Reads', 'AF_Adj']] = [num_reads_close_supporting_indel, 
                                                                                        len(df_reads_close_to_indel),
                                                                                        num_reads_close_supporting_indel / len(df_reads_close_to_indel)
                                                                                       ]

                    else:
                        df_indels.loc[i, ['Indel_Support', 'Total_Reads', 'AF_Adj']] = [num_reads_close_supporting_indel, 
                                                                                        len(df_reads_close_to_indel),
                                                                                        adj_AF
                                                                                       ]

                # there are only reads supporting the indel. Then no need for the KS test
                else:
                    df_indels.loc[i, ['Indel_Support', 'Total_Reads', 'AF_Adj']] = [num_reads_close_supporting_indel, 
                                                                                    len(df_reads_close_to_indel),
                                                                                    adj_AF
                                                                                   ]

        else:
            # there should always be overlapping reads. If not, it means there's an SV or something with few to no reads in the pileup
            # but if that's the case, then there is def no indel, so remove these from the unfixed indels table
            df_indels.loc[i, ['Indel_Support', 'Total_Reads', 'AF_Adj']] = [np.nan, np.nan, np.nan]

    bam.close()

    return df_indels.dropna(subset='Indel_Support')


print(f"Working on {len(unmixed_lineage_samples)} samples")

dist_from_indel = 10

df_indels_results = []

for i, sample in enumerate(unmixed_lineage_samples):
        
    df_variants = pd.read_csv(f"{H37Rv_ref_dir}/{sample}/freebayes/{sample}.cleaned.excludeLowConf.fixedAO.tsv", sep='\t').query("POS not in @rRNA_pos & POS not in @insertion_seqs_phages_pos")
        
    df_indels = df_variants.query("REF.str.len() != ALT.str.len()")
    
    # keep only low frequency variants. Extract fixed variants separately
    df_indels = apply_freebayes_lowAF_QCfilters(df_indels, AF_min=AF_min, AF_max=AF_max)
    
    if len(df_indels) > 0:
    
        df_indels = df_indels[['POS', 'REF', 'ALT', 'AF', 'AO', 'DP', 'ANN[0].GENE', 'ANN[0].HGVS_C', 'ANN[0].HGVS_P']].sort_values('POS').rename(columns={'ANN[0].GENE': 'GENE', 'ANN[0].HGVS_C': 'HGVS_C', 'ANN[0].HGVS_P': 'HGVS_P'})
        
        # require that the depth be at least the global median, so that we don't get low AF indels called in places with huge reference bias
        df_depth = pd.read_csv(f"{H37Rv_ref_dir}/{sample}/bam/{sample}.depth.tsv.gz", sep='\t', compression='gzip', names=['CHROM', 'POS', 'COV'])
        
        min_depth = df_depth.COV.median() / 3
        
        # freebayes nonsense again. Sometimes DP is way lower than the real coverage at the site. Idek why. But AO and DP match up, so use that for AF computation
        # but use COV computed from BAM file to exclude sites with too low coverage
        low_depth_sites = df_depth.query("COV < @min_depth").POS.values
        
        df_indels = df_indels.query("POS not in @low_depth_sites")
        # df_indels = df_indels.query("DP >= @min_depth")
                
        if len(df_indels) > 0:
            
            df_indels_adjusted_AF = compute_adjusted_AF_indels(sample,
                                                               df_indels, 
                                                               'Chromosome', 
                                                               f"{H37Rv_ref_dir}/{sample}/bam/{sample}.dedup.cram", 
                                                               dist_from_indel=dist_from_indel,
                                                              )
            
            df_indels_adjusted_AF['SampleID'] = sample
            
            df_indels_adjusted_AF = df_indels_adjusted_AF.query("Total_Reads >= @min_depth")

            df_indels_results.append(df_indels_adjusted_AF)
            
            # print progress and save intermediate results
            if i % 100 == 0:
                if len(df_indels_results) > 0:
                    pd.concat(df_indels_results).to_csv(out_file, index=False)
                print(sample)
    
    
if len(df_indels_results) > 0:
    df_indels_results = pd.concat(df_indels_results)

    df_indels_results = df_indels_results.merge(df_trust_patients[['SampleID', 'pid']], how='left')
    
    # drop any duplicates that may have accidentally arisen
    df_indels_results = df_indels_results.drop_duplicates().reset_index(drop=True)
    
    df_indels_results['REF_len'] = df_indels_results['REF'].str.len()
    df_indels_results['ALT_len'] = df_indels_results['ALT'].str.len()

    # edge case regarding splitting haplotypes. Sometimes the indel appears twice because the left-normalization part is affected
    # these cases have the exact same AO and DP values, so deduplicate that way
    df_indels_results.drop_duplicates(subset=['SampleID', 'REF_len', 'ALT_len', 'AO', 'DP']).to_csv(out_file, index=False)
else:
    print(f"No indels across {len(unmixed_lineage_samples)} samples")