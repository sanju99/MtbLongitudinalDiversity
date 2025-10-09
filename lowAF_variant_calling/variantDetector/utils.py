import numpy as np
import pandas as pd
import pysam, subprocess, vcf, os


def fasta_length(path):
    length = 0
    with open(path) as f:
        for line in f:
            if not line.startswith(">"):
                length += len(line.strip())
    return length


def apply_freebayes_lowAF_QCfilters(df_variants, AF_min=0.05, AF_max=0.98, MQ_thresh=40, num_support_each_direction=2):
    
    # add AF column
    df_variants['ALT']
    df_variants['AF'] = df_variants['AO'] / df_variants['DP']
    
    df_lowAF_variants = df_variants.query("AF >= @AF_min & AF <= @AF_max & MQM >= @MQ_thresh")

    df_lowAF_variants = pd.concat([df_lowAF_variants.query("(REF.str.len() - ALT.str.len() > 10)"),
                                   df_lowAF_variants.query("~(REF.str.len() - ALT.str.len() > 10) & SAF >= @num_support_each_direction & SAR >= @num_support_each_direction")
                                  ])    
    
    return df_lowAF_variants.reset_index(drop=True)


def get_ALT_allele_base_counts(row):
    
    ref_allele = row['REF']
    alt_allele = row['ALT']
    BC_string = row['BC']
    IC = row['IC']
    DC = row['DC']
    
    BC_lst = BC_string.split(',')
    
    bases_lst = ['A', 'C', 'G', 'T']
    
    if alt_allele in bases_lst:
        keep_idx = bases_lst.index(alt_allele)
        return int(BC_lst[keep_idx])
    # indels won't have a read support value. It will be in IC or DC, so use that
    else:
        if len(alt_allele) > len(ref_allele):
            return IC
        elif len(ref_allele) > len(alt_allele):
            return DC
        else:
            return np.nan


def apply_pilon_lowAF_QCfilters(df_variants, AF_min=0.05, AF_max=0.98, MQ_thresh=40, num_support=5):
    '''
    For columns BC and QP, the order of values is A,C,G,T
    '''
    
    # add the number of bases supporting the alternative allele
    df_variants['ALT_read_count'] = df_variants.apply(lambda row: get_ALT_allele_base_counts(row), axis=1)
     
    # add AF column
    df_variants['AF'] = df_variants['ALT_read_count'] / df_variants['DP']
    
    # also exclude imprecise variants
    df_lowAF_variants = df_variants.query("AF >= @AF_min & AF <= @AF_max & MQ >= @MQ_thresh & IMPRECISE == 0")


    df_lowAF_variants = pd.concat([df_lowAF_variants.query("(REF.str.len() - ALT.str.len() > 10)"),
                                   df_lowAF_variants.query("~(REF.str.len() - ALT.str.len() > 10) & ALT_read_count >= @num_support")
                                  ])    
    
    return df_lowAF_variants.reset_index(drop=True)



def save_table_of_soft_clips(sample, bam_file, ref_genome_file):
    
    dir_name = os.path.dirname(bam_file)
    
    genome_length = fasta_length(ref_genome_file)
    
    bam = pysam.AlignmentFile(bam_file, "rb")
    chrom = bam.references[0]

    records = []
    for read in bam.fetch():

        # skip unmapped reads
        if read.is_unmapped:
            continue
            
        num_left_clip = 0
        num_right_clip = 0

        if read.cigartuples:
            
            # check first operator
            if read.cigartuples[0][0] == 4:  # 4 = soft clip
                num_left_clip = read.cigartuples[0][1]

            # check last operator
            if read.cigartuples[-1][0] == 4:
                num_right_clip = read.cigartuples[-1][1]
                
        # add 1 because it's 0-indexed half-open
        start = read.reference_start + 1

        # don't add 1 because it's half-open at the end
        end = read.reference_end

        records.append({
            "read_name": read.query_name,
            "chrom": chrom,
            "start": start,
            "end": end,
            "cigar_string": read.cigarstring,
            "soft_clipped_bases": num_left_clip + num_right_clip, # total
            "left_clip_start": start - num_left_clip if num_left_clip > 0 else np.nan,
            "left_clip_end": start - 1 if num_left_clip > 0 else np.nan,
            "right_clip_start": end + 1 if num_right_clip > 0 else np.nan,
            "right_clip_end": end + num_right_clip if num_right_clip > 0 else np.nan,
        })

    bam.close()

    df = pd.DataFrame(records)
    
    # there will be clipping at the ends because the genome is circular, but we aligned reads to the linearized genome
    # exclude those from this computation because they're more artifactual
    # THESE ARE 1-INDEXED CLOSED INTERVALS, so the genome ranges from 1 - genome_length, INCLUSIVE
    # so exclude any soft-clipped regions that extend outside of the ref genome AT ALL
    df_soft_clips = df.dropna(subset=['left_clip_start', 'right_clip_start'], how='all').query("~(left_clip_start < 1) & ~(right_clip_end > @genome_length)").reset_index(drop=True)
    
    # convert this to a BED file, then use bedtools genomecov because we're getting a coverage track of soft clipping
    # make sure to get both left and right clips. Expand them like this
    df_soft_clips_BED = pd.concat([df_soft_clips.dropna(subset='left_clip_start')[['chrom', 'left_clip_start', 'left_clip_end']].rename(columns={'left_clip_start': 'BEG', 'left_clip_end': 'END'}),
                                   df_soft_clips.dropna(subset='right_clip_start')[['chrom', 'right_clip_start', 'right_clip_end']].rename(columns={'right_clip_start': 'BEG', 'right_clip_end': 'END'})
                                  ])

    # because we expanded, the length should be at least as long as df_soft_clips. And more than likely longer because some reads have both left and right soft clipping
    assert len(df_soft_clips_BED) >= len(df_soft_clips)
    # print(len(df_soft_clips_BED), len(df_soft_clips))

    # convert to 0-indexed half open intervals
    df_soft_clips_BED['BEG'] -= 1
    df_soft_clips_BED[['BEG', 'END']] = df_soft_clips_BED[['BEG', 'END']].astype(int)
    
    soft_clipping_BED_file = os.path.join(dir_name, f"{sample}.softClips.bed")
    df_soft_clips_BED.to_csv(soft_clipping_BED_file, sep='\t', header=None, index=False)
    
    ref_genome_chrom_lengths_file = f"{ref_genome_file.replace('.fna', '')}.chrom_lengths.txt"
    
    # then run the following to compute the number of soft clipped 
    if not os.path.isfile(ref_genome_chrom_lengths_file):
        subprocess.run(f"cut -f1,2 {fName}.fai > {ref_genome_chrom_lengths_file}", shell=True)

    soft_clips_by_pos_file = os.path.join(dir_name, f"{sample}.softClips.tsv.gz")
    subprocess.run(f"bedtools genomecov -i {soft_clipping_BED_file} -g {ref_genome_chrom_lengths_file} -d | gzip -c > {soft_clips_by_pos_file}", shell=True)
    
    
    
    
def get_orientation(read):
    """Return orientation of the pair: LR, RL, LL, RR, or None if not valid pair."""
    if not (read.is_paired and not read.is_unmapped and not read.mate_is_unmapped):
        return None

    # different chromosomes
    if read.reference_id != read.next_reference_id:
        return "DIFF_CHR"

    mate_start = read.next_reference_start
    this_start = read.reference_start

    # Determine left and right mate
    if this_start <= mate_start:
        left_rev  = read.is_reverse
        right_rev = read.mate_is_reverse
    else:
        left_rev  = read.mate_is_reverse
        right_rev = read.is_reverse

    if not left_rev and right_rev:
        return "LR"   # forward (left) + reverse (right) = normal Illumina
    if left_rev and not right_rev:
        return "RL"   # reverse (left) + forward (right) = IGV green
    if not left_rev and not right_rev:
        return "LL"   # both forward
    if left_rev and right_rev:
        return "RR"   # both reverse
    
    
    
def save_table_of_discordant_reads(sample, bam_file, ref_genome_file):
    
    dir_name = os.path.dirname(bam_file)
    
    genome_length = fasta_length(ref_genome_file)
    
    bam = pysam.AlignmentFile(bam_file, "rb")
    chrom = bam.references[0]

    records = []
    
    for read in bam.fetch():

        # skip unmapped reads
        if read.is_unmapped:
            continue

        orientation = get_orientation(read)

        if orientation != 'LR':
            
            # add 1 because it's 0-indexed half-open
            start = read.reference_start + 1

            # don't add 1 because it's half-open at the end
            end = read.reference_end
        
            records.append({
                "read_name": read.query_name,
                "chrom": chrom,
                "start": start,
                "end": end,
                "orientation": orientation,
                "insert_size": abs(read.template_length)
            })

    bam.close()

    df = pd.DataFrame(records)
    
    df_discordant_reads_BED = df[['chrom', 'start', 'end']]
    
    # convert to 0-indexed half open intervals
    df_discordant_reads_BED['start'] -= 1
    df_discordant_reads_BED[['start', 'end']] = df_discordant_reads_BED[['start', 'end']].astype(int)
    
    # save BED file
    discordant_alns_BED_file = os.path.join(dir_name, f"{sample}.DiscordantReads.bed")
    df_discordant_reads_BED.to_csv(discordant_alns_BED_file, sep='\t', header=None, index=False)
    
    # run bedtools genomecov
    ref_genome_chrom_lengths_file = f"{ref_genome_file.replace('.fna', '')}.chrom_lengths.txt"
    
    # then run the following to compute the number of soft clipped 
    if not os.path.isfile(ref_genome_chrom_lengths_file):
        subprocess.run(f"cut -f1,2 {fName}.fai > {ref_genome_chrom_lengths_file}", shell=True)

    discordant_alns_by_pos_file = os.path.join(dir_name, f"{sample}.DiscordantReads.tsv.gz")
    subprocess.run(f"bedtools genomecov -i {discordant_alns_BED_file} -g {ref_genome_chrom_lengths_file} -d | gzip -c > {discordant_alns_by_pos_file}", shell=True)
    
    


def compute_mean_base_quality_of_variant_support(bam_file, pos, low_freq_allele):
    
    bam = pysam.AlignmentFile(bam_file, "rb")
    
    base_qualities_low_freq_allele = []

    for pileupcolumn in bam.pileup('Chromosome', pos-1, pos, truncate=True):

        for pileupread in pileupcolumn.pileups:
            if pileupread.is_del or pileupread.is_refskip:
                continue  # skip deletions and skipped regions

            base = pileupread.alignment.query_sequence[pileupread.query_position]
            qual = pileupread.alignment.query_qualities[pileupread.query_position]

            if base == low_freq_allele:
                base_qualities_low_freq_allele.append(qual)

    return np.mean(base_qualities_low_freq_allele)




def get_allele_type(record, AF_min=0, AF_max=0.9):
    '''
    Returns "alt" or "ref" if the variant is low-quality or ambiguous. Otherwise this function returns "missing"
    
    Low-quality criteria:
    
        1. FILTER == Del, LowCov
        2. FILTER == Amb and 0.25 < AF <= 0.75
        3. SNP quality < 10

    Criteria for not confident in a variant or can not be reliably inserted, so leave it as reference:

        1. IMPRECISE variant (in the INFO field)
        2. Indels longer than 15 bp where neither the REF nor the ALT are of length 1 (this case is handled in the next function)
        
    If FILTER contains Amb and the alternative allele fraction > presentThresh, then it is a pure alternative call. 
    '''

    ref_allele = str(record.REF)
    alt_allele = "".join(np.array(record.ALT).astype(str))

    # this should not happen in pilon because it is not a haplotype variant caller
    # this would mean that there are 3 alleles present -- reference + 2 alternative alleles
    # haplotype variant callers will often have reference and alternative haplotypes separated by a comma in the ALT field, so this script will not work for them
    if ',' in alt_allele:
        print(fName, record)
        raise ValueError(f"There are multiple alternative alleles in a single record!")

    # fill in things that might be missing
    if "AO" in record.INFO.keys() and "DP" in record.INFO.keys():
        AO_lst = record.INFO["AO"]

        # multiple alternative alleles. Ignore them in this particular check
        if len(AO_lst) > 1:
            return "missing"
        else:
            AF = AO_lst[0] / record.INFO['DP']
    else:
        raise ValueError(f"AO or DP is not in the VCF record for POS = {record.POS}") 

    # QUAL field considers read depth, base quality, mapping quality. But it is also on the Phred scale
    if record.QUAL is None:
        qual = 11
    else:
        qual = record.QUAL
        
    # because IMPRECISE is taken care of above, this should only return missing for cases where REF = N or ALT = N
    if "N" in ref_allele or "N" in alt_allele:
        return "missing"
    
    # check if there are any non alphanumeric characters. This would indicate a heterogeneous alternative allele
    if not alt_allele.isalnum():
        return "missing" 
        
    # base quality, mapping quality, and read depth (measures of certainty about a variant)
    if 'DP' in record.INFO.keys():
        if record.INFO['DP'] < 5:
            return 'missing'

    if 'MQM' in record.INFO.keys():
        if len(record.INFO['MQM']) > 1:
            return "missing"
        else:
            if record.INFO['MQM'][0] < 30:
                return 'missing'

    if AF >= AF_min and AF <= AF_max:
        return "intermediate"
    
    elif AF < AF_min:
        return "ref"
    
    else:
        return "fixed"
    
    

def compute_sites_with_high_lowAF_SNP_density(SNPs_VCF_file, ref_genome_file, AF_min=0, AF_max=0.9, SNP_window_size=100):

    genome_length = fasta_length(ref_genome_file)
    
    df_SNPs = pd.DataFrame(columns = ['POS', 'SNP'])
    i = 0
    
    vcf_reader = vcf.Reader(filename=SNPs_VCF_file)
        
    for record in vcf_reader:
        snp_type = get_allele_type(record, AF_min=AF_min, AF_max=AF_max)
        df_SNPs.loc[i, :] = [record.POS, snp_type]
        i += 1
    
    df_SNPs = df_SNPs.query("SNP == 'intermediate'").drop_duplicates().reset_index(drop=True)

    # add the remaining sites to the dataframe
    ref_sites = list(set(np.arange(1, genome_length+1)) - set(df_SNPs.POS))
    
    df_SNPs = pd.concat([df_SNPs,
                         pd.DataFrame({'POS': ref_sites, 'SNP': 'ref'})
                        ]).sort_values("POS").reset_index(drop=True)
    
    df_SNPs['SNP'] = df_SNPs['SNP'].map({'ref': 0, 'alt': 1})
    
    assert sum(pd.isnull(df_SNPs['SNP'])) == 0
    assert len(df_SNPs) == genome_length

    df_SNPs['SNPS_LEFT_ROLLING_AVG'] = df_SNPs['SNP'].rolling(window=SNP_window_size, min_periods=1, closed='right').mean()
    df_SNPs['SNPS_RIGHT_ROLLING_AVG'] = df_SNPs['SNP'][::-1].rolling(window=SNP_window_size, min_periods=1, closed='right').mean()[::-1]

    # take the maximum at each site
    df_SNPs['SNPS_MAX_ROLLING_AVG'] = np.max([df_SNPs['SNPS_LEFT_ROLLING_AVG'], df_SNPs['SNPS_RIGHT_ROLLING_AVG']], axis=0)

    return df_SNPs




def split_MNPs_into_SNPs(df_variants):
    
    # make sure only SNPs
    df_variants = df_variants.query("REF.str.len() == ALT.str.len()")
    
    df_MNPs = df_variants.query("REF.str.len() > 1").reset_index(drop=True)
    df_SNPs = df_variants.query("REF.str.len() == 1").reset_index(drop=True)

    # all the values that must be duplicated
    duplicate_columns = list(set(df_variants.columns) - set(['POS', 'REF', 'ALT']))

    # helps to control ordering so we know that POS, REF, and ALT are at the end
    df_split_MNPs = pd.DataFrame(columns = duplicate_columns + ['POS', 'REF', 'ALT'])
    k = 0

    for _, row in df_MNPs.iterrows():

        split_ref_alleles = list(row['REF'])
        split_alt_alleles = list(row['ALT'])

        # need to update the positions
        start = row['POS']
        pos_lst = np.arange(start, start + len(row['REF']))

        for i, (ref, alt) in enumerate(zip(split_ref_alleles, split_alt_alleles)):

            # some could be the same because the MNP includes both changed and unchanged sites
            if ref != alt:
                df_split_MNPs.loc[k, :] = [row[col] for col in duplicate_columns] + [pos_lst[i], ref, alt]
                k += 1

    df_variants_split_MNPs = pd.concat([df_SNPs, df_split_MNPs]).sort_values(['POS']).reset_index(drop=True)
    assert len(df_variants_split_MNPs.iloc[df_variants_split_MNPs.index.values[df_variants_split_MNPs.duplicated(subset=['POS'], keep=False)]]) == 0

    return df_variants_split_MNPs




def split_SNPs_indels_same_haplotype(df_variants):

    # in cases where an indel and a SNP occur in close proximity to each other, freebayes will put them on the same record if it finds sufficient
    # evidence that they are part of the same haplotype.
    # for this though, we need to split those so that we keep the SNP part but discard the indel. Otherwise, the whole thing will be detected as as indel
    # and be discarded from the SNP calculation

    # do this so that we can get the indexes of the variants in df_SNP_indels and drop them
    df_variants = df_variants.reset_index(drop=True)
    
    # first, get all indels and restrict to those where both REF and ALT have lengths longer than 1
    # if either has length = 1, then the variant is just an indel, not a SNP + indel in close proximity
    df_SNP_indels = df_variants.query("REF.str.len() > 1 & ALT.str.len() > 1 & REF.str.len() != ALT.str.len()")

    # filter to keep only variants that have a matching nucleotide anywhere. The matching nucleotides (no ALT, just REF) separate the SNP and indel from each other
    df_SNP_indels = df_SNP_indels[
        df_SNP_indels.apply(lambda row: any(r == a for r, a in zip(row['REF'], row['ALT'])), axis=1)
    ]
        
    # next, iterate and split
    # all the values that must be duplicated
    duplicate_columns = list(set(df_SNP_indels.columns) - set(['POS', 'REF', 'ALT']))

    # helps to control ordering so we know that POS, REF, and ALT are at the end
    df_split_SNPs_indels = pd.DataFrame(columns = duplicate_columns + ['POS', 'REF', 'ALT'])
    k = 0

    for _, row in df_SNP_indels.iterrows():

        split_ref_alleles = list(row['REF'])
        split_alt_alleles = list(row['ALT'])

        # need to update the positions
        start = row['POS']
        pos_lst = np.arange(start, start + len(row['REF']))

        # this will exclude indels because when we zip them together, it will trim the ends. This is fine because we left-normalized the variants
        for i, (ref, alt) in enumerate(zip(split_ref_alleles, split_alt_alleles)):

            # some could be the same because the MNP includes both changed and unchanged sites
            if ref != alt:
                df_split_SNPs_indels.loc[k, :] = [row[col] for col in duplicate_columns] + [pos_lst[i], ref, alt]
                k += 1
                
    return pd.concat([df_variants.drop(df_SNP_indels.index.values),
                      df_split_SNPs_indels
                     ]).sort_values('POS').reset_index(drop=True)