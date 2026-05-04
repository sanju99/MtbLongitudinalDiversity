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


def apply_freebayes_lowAF_QCfilters(df_variants, AF_min=0.05, AF_max=0.95, MQ_thresh=40, num_support=5, forward_reverse_support=2):
    '''
    Use 0.98 when doing the validaiton using the personal ref genomes because if there's a variant present at 100% relative to H37Rv, there won't be a variant when called against the
    personal reference genome because it's purely the nucleotide in the assembly. 
    
    But for calling variants against H37Rv, later on, keep everything
    '''
    
    # add AF column
    df_variants['AF'] = df_variants['AO'] / df_variants['DP']
    df_variants['AF'] = df_variants['AF'].astype(float)
    
    df_variants['AF_rounded'] = np.round(df_variants['AF'], 2)
    
    df_lowAF_variants = df_variants.query("(AF >= @AF_min | AF_rounded >= @AF_min) & AF <= @AF_max")
    # df_lowAF_variants = df_variants.query("AF >= @AF_min & AF <= @AF_max")
    
    df_lowAF_variants = df_lowAF_variants.query("MQM >= @MQ_thresh & AO >= @num_support")

    df_lowAF_variants = pd.concat([df_lowAF_variants.query("(REF.str.len() - ALT.str.len() > 10)"),
                                   df_lowAF_variants.query("~(REF.str.len() - ALT.str.len() > 10) & AO >= @num_support")
                                  ])    
    
    df_lowAF_variants = df_lowAF_variants.query("SAF >= @forward_reverse_support & SAR >= @forward_reverse_support")
    
    return df_lowAF_variants.reset_index(drop=True)




def apply_pilon_lowAF_QCfilters(df_variants, AF_min=0.05, AF_max=0.95, MQ_thresh=40, num_support=5):
    '''
    For columns BC and QP, the order of values is A,C,G,T
    '''
        
    # some variants could have high IC or DC in mixed lineage samples. One strain may have the SNP and another may have the deletion
    # idk, we would still need to separate them before being confident in the variants
    df_variants = df_variants.query("MQ >= @MQ_thresh").query("REF.str.len() == ALT.str.len()")
    
    # some variants lie near the 0.05 threshold. If the AF rounds to 0.05 (at two sig figs), keep it too
    df_variants['AF_rounded'] = np.round(df_variants['AF'], 2)
    
    # AF thresholding 
    df_lowAF_variants = df_variants.query("(AF >= @AF_min | AF_rounded >= @AF_min) & AF <= @AF_max")
    
    if 'AO' in df_lowAF_variants.columns:
        assert sum(pd.isnull(df_lowAF_variants['AO'])) == 0
    
        # added after processing the output of the get_alt_allele function
        df_lowAF_variants = df_lowAF_variants.query("AO >= @num_support")

    return df_lowAF_variants.reset_index(drop=True)



def read_pilon_vcf(vcf_path):
    
    # XC is the number of clipped bases at the site
    keep_cols = ['CHROM', 'POS', 'REF', 'ALT', 'QUAL', 'FILTER', 'DP', 'TD', 'BQ', 'MQ', 'BC', 'IC', 'DC', 'XC', 'AF']

    records = []
    info_keys = set()
    format_keys = set()

    with open(vcf_path) as f:
        for line in f:
            line = line.strip()

            # Skip meta-information
            if line.startswith("##"):
                continue

            # Header line
            if line.startswith("#CHROM"):
                header = line.lstrip("#").split("\t")
                continue

            # Variant line
            fields = dict(zip(header, line.split("\t")))

            # --------------------
            # Parse INFO field
            # --------------------
            info_dict = {}
            for item in fields["INFO"].split(";"):
                if "=" in item:
                    k, v = item.split("=", 1)
                    info_dict[k] = v
                else:
                    # Flag field (e.g. IMPRECISE)
                    info_dict[item] = True

            info_keys.update(info_dict.keys())

            # --------------------
            # Parse FORMAT field(s)
            # --------------------
            format_dict = {}
            if "FORMAT" in fields:
                fmt_keys = fields["FORMAT"].split(":")
                sample_cols = [c for c in header if c not in {
                    "CHROM","POS","ID","REF","ALT","QUAL","FILTER","INFO","FORMAT"
                }]

                for sample in sample_cols:
                    fmt_vals = fields[sample].split(":")
                    for k, v in zip(fmt_keys, fmt_vals):
                        format_dict[f"{sample}_{k}"] = v
                        format_keys.add(f"{sample}_{k}")

            # --------------------
            # Combine everything
            # --------------------
            base = {
                "CHROM": fields["CHROM"],
                "POS": int(fields["POS"]),
                "REF": fields["REF"],
                "ALT": fields["ALT"],
                "QUAL": None if fields["QUAL"] == "." else float(fields["QUAL"]),
                "FILTER": fields["FILTER"],
            }

            record = {**base, **info_dict, **format_dict}
            records.append(record)

    # --------------------
    # Build DataFrame
    # --------------------
    df = pd.DataFrame(records)

    # Ensure all INFO / FORMAT columns exist even if missing in some rows
    for col in info_keys.union(format_keys):
        if col not in df.columns:
            df[col] = pd.NA

    return df[keep_cols].reset_index(drop=True)




def get_alt_allele(BC_string, ref_allele, base_order=['A', 'C', 'G', 'T']):
    '''
    Already filtered the VCF to require there to be at least 5 non-REF reads, so there will be an ALT allele
    '''
    
    alt_counts = np.array(BC_string.split(',')).astype(int)
    
    ref_idx = base_order.index(ref_allele)
    ref_count = alt_counts[ref_idx]
    
    # sort after set subtraction. base order is in alpha order
    non_ref_alleles = np.sort(list(set(base_order) - set([ref_allele])))
    non_ref_counts = [count for i, count in enumerate(alt_counts) if i != ref_idx]
    assert len(non_ref_counts) == len(alt_counts) - 1
    
    # if there are multiple ALT alleles with the same support, the first one in alpha-order is returned.
    alt_allele = non_ref_alleles[np.argmax(non_ref_counts)]
    
    return alt_allele, np.max(non_ref_counts)#, ref_count



def save_table_of_soft_clips(sample, bam_file, ref_genome_file):
    
    dir_name = os.path.dirname(bam_file)
    
    genome_length = fasta_length(ref_genome_file)
    
    if bam_file[-4:] == '.bam':
        bam = pysam.AlignmentFile(bam_file, "rb")
    else:
        bam = pysam.AlignmentFile(bam_file, "rc")
    
    chrom = bam.references[0]

    records = []
    for read in bam.fetch():

        # skip unmapped reads or those with secondary or supplementary alignments
        if read.is_unmapped or read.is_secondary or read.is_supplementary:
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
        subprocess.run(f"cut -f1,2 {ref_genome_file}.fai > {ref_genome_chrom_lengths_file}", shell=True)

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
    
    if bam_file[-4:] == '.bam':
        bam = pysam.AlignmentFile(bam_file, "rb")
    else:
        bam = pysam.AlignmentFile(bam_file, "rc")
        
    chrom = bam.references[0]

    records = []
    
    for read in bam.fetch():

        # skip unpaired or unmapped reads or those with secondary or supplementary alignments
        if not read.is_paired or read.is_unmapped or read.is_secondary or read.is_supplementary:
            continue
            
        orientation = get_orientation(read)
        
        # require the insert size to be longer than the read length when counting discordant reads. If insert size < read length, discordant read determination is incorrect
        # it's too aggressive in calling discordant reads because they might start within a few bp of each other        
        insert_size = abs(read.template_length)
        read_length = read.query_length

        if orientation != 'LR' and insert_size > read_length:
            
            # add 1 because it's 0-indexed half-open
            start = read.reference_start + 1

            # don't add 1 because it's half-open at the end
            end = read.reference_end
        
            records.append({
                "read_name": read.query_name,
                "read_length": read_length,
                "chrom": chrom,
                "start": start,
                "end": end,
                "orientation": orientation,
                "insert_size": insert_size
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
        subprocess.run(f"cut -f1,2 {ref_genome_file}.fai > {ref_genome_chrom_lengths_file}", shell=True)

    discordant_alns_by_pos_file = os.path.join(dir_name, f"{sample}.DiscordantReads.tsv.gz")
    subprocess.run(f"bedtools genomecov -i {discordant_alns_BED_file} -g {ref_genome_chrom_lengths_file} -d | gzip -c > {discordant_alns_by_pos_file}", shell=True)
    
    


def compute_mean_base_quality_strand_support_of_variant(bam, pos, low_freq_allele, chrom='Chromosome'):
        
    base_qualities_low_freq_allele = []
    SAF = 0
    SAR = 0

    # default min_base_quality is 13, so bad reads get excluded, which negates the purpose of this function...rolls eyes. min_mapping_quality default is 0
    # but use the mapping quality filter here
    for pileupcolumn in bam.pileup(chrom, pos-1, pos, truncate=True, min_base_quality=0, min_mapping_quality=30):

        for pileupread in pileupcolumn.pileups:
            if pileupread.is_del or pileupread.is_refskip:
                continue  # skip deletions and skipped regions

            aln = pileupread.alignment

            if not aln.is_secondary and not aln.is_supplementary and aln.is_paired and aln.is_proper_pair:
    
                base = aln.query_sequence[pileupread.query_position]
                base_qual = aln.query_qualities[pileupread.query_position]
                # map_qual = aln.mapping_quality

                # include only properly paired reads when computing strand bias and base quality
                if base == low_freq_allele:
                    base_qualities_low_freq_allele.append(base_qual)

                    # only include reads in the SAF and SAR count if they meet base and mapping quality minimums
                    # this is to emulate freebayes behavior, where we use BQ ≥ 30 and MQ ≥ 30
                    if base_qual >= 30:# and map_qual >= 30:
                        if aln.is_reverse:
                            SAR += 1
                        else:
                            SAF += 1
                
    # no reads passed
    if len(base_qualities_low_freq_allele) == 0:
        return 0, SAF, SAR
    else:
        return np.mean(base_qualities_low_freq_allele), SAF, SAR
    
    
    
    
def compute_mean_base_quality_of_variant_support(bam, pos, low_freq_allele, chrom='Chromosome'):
        
    base_qualities_low_freq_allele = []

    # default min_base_quality is 13, so bad reads get excluded, which negates the purpose of this function...rolls eyes. min_mapping_quality default is 0
    # but use the mapping quality filter here
    for pileupcolumn in bam.pileup(chrom, pos-1, pos, truncate=True, min_base_quality=0, min_mapping_quality=30):

        for pileupread in pileupcolumn.pileups:
            if pileupread.is_del or pileupread.is_refskip:
                continue  # skip deletions and skipped regions

            aln = pileupread.alignment

            if not aln.is_secondary and not aln.is_supplementary and aln.is_paired and aln.is_proper_pair:
    
                base = aln.query_sequence[pileupread.query_position]
                base_qual = aln.query_qualities[pileupread.query_position]
                # map_qual = aln.mapping_quality

                # include only properly paired reads when computing strand bias and base quality
                if base == low_freq_allele:
                    base_qualities_low_freq_allele.append(base_qual)
                
    # no reads passed
    if len(base_qualities_low_freq_allele) == 0:
        return 0
    else:
        return np.mean(base_qualities_low_freq_allele)




def get_number_of_soft_clipped_reads_and_variant_location_in_reads(df, bam, chrom='Chromosome'):
    '''
    This function computes the number of reads that support a particular SNV that also have soft clipping.
    The soft clipping is not necessarily at the site of the substitution.

    It can be elsewhere in the read, but if a large number of reads supporting an SNV have soft clipping,
    then there's probably an indel nearby, leading to the clipping, and therefore the substitution is not real.
    '''

    df = df.reset_index(drop=True)
    
    for i, row in df.iterrows():

        pos = int(row['POS'])   # 1-based
        ref = row['REF']
        alt = row['ALT']

        num_soft_clipped_reads = 0

        # list to store the position indexes within each read that support the alternative allele
        variant_idx_in_read = []

        for pileupcolumn in bam.pileup(
            chrom,
            pos - 1,
            pos,
            truncate=True,
            min_base_quality=0,
            min_mapping_quality=30
        ):

            for pileupread in pileupcolumn.pileups:

                # Skip deletions / ref skips at this position
                if pileupread.is_del or pileupread.is_refskip:
                    continue

                read = pileupread.alignment

                # exclude reads with supplementary alignments, secondary alignments, or are discordantly paired
                if read.is_supplementary or read.is_secondary:
                    continue
                
                if read.is_paired and read.is_proper_pair:

                    # query_position is guaranteed to be valid here
                    read_supported_allele = read.query_sequence[pileupread.query_position]

                    # don't need base quality because we already have another variable that is the
                    # average base quality of nucleotides that support the variant
                    # read_pos_BQ = read.query_qualities[pileupread.query_position]

                    if read_supported_allele == alt:
                    
                        # get number of bases soft clipped from this read
                        num_soft_clipped_bases = sum(length for (op, length) in (read.cigartuples or []) if op == 4)

                        if num_soft_clipped_bases > 0:
                            num_soft_clipped_reads += 1

                        # normalized to read length
                        read_length = read.query_length
                        variant_idx_length_norm = pileupread.query_position / read_length

                        # Normalize to the halfway point of the read because close to either of the ends is bad.
                        # Beginning or end of the read doesn't matter, just the distance from the end.
                        # This variable ranges from 0–0.5. Closer to 0.5 is better.
                        if variant_idx_length_norm <= 0.5:
                            variant_idx_in_read.append(variant_idx_length_norm)
                        else:
                            variant_idx_in_read.append(1 - variant_idx_length_norm)                        

#         # weird edge cases when you have an indel and an SNV nearby, the alternate allele can be a little wonky
#         # because different reads may have different nucleotides inserted vs. substituted
#         # but they all support the same net change. It's just bookkeeping
#         # so iterate again, this time allowing read_supported_allele to just be not REF
#         if len(variant_idx_in_read) == 0:
            
#             # an example is 3462145 in MFS-273. There's an insertion close to the SNV. The insertion is truly fixed, and the SNV is truly unfixed
#             # depending on where you put the insertion (and the insertions on reads with the SNV are slightly different reads without the SNV), the SNV is different
#             print(f"Weird case at {pos}, {ref}, {alt}")
            
#             # reinitialize to be safe
#             num_soft_clipped_reads = 0
#             variant_idx_in_read = []

#             for pileupcolumn in bam.pileup(
#                 chrom,
#                 pos - 1,
#                 pos,
#                 truncate=True,
#                 min_base_quality=0,
#                 min_mapping_quality=30
#             ):
                
#                 for pileupread in pileupcolumn.pileups:

#                     # Skip deletions / ref skips at this position
#                     if pileupread.is_del or pileupread.is_refskip:
#                         continue

#                     read = pileupread.alignment

#                     # exclude reads with supplementary alignments, secondary alignments, or are discordantly paired
#                     if read.is_supplementary or read.is_secondary:
#                         continue

#                     if read.is_paired and read.is_proper_pair:

#                         # query_position is guaranteed to be valid here
#                         read_supported_allele = read.query_sequence[pileupread.query_position]

#                         if read_supported_allele != ref:
                                                    
#                             # get number of bases soft clipped from this read
#                             num_soft_clipped_bases = sum(length for (op, length) in (read.cigartuples or []) if op == 4)

#                             if num_soft_clipped_bases > 0:
#                                 num_soft_clipped_reads += 1

#                             # normalized to read length
#                             read_length = read.query_length
#                             variant_idx_length_norm = pileupread.query_position / read_length

#                             # Normalize to the halfway point of the read because close to either of the ends is bad.
#                             # Beginning or end of the read doesn't matter, just the distance from the end.
#                             # This variable ranges from 0–0.5. Closer to 0.5 is better.
#                             if variant_idx_length_norm <= 0.5:
#                                 variant_idx_in_read.append(variant_idx_length_norm)
#                             else:
#                                 variant_idx_in_read.append(1 - variant_idx_length_norm)
                            
        # count the number of reads with ANY bases soft-clipped and take the median normalized index
        df.loc[i, ['NumSoftClippedReads', 'VariantSupportMedianIndex']] = [num_soft_clipped_reads, np.median(variant_idx_in_read)]

    df['NumSoftClippedReads'] = df['NumSoftClippedReads'].astype(int)
    return df




# def get_allele_type(record, AF_min=0, AF_max=0.9):
#     '''
#     Returns "alt" or "ref" if the variant is low-quality or ambiguous. Otherwise this function returns "missing"
    
#     Low-quality criteria:
    
#         1. FILTER == Del, LowCov
#         2. FILTER == Amb and 0.25 < AF <= 0.75
#         3. SNP quality < 10

#     Criteria for not confident in a variant or can not be reliably inserted, so leave it as reference:

#         1. IMPRECISE variant (in the INFO field)
#         2. Indels longer than 15 bp where neither the REF nor the ALT are of length 1 (this case is handled in the next function)
        
#     If FILTER contains Amb and the alternative allele fraction > presentThresh, then it is a pure alternative call. 
#     '''

#     ref_allele = str(record.REF)
#     alt_allele = "".join(np.array(record.ALT).astype(str))

#     # this should not happen in pilon because it is not a haplotype variant caller
#     # this would mean that there are 3 alleles present -- reference + 2 alternative alleles
#     # haplotype variant callers will often have reference and alternative haplotypes separated by a comma in the ALT field, so this script will not work for them
#     if ',' in alt_allele:
#         print(fName, record)
#         raise ValueError(f"There are multiple alternative alleles in a single record!")

#     # fill in things that might be missing
#     if "AF" in record.INFO.keys() and "DP" in record.INFO.keys():
# #         AO_lst = record.INFO["AO"]

# #         # multiple alternative alleles. Ignore them in this particular check
# #         if len(AO_lst) > 1:
# #             return "missing"
# #         else:
# #             AF = AO_lst[0] / record.INFO['DP']

#         alt_base_count = np.max(record.INFO['BC'])
#         AF = alt_base_count / record.INFO['DP']
#     else:
#         raise ValueError(f"AF or DP is not in the VCF record for POS = {record.POS}") 

#     # QUAL field considers read depth, base quality, mapping quality. But it is also on the Phred scale
#     if record.QUAL is None:
#         qual = 11
#     else:
#         qual = record.QUAL
        
#     # because IMPRECISE is taken care of above, this should only return missing for cases where REF = N or ALT = N
#     if "N" in ref_allele or "N" in alt_allele:
#         return "missing"
    
#     # check if there are any non alphanumeric characters. This would indicate a heterogeneous alternative allele
#     if not alt_allele.isalnum():
#         return "missing" 
        
#     # base quality, mapping quality, and read depth (measures of certainty about a variant)
#     if 'DP' in record.INFO.keys():
#         if record.INFO['DP'] < 5:
#             return 'missing'

#     if 'MQM' in record.INFO.keys():
#         if len(record.INFO['MQM']) > 1:
#             return "missing"
#         else:
#             if record.INFO['MQM'][0] < 30:
#                 return 'missing'

#     if AF >= AF_min and AF <= AF_max:
#         return "intermediate"
    
#     elif AF < AF_min:
#         return "ref"
    
#     else:
#         return "fixed"


def get_allele_type(record, AF_min=0.05, AF_max=0.95):
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
    
    base_lst = ['A', 'C', 'G', 'T']
    
    # fill in things that might be missing
    if "AF" in record.INFO.keys() and "DP" in record.INFO.keys():
        alt_base_count = np.max(record.INFO['BC'])
        AF = alt_base_count / record.INFO['DP']
        alt_allele = base_lst[np.argmax(record.INFO['BC'])]
    else:
        raise ValueError(f"AF or DP is not in the VCF record for POS = {record.POS}") 

    if AF >= AF_min and AF <= AF_max:
        return "intermediate"
    
    elif AF < AF_min:
        return "ref"
    
    else:
        if ref_allele == alt_allele:
            return "ref"
        else:
            return "fixed"
    
    

def compute_sites_with_high_lowAF_SNP_density(SNPs_CSV_file, SNPs_VCF_file_bgzipped, AF_min=0, AF_max=0.95, SNP_window_size=25):
    
    df_candidate_vars = pd.read_csv(SNPs_CSV_file)
    df_SNPs = pd.DataFrame(columns = ['POS', 'SNP'])
    i = 0
    
    # to save time, only check regions around the sites with variants in df_candidate_vars, instead of the full genome
    min_pos = df_candidate_vars.POS.min()-SNP_window_size
    max_pos = df_candidate_vars.POS.max()+SNP_window_size
    
    vcf_reader = vcf.Reader(filename=SNPs_VCF_file_bgzipped).fetch('Chromosome', start=min_pos, end=max_pos)
        
    for record in vcf_reader:
        snp_type = get_allele_type(record, AF_min=AF_min, AF_max=AF_max)
        df_SNPs.loc[i, :] = [record.POS, snp_type]
        i += 1
        
    print(f"{i} variants")
        
    df_SNPs = df_SNPs.query("SNP == 'intermediate'").drop_duplicates().reset_index(drop=True)
    df_SNPs['SNP'] = 1

    # add the remaining sites to the dataframe
    ref_sites = list(set(np.arange(min_pos, max_pos+1)) - set(df_SNPs.POS))
    
    df_SNPs = pd.concat([df_SNPs,
                         pd.DataFrame({'POS': ref_sites, 'SNP': 0})
                        ]).sort_values("POS").reset_index(drop=True)
    
    print(len(df_SNPs))
    
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
    # assert len(df_variants_split_MNPs.iloc[df_variants_split_MNPs.index.values[df_variants_split_MNPs.duplicated(subset=['POS'], keep=False)]]) == 0

    if 'SampleID' in df_variants_split_MNPs.columns:
        return df_variants_split_MNPs.drop_duplicates(subset=['SampleID', 'POS']).reset_index(drop=True)
    else:
        return df_variants_split_MNPs.drop_duplicates(subset=['POS']).reset_index(drop=True)



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