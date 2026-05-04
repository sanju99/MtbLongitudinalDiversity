import numpy as np
import pysam
from collections import defaultdict
import argparse

parser = argparse.ArgumentParser()

parser.add_argument("-i", dest='in_vcf', type=str, required=True)
parser.add_argument("-o", dest='out_vcf', type=str, required=True)

cmd_line_args = parser.parse_args()
in_vcf = cmd_line_args.in_vcf
out_vcf = cmd_line_args.out_vcf


def merge_multiallelic_snvs(ref, alts, AOs, MQMs, SAFs, SARs):
    """
    Merge multiallelic SNV/MNV alleles that imply the same base changes.

    Parameters
    ----------
    ref : str
        Reference allele
    alts : list[str]
        Alternate alleles
    ao_list : list[int]
        AO values corresponding to each ALT

    Returns
    -------
    dict
        {(pos, ref_base, alt_base): summed_AO}
        where pos is 0-based relative to REF
    """

    # Sanity check: only SNVs/MNVs
    for alt in alts:
        if len(alt) != len(ref):
            return None  # skip indels / complex length changes

    AO_merged = defaultdict(int)
    MQM_merged = defaultdict(list)
    SAF_merged = defaultdict(int)
    SAR_merged = defaultdict(int)

    for alt, AO, MQM, SAF, SAR in zip(alts, AOs, MQMs, SAFs, SARs):
        for i, (r, a) in enumerate(zip(ref, alt)):
            if r != a:
                AO_merged[(i, r, a)] += AO
                MQM_merged[(i, r, a)].append(MQM) # this is the only where we don't sum because it's not a count. The others are read counts. Average MQM
                SAF_merged[(i, r, a)] += SAF
                SAR_merged[(i, r, a)] += SAR

    return AO_merged, MQM_merged, SAF_merged, SAR_merged



def fix_multiple_haplotypes_substitutions_only(in_vcf, out_vcf):

    vcf_in = pysam.VariantFile(in_vcf)
    vcf_out = pysam.VariantFile(out_vcf, "w", header=vcf_in.header)

    for record in vcf_in:

        alt = record.alts

        # only one haplotype, so write it to the new VCF
        if len(alt) == 1:
            vcf_out.write(record)
        else:        
            pos = record.pos
            ref = record.ref
            alt = record.alts
            depth = record.info['DP']

            alts = list(alt)
            cigars = list(record.info['CIGAR'])
            AOs = list(record.info['AO'])

            MQMs = list(record.info['MQM'])
            SAFs = list(record.info['SAF'])
            SARs = list(record.info['SAR'])

            # of the haplotypes, some could be the same length, while others might not be because they include indels
            # keep only the substitutions
            idx_var_to_change = [idx for idx, var in enumerate(alts) if len(var) == len(ref)]

            if len(idx_var_to_change) > 0:

                alts_to_change = [alts[idx] for idx in idx_var_to_change]
                AOs_to_change = [AOs[idx] for idx in idx_var_to_change]
                MQMs_to_change = [MQMs[idx] for idx in idx_var_to_change]
                SAFs_to_change = [SAFs[idx] for idx in idx_var_to_change]
                SARs_to_change = [SARs[idx] for idx in idx_var_to_change]

                # there are 4 dictionaries, one for each of the variables - AO, MQM, SAF, and SAR
                # within each dictionary, there are N entries, one for each of the nucleotides that has a change
                # nucleotides where REF = ALT for all ALT haplotypes are not included. 
                fixed_outputs = merge_multiallelic_snvs(ref, alts_to_change, AOs_to_change, MQMs_to_change, SAFs_to_change, SARs_to_change)

                # merge all 4 values into a single dictionary, where each key is position, ref, alt, and the value is list of length 4. MQM is a sublist within this
                fixed_outputs_pos_merged = defaultdict(list)

                # for MQM (or anything else that's a list, average the values at this stage)
                for variant_key in fixed_outputs[0].keys():
                    fixed_outputs_pos_merged[variant_key] = [np.mean(fixed_outputs[i][variant_key]) if type(fixed_outputs[i][variant_key]) == list else fixed_outputs[i][variant_key] for i in range(len(fixed_outputs))]

                assert len(fixed_outputs_pos_merged) == len(fixed_outputs[0])

                # for each change, the position needs to be updated to make a new VCF entry
                # iterate through the merged dictionary
                for variant_key, variant_new_vals in fixed_outputs_pos_merged.items():

                    # need a new VCF entry for each one
                    # empty record with the same header so that we don't break the VCF format
                    new_record = vcf_in.new_record()

                    # fields that are the same for all of them
                    new_record.contig = record.contig
                    new_record.qual = record.qual
                    new_record.info['DP'] = record.info['DP']

                    # also need cigar string, but since these are all just substitutions, just put the allele length followed by X (X = substitution)
                    new_record.info['CIGAR'] = str(len(variant_key[2])) + 'X'

                    new_record.filter.clear()
                    for f in record.filter:
                        new_record.filter.add(f)

                    new_AO, new_MQM, new_SAF, new_SAR = variant_new_vals

                    # also update these using pos_offset, ref, and alt in variant_key. Even for length 1, most are tuples
                    new_record.pos = pos + variant_key[0] # not a tuple
                    new_record.ref = variant_key[1] # not a tuple
                    new_record.alts = (variant_key[2],)

                    new_record.info['AO'] = (new_AO,)
                    new_record.info['MQM'] = (new_MQM,)
                    new_record.info['SAF'] = (new_SAF,)
                    new_record.info['SAR'] = (new_SAR,)

                    vcf_out.write(new_record)

                # for the variants that we're not going to change write the values as is
                idx_no_change = [idx for idx, var in enumerate(alts) if len(var) != len(ref)]

                for idx in idx_no_change:

                    # need a new VCF entry for each one
                    # empty record with the same header so that we don't break the VCF format
                    new_record = vcf_in.new_record()

                    # fields that are the same for all of them
                    new_record.contig = record.contig
                    new_record.qual = record.qual
                    new_record.info['DP'] = record.info['DP']
                    new_record.pos = pos # same pos
                    new_record.ref = ref # same ref

                    new_record.filter.clear()
                    for f in record.filter:
                        new_record.filter.add(f)

                    # get the following from the original lists with info from all the haplotypes at this site
                    new_record.info['CIGAR'] = cigars[idx]
                    new_record.alts = (alts[idx],)
                    new_record.info['AO'] = (AOs[idx],)
                    new_record.info['MQM'] = (MQMs[idx],)
                    new_record.info['SAF'] = (SAFs[idx],)
                    new_record.info['SAR'] = (SARs[idx],)

                    vcf_out.write(new_record)
                    
            # nothing to be changed if none of the lengths match between REF and ALT, so just write the record out as is
            else:
                vcf_out.write(record)
                
                
fix_multiple_haplotypes_substitutions_only(in_vcf, out_vcf)