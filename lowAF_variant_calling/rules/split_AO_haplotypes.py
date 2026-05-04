import os, glob, pysam, argparse, re
import numpy as np
import pandas as pd


parser = argparse.ArgumentParser()

parser.add_argument("-i", dest='in_vcf', type=str, required=True)
parser.add_argument("-o", dest='out_vcf', type=str, required=True)

cmd_line_args = parser.parse_args()
in_vcf = cmd_line_args.in_vcf
out_vcf = cmd_line_args.out_vcf


def collapse_cigar_tuples(cigar_tuples):
    """
    Combine consecutive cigar operations with the same operator.
    Example:
        [('2', 'X'), ('3', 'M'), ('1', 'X')] → [('3', 'X'), ('3', 'M')]
    """
    
    df_cigar_tuples = pd.DataFrame(cigar_tuples, columns=['NumBases', 'Operator'])
    df_cigar_tuples['NumBases'] = df_cigar_tuples['NumBases'].astype(int)

    df_cigar_tuples_collapsed = pd.DataFrame(df_cigar_tuples.groupby('Operator')['NumBases'].sum()).reset_index()
    
    return [(row['NumBases'], row['Operator']) for i, row in df_cigar_tuples_collapsed.iterrows()]


def match_variant_to_cigar(ref, alt, cigar_string, collapse=True):
    
    # X = substitution, D = deletion, I = insertion
    if len(ref) == len(alt):
        aln_symbol = 'X'
    elif len(ref) > len(alt):
        aln_symbol = 'D'
    else:
        aln_symbol = 'I'
        
    if aln_symbol == 'X':
        variant_length = len(ref)
    else:
        variant_length = abs(len(ref) - len(alt))
        
    # convert CIGAR string to tuples
    cigar_tuples = re.findall(r"(\d+)([MIDNSHP=X])", cigar_string)
    
    # add numbers from the same operator, so 2X3M1X would be [('3', 'X'), ('3', 'M')]
    if collapse:
        cigar_tuples = collapse_cigar_tuples(cigar_tuples)
            
    match = False
    
    for num_bases, operator in cigar_tuples:
        if operator == aln_symbol and int(num_bases) == variant_length:
            return True, cigar_string
    
    return match, None


def fix_AO_split_haplotypes(in_vcf, out_vcf):

    vcf_in = pysam.VariantFile(in_vcf)
    vcf_out = pysam.VariantFile(out_vcf, "w", header=vcf_in.header)

    for record in vcf_in:

        new = record.copy()

        pos = record.pos
        ref = record.ref
        alt = record.alts
        depth = record.info['DP']
        ao = record.info['AO']
        mqm = record.info['MQM']
        saf = record.info['SAF']
        sar = record.info['SAR']

        # vcfwave splits multiallelics, so there should only be one ALT
        assert len(alt) == 1
        alt = alt[0]

        cigar_strings = list(record.info['CIGAR'])
        
        matched_cigar_strings = []
        
        keep_idx = None

        if len(cigar_strings) > 1 and len(ao) > 1:

            found_match_global = False

            for cigar_string in cigar_strings:
                found_match, matched_cigar = match_variant_to_cigar(ref, alt, cigar_string, collapse=True)

                if found_match:
                    matched_cigar_strings.append(matched_cigar)
                    found_match_global = True

            # if no match was found after going through the collapsed versions, then check the uncollapsed cigar tuples.
            if not found_match_global:
                # the collapsed one is more accurate because of the problems of reporting multiple variants at different AFs on the same haplotype.
                # see 163705 and 163710 in MFS-172 for an example. 163705 is at high frequency, 163710 is at unfixed frequency
                for cigar_string in cigar_strings:
                    found_match, matched_cigar = match_variant_to_cigar(ref, alt, cigar_string, collapse=False)

                    if found_match:
                        matched_cigar_strings.append(matched_cigar)

            if len(matched_cigar_strings) > 0:
                keep_idx = [list(cigar_strings).index(matched_cigar) for matched_cigar in np.unique(matched_cigar_strings)]
                # print(pos, ref, alt, ao[keep_idx], depth)
            # else:
            # ignore. Usually low quality variants. Very rarely haplotype variants of different AFs that can not be resolved without writing a whole new variant caller :(
            # raise ValueError(pos, ref, alt, cigar_strings, f"not found in {in_vcf}") 

            # get individual AO. If the index is longer than AO, then that variant is probably not real. So set AO = 0
            # need to also split MQM because we use that for filtering. For completeness, also split mean alt
            if keep_idx is not None:
                new.info['AO'] = int(np.sum(np.array(ao)[keep_idx]))
                new.info['MQM'] = np.sum(np.array(mqm)[keep_idx])
                new.info['SAF'] = int(np.sum(np.array(saf)[keep_idx]))
                new.info['SAR'] = int(np.sum(np.array(sar)[keep_idx]))
            else:
                new.info['AO'] = 0
                new.info['MQM'] = 0
                new.info['SAF'] = 0
                new.info['SAR'] = 0

        vcf_out.write(new)

    vcf_out.close()
    
    
fix_AO_split_haplotypes(in_vcf, out_vcf)