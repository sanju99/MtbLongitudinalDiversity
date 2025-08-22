import numpy as np
import pandas as pd
import glob, os, vcf, warnings, shutil, subprocess, re, argparse
from Bio import Seq, SeqIO
warnings.filterwarnings('ignore')

h37Rv_path = "/n/data1/hms/dbmi/farhat/Sanjana/H37Rv"
h37Rv_seq = SeqIO.read(os.path.join(h37Rv_path, "GCF_000195955.2_ASM19595v2_genomic.gbff"), "genbank")
h37Rv_coords = pd.read_csv(os.path.join(h37Rv_path, "h37Rv_coords_to_gene.csv"))
h37Rv_coords_dict = dict(zip(h37Rv_coords["pos"].values, h37Rv_coords["region"].values))

parser = argparse.ArgumentParser()

# Add a required string argument for the paths file
parser.add_argument('--cov-window-size', type=int, dest='COV_WINDOW_SIZE', default=50, help='Window size for computing the rolling average of coverage')
parser.add_argument('--SNP-window-size', type=int, dest='SNP_WINDOW_SIZE', default=25, help='Window size for computing the rolling average of low-AF SNPs')
parser.add_argument('--AF-thresh', type=float, dest='AF_THRESH', default=0.05, help='Alternative allele frequency threshold (exclusive) to consider variants present')
parser.add_argument('--SNP-density-thresh', type=float, dest='SNP_DENSITY_PROP_THRESH', default=0.05, help='Threshold for considering a region as having a high density of SNPs. 0.05 means 5% of sites in the window are a SNP relative to reference')
parser.add_argument("-d", dest='sample_dir', type=str, required=True, help='Directory with output files')

cmd_line_args = parser.parse_args()

# required arguments
COV_WINDOW_SIZE = cmd_line_args.COV_WINDOW_SIZE
SNP_WINDOW_SIZE = cmd_line_args.SNP_WINDOW_SIZE
AF_THRESH = cmd_line_args.AF_THRESH
SNP_DENSITY_PROP_THRESH = cmd_line_args.SNP_DENSITY_PROP_THRESH
sample_dir = cmd_line_args.sample_dir

def compute_coverage_derivative(depth_file, window_size=50):
    '''
    This function computes the derivative of coverage at each site in the H37Rv reference genome, normalized to the moving average of coverage.
    '''
    
    df_depth = pd.read_csv(depth_file, sep='\t', header=None, compression='gzip')
    df_depth.columns = ['CHROM', 'POS', 'COV']
    del df_depth['CHROM']

    # get the change in coverage at each site
    deriv_cov = np.diff(df_depth['COV'], n=1)
    df_depth['DERIV_COV'] = [np.nan] + list(deriv_cov)

    # then normalize it to rolling average at each site
    # this computes the sliding window average. closed='right' means to exclude the last number, which is like how [:50] means to exclude the value at index = 50 (the 51st value)
    df_depth['COV_ROLLING_AVG'] = df_depth['COV'].rolling(window=window_size, min_periods=1, closed='right').mean()

    df_depth['NORM_DERIV_COV_COV_ROLLING_AVG'] = df_depth['DERIV_COV'] / df_depth['COV_ROLLING_AVG']
    
    # take sites with the top 1% of derivatives, which are the most variable, as edges
    # take sites with the bottom 1% of derivatives, which are the most constant, as plateaus
    df_depth['REGION'] = df_depth['POS'].map(h37Rv_coords_dict)

    return df_depth




def get_candidate_coverage_plateaus(df_depth, deriv_threshold_percent=0.1, plateau_length=100):

    deriv_col = 'NORM_DERIV_COV_COV_ROLLING_AVG'
    
    # deriv_threshold = df_depth[deriv_col].mean() + 3 * df_depth[deriv_col].std()
    # take the top 1% of sites
    deriv_threshold = np.percentile(np.abs(df_depth[deriv_col].dropna()), 100-deriv_threshold_percent)
    
    if 'EDGE' in df_depth.columns:
        del df_depth['EDGE']
    
    df_depth['EDGE'] = np.nan
    df_depth.loc[df_depth[deriv_col] > deriv_threshold, 'EDGE'] = 'JUMP'
    df_depth.loc[df_depth[deriv_col] < -deriv_threshold, 'EDGE'] = 'DROP'
    
    # we want to get coverage plateaus after jumps only. The read depth filters will probably exclude low plateaus.
    print(df_depth.EDGE.value_counts())
    
    # keep only high derivative sites that are also in the regions of interest
    df_high_deriv = df_depth.dropna(subset='EDGE').reset_index(drop=True)
    
    # we're interested in finding coverage jumps followed by drops within 200 base pairs, let's say
    for i, row in df_high_deriv.iterrows():
    
        if i < len(df_high_deriv) - 1:
    
            # if a jump is not followed by a drop within N base pairs, then exclude it
            if row['EDGE'] == 'JUMP':
                
                # first, if the next DROP happens to be the next nucleotide over, skip that because it's probably just noise
                if df_high_deriv['EDGE'].values[i+1] == 'DROP':
                    
                    if df_high_deriv['POS'].values[i+1] - row['POS'] == 1:
    
                        # go to the next one, and if it occurs within N base pairs, include this pair
                        if df_high_deriv['EDGE'].values[i+2] == 'DROP' and df_high_deriv['POS'].values[i+2] - row['POS'] <= plateau_length:
    
                            # include both
                            df_high_deriv.loc[i, 'INCLUDE'] = 1
                            df_high_deriv.loc[i + 2, 'INCLUDE'] = 1
    
                    # go with i + 1, not i + 2
                    else:
                        # go to the next one, and if it occurs within N base pairs, include this pair
                        if df_high_deriv['POS'].values[i+1] - row['POS'] <= plateau_length:
    
                            # include both
                            df_high_deriv.loc[i, 'INCLUDE'] = 1
                            df_high_deriv.loc[i + 1, 'INCLUDE'] = 1
                else:
                    # exclude
                    df_high_deriv.loc[i, 'INCLUDE'] = 0

    df_candidate_regions = pd.DataFrame({'START': df_high_deriv.query("INCLUDE==1").query("EDGE=='JUMP'").POS.values,
                                     # subtract 1 from the end because currently it is the position after a candidate plateau, when the coverage drops
                                     'END': df_high_deriv.query("INCLUDE==1").query("EDGE=='DROP'").POS.values - 1
                                    }).query("START != END")

    for i, row in df_candidate_regions.iterrows():
    
        start = row['START']
        end = row['END']
    
        # get the maximum derivative in the region, excluding the first site because it's derivative will be relative to the previous site, which is very different
        cov_intervening = df_depth.query("POS > @start & POS <= @end")['COV'].mean()
        max_deriv = df_depth.query("POS > @start & POS <= @end")[deriv_col].abs().max()
        df_candidate_regions.loc[i, ['AVG_COV_INTERVENING', 'MAX_DERIV_INTERVENING']] = [cov_intervening, max_deriv]
    
        # then also get the rolling coverage average just outside the plateau and the average coverage within the plateau
        cov_rolling_avg_before = df_depth.query("POS == @start - 1").COV_ROLLING_AVG.values[0]
    
        # the rolling average is for the i-window_size:i positions, where i is the position of interest
        # so to get the first rolling average that DOESN'T include the plateau, need to get it window_size bases after
        cov_rolling_avg_after = df_depth.query("POS == @end + @COV_WINDOW_SIZE").COV_ROLLING_AVG.values[0]
    
        df_candidate_regions.loc[i, 'COV_ROLLING_AVG_BEFORE'] = cov_rolling_avg_before
        df_candidate_regions.loc[i, 'COV_ROLLING_AVG_AFTER'] = cov_rolling_avg_after

    # require that the average coverage in the plateau is greater than the nearest rolling averages before and after that do not include the plateau
    # also require that maximum rolling average derivative within the plateau is less than the threshold used for identifying the edges of plateaus
    df_final_regions = df_candidate_regions.query("(AVG_COV_INTERVENING > COV_ROLLING_AVG_BEFORE) & (AVG_COV_INTERVENING > COV_ROLLING_AVG_AFTER) & (MAX_DERIV_INTERVENING > -@deriv_threshold) & (MAX_DERIV_INTERVENING < @deriv_threshold)")
    print(f"Keeping {len(df_final_regions)} / {len(df_candidate_regions)} candidate plateaus")

    extra_bp_exclude = 50
    
    coverage_plateau_sites = []
    for i, row in df_final_regions.iterrows():
        # also include N base pairs on each end. Add 1 to the end because row[1] is inclusive currently, but np.arange will treat it as exclusive
        coverage_plateau_sites += list(np.arange(row[0] - extra_bp_exclude, row[1] + 1 + extra_bp_exclude))
        
    return df_depth, coverage_plateau_sites





def get_allele_type(record, AF_present_thresh=0.05):
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
        
    # don't include IMPRECISE variants because they are difficult to reliably insert and often aren't reliable calls anyway
    # unreliability can be due to ambiguous alignments, complex genomic regions, low sequencing coverage, assembly gaps, or segmental duplications
    # basically these are breakpoints that the variant caller is not confident in. If we put Ns, often we get huge runs of Ns, which causes too much noise for the model.
    # pilon was not able to resolve the variants (usually due to large deletions), so leave as reference because we don't know what the variant is with high confidence
    if "IMPRECISE" in record.INFO.keys():
        return "ref"
        
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

    # at this point, we have already checked all of the low-quality criteria. If a variant has made it this far without returning anything, then it is high quality
    # ≤ 25%, always absent
    if AF <= 0.05:
        return "ref"
    # present depends on the threshold passed in
    elif AF > AF_present_thresh:
        return "alt"

    # if nothing has been returned, then the variant is high quality (there are no REF = ALT records in the input VCF files, so return the alternative variant)
    # the reference variant only gets returned above if FILTER == Amb and AF <= 0.25
    return "alt"




def compute_sites_with_high_SNP_density(sample, AF_present_thresh=0.05, SNP_window_size=25, snp_density_proportion_thresh=0.05):

    df_SNPs = pd.DataFrame(columns = ['POS', 'SNP'])
    i = 0
    
    SNPs_VCF = f'{sample_dir}/{sample}/freebayes/{sample}.SNPs.vcf'
    vcf_reader = vcf.Reader(filename=SNPs_VCF)
        
    for record in vcf_reader:
        snp_type = get_allele_type(record, AF_present_thresh=AF_present_thresh)
        df_SNPs.loc[i, :] = [record.POS, snp_type]
        i += 1
    
    df_SNPs = df_SNPs.query("SNP == 'alt'").drop_duplicates().reset_index(drop=True)
    print(f"{len(df_SNPs)} SNPs with AF > {AF_present_thresh} in {sample}")

    # add the remaining sites to the dataframe
    ref_sites = list(set(np.arange(1, len(h37Rv_seq.seq)+1)) - set(df_SNPs.POS))
    
    df_SNPs = pd.concat([df_SNPs,
                         pd.DataFrame({'POS': ref_sites, 'SNP': 'ref'})
                        ]).sort_values("POS").reset_index(drop=True)
    
    df_SNPs['SNP'] = df_SNPs['SNP'].map({'ref': 0, 'alt': 1})
    df_SNPs['REGION'] = df_SNPs['POS'].map(h37Rv_coords_dict)
    
    assert sum(pd.isnull(df_SNPs['SNP'])) == 0
    assert len(df_SNPs) == len(h37Rv_seq.seq)

    df_SNPs['SNP_density_rolling_avg'] = df_SNPs['SNP'].rolling(window=SNP_window_size, min_periods=1, closed='right').mean()

    # get sites where the rolling average is at least 10%
    high_density_SNP_regions = df_SNPs.query("SNP_density_rolling_avg > @snp_density_proportion_thresh")
    
    high_density_SNP_sites = list(high_density_SNP_regions['POS'].values)

    extra_bp_exclude = 50
    
    # add the sites that are also part of the high rolling average, so subtract window_size from each position
    for pos in high_density_SNP_regions.POS.values:
        # add 1 to make it 1-indexed
        high_density_SNP_sites += list(np.arange(pos - SNP_window_size - extra_bp_exclude, pos + extra_bp_exclude) + 1)
    
    high_density_SNP_sites = np.unique(high_density_SNP_sites)
    
    # because the rolling average takes the sites from i-window_size:i, exclude the window_size sites that are upstream of each site with a high rolling average as well
    # print(f"{len(high_density_SNP_sites)} high-density SNP sites across {df_SNPs.query('POS in @high_density_SNP_sites').REGION.nunique()} genes/regions: {df_SNPs.query('POS in @high_density_SNP_sites').REGION.unique()}")
 
    print(f"{len(high_density_SNP_sites)} high-density SNP sites across {df_SNPs.query('POS in @high_density_SNP_sites').REGION.nunique()} genes/regions")

    return high_density_SNP_sites



# use the output from script 00_get_pids_for_analysis.py, which is pids_WGS_data.csv
samples = os.listdir(sample_dir)
print(f"{len(samples)} samples to process")

for i, SAMPLE in enumerate(samples):
    
    if os.path.isfile(f"{sample_dir}/{SAMPLE}/freebayes/coverage_plateau_sites.npy") and os.path.isfile(f"{sample_dir}/{SAMPLE}/freebayes/high_density_SNP_sites.npy"):
        print(f"Already finished {SAMPLE}")
    else:
        if not os.path.isfile(f"{sample_dir}/{SAMPLE}/freebayes/coverage_plateau_sites.npy"):
            
            df_depth = compute_coverage_derivative(f"{sample_dir}/{SAMPLE}/bam/{SAMPLE}.depth.tsv.gz", 
                                                   COV_WINDOW_SIZE
                                                  )
            
            _, coverage_plateau_sites = get_candidate_coverage_plateaus(df_depth)
            
            np.save(f"{sample_dir}/{SAMPLE}/freebayes/coverage_plateau_sites.npy", coverage_plateau_sites)
            
        
        if not os.path.isfile(f"{sample_dir}/{SAMPLE}/freebayes/high_density_SNP_sites.npy"):
                
            high_density_SNP_sites = compute_sites_with_high_SNP_density(SAMPLE, 
                                                                         AF_present_thresh=AF_THRESH, 
                                                                         SNP_window_size=SNP_WINDOW_SIZE, 
                                                                         snp_density_proportion_thresh=SNP_DENSITY_PROP_THRESH
                                                                        )
            
            np.save(f"{sample_dir}/{SAMPLE}/freebayes/high_density_SNP_sites.npy", high_density_SNP_sites)
    
        print(f"Finished {SAMPLE}")