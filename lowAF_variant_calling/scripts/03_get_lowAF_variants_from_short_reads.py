import numpy as np
import pandas as pd
import glob, os, argparse, time, vcf, warnings
warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser()

# "/n/data1/hms/dbmi/farhat/rollingDB/TRUST/clinical_data/20240826_raw_data.csv"
parser.add_argument("-d", "--H37Rv_ref_dir", dest="H37Rv_ref_dir", type=str, default="/n/data1/hms/dbmi/farhat/Sanjana/TRUST_lowAF", help='Directory where alignments to H37Rv are stored')
parser.add_argument("-F2", dest='F2_score_max', type=float, required=False, default=0.03, help='F2 score above which a sample is considered to have mixed lineages')

cmd_line_args = parser.parse_args()
H37Rv_ref_dir = cmd_line_args.H37Rv_ref_dir
F2_score_max = cmd_line_args.F2_score_max


def convert_freebayes_VCF_records_to_CSV(fName):
    
    # SRF = # of reference observations on the forward strand
    # SAF = # of alternate observations on the forward strand
    # SRR = # of reference observations on the reverse strand
    # SAR = # of alternate observations on the reverse strand
    # SRP and SAP are strand balance proabilities for the reference and alternate probabilities. They are Phred-scaled upper-bounds estimate of the probability of observing the deviation between the forward and reverse strands
    # The higher this quantity the better the site as it diminishes the chance of the site having significant strand bias
    df_variants = pd.DataFrame(columns = ['POS', 'REF', 'ALT', 'QUAL', 'FILTER', 'DP', 'DPB', 'RO', 'AO', 'AF', 'AF_freebayes', 'MQM', 'MQMR', 'SRF', 'SRR', 'SAF', 'SAR', 'SRP', 'SAP', 'RPP', 'RPPR', 'RPL', 'RPR', 'ANN'])
    i = 0

    vcf_reader = vcf.Reader(filename=fName)
    
    for record in vcf_reader:

        vals_dict = {}

        # there should only be 1 value for all of these because we split multiallelic sites to different lines. So even if it's a list, the list should have length 1
        for field in ['DPB', 'AF', 'AO', 'MQM', 'MQMR', 'SAF', 'SAR', 'SAP', 'RPP', 'RPPR', 'RPL', 'RPR']:

            if type(record.INFO[field]) == int or type(record.INFO[field]) == float:
                vals_dict[field] = record.INFO[field]
            
            elif type(record.INFO[field]) == list:
                assert len(record.INFO[field]) == 1
                vals_dict[field] = float(record.INFO[field][0])
                
        # not sure why this is, but sometimes a random variant doesn't get annotated?? Oh well, if they all get excluded by the low AF filters, ignore them
        if 'ANN' not in record.INFO.keys():
            ANN = np.nan
        else:
            ANN = record.INFO['ANN']
        
        df_variants.loc[i, :] = [record.POS, 
                                 record.REF, 
                                 ','.join(np.array(record.ALT).astype(str)), 
                                 record.QUAL, 
                                 record.FILTER, 
                                 record.INFO['DP'],
                                 record.INFO['DPB'],
                                 record.INFO['RO'],
                                 vals_dict['AO'],
                                 vals_dict['AO'] / record.INFO['DP'],
                                 vals_dict['AF'],
                                 vals_dict['MQM'], 
                                 vals_dict['MQMR'], 
                                 record.INFO['SRF'], 
                                 record.INFO['SRR'], 
                                 vals_dict['SAF'],
                                 vals_dict['SAR'], 
                                 record.INFO['SRP'], 
                                 vals_dict['SAP'], 
                                 vals_dict['RPP'],
                                 vals_dict['RPPR'],
                                 vals_dict['RPL'],
                                 vals_dict['RPR'],
                                 ANN,
                                ]
        i += 1
    
    # split the annotation column
    df_variants.loc[~pd.isnull(df_variants['ANN']), 'GENE'] = df_variants['ANN'].str[0].str.split('|').str[3]
    df_variants.loc[~pd.isnull(df_variants['ANN']), 'EFFECT'] = df_variants['ANN'].str[0].str.split('|').str[1]
    df_variants.loc[~pd.isnull(df_variants['ANN']), 'HGVS_C'] = df_variants['ANN'].str[0].str.split('|').str[9]
    df_variants.loc[~pd.isnull(df_variants['ANN']), 'HGVS_P'] = df_variants['ANN'].str[0].str.split('|').str[10]
    
    # the one weird variant that didn't get annotated in MFS-618. All other variants are annotated
    # it's right between two genes (which are separated by only one nucleotide, 3977061). Not sure why it didn't get annotated as intergenic.
    if 'MFS-618' in fName:
        df_variants['ANN'] = df_variants['ANN'].fillna('')
        df_variants.loc[pd.isnull(df_variants['ANN']), 'GENE'] = 'Rv3537-Rv3538'
        df_variants.loc[pd.isnull(df_variants['ANN']), 'EFFECT'] = 'intergenic_region'
        df_variants.loc[pd.isnull(df_variants['ANN']), 'HGVS_C'] = 'n.3977061T>C'
        
    return df_variants




# set AF_max = 1.2 in case there are AFs > 1 (which sometimes weirdly happen). Here, we want to keep all variants so that we can see the change in them
def apply_lowAF_QCfilters(df_variants, AF_thresh=0.05, AF_max=1.2, MQ_thresh=40, num_support_each_direction=2):
    
    # Phred-scaled upper-bounds estimate of the probability of observing the deviation between SRF and SRR given E(SRF/SRR) ~ 0.5, derived using Hoeffding's inequality">
    # The higher this quantity the better the site as it diminishes the chance of the site having significant strand bias.
    # df_variants['SRP_prob'] = 10**(-df_variants['SRP']/10)
    # df_variants['SAP_prob'] = 10**(-df_variants['SAP']/10)

    df_variants = df_variants.query("AF > @AF_thresh & AF <= @AF_max & MQM >= @MQ_thresh")

    # for long indels, you may not have forward and reverse strands covering both, but those are often real, so don't exclude by those
    df_lowAF_variants = pd.concat([df_variants.query("(REF.str.len() - ALT.str.len() > 10)"),
                                   df_variants.query("~(REF.str.len() - ALT.str.len() > 10) & SAF >= @num_support_each_direction & SAR >= @num_support_each_direction")
                                  ])    
    
    return df_lowAF_variants.reset_index(drop=True)




samples_lst = os.listdir(H37Rv_ref_dir)
# samples_lst = ['MFS-95', 'MFS-96']
# ['MFS-1', 'MFS-2', 'MFS-9', 'MFS-10']
print(f"Getting low AF variants for {len(samples_lst)} samples")

keep_cols = ['POS', 'REF', 'ALT', 'QUAL', 'FILTER', 'DP', 'RO', 'AO', 'AF', 'MQM', 'MQMR', 'SRF', 'SRR', 'SAF', 'SAR', 'SRP', 'SAP', 'RPP', 'RPPR', 'RPL', 'RPR', 'ANN', 'GENE', 'EFFECT', 'HGVS_C', 'HGVS_P', 'LowCov']

for sample in samples_lst:
    
    # only SNPs for right now
    out_fName = f"{H37Rv_ref_dir}/{sample}/lowAF_SNPs.csv"
    
    # if not os.path.isfile(out_fName):
        
    # get the dataframe of variants from the freebayes VCF of Illumina reads aligned to H37Rv
    df_lowAF_variants_H37Rv_asm = convert_freebayes_VCF_records_to_CSV(f"{H37Rv_ref_dir}/{sample}/freebayes/{sample}.excludeLowConf.vcf")

    # then after the switch above, apply QC filters to keep only the high quality ones and remove those with fixed allele frequencies
    df_lowAF_variants_H37Rv_asm = apply_lowAF_QCfilters(df_lowAF_variants_H37Rv_asm)
    
    # keep only SNPs for now
    df_lowAF_variants_H37Rv_asm = df_lowAF_variants_H37Rv_asm.query("REF.str.len() == ALT.str.len()")

    # don't apply the coverage_plateau_sites and high_SNP_density_sites filters for samples with mixed lineages. Causes too many false negatives
    with open(f"{H37Rv_ref_dir}/{sample}/lineage/F2_Coll2014.txt", "r") as file:
        F2_score = float(file.readlines()[0].strip())
        
    # exclude low frequency variants occurring at sites where the coverage drops to < 1/10 of the median coverage
    df_depth = pd.read_csv(f"{H37Rv_ref_dir}/{sample}/bam/{sample}.depth.tsv.gz", compression='gzip', sep='\t', header=None, names=['CHROM', 'POS', 'COV'])
    
    # coverage_min = df_depth['COV'].median() / 10    
    # compute rolling average of coverage. Smaller than 100 is too small, not enough smoothing
    window_size = 100
    df_depth['COV_LEFT_ROLLING_AVG'] = df_depth['COV'].rolling(window=window_size, min_periods=1, closed='right').mean()
    
    # for this one, compute the rolling average as you would from the left, but reverse the values beforehand. This is the easiest way. Then reverse them again
    df_depth['COV_RIGHT_ROLLING_AVG'] = df_depth['COV'][::-1].rolling(window=window_size, min_periods=1, closed='right').mean()[::-1]

    # low_cov_sites = df_depth.query("COV < @coverage_min").POS.unique()
    
    # exclude sites where the coverage drops to less than 1/2 the rolling average. Means we're at a valley or something
    left_low_cov_sites = df_depth.loc[df_depth['COV'] < df_depth['COV_LEFT_ROLLING_AVG'] / 2].POS.values
    right_low_cov_sites = df_depth.loc[df_depth['COV'] < df_depth['COV_RIGHT_ROLLING_AVG'] / 2].POS.values
    low_cov_sites = np.concatenate([right_low_cov_sites, left_low_cov_sites])
        
    # exclude the first 100 bp and last 100 bp of the genome because read splitting due to aligning to a linearized version of a circular genome
    # causes artificially low coverage at the beginning and end, which can cause false variants. Sometimes there are true variants, but they won't be
    # reliably detected in all samples of the same pid, so it can wrongly seem like a variant was gained or lost
    end_sites = list(df_depth.iloc[:100].POS.values) + list(df_depth.iloc[-100:].POS.values)
    
    # idk about high F2 right now. May have to use pilon VCFs for those because freebayes mistakenly puts multiple SNPs onto the same haplotype when they shouldn't be
    # because there are so many more SNPs in mixed-lineage samples and at different frequencies (could be in one or all lineages), ideally each variant should be
    # its own line, not combined into haplotypes
    if F2_score <= F2_score_max:

        coverage_plateau_sites = np.load(f"{H37Rv_ref_dir}/{sample}/freebayes/coverage_plateau_sites.npy")
        high_density_SNP_sites = np.load(f"{H37Rv_ref_dir}/{sample}/freebayes/high_density_SNP_sites.npy")

        df_lowAF_variants_H37Rv_asm['Cov_Plateau'] = df_lowAF_variants_H37Rv_asm['POS'].isin(coverage_plateau_sites).astype(int)
        df_lowAF_variants_H37Rv_asm['High_SNP_Density'] = df_lowAF_variants_H37Rv_asm['POS'].isin(high_density_SNP_sites).astype(int)

        # keep track of low coverage sites. If it occurs in one sample, remove it from the other as well? Seeing lots of issues where the coverage drops
        # substantially, but not enough to trigger the 1/2 rolling average filter, and the variant is retained in 1 sample but not the other
        df_lowAF_variants_H37Rv_asm['LowCov'] = df_lowAF_variants_H37Rv_asm['POS'].isin(low_cov_sites).astype(int)
        
        # only exclude low AF variants according to the above. Otherwise, you risk missing fixed variants
        # don't exclude the LowCov sites. Just keep track of them for now, then remove them when you integrate two samples from the same pid together
        df_lowAF_variants_H37Rv_asm = df_lowAF_variants_H37Rv_asm.query("(AF > 0.75 | (Cov_Plateau != 1 & High_SNP_Density != 1)) & POS not in @end_sites")

    else:
        df_lowAF_variants_H37Rv_asm['LowCov'] = np.nan
        
    df_lowAF_variants_H37Rv_asm[keep_cols].to_csv(out_fName, index=False)
    print(f"Finished {sample}")