import numpy as np
import pandas as pd
import glob, os, vcf, warnings, shutil, subprocess, re, argparse

parser = argparse.ArgumentParser()

# use edirect_env environment
parser.add_argument("-d", dest='sample_dir', type=str, required=True, help='Directory with output files')
# parser.add_argument("-o", "--output", dest='output_file', type=str, required=True, help='Output CSV file for the returned metadata')

cmd_line_args = parser.parse_args()

sample_dir = cmd_line_args.sample_dir


def get_all_variant_info(fName):
    
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
                # vals_dict[field] = ','.join(np.array(record.INFO[field]).astype(str))
                
        # not sure why this is, but sometimes a random variant doesn't get annotated?? Oh well, if they all get excluded by the low AF filters, ignore them
        if 'ANN' not in record.INFO.keys():
            ANN = np.nan
        else:
            ANN = record.INFO['ANN']

        # AF_lst = np.array(AF_lst).astype(str)
        # AF_lst_AC = np.array(AF_lst_AC).astype(str)
        
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
        # print(df_variants.loc[pd.isnull(df_variants['ANN'])])
        df_variants.loc[pd.isnull(df_variants['ANN']), 'GENE'] = 'Rv3537-Rv3538'
        df_variants.loc[pd.isnull(df_variants['ANN']), 'EFFECT'] = 'intergenic_region'
        df_variants.loc[pd.isnull(df_variants['ANN']), 'HGVS_C'] = 'n.3977061T>C'
    
    return df_variants


samples = os.listdir(sample_dir)
print(f"{len(samples)} samples to process")

for i, sample in enumerate(samples):

    # do both the ROI and the full list of variants

    # these contain the snpEff annotations too
    full_in_file = f"{sample_dir}/{sample}/freebayes/{sample}.excludeLowConf.vcf"
    full_out_file = f"{sample_dir}/{sample}/freebayes/{sample}.csv"

    ROI_in_file = f"{sample_dir}/{sample}/freebayes/{sample}.excludeLowConf.regionsOfInterest.vcf"
    ROI_out_file = f"{sample_dir}/{sample}/freebayes/{sample}.regionsOfInterest.csv"

    if not os.path.isfile(full_out_file) or not os.path.isfile(ROI_out_file):
    
        df_variants_full = get_all_variant_info(full_in_file)
        df_variants_ROI = get_all_variant_info(ROI_in_file)

        # delete columns that are NA everywhere
        for col in df_variants_full.columns:
            if sum(~pd.isnull(df_variants_full[col])) == 0:
                del df_variants_full[col]

        # delete columns that are NA everywhere
        for col in df_variants_ROI.columns:
            if sum(~pd.isnull(df_variants_ROI[col])) == 0:
                del df_variants_ROI[col]

        df_variants_full.to_csv(full_out_file, index=False)
        df_variants_ROI.to_csv(ROI_out_file, index=False)
        print(f"Finished {sample}")
        
    else:
        print(f"Already finished {sample}")