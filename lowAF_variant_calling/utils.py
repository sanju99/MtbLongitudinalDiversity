import os, glob
import pandas as pd
import numpy as np
import vcf


def convert_freebayes_VCF_records_to_CSV(fName):
    
    # SRF = # of reference observations on the forward strand
    # SAF = # of alternate observations on the forward strand
    # SRR = # of reference observations on the reverse strand
    # SAR = # of alternate observations on the reverse strand
    # SRP and SAP are strand balance proabilities for the reference and alternate probabilities. They are Phred-scaled upper-bounds estimate of the probability of observing the deviation between the forward and reverse strands
    # The higher this quantity the better the site as it diminishes the chance of the site having significant strand bias
    df_variants = pd.DataFrame(columns = ['POS', 'REF', 'ALT', 'QUAL', 'FILTER', 'DP', 'DPB', 'RO', 'AO', 'AF', 'AF_freebayes', 'MQM', 'MQMR', 'SRF', 'SRR', 'SAF', 'SAR', 'SRP', 'SAP', 'RPP', 'RPPR', 'RPL', 'RPR'])#, 'ANN'])
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
                
        # # not sure why this is, but sometimes a random variant doesn't get annotated?? Oh well, if they all get excluded by the low AF filters, ignore them
        # if 'ANN' not in record.INFO.keys():
        #     ANN = np.nan
        # else:
        #     ANN = record.INFO['ANN']

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
                                 # ANN,
                                ]
        i += 1
    
    # # split the annotation column
    # df_variants.loc[~pd.isnull(df_variants['ANN']), 'GENE'] = df_variants['ANN'].str[0].str.split('|').str[3]
    # df_variants.loc[~pd.isnull(df_variants['ANN']), 'EFFECT'] = df_variants['ANN'].str[0].str.split('|').str[1]
    # df_variants.loc[~pd.isnull(df_variants['ANN']), 'HGVS_C'] = df_variants['ANN'].str[0].str.split('|').str[9]
    # df_variants.loc[~pd.isnull(df_variants['ANN']), 'HGVS_P'] = df_variants['ANN'].str[0].str.split('|').str[10]
    
    return df_variants





def apply_lowAF_QCfilters(df_variants, AF_thresh=0.05, AF_max=0.25, MQ_thresh=40, num_support_each_direction=2):
    
    # Phred-scaled upper-bounds estimate of the probability of observing the deviation between SRF and SRR given E(SRF/SRR) ~ 0.5, derived using Hoeffding's inequality">
    # The higher this quantity the better the site as it diminishes the chance of the site having significant strand bias.
    # df_variants['SRP_prob'] = 10**(-df_variants['SRP']/10)
    df_variants['SAP_prob'] = 10**(-df_variants['SAP']/10)

    # when SRF = SRR = 0, there are no reads supporting the reference. If quality is 0, then the error probability is 1, which isn't true. They're actually NA
    df_variants.loc[(df_variants['SRF']==0) | (df_variants['SRR']==0), 'SRP_prob'] = np.nan

    # require that SAP_prob > 0.05, means that the probability of observing the deviation between SAF and SAR by chance is greater than 5%
    # <= 5% would indicate that there is significant strand bias for the alternative allele at the alpha = 0.05 level
    df_variants = df_variants.query("AF > @AF_thresh & AF <= @AF_max & MQM >= @MQ_thresh & SAP_prob > 0.05")

    df_lowAF_variants = pd.concat([df_variants.query("(REF.str.len() - ALT.str.len() > 10)"),
                                   df_variants.query("~(REF.str.len() - ALT.str.len() > 10) & SAF >= @num_support_each_direction & SAR >= @num_support_each_direction")
                                  ])    
    
    return df_lowAF_variants.reset_index(drop=True)
