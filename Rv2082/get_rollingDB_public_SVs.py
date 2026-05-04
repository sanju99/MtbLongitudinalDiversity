import os, glob, vcf
import numpy as np
import pandas as pd

genomic_data_dir = "/n/data1/hms/dbmi/farhat/rollingDB/genomic_data"


# delly_fNames = glob.glob(f"{genomic_data_dir}/*/delly/*.vcf")
delly_fNames = pd.read_csv("delly_fNames.txt", sep='\t', header=None)[0].values
print(len(delly_fNames))

df_rollingDB_lineages = pd.read_csv("rollingDB_lineages.csv")

def read_in_delly_SVs(fName, START, END):
    
    df_SV = []

    vcf_reader = vcf.Reader(filename=fName)

    for record in vcf_reader:
        
        pos = record.POS
        
        if pos >= START and pos <= END:
            
            ref = record.REF
            alt = ''.join([str(a) for a in record.ALT]).strip('<').strip('>')      # ALT is a list of objects
            qual = record.QUAL
            flt = record.FILTER if record.FILTER else "PASS"
            
            info = {k: record.INFO[k] for k in record.INFO}

            if 'IMPRECISE' in info.keys():
                imprecise_bool = True
            else:
                imprecise_bool = False
        
            df_SV.append(pd.DataFrame({'POS': pos, 'REF': ref, 'ALT': alt, 'QUAL': qual, 'FILTER': flt, 'IMPRECISE': imprecise_bool, 
                                       'SVTYPE': info['SVTYPE'], 'END': info['END'], 'MQ': info['MAPQ'], 'PE': info['PE'], 'DV': record.samples[0]['DV'],
                                       'CIPOS_LB': info['CIPOS'][0], 'CIPOS_UB': info['CIPOS'][1], 'CIEND_LB': info['CIEND'][0], 'CIEND_UB': info['CIEND'][1]
                                      }, index=[0])
                        )
    
    if len(df_SV) > 0:
        df_SV = pd.concat(df_SV)
        df_SV['ROLLINGDB_ID'] = os.path.basename(fName).split('.vcf')[0]
        df_SV['SVLEN'] = df_SV['END'] - df_SV['POS']

        return df_SV.set_index('ROLLINGDB_ID').reset_index()
    else:
        return pd.DataFrame()
    
    
# this is the full Rv2081c-Rv2082 region
start = 2338065
end = 2340874

df_SV_SR_rollingDB = []

for i, fName in enumerate(delly_fNames):
    
    # only public samples for now
    if os.path.basename(fName)[:3] == 'SAM':
    
        df_test = read_in_delly_SVs(fName, start, end)

        if len(df_test) > 0:
            df_SV_SR_rollingDB.append(df_test)

    if i % 1000 == 0:
        print(i)
        
        if len(df_SV_SR_rollingDB) > 0:
            pd.concat(df_SV_SR_rollingDB).to_csv(f"rollingDB_Rv2082_SV.csv", index=False)
        
df_SV_SR_rollingDB = pd.concat(df_SV_SR_rollingDB)