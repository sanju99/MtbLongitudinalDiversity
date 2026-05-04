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

parser.add_argument("--AF_min",  type=float, default=0.95)
parser.add_argument("-o",  dest='out_file', type=str, required=True)

cmd_line_args = parser.parse_args()

AF_min = cmd_line_args.AF_min
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

print(f"Working on {len(unmixed_lineage_samples)} samples")

dist_from_indel = 10

df_indels_results = []

for i, sample in enumerate(unmixed_lineage_samples):
        
    df_variants = pd.read_csv(f"{H37Rv_ref_dir}/{sample}/freebayes/{sample}.cleaned.excludeLowConf.fixedAO.tsv", sep='\t').query("POS not in @rRNA_pos & POS not in @insertion_seqs_phages_pos")
        
    df_indels = df_variants.query("REF.str.len() != ALT.str.len()")
    
    df_indels['AF'] = df_indels['AO'] / df_indels['DP']
    df_indels['AF'] = df_indels['AF'].astype(float)
    
    df_indels = df_indels.query("MQM >= 40 & AF > @AF_min")
    
    if len(df_indels) > 0:
    
        df_indels = df_indels[['POS', 'REF', 'ALT', 'AF', 'AO', 'DP']].sort_values('POS')
            
        df_indels['SampleID'] = sample
            
        df_indels_results.append(df_indels)

        # print progress and save intermediate results
        if i % 100 == 0:
            if len(df_indels_results) > 0:
                pd.concat(df_indels_results).to_csv(out_file, index=False)
            print(sample)

    
if len(df_indels_results) > 0:
    df_indels_results = pd.concat(df_indels_results)

    df_indels_results = df_indels_results.merge(df_trust_patients[['SampleID', 'pid']], how='left')

    # have to do AF > AF_min here because apply_freebayes_lowAF_QCfilters treats AF_min as inclusive, but we need exclusive
    df_indels_results.query("AF > @AF_min").to_csv(out_file, index=False)
else:
    print(f"No indels across {len(unmixed_lineage_samples)} samples")