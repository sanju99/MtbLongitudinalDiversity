import numpy as np
import pandas as pd
import glob, os, warnings, shutil, subprocess, re, sys
from Bio import Seq, SeqIO
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as st
import scipy

sys.path.append("/home/sak0914/TRUST_data_processing/scripts")
from utils import *

TRUST_data_dir = "/home/sak0914/TRUST_data_processing"

F2_thresh = 0.03


############################### THIS SCRIPT DETERMINES THE PIDS TO USE FOR LONGITUDINAL ANALYSIS ###############################


df_patient_WGS = pd.read_csv(f"{TRUST_data_dir}/processed_data/20250818_combined_patient_WGS_data.csv")[['pid', 'Original_ID', 'SampleID', 'Sampling_Week']]

pids_longitudinal = pd.DataFrame(df_patient_WGS.groupby('pid')['Sampling_Week'].nunique()).query("Sampling_Week > 1").index.values
print(f"{len(pids_longitudinal)} pids have multiple timepoints")

df_longitudinal = df_patient_WGS.query("pid in @pids_longitudinal")

df_lineages = extract_lineages(df_longitudinal, 'SampleID', "/n/data1/hms/dbmi/farhat/Sanjana/TRUST_lowAF")

# combine with lineages
df_longitudinal = df_longitudinal.merge(df_lineages, on='SampleID', how='inner')

# keep only samples taken in the first 12 weeks (during treatment). We don't want to consider follow-up samples
df_longitudinal = df_longitudinal.query("Sampling_Week <= 12")

df_longitudinal['Paired_Sample_Num'] = df_longitudinal.sort_values(['pid', 'Sampling_Week', 'Original_ID', 'SampleID']).groupby("pid").cumcount() + 1

# Assign sample_num (total per patient)
df_longitudinal["total_samples"] = df_longitudinal.groupby("pid")["Sampling_Week"].transform("size")

# remove pids with only 1 sample now, after removing contaminated samples
df_longitudinal = df_longitudinal.query("total_samples > 1")

print(f"    {df_longitudinal.pid.nunique()} pids have at least 2 uncontaminated WGS samples")


############################### HANDLE PIDS WITH SAMPLES WITH DIFFERENT LINEAGES ###############################


df_num_unique_lineages_by_pid = pd.DataFrame(df_longitudinal.groupby('pid')['Lineage'].nunique()).reset_index()

pids_with_multiple_lineages = df_num_unique_lineages_by_pid.query("Lineage > 1").pid.values
pids_with_one_lineage = df_num_unique_lineages_by_pid.query("Lineage == 1").pid.values

print(f"    {len(pids_with_multiple_lineages)} pids have multiple lineages")

def select_samples_to_keep_single_pid(df, pid):
    
    df_single_pid = df.query("pid==@pid")
    
    num_samples = len(df_single_pid)
    expanded_sample_lineages = [lineage.split(',') for lineage in df_single_pid['Lineage'].values]
    
    # simplest case
    if num_samples == 2:
        
        # if there is overlap between the lineages, then there is evidence of mixing, and we keep both
        if len(set(expanded_sample_lineages[0]).intersection(expanded_sample_lineages[1])) > 0:
            return df_single_pid.SampleID.values
        else:
            # check if any have high F2. If so, keep
            if len(df_single_pid.query("F2 > @F2_thresh")) > 0:
                return df_single_pid.SampleID.values
            else:
                return []
    
    else:
        # get the majority lineage
        flattened_sample_lineages = list(itertools.chain.from_iterable(expanded_sample_lineages))
    
        vals, counts = np.unique(flattened_sample_lineages, return_counts=True)
        majority_lineage = vals[np.argmax(counts)]
        
        return df_single_pid.query("Lineage.str.contains(@majority_lineage)").SampleID.values
    
    
pids_with_multiple_lineages_samples_to_keep = []

for pid in pids_with_multiple_lineages:
    
    first_sample_lineage = df_longitudinal.query("pid==@pid & Paired_Sample_Num == 1")['Lineage'].values[0]
    

    other_sample_lineages = df_longitudinal.query("pid==@pid & Paired_Sample_Num != 1")['Lineage'].values
    
    # expand any mixed lineage cases
    expanded_sample_lineages = [lineage.split(',') for lineage in df_longitudinal.query("pid==@pid")['Lineage'].values]
    
    samples_to_keep = list(select_samples_to_keep_single_pid(df_longitudinal, pid))
    pids_with_multiple_lineages_samples_to_keep += samples_to_keep
    
    
    
df_longitudinal_keep = pd.concat([df_longitudinal.query("pid in @pids_with_one_lineage"),
                                  df_longitudinal.query("pid in @pids_with_multiple_lineages & SampleID in @pids_with_multiple_lineages_samples_to_keep")
                                 ])

df_longitudinal_keep['Paired_Sample_Num'] = df_longitudinal_keep.sort_values(['pid', 'Sampling_Week', 'Original_ID', 'SampleID']).groupby("pid").cumcount() + 1

# Assign sample_num (total per patient)
df_longitudinal_keep["total_samples"] = df_longitudinal_keep.groupby("pid")["Sampling_Week"].transform("size")

# remove pids with only 1 sample now, after removing samples with confidently called different lineages
df_longitudinal_keep = df_longitudinal_keep.query("total_samples > 1")

# save a table of the removed pids to another file to inspect manually later
df_discordant_pids = df_longitudinal.query("pid not in @df_longitudinal_keep.pid")[['pid', 'Original_ID', 'SampleID', 'Sampling_Week', 'Coll2014', 'Freschi2020']]

print(f"    {df_discordant_pids.pid.nunique()} pids have multiple unmixed lineages")
df_discordant_pids.to_csv("../data/pids_discordant_unmixed_lineages.csv", index=False)

print(f"{df_longitudinal_keep.pid.nunique()} pids have at least 2 WGS samples without confidently called discordant lineages")

# Keep only first and last samples per patient.
df_longitudinal_keep = df_longitudinal_keep.query("Paired_Sample_Num == 1 | Paired_Sample_Num == total_samples").sort_values(['pid', 'Paired_Sample_Num']).reset_index(drop=True)

# replace the Paired_Sample_Num second sample with the number 2
df_longitudinal_keep.loc[df_longitudinal_keep['Paired_Sample_Num'] != 1, 'Paired_Sample_Num'] = 2

assert len(df_longitudinal_keep) == 2 * df_longitudinal_keep.pid.nunique()

df_longitudinal_keep.to_csv("../data/pids_WGS_data.csv", index=False)