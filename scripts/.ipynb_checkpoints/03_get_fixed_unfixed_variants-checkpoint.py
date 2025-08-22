########## PROCESSING SCRIPT

# This script processes the available data to keep only patients with
# longitudinal sampling
# matching lineages at the two timepoints (unless there is evidence of lineage mixing, and it is consistent, in which case the unmixed and mixed samples will be kept)

# It will also remove samples for patients who have more than 2 WGS samples, keeping the first and last ones to maximize time distance between them


import numpy as np
import pandas as pd
import glob, os, warnings, shutil, subprocess, re, sys, argparse
from Bio import Seq, SeqIO
warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser()

parser.add_argument('--AF_thresh', dest='AF_thresh', type=float, default=0.75, help='AF threshold to be considered a fixed variant')
parser.add_argument("-d", dest='sample_dir', type=str, required=True, help='Directory with output files')
parser.add_argument("-o", dest='output_dir', type=str, required=True, help="Name of the directory to write the combined variants CSV files to. It will be created  if it doesn't exist.")

cmd_line_args = parser.parse_args()

AF_thresh = cmd_line_args.AF_thresh
sample_dir = cmd_line_args.sample_dir
output_dir = cmd_line_args.output_dir


################################################## STEP 1: GET ALL VARIANTS FOR ALL SAMPLES AND SAVE THEM ##################################################


coverage_plateau_results_fNames = glob.glob(f"{sample_dir}/*/freebayes/coverage_plateau_sites.npy")
high_SNP_density_results_fNames = glob.glob(f"{sample_dir}/*/freebayes/high_density_SNP_sites.npy")

finished_samples = list(set([re.search(r'MFS-\d{1,3}', fName).group() for fName in coverage_plateau_results_fNames]).intersection([re.search(r'MFS-\d{1,3}', fName).group() for fName in high_SNP_density_results_fNames]))

print(f"{len(finished_samples)} samples have finished annotation files for low-quality unfixed variants")


def annotate_lowAF_variants(sample, fName, fixed_AF_thresh=0.75):

    # 0.05 is the minimum to be considered an unfixed variant, and so the maximum must be greater than it so that df_unfixed_variants below doesn't have length 0
    assert fixed_AF_thresh > 0.05

    df_variants = pd.read_csv(fName)

    num_support_each_direction = 2
    
    # these are upper bounds on the probability of observing the observed deviation between SRF and SRR or between SAF and SAR, given that the expected probability 
    df_variants['SRP_prob'] = 10**(-df_variants['SRP']/10)
    df_variants['SAP_prob'] = 10**(-df_variants['SAP']/10)
    
    # when SRF = SRR = 0, there are no reads supporting the reference. If quality is 0, then the error probability is 1, which isn't true. They're actually NA
    df_variants.loc[(df_variants['SRF']==0) | (df_variants['SRR']==0), 'SRP_prob'] = np.nan
    
    df_unfixed_variants = df_variants.query("AF >= 0.05 & AF <= @fixed_AF_thresh & SAF >= @num_support_each_direction & SAR >= @num_support_each_direction & SAP_prob > 0.05")
    df_fixed_variants = df_variants.query("AF > @fixed_AF_thresh & SAF >= @num_support_each_direction & SAR >= @num_support_each_direction & SAP_prob > 0.05")

    # some samples have no low AF variants, and so the .loc method to annotate low quality low AF sites will fail
    if len(df_unfixed_variants) > 0:
        
        if not os.path.isfile(f"{sample_dir}/{sample}/freebayes/coverage_plateau_sites.npy") or not os.path.isfile(f"{sample_dir}/{sample}/freebayes/high_density_SNP_sites.npy"):
            raise ValueError(f"Low quality site annotations have not yet finished for {sample}")
        
        coverage_plateau_sites = np.load(f"{sample_dir}/{sample}/freebayes/coverage_plateau_sites.npy")
        high_density_SNP_sites = np.load(f"{sample_dir}/{sample}/freebayes/high_density_SNP_sites.npy")
    
        # only annotate indels in this way
        df_unfixed_variants.loc[(df_unfixed_variants['REF'].str.len() != df_unfixed_variants['ALT'].str.len()) & ((df_unfixed_variants['POS'].isin(coverage_plateau_sites)) | (df_unfixed_variants['POS'].isin(high_density_SNP_sites))), 'Low_Qual'] = 1
        
        df_unfixed_variants['Low_Qual'] = df_unfixed_variants['Low_Qual'].fillna(0).astype(int)

    return df_fixed_variants, df_unfixed_variants




def save_both_fixed_unfixed_variants(finished_samples, save_dir, regions_of_interest=False):

    os.makedirs(save_dir, exist_ok=True)
    
    df_fixed_variants_all = []
    df_unfixed_variants_annotated = []
    
    for i, sample in enumerate(finished_samples):

        if regions_of_interest:
            fName = f"{sample_dir}/{sample}/freebayes/{sample}.regionsOfInterest.csv"
        else:
            fName = f"{sample_dir}/{sample}/freebayes/{sample}.csv"
        
        df_fixed_variants, df_unfixed_variants = annotate_lowAF_variants(sample, fName, AF_thresh)
    
        df_fixed_variants['SampleID'] = sample
        df_unfixed_variants['SampleID'] = sample
    
        df_fixed_variants_all.append(df_fixed_variants)
        df_unfixed_variants_annotated.append(df_unfixed_variants)
    
        if i % 100 == 0:
            print(i)
    
    df_fixed_variants_all = pd.concat(df_fixed_variants_all)
    df_unfixed_variants_annotated = pd.concat(df_unfixed_variants_annotated)
    
    df_fixed_variants_all.set_index('SampleID').to_csv(f"{save_dir}/fixed_variants.csv")
    df_unfixed_variants_annotated.set_index('SampleID').to_csv(f"{save_dir}/unfixed_variants.csv")


# only the phase variation regions of interest. It's more accurate to do this with a BED file because it will get all variants that affect the regions of interest
# rather than a python filter will only get variants with a POS value that falls within the regions of interest
save_both_fixed_unfixed_variants(finished_samples, f"{output_dir}/phase_variation_regions", regions_of_interest=True)

# full genome
save_both_fixed_unfixed_variants(finished_samples, f"{output_dir}/full_genome", regions_of_interest=False)