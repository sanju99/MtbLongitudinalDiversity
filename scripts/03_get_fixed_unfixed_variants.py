########## GET UNFIXED AND FIXED VARIANTS FOR ALL SAMPLES AND MERGE INTO ONE CSV FOR EACH TYPE ##########

import numpy as np
import pandas as pd
import glob, os, warnings, shutil, subprocess, re, sys, argparse
from Bio import Seq, SeqIO
warnings.filterwarnings('ignore')
from variant_analysis_utils import *

parser = argparse.ArgumentParser()

parser.add_argument('--AF_thresh', dest='AF_thresh', type=float, default=0.95, help='AF threshold to be considered a fixed variant')
parser.add_argument("-i", dest='input_file', type=str, required=True, help="Name of the input file of pids and WGS metadata to merge with the fixed and unfixed variants.")
parser.add_argument("-d", dest='sample_dir', type=str, required=True, help='Directory with output files')
parser.add_argument("-o", dest='output_dir', type=str, required=True, help="Name of the directory to write the combined variants CSV files to. It will be created if it doesn't exist.")

cmd_line_args = parser.parse_args()

AF_thresh = cmd_line_args.AF_thresh
sample_dir = cmd_line_args.sample_dir
input_file = cmd_line_args.input_file
output_dir = cmd_line_args.output_dir


################################################## STEP 1: GET ALL VARIANTS FOR ALL SAMPLES AND SAVE THEM ##################################################


coverage_plateau_results_fNames = glob.glob(f"{sample_dir}/*/freebayes/coverage_plateau_sites.npy")
high_SNP_density_results_fNames = glob.glob(f"{sample_dir}/*/freebayes/high_density_SNP_sites.npy")

finished_samples = list(set([re.search(r'MFS-\d{1,3}', fName).group() for fName in coverage_plateau_results_fNames]).intersection([re.search(r'MFS-\d{1,3}', fName).group() for fName in high_SNP_density_results_fNames]))

print(f"{len(finished_samples)} samples have finished annotation files for low-quality unfixed variants")

# read in pid / WGS metadata table
df_longitudinal_pids = pd.read_csv(input_file)



def save_both_fixed_unfixed_variants(finished_samples, save_dir, regions_of_interest=False):

    os.makedirs(save_dir, exist_ok=True)
    
    df_fixed_variants_all = []
    df_unfixed_variants_annotated = []
    
    for i, sample in enumerate(finished_samples):

        if regions_of_interest:
            fName = f"{sample_dir}/{sample}/freebayes/{sample}.regionsOfInterest.csv"
        else:
            fName = f"{sample_dir}/{sample}/freebayes/{sample}.csv"
        
        df_fixed_variants = get_fixed_variants(fName, df_longitudinal_pids, AF_thresh=AF_thresh)
        df_unfixed_variants = get_unfixed_variants(fName, df_longitudinal_pids, AF_thresh=0.05, AF_max=AF_thresh)
    
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
save_both_fixed_unfixed_variants(finished_samples, f"{output_dir}/ROI", regions_of_interest=True)

# full genome
save_both_fixed_unfixed_variants(finished_samples, f"{output_dir}/genome", regions_of_interest=False)