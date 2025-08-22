import numpy as np
import pandas as pd
import glob, os, warnings, shutil, subprocess, re, sys
from Bio import Seq, SeqIO
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as st
import scipy

sys.path.append("utils")
from epi_utils import *
from variant_analysis_utils import *

h37Rv_path = "/n/data1/hms/dbmi/farhat/Sanjana/H37Rv"
h37Rv_regions = pd.read_csv(os.path.join(h37Rv_path, "mycobrowser_h37rv_v4.csv"))

# these are promoters, transcriptional signals, or RNAs. Exclude these
non_coding_regions = h37Rv_regions.query("Feature != 'CDS'").Name.values

# if these remain in the dataframe, then there will be multiple entires for a single gene name, which will cause process_intergenic_variant_WHO_catalog_coord to fail
h37Rv_regions = h37Rv_regions.query("~Feature.str.contains('|'.join(['promoter', 'signal']), case=False)")
assert len(h37Rv_regions) == h37Rv_regions.Name.nunique()

h37Rv_coords = pd.read_csv(os.path.join(h37Rv_path, "h37Rv_coords_to_gene.csv"))
h37Rv_coords_dict = dict(zip(h37Rv_coords["pos"].values, h37Rv_coords["region"].values))

# keep only the 727 patients with high quality WGS and matched patients
# df_WGS = pd.read_csv("/n/data1/hms/dbmi/farhat/rollingDB/TRUST/Illumina_culture_WGS_summary.csv")
df_trust_patients = pd.read_csv("./patient_data/combined_patient_WGS_data_contam_control.csv").dropna(subset=['pid', 'Lineage']).reset_index(drop=True)
print(df_trust_patients.pid.nunique(), df_trust_patients.SampleID.nunique())

for i, row in df_trust_patients.iterrows():
    if not pd.isnull(row['Lineage']):
        if type(row['Lineage']) == str:
            if ',' not in row['Lineage']:
                df_trust_patients.loc[i, 'Lineage'] = str(int(float(row['Lineage'])))
        else:
            df_trust_patients.loc[i, 'Lineage'] = str(int(row['Lineage']))
                
df_trust_patients.loc[df_trust_patients['bl_hiv']==0, 'HIV_CD4'] = 0
df_trust_patients.loc[(df_trust_patients['bl_hiv']==1) & (df_trust_patients['bl_cd4'] >= 200), 'HIV_CD4'] = 1
df_trust_patients.loc[(df_trust_patients['bl_hiv']==1) & (df_trust_patients['bl_cd4'] < 200), 'HIV_CD4'] = 2

# isolates with matched LR sequencing (including samples from the same patient where only 1 sample had LR done)
isolates_with_LR = pd.read_csv("TRUST_isolates_long_read_aln.tsv", sep='\t', header=None)

processed_data_dir = './processed_data'
CNN_results_dir = "/n/data1/hms/dbmi/farhat/Sanjana/CNN_results"

df_TCC = pd.read_csv("./imputation/TCC.csv")
df_final_outcomes = pd.read_csv("./patient_data/tx_outcomes_table.csv")


############################# STEP 1: GET THE INPUT FILES #############################


# pids_for_analysis = pd.read_csv(f"{processed_data_dir}/pids_for_analysis.csv")
pids_for_analysis = df_trust_patients.copy()
pids_for_analysis['Lineage'] = pids_for_analysis['Lineage'].astype(str)
pids_for_analysis.pid.nunique(), pids_for_analysis.SampleID.nunique(), len(pids_for_analysis)

# exclude_pids = pd.DataFrame(pids_for_analysis.groupby('pid')['SampleID'].nunique()).query("SampleID < 2").index.values
# pids_for_analysis = pids_for_analysis.query("pid not in @exclude_pids").reset_index(drop=True)
print(pids_for_analysis.pid.nunique(), pids_for_analysis.SampleID.nunique(), len(pids_for_analysis))

F2_thresh = 0.03
pids_for_analysis['high_F2'] = (pids_for_analysis['F2'] > F2_thresh).astype(int)

# at least 1 of the 2 WHO samples has a high F2 score
pids_for_analysis['pid_high_F2'] = pids_for_analysis.groupby('pid')['high_F2'].transform(lambda x: np.max(x))

low_F2_pids = pids_for_analysis.query("pid_high_F2 == 0")
high_F2_pids = pids_for_analysis.query("pid_high_F2 == 1")

print(f"{low_F2_pids.pid.nunique()} pids with F2 ≤ {F2_thresh}")
print(f"{high_F2_pids.pid.nunique()} pids with F2 > {F2_thresh}")


############################# STEP 2: GET ALL VARIANTS AND KEEP ONLY PHASE VARIANTS #############################


# HT_regions = find_HT_regions(str(h37Rv_seq.seq))
# HT_regions['Region'] = HT_regions['POS'].map(h37Rv_coords_dict)
HT_regions = pd.read_csv("nucleotide_runs_all_lengths.csv")
print(f"{len(HT_regions)} runs of at least 2")

HT_nucs = []

for i, row in HT_regions.query("Length >= 3").iterrows():
    # don't need to add 1 to the end because the start is one of the N nucleotides
    # BUT add 1 bp to the front and back so that you include variants that occur adjacent to an HT region, not just within it
    HT_nucs += list(np.arange(row['POS'] - 1, row['POS'] - 1 + row['Length'] + 1))

HT_nucs = np.unique(HT_nucs)
print(f"{len(HT_nucs)} nucleotides are in homopolymeric tracts")


df_variants, sample_1_variants = get_matrix_of_HT_indels(f"{processed_data_dir}/full_genome", 
                                                        pids_for_analysis, # include only pids with an F2 score below the threshold
                                                        HT_nucs,
                                                        sample_2=False,
                                                        binarize_fixed_variants=False, 
                                                        fixed_thresh=0.95, 
                                                        absent_thresh=0.05
                                                       )


def combine_frameshifts_single_gene(df, difference=False):

    df_grouped_by_gene = df.copy()
    
    # only keep track of the genes with indels in coding regions
    grouped_variants_dict = {}
    
    for col in df_grouped_by_gene.columns:
    
        if '_p.' in col:
            gene = col.split('_p.')[0]
            
            if gene not in grouped_variants_dict.keys():
                grouped_variants_dict[gene] = [col]
            else:
                grouped_variants_dict[gene] += [col]
    
    for gene, variants in grouped_variants_dict.items():

        if difference:
#             # if there are samples with more than 1 variant per gene, trigger this
#             unique_variants_per_gene = df_grouped_by_gene[variants].replace(0, np.nan).count(axis=1).unique()
        
#             if np.max(unique_variants_per_gene) > 1:
#                 raise ValueError(f"Some isolates have multiple variants per {gene}")
#             else:
#                 # create a combined variable.
#                 # replace 0s with NAs because NA is always returned by np.max() and np.min() in python
#                 df_grouped_by_gene[gene] = df_grouped_by_gene[variants].replace(0, np.nan).max(axis=1).fillna(0)

            # create a combined variable by averaging the differences. 
            # Average because sometimes 1 could be positive and 1 could be negative, but the magnitudes are really small, so they're both considered fixed
            
            df_grouped_by_gene[gene] = df_grouped_by_gene[variants].mean(axis=1).fillna(0)
        
        # do a simple maximum because these dataframes are not measuring differences (so they can't be negative). 
        else:
            # create a combined variable
            df_grouped_by_gene[gene] = df_grouped_by_gene[variants].max(axis=1)

        # remove the component variables
        df_grouped_by_gene = df_grouped_by_gene.drop(columns=variants, axis=1)

    print(f"Collapsed {df.shape[1]} variants into {df_grouped_by_gene.shape[1]}")

    return df_grouped_by_gene, list(grouped_variants_dict.keys())


sample_1_variants, genes_with_frameshifts = combine_frameshifts_single_gene(sample_1_variants, difference=False)
# sample_2_variants, _ = combine_frameshifts_single_gene(sample_2_variants, difference=False)
# indels_difference, genes_with_frameshifts = combine_frameshifts_single_gene(indels_difference, difference=True)



############################# STEP 3: GET THE INPUT FILES #############################


df_full = pd.read_csv("patient_data/20240826_metadata_MIC_method_updates.csv")
df_full = add_household_numbers(df_full)

df_trust_patients, TRUST_phenos, df_pred_combined = read_combine_all_TRUST_data("./patient_data/combined_patient_WGS_data_contam_control.csv", 
                                                                                {'RIF': 'lineage_amino_acid', 
                                                                                 'INH': 'lineage_amino_acid', 
                                                                                 'EMB': 'lineage_amino_acid', 
                                                                                 'PZA': 'amino_acid'}, # dictionary for which model to get predictions for
                                                                                CNN_results_dir="/n/data1/hms/dbmi/farhat/Sanjana/CNN_results", 
                                                                                baseline_only=False
                                                                               )

# also run on this so that the variables are in better encodings
df_trust_patients = process_patient_metadata_better_encodings(df_trust_patients, TRUST_phenos, include_TTP=False)
df_trust_patients = add_household_numbers(df_trust_patients)

print(f"{df_trust_patients.merge(df_TCC, on='pid').pid.nunique()}/{df_trust_patients.pid.nunique()} patients with valid TCC, WGS, and MICs")


def drop_duplicate_patients(imputed_data_fName, imputed_TCC_fName):

    # read in the imputed patient data
    df_imputed = pd.read_csv(imputed_data_fName).drop_duplicates()
    
    # read the imputed TCC results (this is only TCC, not the rest of the patient data)
    df_TCC_imputed_all = pd.read_csv(imputed_TCC_fName).rename(columns={'culture_convert_imputed': 'culture_convert', 'TCC_imputed': 'TCC'})
    
    # df_full.dropna(subset='esp_reasonoth_end_of_study_parti').query("esp_reasonoth_end_of_study_parti.str.contains('|'.join(['resist', 'regimen']), case=False)")[['pid', 'to_studyto_treatment_outcome', 'to_comments_treatment_outcome', 'esp_reason_end_of_study_parti', 'esp_reasonoth_end_of_study_parti']]
    
    # T0122 was diagnosed with MDR. T0137 died within 3 weeks of starting treatment, but their ESP reason says 'Changed to "liver-friendly regimen". No longer on first line drugs'
    # they are removed by the TCC exclusion criterion because they died so soon and don't have enough cultures sampled.
    exclude_pids_resistance = ['T0122', 'T0137']
        
    # T0322 says they were started on RHZE + levofloxacin
    # T0330 is fine, says they had a positive month 5 culture and were RIF- and INH-sensitive
    # T0355 started MDR regimen upon incarceration
    exclude_pids_diff_regimen = ['T0322', 'T0355']
    
    # add the household numbers
    df_TCC_imputed_all = df_TCC_imputed_all.merge(df_full[['pid', 'household_num']])
    
    # add the unique patient identifiers (just integers)
    unique_cluster_dict = get_unique_patient_ID_dict(df_full)

    df_TCC_imputed_all['unique_patient'] = df_TCC_imputed_all['pid'].map(unique_cluster_dict).astype(int)
    
    # keep only the first instance for re-enrolled patients
    df_TCC_imputed_all = df_TCC_imputed_all.sort_values(['imp_num', 'pid', 'unique_patient']).drop_duplicates(['imp_num', "unique_patient"], keep='first').merge(df_trust_patients[['pid']].drop_duplicates())
    print(f"{df_TCC_imputed_all.pid.nunique()} unique patients across {df_TCC_imputed_all.household_num.nunique()} households")
    
    # 3 of the above excluded pids are in here
    df_TCC_imputed_all = df_TCC_imputed_all.query("pid not in @exclude_pids_resistance and pid not in @exclude_pids_diff_regimen").merge(df_trust_patients[['pid']].drop_duplicates()).reset_index(drop=True)
    
    print(f"{df_TCC_imputed_all.pid.nunique()} unique patients across {df_TCC_imputed_all.household_num.nunique()} households")
    
    df_imputed = df_imputed.query("pid in @df_TCC_imputed_all.pid")

    return df_imputed, df_TCC_imputed_all



df_imputed_patient_data, df_imputed_TCC = drop_duplicate_patients("imputation/data_TCC.csv", "imputation/TCC.csv")


# df_imputed_patient_data = df_imputed_patient_data.query("pid in @low_F2_pids.pid")
print(f"{df_imputed_patient_data.pid.nunique()} pids for the analysis")


def add_change_to_indel_to_patient_data(df, sample_1_variants, sample_2_variants, indels_difference, variant, freq_thresh=3, include_interaction=True):

    num_samples_with_indel_change = len(indels_difference.loc[indels_difference[variant] != 0])
    
    num_samples_variant_initial = sample_1_variants[variant].sum()
    num_samples_variant_final = sample_2_variants[variant].sum()

    if num_samples_variant_initial < freq_thresh and num_samples_variant_final < freq_thresh:
        print(f"    Fewer than {freq_thresh} patients have {variant} at the beginning and end. Skipping this variant...")
        return None
    
#     if num_samples_with_indel_change < freq_thresh:
#         print(f"    Only {num_samples_with_indel_change} patients have a change in {variant}. Skipping this case...")
#         return None

    # add the indel to the dataframe. Change the name of the column because the format like _p. can mess with the string encoding the formula in the Cox model
    df_with_variant = df.merge(indels_difference[[variant]], left_on='pid', right_index=True).rename(columns={variant: 'Indel_Change'})

    # also add an AF variable for whether the variable was present at baseline
    df_with_variant = df_with_variant.merge(sample_1_variants[[variant]], left_on='pid', right_index=True).rename(columns={variant: 'Indel_Sample1'})

    # include interaction between the two
    if include_interaction:
        df_with_variant['Indel_Sample1_x_Change'] = df_with_variant['Indel_Sample1'] * df_with_variant['Indel_Change']

    return df_with_variant



def add_indel_at_baseline_to_patient_data(df, sample_1_variants, variant, freq_thresh=3):
    
    num_samples_variant_initial = sample_1_variants[variant].sum()

    if num_samples_variant_initial < freq_thresh:
        print(f"    Fewer than {freq_thresh} patients have {variant} at at baseline. Skipping this variant...")
        return None
    
    # also add an AF variable for whether the variable was present at baseline
    df_with_variant = df.merge(sample_1_variants[[variant]], left_on='pid', right_index=True).rename(columns={variant: 'Indel_Sample1'})

    return df_with_variant



cols_lst = [
            'screen_sex',
            'screen_years',
            'bl_bmi', # numerical BMI
            'fstrom1_baseline', # smoking yes or no
            # 'smoked_substance_use',
            'HIV_CD4',
            'bl_prevtb',
            'smear_pos_no_contam_sputum_specimen_1', # binary variable for smear positivity at baseline
            'diabetes',
            'high_lung_involvement',
            'adherence_12week',
            'cxr_cavity_chest_radiograph_1',
            # 'TTP',
            # 'underweight',
            'RIF_AUC',
            'INH_AUC',
            'EMB_AUC',
            'PZA_AUC',
            'F2'
           ]

# try these first, as they were required for the full TCC model
stratify_covariates = ['smear_pos_no_contam_sputum_specimen_1', 'high_lung_involvement', 'cxr_cavity_chest_radiograph_1', 'screen_sex', 'HIV_CD4']

df_LRT_pvals_TCC = pd.DataFrame(columns = ['pval'])

for covar in cols_lst:

    # need to stratify by these variables
    if covar not in stratify_covariates:

        print(f"Testing {covar}")
    
        pval = run_LRT_single_predictor(covar, 
                                        df_trust_patients,
                                        df_pred_combined,
                                         TRUST_phenos,
                                        df_imputed_patient_data, 
                                        df_imputed_TCC, 
                                        cols_lst, 
                                        event_col='culture_convert',
                                        time_col='TCC',
                                        MIC_type='none',
                                        stratify_variables=stratify_covariates,
                                       )
        
        df_LRT_pvals_TCC.loc[covar, 'pval'] = pval

df_LRT_pvals_TCC.reset_index().rename(columns={'index': 'covariate'}).to_csv("./results/TCC/LRT_results_baseline_variant_only.csv", index=False)
df_LRT_pvals_TCC = pd.read_csv("./results/TCC/LRT_results_baseline_variant_only.csv")

final_patient_predictors = list(df_LRT_pvals_TCC.query("pval <= 0.2").covariate.values) + stratify_covariates
print(f"Final patient predictors: {final_patient_predictors}")

df_results_all_indels = []
# num_to_test = 0

for i, variant in enumerate(sample_1_variants.columns): #:#indels_difference.columns:

    print(f"Testing {variant}")

    df_imputed_patient_data_single_variant = add_indel_at_baseline_to_patient_data(df_imputed_patient_data, 
                                                                                     sample_1_variants,
                                                                                     variant,
                                                                                     freq_thresh=10,
                                                                                  )
    
    # if df_imputed_patient_data_single_variant is not None:
    #     num_to_test += 1

    # will be None if there are not enough patients with the specified change in the indel
    if df_imputed_patient_data_single_variant is not None:
        
#         pids_with_change = indels_difference.loc[(indels_difference[variant] >= 0.1) | (indels_difference[variant] <= -0.1)].index.values

#         # only include the Indel_Change variable if is at least 1 patient in each even / no event group with an AF change. Otherwise, you get perfect separation
#         # of the groups, and the hazard ratio estimates will be huge and unreliable
#         if df_imputed_TCC.loc[df_imputed_TCC['imp_num']==0].query("pid in @pids_with_change")['culture_convert'].nunique() == 2:
#             predictors_lst = final_patient_predictors + ['inh_resistant', 'Lineage'] + ['Indel_Sample1', 'Indel_Change']
#         else:
#             predictors_lst = final_patient_predictors + ['inh_resistant', 'Lineage'] + ['Indel_Sample1']

        try:
            df_test_results_indel, _, _ = fit_cox_models_all_imputations(df_trust_patients,
                                                                      df_pred_combined,
                                                                       TRUST_phenos,
                                                                       df_imputed_patient_data_single_variant,
                                                                      df_imputed_TCC,
                                                                      final_patient_predictors + ['inh_resistant', 'Lineage', 'Indel_Sample1'],
                                                                      event_col = 'culture_convert',
                                                                      time_col = 'TCC',
                                                                      invert_OR=True,
                                                                      MIC_type='none',
                                                                      include_drugs=['INH'],
                                                                      stratify_variables=stratify_covariates,
                                                                      alpha=0.05,
                                                                     )

            df_test_results_indel['Indel'] = variant

            df_results_all_indels.append(df_test_results_indel)

            pd.concat(df_results_all_indels).to_csv("./results/TCC/indels_baseline_variants_only.csv", index=False)
            
        except:
            print(f"    {variant} failed")


df_results_all_indels = pd.concat(df_results_all_indels)
df_results_all_indels.to_csv("./results/TCC/indels_baseline_variants_only.csv", index=False)