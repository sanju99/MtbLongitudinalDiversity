import numpy as np
import pandas as pd
import argparse, os, glob, warnings, sys
warnings.filterwarnings('ignore')
sys.path.append(os.path.dirname(__file__))
from data_utils import *
from epi_utils import *

parser = argparse.ArgumentParser()

parser.add_argument("-i", dest='input_file', type=str, required=True, help='RedCap data from BMC group')
parser.add_argument("-o", dest='output_dir', type=str, required=True, help='Directory to save files to')

cmd_line_args = parser.parse_args()

input_file = cmd_line_args.input_file
output_dir = cmd_line_args.output_dir


# ordinal encoding: bl_afbprog --> smear
smear_encoding_dict = {6: np.nan, # I think this was already done in their data cleaning
                       5: np.nan, # I think this was already done in their data cleaning
                       0: 0, # no AFB
                       4: 1, # scanty
                       1: 2, # +
                       2: 3, # ++
                       3: 4, # +++
                      }



def get_time_to_culture_conversion(single_sample_combined_sputum_results):
    '''
    This function computes the TCC for a single sample in the sputum results dataframe. A TTP is only valid if the culture result is tb_positive.
    '''

    # keep only samples up to 12 (in case the month 5 samples are still in the table)
    single_sample_combined_sputum_results = single_sample_combined_sputum_results.query("sample_num <= 12")

    # replacement that the BMC group did. This is only for the TCC calculation. Keep the original samples unchanged
    single_sample_combined_sputum_results['result'] = single_sample_combined_sputum_results['result'].replace('tb_positive_contaminated', 'tb_positive')
    
    start_positive = single_sample_combined_sputum_results.query("result=='tb_positive'").sample_num.min()

    if start_positive is None:
        raise ValueError(f"There is no start culture positivity time for {pid}")
        
    # the TCC will be the first of two negative result that are not followed by a positive result
    # don't consider positive smear because smear test can detect dead bacteria, which won't grow in the culture.
    end_positive = single_sample_combined_sputum_results.query("result=='tb_positive'").sample_num.max()

    # exclude the month 5 culture (sample_num = 20) from the TCC computation
    post_last_positive_results = single_sample_combined_sputum_results.query("sample_num > @end_positive").reset_index(drop=True)

    # initialize as None variable
    start_negative = None

    # check that there are at least 2 negative results, otherwise don't do the search below
    if len(post_last_positive_results.query("result=='tb_negative'")) >= 2:

        # check that they are consecutive results
        for i, row in post_last_positive_results.iterrows():

            # check that it's not the last culture, in which case there won't be a second negative afterwards
            if row['result'] == 'tb_negative' and i != len(post_last_positive_results) - 1:
                
                if post_last_positive_results.result.values[i+1] == 'tb_negative':
                    
                    # get the sample number of the first negative sample
                    start_negative = row['sample_num']

                    # can break because we already checked above that there are no positive cultures or smear grades afterwards
                    break

    # if no culture conversion (no event), then the patient did not culture convert, so take the maximum number of weeks
    if start_negative is None:
        # start_negative = 12 # the last culture sample in the treatment window. Don't consider 
        #start_negative = single_sample_combined_sputum_results.sample_num.max()

        # take the time of the last known positive culture. They will be censored at this time. If there are contaminated or single negative cultures after this time,
        # we can't interpret them because they are inconclusive. 
        # Exclude the values above 12 because those aren't weeks

        # if you take the last negative sample, sometimes you take a negative sample that occurs before a positive sample. The time of the last positive sample is probably the most informative time
        # take the last known time when the patient was smear positive or culture positive
        start_negative = end_positive
        
        culture_convert = 0
    else:
        culture_convert = 1

    # keep track of patients who culture converted
    # all patients in this study have TB (microbiologically confirmed), so take week 1 as the starting time
    return culture_convert, start_negative



def get_combined_smear_and_culture_results_single_pid(df_trust_patients, pid):

    ########################################## STEP 1: CULTURE POSITIVITIY ########################################## 

    # get all sputum culture results for a single pid 
    culture_cols = list(df_trust_patients.columns[(df_trust_patients.columns.str.contains('culture_conversion')) & (~df_trust_patients.columns.str.contains('additional'))])
    single_pid_culture_results = pd.DataFrame(df_trust_patients.drop_duplicates(subset='pid')[['pid'] + culture_cols].set_index('pid').loc[pid]).reset_index()
    
    single_pid_culture_results.columns = ['column', 'result']
    
    # get the sample week and sort by that. Can't sort by the raw column name itself because _2 will be considered greater than _10. So need to convert them to integers
    single_pid_culture_results['sample_num'] = single_pid_culture_results['column'].str.split('_').str[-1].astype(int)
    del single_pid_culture_results['column'] # original column name, don't need anymore
    single_pid_culture_results = single_pid_culture_results.sort_values('sample_num').reset_index(drop=True)

    ########################################## STEP 2: TIME TO CULTURE POSITIVITY ########################################## 
    
    # get all TTP culture results for a single pid
    TTP_cols = list(df_trust_patients.columns[(df_trust_patients.columns.str.contains('ttp')) & (~df_trust_patients.columns.str.contains('|'.join(['analysis', 'hour', 'day', 'additional'])))])
    single_pid_TTP_results = pd.DataFrame(df_trust_patients.drop_duplicates(subset='pid')[['pid'] + TTP_cols].set_index('pid').loc[pid]).reset_index()
    
    # BMC Group combined TTP in days with TTP hours (so days + 24 * hours) to get this column
    single_pid_TTP_results.columns = ['column', 'hours']
    
    # get the sample week and sort by that. Can't sort by the raw column name itself because _2 will be considered greater than _10. So need to convert them to integers
    single_pid_TTP_results['sample_num'] = single_pid_TTP_results['column'].str.split('_').str[-1].astype(int)
    del single_pid_TTP_results['column'] # original column name, don't need anymore
    single_pid_TTP_results = single_pid_TTP_results.sort_values('sample_num').reset_index(drop=True)
    
    ########################################## STEP 3: SMEAR GRADE ##########################################
            
    # get all sputum culture results for a single pid 
    smear_grade_cols = list(df_trust_patients.columns[(df_trust_patients.columns.str.contains('s_concafb_sputum_specimen')) & (~df_trust_patients.columns.str.contains('additional'))])
    single_pid_smear_results = pd.DataFrame(df_trust_patients.drop_duplicates(subset='pid')[['pid'] + smear_grade_cols].set_index('pid').loc[pid]).reset_index()
    
    single_pid_smear_results.columns = ['column', 'smear_grade']
    
    # get the sample week and sort by that. Can't sort by the raw column name itself because _2 will be considered greater than _10. So need to convert them to integers
    single_pid_smear_results['sample_num'] = single_pid_smear_results['column'].str.split('_').str[-1].astype(int)
    del single_pid_smear_results['column'] # original column name, don't need anymore
    single_pid_smear_results = single_pid_smear_results.sort_values('sample_num').reset_index(drop=True)

    # change to proper ordinal encoding
    single_pid_smear_results['smear_grade'] = single_pid_smear_results['smear_grade'].map(smear_encoding_dict)

    ########################################## STEP 4: COMBINE ALL SPUTUM RESULTS ##########################################
    
    # combine culture results (positive, negative, contaminated) with TTP results
    combined_sputum_results = single_pid_culture_results.merge(single_pid_TTP_results, on='sample_num', how='outer').merge(single_pid_smear_results, on='sample_num', how='outer')

    # for contaminated samples, the TTP is not valid, so replace with NaN. BMC group did this in their data cleaning as well
    combined_sputum_results.loc[combined_sputum_results['result'] != 'tb_positive', 'hours'] = np.nan

    # check that there are no duplicates. This would occur if there are other smear grade / TTP / culture columns like for additional visits. These should be excluded
    assert len(combined_sputum_results) == combined_sputum_results.sample_num.nunique()
    
    return combined_sputum_results



def get_combined_culture_results(df_trust_patients):

    # combined_TTP_results = []
    df_TCC = pd.DataFrame(columns = ['pid', 'culture_convert', 'TCC'])
    df_TTP_smear = pd.DataFrame(columns = ['pid', 'culture_sample_num', 'TTP', 'smear_sample_num', 'smear_grade'])
    i = 0

    df_combined_culture = []
    
    for pid in df_trust_patients.pid.unique():

        # use the function above to get all smear and culture results for a single pid
        combined_sputum_results = get_combined_smear_and_culture_results_single_pid(df_trust_patients, pid).query("sample_num <= 13")

        # week 13 is actually month 5, so replace with 20
        combined_sputum_results.loc[combined_sputum_results['sample_num']==13, 'sample_num'] = 20
        
        combined_sputum_results['pid'] = pid
        df_combined_culture.append(combined_sputum_results)
        
        # get the first measured smear grade (so not NA). Smear test doesn't require culturing, so this is separate from the TTP calculation
        smear_grade_baseline = combined_sputum_results.dropna(subset='smear_grade')['smear_grade'].values[0]
        smear_grade_sample = combined_sputum_results.dropna(subset='smear_grade').sample_num.values[0]
        
        # get the first time to culture positivity (in hours) for a single sample in the sputum results dataframe
        baseline_positive_sample = combined_sputum_results.query("result=='tb_positive'").sample_num.min()

        # no positive culture sample for this pid. Probably only contaminated positive samples
        # similarly, get the smear grade at the first tb_positive culture
        if pd.isnull(baseline_positive_sample):
            TTP = np.nan
        else:
            TTP = combined_sputum_results.query("sample_num==@baseline_positive_sample")['hours'].values[0]
            
        # from BMC inclusion/exclusion criteria for TCC analysis:
        # 1. exclude patients with fewer than 3 culture samples because TCC analysis requires at least 1 positive and 2 negatives. Many of these withdrew. Some just have missing cultures
        # 2. exclude patients who don't have at least one negative culture because it's hard to reliably tell when they culture converted if you don't have that
        # 3. exclude patients who didn't have a positive culture in the first 5 weeks. This is because we assume that if they had a positive sample within the first 5 weeks, they were
        # positive at baseline, and we don't have to adjust the TCC timeline for them
        exclude_patient = False

        # first check that they had at least 3 total cultures. tb_positive_contaminated is okay as it is is replaced with tb_positive above. This is what they determined. 
        if len(combined_sputum_results.query("result in ['tb_negative', 'tb_positive']")) < 3:
            exclude_patient = True
            
        # check that they had a positive TB culture within the first 5 weeks
        elif 'tb_positive' not in combined_sputum_results.query("sample_num <= 5")['result'].values:
            exclude_patient = True

        if exclude_patient == True:
            # print(f"{pid} needs to be excluded!")
            culture_convert = np.nan
            TCC = np.nan
        else:
            culture_convert, TCC = get_time_to_culture_conversion(combined_sputum_results)
        
        df_TCC.loc[i, :] = [pid, culture_convert, TCC]
        df_TTP_smear.loc[i, :] = [pid, baseline_positive_sample, TTP, smear_grade_sample, smear_grade_baseline]
        
        i += 1

    return df_TTP_smear, df_TCC.dropna(subset='TCC').reset_index(drop=True), pd.concat(df_combined_culture).dropna(subset='result').reset_index(drop=True)



df_RedCap = pd.read_csv(input_file)

os.makedirs(output_dir, exist_ok=True)

df_TTP_smear, df_TCC, df_combined_cultures = get_combined_culture_results(df_RedCap)

print(f"{df_combined_cultures.pid.nunique()} pids with culture results")
print(f"{df_TTP_smear.pid.nunique()} pids with TTP and smear results")
print(f"{df_TCC.pid.nunique()} pids with valid TCC")

df_combined_cultures.set_index('pid').to_csv(f"{output_dir}/combined_culture_results.csv")
df_TTP_smear.set_index('pid').to_csv(f"{output_dir}/TTP_smear_results.csv")
df_TCC.set_index('pid').to_csv(f"{output_dir}/TCC.csv")

########### pivot to matrix, which is used by the imputation R script ###########
########### restrict to only pids that have a valid TCC ###########

# positive contaminated = positive. negative contaminated = unknown
df_combined_cultures['result'] = df_combined_cultures['result'].replace('tb_positive_contaminated', 'tb_positive')
df_combined_cultures['result'] = df_combined_cultures['result'].replace('tb_negative_contaminated', np.nan)
df_combined_cultures['time_col'] = 'culture_' + df_combined_cultures['sample_num'].astype(str)

cultures_matrix = df_combined_cultures.query("pid in @df_TCC.pid & sample_num <= 12").pivot(index='pid', columns = 'time_col', values='result')

df_combined_cultures['smear_grade'] = df_combined_cultures['smear_grade'].astype(int)
df_combined_cultures['smear_positivity'] = (df_combined_cultures['smear_grade'] > 0).astype(int)

df_combined_cultures['smear_grade_time_col'] = 'smear_grade_' + df_combined_cultures['sample_num'].astype(str)
df_combined_cultures['smear_positivity_time_col'] = 'smear_positivity_' + df_combined_cultures['sample_num'].astype(str)

smear_grade_matrix = df_combined_cultures.query("pid in @df_TCC.pid & sample_num <= 12").pivot(index='pid', columns = 'smear_grade_time_col', values='smear_grade')
smear_positivity_matrix = df_combined_cultures.query("pid in @df_TCC.pid & sample_num <= 12").pivot(index='pid', columns = 'smear_positivity_time_col', values='smear_positivity')

cultures_matrix.to_csv(f"{output_dir}/cultures.csv")
smear_grade_matrix.to_csv(f"{output_dir}/smear_grade.csv")
smear_positivity_matrix.to_csv(f"{output_dir}/smear_positivity.csv")

# also save this. Annotate it with TTP_baseline and inh_resistant though
TRUST_phenos = get_measured_MICs(df_RedCap, id_vars=['pid'], baseline_only=True)
TRUST_phenos['inh_resistant'] = (TRUST_phenos['INH_lower_bound'] >= 0.1).astype(int)
TRUST_phenos.to_csv(f"{output_dir}/measured_MICs.csv", index=False)

df_RedCap = df_RedCap.merge(TRUST_phenos[['pid', 'inh_resistant']], how='left')

# only keep TTP and smear measured within the first 2 weeks. Adjust some other variables to have more intuitive encodings
df_RedCap = process_patient_metadata_better_encodings(df_RedCap, df_TTP_smear=df_TTP_smear, include_TTP=True)

df_RedCap.to_csv(f"{output_dir}/patient_data.csv", index=False)