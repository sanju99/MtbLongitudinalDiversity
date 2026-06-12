import pandas as pd
import numpy as np
import seaborn as sns
import glob, os, re, warnings, argparse
warnings.filterwarnings("ignore")
from data_utils import *


def combine_TRUST_patient_samples(df_trust_patient_data, WGS_metadata):

    print(f"{len(df_trust_patient_data.patient_num.unique())} patients in phenotypes dataframe")
    
    # each sample ID is in the format like 260-04, S0299-01, or S0271-01A. So use re to remove all alpha characters and keep only numbers.
    # first number is the patient number, second is the sample number
    # merge the patient number extracted from each sample with the patient numbers in df_trust_patient_data
    for i, row in WGS_metadata.iterrows():
        try:
            patient_num = int(re.findall(r'\d+', row["Original_ID"].split("-")[0])[0])
        # weird names like T-7, T-5, and P-T- 1
        except:
            patient_num = row["Original_ID"]

        try:
            sample_week = int(re.findall(r'\d+', row["Original_ID"].split("-")[1])[0])
        # weird names like T-7, T-5, and P-T- 1
        except:
            sample_week = row['Original_ID']
            
        WGS_metadata.loc[i, ['patient_num', 'sample_week']] = [patient_num, sample_week]
            
    # WGS_metadata["patient_num"] = [int(re.findall(r'\d+', val.split("-")[0])[0]) for val in WGS_metadata["Original_ID"].values]
    # WGS_metadata["sample_week"] = [int(re.findall(r'\d+', val.split("-")[1])[0]) for val in WGS_metadata["Original_ID"].values]

    # keep only samples for the patients in df_trust_patient_data
    WGS_metadata = WGS_metadata.merge(df_trust_patient_data, on="patient_num", how="inner")
    print(f"Found {WGS_metadata.patient_num.nunique()}/{df_trust_patient_data.patient_num.nunique()} patients in the WGS metadata files")
    
    # # rename some columns that have spaces in them
    # WGS_metadata = WGS_metadata.rename(columns={"Cov Any Mean": "Cov_Any_Mean",
    #                                             "Cov Unam Perc": "Cov_Unam_Perc",
    #                                             "Perc. Reads Mapped": "Perc_Reads_Mapped",
    #                                             "phylogenetic classification (Coll et al., 2014)": "Coll2014_Annotated"
    #                                    })

    return WGS_metadata




parser = argparse.ArgumentParser()

parser.add_argument("-i", "--patient_input", dest='patient_data_fName', type=str, required=True, help='Full path to a filename for the RedCap data from the TRUST study. This has had some data cleaning done on it')
parser.add_argument("-I", "--WGS_input", dest='WGS_data_fName', type=str, required=True, help='Full path to a filename with all WGS data and QC.')
parser.add_argument("-o", "--output", dest='out_fName', type=str, required=True, help='Full path to a file name of the combined WGS and patient data results')
parser.add_argument("-d1", dest='WGS_data_dir', default="/n/data1/hms/dbmi/farhat/rollingDB/TRUST/Illumina_culture_WGS_processed", type=str, help='Directory where the processed sequencing is')
parser.add_argument("-d2", dest='WGS_report_directory', default="/n/data1/hms/dbmi/farhat/rollingDB/TRUST/WGS_metadata_reports", type=str, help='Directory where the Excel files of WGS metadata reports are')

cmd_line_args = parser.parse_args()
patient_data_fName = cmd_line_args.patient_data_fName
WGS_data_fName = cmd_line_args.WGS_data_fName
out_fName = cmd_line_args.out_fName
WGS_data_dir = cmd_line_args.WGS_data_dir
WGS_report_directory = cmd_line_args.WGS_report_directory

# first get all the WGS QC
if not os.path.isfile(WGS_data_fName):
    
    # this is all samples that we received FASTQ files for
    df_samples = pd.DataFrame({'SampleID': os.listdir(WGS_data_dir), 'Run': os.listdir(WGS_data_dir)})

    # get kraken-classification metrics
    df_kraken = extract_kraken_reports(df_samples, 'SampleID', 'Run', out_dir = WGS_data_dir)

    # get BAM metrics
    df_depth_metrics = compute_BAM_depth_metrics(df_samples, 'SampleID', 'Run', out_dir = WGS_data_dir)

    # get lineage information
    df_lineages = extract_lineages(df_samples, 'SampleID', out_dir = WGS_data_dir)

    # merge
    df_results = df_kraken.merge(df_depth_metrics, on=['SampleID', 'Run'], how='outer').merge(df_lineages, on='SampleID', how='outer')
    
    del df_results['Run']
    
    print(df_results.Lineage.value_counts())
    
    df_results.to_csv(WGS_data_fName, index=False)
    
    
df_trust_patient_data = pd.read_csv(patient_data_fName, low_memory=False)

# get the patient number to match WGS IDs and pids
df_trust_patient_data["patient_num"] = [int(patient_id.replace("T0", "")) for patient_id in df_trust_patient_data["pid"].values]

# get all Excel files from this directory
trust_report_fNames = glob.glob(f"{WGS_report_directory}/*.xlsx")
print(f"{len(trust_report_fNames)} WGS metadata Excel files")

df_trust_WGS_metadata = []

# sort chronologically because below, we preferentially keep the later one
for fName in np.sort(trust_report_fNames)[::-1]:

    # read in single Excel file
    df = pd.read_excel(fName, sheet_name=None)

    # remove spaces from column name to make querying easier. Also there could be NaN rows if there are additional empty rows in the Excel sheet, so drop them
    df = df['summary'].rename(columns={'original ID': 'Original_ID'}).dropna(axis=0, how='all')

    # append to list for concatenation
    df_trust_WGS_metadata.append(df)

# the Excel files above are running totals, so the most recent file has data that is also in the older files. So drop duplicates, keeping the most recent (last) one
df_trust_WGS_metadata = pd.concat(df_trust_WGS_metadata).drop_duplicates('SampleID', keep='last')#.query("status!='failed'")

# combine patient and sample IDs (pid = patient, Original_ID and SampleID = WGS)
df_trust_combined = combine_TRUST_patient_samples(df_trust_patient_data, df_trust_WGS_metadata.reset_index(drop=True))

# combine with WGS QC data
df_geno = pd.read_csv(WGS_data_fName)
print(f"{len(df_geno)} WGS samples that pass QC")

df_trust_combined = df_trust_combined.merge(df_geno[['SampleID', 'F2', 'Coll2014', 'Freschi2020', 'Lineage', 'Kraken_Unclassified_Percent', 'Median_Depth', 'Perc_Sites_20x']], on='SampleID', how='left').reset_index(drop=True)

# Remove columns that are NaN everywhere to clean the dataframe
df_trust_combined = df_trust_combined.dropna(how='all', axis=1)

# get the sample week
df_trust_combined['Sampling_Week'] = df_trust_combined['Original_ID'].str.split('-').str[1]

# replace month 5 with 20 for weeks
df_trust_combined['Sampling_Week'] = df_trust_combined['Sampling_Week'].replace('01A', '01').replace('m5', '20').replace('M5', '20')

# drop these. They are not TRUST samples
df_trust_combined = df_trust_combined.dropna(subset='Sampling_Week')

# PMN means post-treatment month N. Because it's months, multiply by 4 to get weeks, but also add 24 weeks for the 6 months of treatment
# PMN means months since treatment end, not months since treatment start.
df_trust_combined.loc[df_trust_combined['Sampling_Week'].str.startswith('PM'), 'Sampling_Week'] = df_trust_combined['Sampling_Week'].str.replace('PM', '').astype(int) * 4 + 24

df_trust_combined['Sampling_Week'] = df_trust_combined['Sampling_Week'].astype(int)

# if there are multiple lineages per timepoint, then keep the samples belonging to the lineage with the most support
df_trust_combined = df_trust_combined.reset_index(drop=True)

for i, row in df_trust_combined.iterrows():
    if not row['Original_ID'].startswith('S'):
        
        length_numerical = len(row['Original_ID'].split('-')[0])
            
        newID = 'S' + '0' * (4 - length_numerical) + row['Original_ID']
        df_trust_combined.loc[i, 'Original_ID'] = newID
        
# samples_check_multiple_lineages = pd.DataFrame(df_trust_combined.query("F2 <= 0.03").groupby('Original_ID').Coll2014.nunique()).query("Coll2014 > 1").index.values
samples_check_multiple_lineages = pd.DataFrame(df_trust_combined.groupby('Original_ID').Coll2014.nunique()).query("Coll2014 > 1").index.values

exclude_WGS_samples = []

for original_ID in samples_check_multiple_lineages:
    
    major_lineage = df_trust_combined.query("Original_ID == @original_ID").Coll2014.mode().values[0]
    
    exclude_WGS_samples += list(df_trust_combined.query("Original_ID == @original_ID & Coll2014 != @major_lineage").SampleID.values)
    
print(f"Excluding {len(exclude_WGS_samples)} samples: {','.join(exclude_WGS_samples)} with discrepant lineages at the same timepoint")

df_trust_combined = df_trust_combined.query("SampleID not in @exclude_WGS_samples").reset_index(drop=True)

# do it for pids as well. If there > 2 unmixed discordant lineages, drop the ones that do not belong to the mode 
pids_check_multiple_lineages = pd.DataFrame(df_trust_combined.dropna(subset='Coll2014').query("~Coll2014.str.contains(',')").groupby('pid').Coll2014.nunique()).query("Coll2014 > 1").index.values

exclude_WGS_samples = []

for pid in pids_check_multiple_lineages:
    
    num_samples = len(df_trust_combined.dropna(subset='Coll2014').query("pid == @pid"))
    
    if num_samples > 2:
    
        major_lineage = df_trust_combined.dropna(subset='Coll2014').query("pid == @pid").Coll2014.mode().values[0]

        exclude_WGS_samples += list(df_trust_combined.dropna(subset='Coll2014').query("pid == @pid & Coll2014 != @major_lineage").SampleID.values)

print(f"Excluding {len(exclude_WGS_samples)} samples: {','.join(exclude_WGS_samples)} with lineages that don't agree with the other lineages for the same participants")
df_trust_combined = df_trust_combined.query("SampleID not in @exclude_WGS_samples").reset_index(drop=True)

# put the IDs at the front, then save
print(f"{df_trust_combined.pid.nunique()} patients with sequencing data")
df_trust_combined.set_index(['pid', 'Original_ID', 'SampleID']).to_csv(out_fName)