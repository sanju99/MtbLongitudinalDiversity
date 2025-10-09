import pymc as pm
import numpy as np
import pandas as pd
import sklearn.model_selection
import argparse, os, glob, re, warnings
import arviz as az
warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser()

parser.add_argument("-d1", dest='lowAF_directory', type=str, default='/n/data1/hms/dbmi/farhat/Sanjana/TRUST_lowAF', help='Directory in which candidate SNP files are')
parser.add_argument("-d2", dest='ground_truth_directory', type=str, default='/n/data1/hms/dbmi/farhat/Sanjana/TRUST_aln_personal_assembly', help='Directory that contains the ground truth variants for the samples with personal reference genomes')
parser.add_argument("-o", dest='output_dir', type=str, help='Output directory model files')

df_trust_patients = pd.read_csv("~/TRUST_data_processing/processed_data/20250904_cleaned_patient_outcomes_data.csv")
unmixed_lineage_samples = df_trust_patients.query("F2 <= 0.03").SampleID.unique()

cmd_line_args = parser.parse_args()

lowAF_directory = cmd_line_args.lowAF_directory
ground_truth_directory = cmd_line_args.ground_truth_directory
output_dir = cmd_line_args.output_dir

if not os.path.isdir(lowAF_directory):
    raise ValueError(f"{lowAF_directory} is not an available directory")
    
if not os.path.isdir(ground_truth_directory):
    raise ValueError(f"{ground_truth_directory} is not an available directory")

SNP_data_files = glob.glob(f"{lowAF_directory}/*/Bayesian_detector/lowAF_SNPs.csv")

df_candidate_SNPs = []

for fName in SNP_data_files:
    
    match = re.search(r'MFS-\d{1,3}', fName)
    
    if match.group() in unmixed_lineage_samples:
        
        df = pd.read_csv(fName).query("REF.str.len() == ALT.str.len()")        
        df['SampleID'] = match.group()
        df_candidate_SNPs.append(df)
    
df_candidate_SNPs = pd.concat(df_candidate_SNPs).drop_duplicates(subset=['SampleID', 'CHROM', 'POS', 'REF', 'ALT'])

# these are the samples with ground truth data (hybrid assemblies)
df_personal_assemblies = pd.read_csv("../data/personal_assemblies_samples.tsv", sep='\t', header=None)

samples_with_ground_truth = df_personal_assemblies[0].values

ground_truth_SNP_data_files = glob.glob(f"{ground_truth_directory}/*/lowAF_comparison/ground_truth.csv")

df_ground_truth = []

for fName in ground_truth_SNP_data_files:
    
    match = re.search(r'MFS-\d{1,3}', fName)
    
    if match.group() in unmixed_lineage_samples:
        
        df = pd.read_csv(fName).query("REF.str.len() == ALT.str.len()")
        df['SampleID'] = match.group()
        df_ground_truth.append(df)
    
df_ground_truth = pd.concat(df_ground_truth)
df_ground_truth['Real'] = 1

# merge the candidate SNPs from Illumina sequencing with the ground truth information from the hybrid assemblies, but only for samples with hybrid assemblies
# only merge on POS because of differences in the minor allele relative to the personal assembly vs. H37Rv
df_candidate_SNPs_ground_truth = df_candidate_SNPs.query("SampleID in @samples_with_ground_truth").merge(df_ground_truth[['SampleID', 'POS', 'Real']], how='outer', on=['POS', 'SampleID'])

# merged outer to keep everything, so anything that didn't have overlap between candidate SNPs and ground truth is not real
df_candidate_SNPs_ground_truth['Real'] = df_candidate_SNPs_ground_truth['Real'].fillna(0).astype(int)

# save training and validation data
df_candidate_SNPs_ground_truth.to_csv(f"{output_dir}/training_data.csv", index=False)
df_candidate_SNPs.query("SampleID not in @samples_with_ground_truth").to_csv(f"{output_dir}/validation_data.csv", index=False)

# # ------------------------------------------------------------------------------------------------------------------------
# # Build model using the samples with ground truth information
# # ------------------------------------------------------------------------------------------------------------------------
# predictors = ["COV_RATIO", "CLIPPED_BASES_RATIO", "DISCORDANT_READS_RATIO", "Mean_BQ_ALT_allele"]

# # NAs are false negatives. Print them as sanity checks. The two that have been found so far are variants ~95%/5%, and the H37Rv calls are <5%, so they get filtered out
# # basically a fixed variant though, so not really a false negative
# print(df_candidate_SNPs_ground_truth.loc[pd.isnull(df_candidate_SNPs_ground_truth['COV_RATIO'])]['SampleID'].value_counts())

# # assert len(df_candidate_SNPs_ground_truth.dropna(subset=predictors)) == len(df_candidate_SNPs_ground_truth)

# df_candidate_SNPs_ground_truth = df_candidate_SNPs_ground_truth.dropna(subset=predictors)

# # make sure all numeric types
# df_candidate_SNPs_ground_truth[predictors + ['Real']] = df_candidate_SNPs_ground_truth[predictors + ['Real']].apply(pd.to_numeric, errors='coerce')

# X = df_candidate_SNPs_ground_truth[predictors].values 
# y = df_candidate_SNPs_ground_truth["Real"].values

# # Train-test split, stratifying by real or false variants
# # X_train, X_val, y_train, y_val = sklearn.model_selection.train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

# if not os.path.isfile("train_idx.npy"):
#     train_idx, _ = sklearn.model_selection.train_test_split(np.arange(X.shape[0]), test_size=0.2, random_state=42, stratify=y)

#     np.save("train_idx.npy", train_idx)
    
# train_idx = np.load("train_idx.npy")

# X_train = X[train_idx]
# y_train = y[train_idx]

# X_val = np.delete(X, train_idx, axis=0)
# y_val = np.delete(y, train_idx, axis=0)

# # scale X_train. Save mean and standard deviation to scale X_val for later validation
# X_train_mean = X_train.mean(axis=0)
# X_train_sd = X_train.std(axis=0)

# np.save("X_train_mean.npy", X_train_mean)
# np.save("X_train_sd.npy", X_train_sd)

# # standard scale
# X_train = (X_train - X_train_mean) / X_train_sd

# print(f"Train: N = {len(y_train)}, Real % = {np.round(y_train.mean()*100, 2)}")
# print(f"Validation: N = {len(y_val)}, Real % = {np.round(y_val.mean()*100, 2)}")

# # ------------------------------------------------------------------------------------------------------------------------
# # Bayesian logistic regression model
# # ------------------------------------------------------------------------------------------------------------------------
# def create_model_instance(X_train, y_train):
    
#     with pm.Model() as model:
#         # Priors for betas (directional expectations)
#         beta_cov = pm.Normal("beta_cov", mu=0.5, sigma=2)       # coverage ratio (positive)
#         beta_soft = pm.Normal("beta_soft", mu=-0.5, sigma=2)    # soft clips (negative)
#         beta_disc = pm.Normal("beta_disc", mu=-0.5, sigma=2)    # discordant reads (negative)
#         beta_bq = pm.Normal("beta_bq", mu=0.5, sigma=2)         # base quality (positive)

#         # Intercept
#         intercept = pm.Normal("intercept", mu=0, sigma=2)

#         # Linear predictor (logit scale)
#         logit_p = (
#             intercept
#             + beta_cov * X_train[:, 0]
#             + beta_soft * X_train[:, 1]
#             + beta_disc * X_train[:, 2]
#             + beta_bq * X_train[:, 3]
#         )

#         # Probability via logistic link
#         p = pm.Deterministic("p", pm.math.sigmoid(logit_p))

#         # Likelihood
#         y_obs = pm.Bernoulli("y_obs", p=p, observed=y_train)
        
#     return model

        
# model = create_model_instance(X_train, y_train)

# if not os.path.isfile("bayesian_model_trace.nc"):

#     with model:
#         # ------------------------
#         # Sampling
#         # ------------------------
#         trace = pm.sample(2000, tune=1000, target_accept=0.9, random_seed=42)

#         # Save full trace (posterior, log-likelihoods, etc.)
#         az.to_netcdf(trace, "bayesian_model_trace.nc")

# trace = az.from_netcdf(f"{output_dir}/bayesian_model_trace.nc")

# # save just the posterior draws
# trace.posterior.to_dataframe().to_csv(f"{output_dir}/posterior_samples.csv")

# # save the summary
# summary = pm.summary(trace, hdi_prob=0.95)
# summary.to_csv(f"{output_dir}/posterior_summary.csv")