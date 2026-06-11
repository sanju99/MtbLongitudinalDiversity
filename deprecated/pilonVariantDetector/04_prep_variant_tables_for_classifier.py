import numpy as np
import pandas as pd
import glob, os, argparse, re, sys
import warnings
from pathlib import Path
warnings.filterwarnings('ignore')
sys.path.append(
    str(Path(__file__).resolve().parent.parent / "variantDetector")
)

h37Rv_path = "~/MtbLongitudinalDiversity/H37Rv"
h37Rv_regions = pd.read_csv(os.path.join(h37Rv_path, "mycobrowser_h37rv_v4.csv"))

# remove rRNAs, which are highly conserved. rrs, rrl, and rrf
rRNA_pos = []

for i, row in h37Rv_regions.query("Functional_Category=='stable RNAs' & Feature=='rRNA'").iterrows():
    # print(row['Name'])
    rRNA_pos += list(np.arange(row['Start'], row['Stop'] + 1))
    
# remove insertion seqs and phages because too much short-read mismapping
insertion_seqs_phages_pos = []

for i, row in h37Rv_regions.query("Functional_Category=='insertion seqs and phages'").iterrows():
    insertion_seqs_phages_pos += list(np.arange(row['Start'], row['Stop'] + 1))
    
insertion_seqs_phages_pos = np.unique(insertion_seqs_phages_pos)
    
parser = argparse.ArgumentParser()

parser.add_argument("-o", dest='dir_name', type=str, required=True, help="Directory to write tables to. Will be created if it doesn't exist.")
parser.add_argument("-d", "--H37Rv_ref_dir", type=str, default='/n/data1/hms/dbmi/farhat/Sanjana/TRUST_lowAF', help='Directory with Illumina samples aligned to H37Rv')
parser.add_argument("-D", "--personal_ref_dir", type=str, default='/n/data1/hms/dbmi/farhat/Sanjana/TRUST_aln_personal_assembly', help='Directory with Illumina samples aligned to their own personal genomes')
parser.add_argument("--F2_max", type=float, default=0.03, help='Maximum F2 (inclusive) to keep samples for')
parser.add_argument("--fixed_thresh", type=float, default=0.95, help='AF above which a variant is considered fixed')
parser.add_argument("--patient_WGS_data", type=str, default='/home/sak0914/TRUST_data_processing/processed_data/combined_patient_WGS_data.csv', help='combined patient + WGS data to merge metadata in with')

cmd_line_args = parser.parse_args()
model_dir = cmd_line_args.dir_name
H37Rv_ref_dir = cmd_line_args.H37Rv_ref_dir
personal_ref_dir = cmd_line_args.personal_ref_dir
F2_max = cmd_line_args.F2_max
fixed_thresh = cmd_line_args.fixed_thresh

df_trust_patients = pd.read_csv(cmd_line_args.patient_WGS_data)

ground_truth_samples = os.listdir(personal_ref_dir)

num_ground_truth_samples = len(df_trust_patients.query("SampleID in @ground_truth_samples & F2 <= @F2_max"))

non_ground_truth_samples = set(os.listdir(H37Rv_ref_dir)) - set(ground_truth_samples)

num_non_ground_truth_samples = len(df_trust_patients.query("F2 <= @F2_max & SampleID in @non_ground_truth_samples"))

print(f"{num_ground_truth_samples} ground truth samples, {num_non_ground_truth_samples} validation samples")


def write_lowAF_variant_table(model_dir, fixed_thresh=0.95):

    os.makedirs(model_dir, exist_ok=True)

    if not os.path.isdir(H37Rv_ref_dir):
        raise ValueError(f"{H37Rv_ref_dir} is not an available directory")

    if not os.path.isdir(personal_ref_dir):
        raise ValueError(f"{personal_ref_dir} is not an available directory")

    SNP_data_files = glob.glob(f"{H37Rv_ref_dir}/*/pilon/lowAF_SNPs_aln_stats.csv")

    df_candidate_SNPs = []

    for fName in SNP_data_files:

        match = re.search(r'MFS-\d{1,3}', fName)

        df = pd.read_csv(fName).query("REF.str.len() == ALT.str.len()")        
        df['SampleID'] = match.group()
        df_candidate_SNPs.append(df)

    df_candidate_SNPs = pd.concat(df_candidate_SNPs).query("DP >= 5")

    ground_truth_SNP_data_files = glob.glob(f"{personal_ref_dir}/*/pilon/ground_truth.csv")
    samples_with_ground_truth = os.listdir(personal_ref_dir)

    df_ground_truth = []

    for fName in ground_truth_SNP_data_files:

        match = re.search(r'MFS-\d{1,3}', fName)

        df = pd.read_csv(fName)
        
        df['SampleID'] = match.group()
        df_ground_truth.append(df)

    df_ground_truth = pd.concat(df_ground_truth).merge(df_trust_patients[['pid', 'SampleID', 'Coll2014', 'Freschi2020', 'Lineage', 'F2']], on='SampleID', how='left').query("F2 <= @F2_max")
    
    assert sum(pd.isnull(df_ground_truth['pid'])) == 0
    df_ground_truth['Real'] = 1
    
    # remove rRNA pos, which have too many false positives due to contamination, and transposable elements and phages, which can integrate multiple times, making alns ambiguous
    df_ground_truth = df_ground_truth.query("POS not in @rRNA_pos & POS not in @insertion_seqs_phages_pos")
    df_candidate_SNPs = df_candidate_SNPs.query("POS not in @rRNA_pos & POS not in @insertion_seqs_phages_pos")
    
    # also exclude Rv2081c-Rv2082 region. We know they're all false variants, and they will swamp the signal
    # there are discordantly paired reads at the edges of the duplicated region, but most of the false unfixed SNVs are in the interior of the region
    # and they have fewer discordantly paired reads
    df_ground_truth = df_ground_truth.query("~(POS >= 2338065 & POS <= 2340874)")
    df_candidate_SNPs = df_candidate_SNPs.query("~(POS >= 2338065 & POS <= 2340874)")

    # weird characters in the columns
    df_candidate_SNPs.rename(columns={'ANN[0].GENE': 'GENE', 'ANN[0].HGVS_C': 'HGVS_C', 'ANN[0].HGVS_P': 'HGVS_P'}, inplace=True)

    # add pid
    df_candidate_SNPs = df_candidate_SNPs.merge(df_trust_patients[['pid', 'SampleID', 'Coll2014', 'Freschi2020', 'Lineage', 'F2']], on='SampleID', how='left').query("F2 <= @F2_max")
        
    assert sum(pd.isnull(df_candidate_SNPs['pid'])) == 0
    # df_candidate_SNPs['Lineage'] = df_candidate_SNPs['Lineage'].astype(int)
    
    print(f"{df_candidate_SNPs.SampleID.nunique()} WGS samples across {df_candidate_SNPs.pid.nunique()} pids")

    # merge the candidate SNPs from Illumina sequencing with the ground truth information from the hybrid assemblies, but only for samples with hybrid assemblies
    # only merge on POS because of differences in the minor allele relative to the personal assembly vs. H37Rv
    df_candidate_SNPs_ground_truth = df_candidate_SNPs.query("SampleID in @samples_with_ground_truth & AF <= @fixed_thresh").merge(df_ground_truth[['SampleID', 'POS', 'Real']], how='outer', on=['POS', 'SampleID'])

    # merged outer to keep everything, so anything that didn't have overlap between candidate SNPs and ground truth is not real
    df_candidate_SNPs_ground_truth['Real'] = df_candidate_SNPs_ground_truth['Real'].fillna(0).astype(int)

    # assert df_candidate_SNPs_ground_truth.Lineage.nunique() == 3

    # get a list of positions that are never Real
    # restrict to low AF sites because sometimes they are not considered real if they were highly fixed (AF > 0.95). They are not found as a variant in the personal ref genome
    # exclude mixed infections when computing this because we're using this variable on the validation data, which is only clonal infections
    pos_only_false_variants = pd.DataFrame(df_candidate_SNPs_ground_truth.query("F2 <= @F2_max").groupby('POS').Real.sum()).query("Real==0").index.values

    # require that they also occurred in at least 2 pids
    # pos_multiple_pids = pd.DataFrame(df_candidate_SNPs_ground_truth.query("F2 <= @F2_max & Real == 0").groupby('POS').pid.nunique()).query("pid > 1").index.values

    # pos_only_false_variants = list(set(pos_only_false_variants).intersection(pos_multiple_pids))
    print(f"Excluding {len(pos_only_false_variants)} additional positions: {np.sort(pos_only_false_variants)}")

    df_candidate_SNPs_ground_truth['False_in_all_Ground_Truth_Samples'] = df_candidate_SNPs_ground_truth['POS'].isin(pos_only_false_variants).astype(int)
    df_candidate_SNPs['False_in_all_Ground_Truth_Samples'] = df_candidate_SNPs['POS'].isin(pos_only_false_variants).astype(int)

    # ------------------------------------------------------------------------------------------------------------------------
    # Build model using the samples with ground truth information
    # ------------------------------------------------------------------------------------------------------------------------
    predictors = ["COV_RATIO", "CLIPPED_BASES_RATIO", "DISCORDANT_READS_RATIO", "Mean_BQ_ALT_allele", "SAF_prop_deviation_from_half", 'VariantSupportMedianIndex', 'Soft_Clipped_Read_Support']

    # NAs are false negatives. Print them as sanity checks. The two that have been found so far are variants ~95%/5%, and the H37Rv calls are <5%, so they get filtered out
    # basically a fixed variant though, so not really a false negative
    print(df_candidate_SNPs_ground_truth.loc[pd.isnull(df_candidate_SNPs_ground_truth['COV_RATIO'])]['SampleID'].value_counts())

    # assert len(df_candidate_SNPs_ground_truth.dropna(subset=predictors)) == len(df_candidate_SNPs_ground_truth)

    df_candidate_SNPs_ground_truth = df_candidate_SNPs_ground_truth.dropna(subset=predictors)

    df_val = df_candidate_SNPs.query("SampleID not in @samples_with_ground_truth")
    assert len(df_val.dropna(subset=predictors)) == len(df_val)

    # make sure all numeric types
    df_candidate_SNPs_ground_truth[predictors + ['Real']] = df_candidate_SNPs_ground_truth[predictors + ['Real']].apply(pd.to_numeric, errors='coerce')

    df_candidate_SNPs_ground_truth = df_candidate_SNPs_ground_truth.query("AF <= @fixed_thresh").drop_duplicates()
    
    print(f"{df_candidate_SNPs_ground_truth.SampleID.nunique()} training samples with {len(df_candidate_SNPs_ground_truth)} low-AF variants")
    print(df_candidate_SNPs_ground_truth.Real.value_counts())
    
    df_candidate_SNPs_ground_truth.to_csv(f"{model_dir}/training_data.csv", index=False)

    # save all values to get predictions
    df_val = df_val.query("AF <= @fixed_thresh").query("F2 <= @F2_max").drop_duplicates()
    print(f"{df_val.SampleID.nunique()} validation samples with {len(df_val)} low-AF variants")
    
    df_val.to_csv(f"{model_dir}/validation_data.csv", index=False)

    print(f"{df_candidate_SNPs.SampleID.nunique()} samples with lowAF SNPs")
    
    
    
def write_fixed_variable_table(model_dir, fixed_thresh=0.95):
    
    df_fixed_SNPs = []

    fixed_SNP_data_files = glob.glob(f"{H37Rv_ref_dir}/*/pilon/fixed_SNPs.csv")

    for fName in fixed_SNP_data_files:

        match = re.search(r'MFS-\d{1,3}', fName)

        df = pd.read_csv(fName)  
        df['SampleID'] = match.group()
        df_fixed_SNPs.append(df)

    df_fixed_SNPs = pd.concat(df_fixed_SNPs).query("AF > @fixed_thresh").query("POS not in @rRNA_pos & POS not in @insertion_seqs_phages_pos")

    df_fixed_SNPs = df_fixed_SNPs.merge(df_trust_patients[['pid', 'SampleID', 'Coll2014', 'Freschi2020', 'Lineage', 'F2']], on='SampleID', how='left').query("F2 <= @F2_max")
    assert sum(pd.isnull(df_fixed_SNPs['pid'])) == 0

    df_fixed_SNPs.rename(columns={'ANN[0].GENE': 'GENE', 'ANN[0].HGVS_C': 'HGVS_C', 'ANN[0].HGVS_P': 'HGVS_P'}).drop_duplicates().to_csv(f"{model_dir}/fixed_variants.csv.gz", index=False, compression='gzip')
    
    
print(f"Fixed variant = AF > {fixed_thresh}")
print(f"Keeping only WGS samples with F2 ≤ {F2_max}")
    
write_lowAF_variant_table(model_dir, fixed_thresh=fixed_thresh)
# write_fixed_variable_table(model_dir, fixed_thresh=fixed_thresh)