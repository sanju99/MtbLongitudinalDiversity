# Inferring Transmission Dynamics of L4 Samples using Phybreak

```python
TRUST_data_dir = "/home/sak0914/TRUST_data_processing/processed_data"

df_tx_outcomes = pd.read_csv(f"{TRUST_data_dir}/tx_outcomes.csv")
df_trust_patients = pd.read_csv(f"{TRUST_data_dir}/20250818_combined_patient_WGS_data.csv")

L4_samples = pd.DataFrame([(seq.id, seq.seq) for seq in SeqIO.parse("./data/trees/L4_samples/SNP_concatenate_filtered.fasta", "fasta")])

# 309 samples. Includes multiple WGS samples per patient. Each has 10,323 SNPs
len(L4_samples)

culture_date_cols = list(df_trust_patients.columns[(df_trust_patients.columns.str.contains('cultdate')) & (~df_trust_patients.columns.str.contains('additional'))])

L4_outcomes = df_tx_outcomes.merge(df_trust_patients.query("SampleID in @L4_samples[0]")[['pid', 'SampleID', 'Sampling_Week'] + culture_date_cols], on='pid')

# get the culture sampling date for each sample
for i, row in L4_outcomes.iterrows():
    
    sample_week = row['Sampling_Week']
    
    assert f"s_cultdate_sputum_specimen_{sample_week}" in L4_outcomes.columns
    
    L4_outcomes.loc[i, 'Sample_Date'] = row[f"s_cultdate_sputum_specimen_{sample_week}"]
    
# keep only the first WGS sample per patient
L4_outcomes_one_sample_per_pid = L4_outcomes[['pid', 'screen_date', 'SampleID', 'Sampling_Week', 'Sample_Date']].sort_values(['pid', 'Sample_Date']).drop_duplicates(subset='pid', keep='first').reset_index(drop=True)

# save so you can get the dates and import them into phybreak
L4_outcomes_one_sample_per_pid.to_csv("./transmission/L4_samples/dates.csv", index=False)

# write to new FASTA file
with open("./transmission/L4_samples/SNP_concatenate.fasta", "w+") as file:
    for i, row in L4_samples.iterrows():
        
        if row[0] in L4_outcomes_one_sample_per_pid.SampleID.values:
            file.write(f">{row[0]}\n")
            file.write(f"{str(row[1])}\n")
```

```bash
# drop SNPs that are the same everywhere. Kept 481/10,323 SNPs
cd ../scripts
python3 drop_constant_ambiguous_sites_FASTA.py -i ../transmission/L4_samples/SNP_concatenate.fasta -o ../transmission/L4_samples/SNP_concatenate_filtered.fasta

# 10,193 SNPs
python3 drop_constant_ambiguous_sites_FASTA.py -i ../transmission/L4_samples/SNP_concatenate_all.fasta -o ../transmission/L4_samples/SNP_concatenate_all_filtered.fasta
```

Phybreak is not available for R version 4.4.2, which was installed on O2 at the time of this analysis. Therefore, I installed it from Github:

```R
install.packages("remotes")

remotes::install_github("donkeyshot/phybreak")
```

The estimated mutation rate for Mtb is 0.5 SNPs/genome/year. L2 appears to be the fastest evolving, and others have used 0.3 SNPs/genome/year for L4, so start with that. Maybe use a log-normal distribution as a prior due to the uncertainty.

## Transphylo

```bash
cd /home/sak0914/MtbLongitudinalDiversity/transmission/L4_samples

treetime --aln SNP_concatenate_filtered.fasta --tree SNP_concatenate_filtered.nwk --dates dates.csv --name-column SampleID --date-column Sample_Date --max-iter 100 --keep-root --outdir treetime_results
```