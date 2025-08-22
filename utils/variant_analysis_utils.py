import numpy as np
import pandas as pd
import glob, os, warnings, shutil, subprocess, re, sys
from Bio import Seq, SeqIO
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as st
import scipy

h37Rv_path = "/n/data1/hms/dbmi/farhat/Sanjana/H37Rv"
h37Rv_seq = SeqIO.read(os.path.join(h37Rv_path, "GCF_000195955.2_ASM19595v2_genomic.gbff"), "genbank")
h37Rv_genes = pd.read_csv(os.path.join(h37Rv_path, "mycobrowser_h37rv_genes_v4.csv"))
h37Rv_regions = pd.read_csv(os.path.join(h37Rv_path, "mycobrowser_h37rv_v4.csv"))

# these are promoters, transcriptional signals, or RNAs. Exclude these
# non_coding_regions = h37Rv_regions.query("Feature != 'CDS'").Name.values
promoter_transcriptional_signals = h37Rv_regions.query("Feature in ['promoter', '-10_signal', '-35_signal']").Name.values

# if these remain in the dataframe, then there will be multiple entires for a single gene name, which will cause process_intergenic_variant_WHO_catalog_coord to fail
h37Rv_regions = h37Rv_regions.query("~Feature.str.contains('|'.join(['promoter', 'signal']), case=False)")
assert len(h37Rv_regions) == h37Rv_regions.Name.nunique()

# h37Rv_regions = h37Rv_regions.query("Feature == 'CDS'").reset_index(drop=True)
h37Rv_coords = pd.read_csv(os.path.join(h37Rv_path, "h37Rv_coords_to_gene.csv"))
h37Rv_coords_dict = dict(zip(h37Rv_coords["pos"].values, h37Rv_coords["region"].values))

silent_lst = ['synonymous_variant', 'initiator_codon_variant', 'stop_retained_variant']


def find_HT_regions(seq):
    '''
    Find homopolymeric tracts (HTs) in the H37Rv reference genome. These are runs of the same nucleotide, probably up to 7 bp in length.
    '''

    assert type(seq) == str
    
    # convert the string to an array
    seq_array = np.array(list(seq))

    change_points = np.where(seq_array[:-1] != seq_array[1:])[0] + 1

    starts = np.insert(change_points, 0, 0)  # Start positions
    ends = np.append(change_points, len(seq_array))  # End positions

    # the values in starts are 0-indexed, so add 1 to get coordinates
    df_runs = pd.DataFrame([(seq_array[s], e - s, s + 1) for s, e in zip(starts, ends)])
    df_runs.columns = ['NT', 'Length', 'POS']

    # a run of 1 isn't really a run, so exclude
    df_runs = df_runs.query("Length > 1")
    
    return df_runs


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



def convert_H37Rv_upstream_coordinate(pos, gene):
    '''
    Use this only if POS in HGVS_C (so not a protein-changing variant)
    '''

    start, end, sense = h37Rv_regions.query("Name==@gene")[['Start', 'Stop', 'Strand']].values[0]

    if type(pos) == str:
        pos = int(pos)

    # must be intergenic, so should be outside the gene bounds
    if sense == '+':
        upstream_dist = start - pos
    else:
        upstream_dist = pos - end

    # could be negative if a variant starts upstream of the gene but then extends into the gene
    # assert upstream_dist > 0
    return -upstream_dist



def reverse_complement(seq):
    complement = {'A': 'T', 'T': 'A', 'C': 'G', 'G': 'C'}
    return ''.join(complement[base] for base in reversed(seq))




def process_intergenic_variant_WHO_catalog_coord(row):

    pos = row['POS']
    ref = row['REF']
    alt = row['ALT']
    nt_change = row['HGVS_C']
    
    gene_1, gene_2 = row['GENE'].split('-')

    # sometimes inconsistencies here, so change the prefix just to be safe
    if gene_1 in ['rrs', 'rrl'] or gene_2 in ['rrs', 'rrl']:
        letter_prefix = 'n'
    else:
        letter_prefix = 'c'
        
    # gene_1 is the gene with smaller coordinates. gene_2 is larger
    gene_1 == h37Rv_regions.query("Stop <= @pos")['Name'].values[-1]
    
    # this means that CHR_END is in the gene name
    if len(h37Rv_regions.query("Stop >= @pos")) == 0:
        gene_2 = gene_1
    else:
        gene_2 == h37Rv_regions.query("Stop >= @pos")['Name'].values[0]

    # if negative sense, then the downstream gene is the one that comes earlier in H37Rv coordinate space
    gene_1_data = h37Rv_regions.query("Name==@gene_1 | Locus==@gene_1")
    
    if len(gene_1_data) > 1:
        gene_1_data = gene_1_data.query("Feature=='CDS'")
        assert len(gene_1_data) == 1
        
    gene_2_data = h37Rv_regions.query("Name==@gene_2 | Locus==@gene_2")
    
    if len(gene_2_data) > 1:
        gene_2_data = gene_2_data.query("Feature=='CDS'")
        assert len(gene_2_data) == 1
    
    sense_1 = gene_1_data['Strand'].values[0]
    sense_2 = gene_2_data['Strand'].values[0]

    # downstream of both genes, so probably doesn't affect their transcription
    if sense_1 == '+' and sense_2 == '-':
        return None

    # first remove the letter prefix. Shouldn't be p. in here, but do it anyway
    nt_change = nt_change.replace('n.', '').replace('c.', '').replace('p.', '')

    # remove the position from the nt_change variable
    # the only numbers should be the coordinate
    str_pos = ''.join([char for char in nt_change if char.isdigit()])

    # variant is on a single position
    if str_pos in nt_change:
        
        nt_change = nt_change.replace(str_pos, '')

        # genes are in the same sense direction, so return upstream of the later gene
        if sense_1 == sense_2:

            if sense_1 == '+':
                primary_gene = gene_2
            else:
                primary_gene = gene_1
                
            new_coord = convert_H37Rv_upstream_coordinate(str_pos, primary_gene)

            if sense_1 == '+':
                return f"{primary_gene}_{letter_prefix}.{new_coord}{nt_change}"
            else:
                return reverse_complement_variant(f"{primary_gene}_{letter_prefix}.{new_coord}{nt_change}")
    
        # upstream of both genes, so need to return the variant relative to both genes
        elif sense_1 == '-' and sense_2 == '+':

            new_coord_gene_1 = convert_H37Rv_upstream_coordinate(str_pos, gene_1)
            new_coord_gene_2 = convert_H37Rv_upstream_coordinate(str_pos, gene_2)
            
            # return a comma-separated string with both mutations -- will split them into two lines later
            # only reverse complement the first variant because it's negative sense
            final_variant_1 = reverse_complement_variant(f"{gene_1}_{letter_prefix}.{new_coord_gene_1}{nt_change}")
            
            return f"{final_variant_1},{gene_2}_{letter_prefix}.{new_coord_gene_2}{nt_change}"
            
    # this means that there is a range of positions, and the variant needs to be of the form rpoA_c.-11_-4delATGCATGC
    else:
        # this will be the form for insertions, deletions, and duplications
        assert '_' in nt_change
        assert len(nt_change.split('_')) == 2
    
        # note that the nucleotide change will be on the second coordinate
        pos_1, pos_2 = nt_change.split('_')

        # then remove that underscore spacer between the two positions. Already verified that there's only one by checking that length of the split is 2
        nt_change = nt_change.replace('_', '')
    
        str_pos_1 = ''.join([char for char in pos_1 if char.isdigit()])
        str_pos_2 = ''.join([char for char in pos_2 if char.isdigit()])
    
        assert str_pos_1 in nt_change
        assert str_pos_2 in nt_change
    
        # replace both strings with the empty string
        nt_change = nt_change.replace(str_pos_1, '').replace(str_pos_2, '')

        if sense_1 == sense_2:
            
            if sense_2 == '+':
                primary_gene = gene_2
            else:
                primary_gene = gene_1
        
            new_coord_pos_1 = convert_H37Rv_upstream_coordinate(str_pos_1, primary_gene)
            new_coord_pos_2 = convert_H37Rv_upstream_coordinate(str_pos_2, primary_gene)

            # order them in increasing order of the upstream coordinate (not the H37Rv coordinate), which for negative sense genes will be the opposite order
            if sense_2 == '+':
                return f"{primary_gene}_{letter_prefix}.{new_coord_pos_1}_{new_coord_pos_2}{nt_change}"
            else:
                return reverse_complement_variant(f"{primary_gene}_{letter_prefix}.{new_coord_pos_2}_{new_coord_pos_1}{nt_change}")
    
        # upstream of both genes, so need to return the variant relative to both genes
        elif sense_1 == '-' and sense_2 == '+':

            new_coord_pos_1_gene_1 = convert_H37Rv_upstream_coordinate(str_pos_1, gene_1)
            new_coord_pos_2_gene_1 = convert_H37Rv_upstream_coordinate(str_pos_2, gene_1)
            
            new_coord_pos_1_gene_2 = convert_H37Rv_upstream_coordinate(str_pos_1, gene_2)
            new_coord_pos_2_gene_2 = convert_H37Rv_upstream_coordinate(str_pos_2, gene_2)

            # return a comma-separated string with both mutations -- will split them into two lines later
            # order them in increasing order of the upstream coordinate (not the H37Rv coordinate), which for negative sense genes will be the opposite order
            if sense_2 == '+':
                return f"{gene_1}_{letter_prefix}.{new_coord_pos_1_gene_1}_{new_coord_pos_2_gene_1}{nt_change},{gene_2}_{letter_prefix}.{new_coord_pos_1_gene_2}_{new_coord_pos_2_gene_2}{nt_change}"

            # reverse complement
            else:
                final_variant_1 = reverse_complement_variant(f"{gene_1}_{letter_prefix}.{new_coord_pos_2_gene_1}_{new_coord_pos_1_gene_1}{nt_change}")
                final_variant_2 = reverse_complement_variant(f"{gene_2}_{letter_prefix}.{new_coord_pos_2_gene_2}_{new_coord_pos_1_gene_2}{nt_change}")
                
                return final_variant_1, final_variant_2



def reverse_complement_variant(variant):

    # separate the gene from the rest of the variant so that no letters in the gene name are altered
    gene, mutation = variant.split('_', maxsplit=1)
    
    # Split the mutation into components using a regular expression
    # This pattern assumes nucleotide sequences are in uppercase letters
    components = re.split(r'([ATGC]+)', mutation)
    
    # Iterate through the components and reverse complement the nucleotide sequences
    for i, component in enumerate(components):
        if re.match(r'^[ATGC]+$', component):
            components[i] = reverse_complement(component)
    
    # Join the components back together
    reversed_mutation = ''.join(components)
    return gene + '_' + reversed_mutation





def get_fixed_variants(fName, df_pid_metadata, AF_thresh=0.75, num_support_each_direction=5, MQ_thresh=40):
        
    df_variants = pd.read_csv(fName)
    df_variants['SampleID'] = os.path.basename(fName).split('.')[0]
    
    # these are upper bounds on the probability of observing the observed deviation between SRF and SRR or between SAF and SAR, given that the expected probability 
    df_variants['SRP_prob'] = 10**(-df_variants['SRP']/10)
    df_variants['SAP_prob'] = 10**(-df_variants['SAP']/10)
    
    # when SRF = SRR = 0, there are no reads supporting the reference. If quality is 0, then the error probability is 1, which isn't true. They're actually NA
    df_variants.loc[(df_variants['SRF']==0) | (df_variants['SRR']==0), 'SRP_prob'] = np.nan
    
    # keep only fixed variants and add metadata. SAP_prob > 0.05
    df_variants = df_variants.query("AF > @AF_thresh & SAF >= @num_support_each_direction & SAR >= @num_support_each_direction & MQM >= @MQ_thresh").merge(df_pid_metadata[['pid', 'SampleID', 'Sampling_Week', 'Paired_Sample_Num', 'F2', 'Lineage', 'Coll2014']], on='SampleID')
    
    # sort for readability
    df_variants = df_variants.sort_values(["pid", 'Sampling_Week']).set_index(["pid", 'SampleID', 'Sampling_Week', 'Paired_Sample_Num']).reset_index()

    # add homopolymeric tract annotations
    df_variants['HT'] = df_variants['POS'].isin(HT_nucs).astype(int)
    
    # set to NA here so that they remain NA if never filled in, instead of being converted to the string "nan"
    df_variants[['variant', 'Diff_NT', 'Unique_Change_NT', 'Phase_Variant', 'SNP']] = np.nan

    # to be a phase variant, the difference between REF and ALT must be only one type of nucleotide. Otherwise, it's just a regular indel
    # there are some cases where a large frameshift encompasses a homopolymeric tract, but it's not a phase variation because many more nucleotides were included in that variant
    for i, row in df_variants.iterrows():

        # first, make names easier to read for the intergenic variants. Unrelated to the phase variant issue but need to iterate for this too
        if '-' in row['GENE']:
            
            new_variant = process_intergenic_variant_WHO_catalog_coord(row)
        
            # returns None if an intergenic variant is not upstream of either flanking gene. Don't change these because there's no naming convention when they occur downstream
            if not pd.isnull(new_variant):
                # some of these may have commas in them if the variants occur upstream of two genes
                df_variants.loc[i, 'variant'] = new_variant
            else:
                # just combine with HGVS_C. This is the case when the variant is intergenic but downstream of both flanking genes
                df_variants.loc[i, 'variant'] = row['GENE'] + '_' + row['HGVS_C']
        
        # HGVS_P will be NA for non-coding variants
        elif row['EFFECT'] in silent_lst or pd.isnull(row['HGVS_P']):
            df_variants.loc[i, 'variant'] = row['GENE'] + '_' + row['HGVS_C']
            
        else:
            df_variants.loc[i, 'variant'] = row['GENE'] + '_' + row['HGVS_P']

        ref = row['REF']
        alt = row['ALT']

        # deletion, so remove ALT from the right side of REF
        if len(ref) > len(alt):
            # replace only 1 instance
            diff_nucs = ref.replace(alt, "", 1)

        # insertion, so remove REF from the left side of ALT
        elif len(alt) > len(ref):
            # replace only 1 instance
            diff_nucs = alt.replace(ref, "", 1)

        # ignore SNPs
        else:
            continue

        df_variants.loc[i, ['Diff_NT', 'Unique_Change_NT']] = [diff_nucs, len(np.unique(list(diff_nucs)))]

    # annotate phase variants
    df_variants.loc[(df_variants['HT']==1) & (df_variants['Unique_Change_NT']==1) & (df_variants.REF.str.len() != df_variants.ALT.str.len()), 'Phase_Variant'] = 1
    df_variants['Phase_Variant'] = df_variants['Phase_Variant'].fillna(0).astype(int)

    df_variants.loc[(df_variants['REF'].str.len() == df_variants['ALT'].str.len()), 'SNP'] = 1
    df_variants['SNP'] = df_variants['SNP'].fillna(0).astype(int)

    # check that there is a value for variant in every row
    assert sum(pd.isnull(df_variants['variant'])) == 0
    
    return df_variants

    
    
    
    
def get_unfixed_variants(fName, df_pid_metadata, AF_thresh=0.05, AF_max=0.75, num_support_each_direction=2, MQ_thresh=40):
        
    df_variants = pd.read_csv(fName)
    df_variants['SampleID'] = os.path.basename(fName).split('.')[0]
    
    # the one weird variant that didn't get annotated in MFS-618. All other variants are annotated
    # it's right between two genes (which are separated by only one nucleotide, 3977061). Not sure why it didn't get annotated as intergenic.
    if 'MFS-618' in fName:
        df_variants['ANN'] = df_variants['ANN'].fillna('')
        df_variants.loc[pd.isnull(df_variants['ANN']), 'GENE'] = 'Rv3537-Rv3538'
        df_variants.loc[pd.isnull(df_variants['ANN']), 'EFFECT'] = 'intergenic_region'
        df_variants.loc[pd.isnull(df_variants['ANN']), 'HGVS_C'] = 'n.3977061T>C'
        
    high_homology_genes = ["clpB", "hsp", "rpoB", "rpoC", "tuf", "rpsC", "aspT", "rrs", "rrl"]
    df_variants = df_variants.query("~GENE.str.contains('|'.join(@high_homology_genes))")
    
    ###### TODO: ADD ANNOTATIONS FOR POSSIBLE FALSE LOW AF VARIANTS USING THE COVERAGE AND SNP DENSITY ROLLING AVERAGES ######
    
    # these are upper bounds on the probability of observing the observed deviation between SRF and SRR or between SAF and SAR, given that the expected probability 
    df_variants['SRP_prob'] = 10**(-df_variants['SRP']/10)
    df_variants['SAP_prob'] = 10**(-df_variants['SAP']/10)
    
    # when SRF = SRR = 0, there are no reads supporting the reference. If quality is 0, then the error probability is 1, which isn't true. They're actually NA
    df_variants.loc[(df_variants['SRF']==0) | (df_variants['SRR']==0), 'SRP_prob'] = np.nan
    
    # keep only unfixed variants and add metadata. SAP_prob > 0.05
    df_variants = df_variants.query("AF >= @AF_thresh & AF <= @AF_max & SAF >= @num_support_each_direction & SAR >= @num_support_each_direction & MQM >= @MQ_thresh").merge(df_pid_metadata[['pid', 'SampleID', 'Sampling_Week', 'Paired_Sample_Num', 'F2', 'Lineage', 'Coll2014']], on='SampleID')
    
    # sort for readability
    df_variants = df_variants.sort_values(['Sampling_Week', 'POS']).set_index(["pid", 'SampleID', 'Sampling_Week', 'Paired_Sample_Num']).reset_index()

    # add homopolymeric tract annotations
    df_variants['HT'] = df_variants['POS'].isin(HT_nucs).astype(int)
    
    # set to NA here so that they remain NA if never filled in, instead of being converted to the string "nan"
    df_variants[['variant', 'Diff_NT', 'Unique_Change_NT', 'Phase_Variant', 'SNP']] = np.nan

    # to be a phase variant, the difference between REF and ALT must be only one type of nucleotide. Otherwise, it's just a regular indel
    # there are some cases where a large frameshift encompasses a homopolymeric tract, but it's not a phase variation because many more nucleotides were included in that variant
    for i, row in df_variants.iterrows():

        # first, make names easier to read for the intergenic variants. Unrelated to the phase variant issue but need to iterate for this too
        if '-' in row['GENE']:
            
            new_variant = process_intergenic_variant_WHO_catalog_coord(row)

            # returns None if an intergenic variant is not upstream of either flanking gene. Don't change these because there's no naming convention when they occur downstream
            if not pd.isnull(new_variant):
                # some of these may have commas in them if the variants occur upstream of two genes
                df_variants.loc[i, 'variant'] = new_variant
            else:
                # just combine with HGVS_C. This is the case when the variant is intergenic but downstream of both flanking genes
                df_variants.loc[i, 'variant'] = row['GENE'] + '_' + row['HGVS_C']
        
        # HGVS_P will be NA for non-coding variants
        elif row['EFFECT'] in silent_lst or pd.isnull(row['HGVS_P']):
            df_variants.loc[i, 'variant'] = row['GENE'] + '_' + row['HGVS_C']
            
        else:
            df_variants.loc[i, 'variant'] = row['GENE'] + '_' + row['HGVS_P']

        ref = row['REF']
        alt = row['ALT']

        # deletion, so remove ALT from the right side of REF
        if len(ref) > len(alt):
            # replace only 1 instance
            diff_nucs = ref.replace(alt, "", 1)

        # insertion, so remove REF from the left side of ALT
        elif len(alt) > len(ref):
            # replace only 1 instance
            diff_nucs = alt.replace(ref, "", 1)

        # ignore SNPs
        else:
            continue

        df_variants.loc[i, ['Diff_NT', 'Unique_Change_NT']] = [diff_nucs, len(np.unique(list(diff_nucs)))]

    # annotate phase variants
    df_variants.loc[(df_variants['HT']==1) & (df_variants['Unique_Change_NT']==1) & (df_variants.REF.str.len() != df_variants.ALT.str.len()), 'Phase_Variant'] = 1
    df_variants['Phase_Variant'] = df_variants['Phase_Variant'].fillna(0).astype(int)

    df_variants.loc[(df_variants['REF'].str.len() == df_variants['ALT'].str.len()), 'SNP'] = 1
    df_variants['SNP'] = df_variants['SNP'].fillna(0).astype(int)
    
    # finally, annotate low quality unfixed variants using the coverage and SNP density information
    if not os.path.isfile(f"{sample_dir}/{sample}/freebayes/coverage_plateau_sites.npy") or not os.path.isfile(f"{sample_dir}/{sample}/freebayes/high_density_SNP_sites.npy"):
        raise ValueError(f"Low quality site annotations have not yet finished for {sample}")

    coverage_plateau_sites = np.load(f"{sample_dir}/{sample}/freebayes/coverage_plateau_sites.npy")
    high_density_SNP_sites = np.load(f"{sample_dir}/{sample}/freebayes/high_density_SNP_sites.npy")

    # only annotate indels in this way
    df_variants.loc[(df_variants['REF'].str.len() != df_variants['ALT'].str.len()) & ((df_variants['POS'].isin(coverage_plateau_sites)) | (df_variants['POS'].isin(high_density_SNP_sites))), 'Low_Qual'] = 1

    df_variants['Low_Qual'] = df_variants['Low_Qual'].fillna(0).astype(int)

    # check that there is a value for variant in every row
    assert sum(pd.isnull(df_variants['variant'])) == 0
    
    return df_variants





def get_matrix_of_HT_indels(data_dir, pids_for_analysis, HT_nucs, sample_2=True, binarize_fixed_variants=False, fixed_thresh=0.95, absent_thresh=0.05):

    # previously annotated phase variants in homopolymeric tracts
    df_variants = pd.concat([pd.read_csv(f"{data_dir}/fixed_variants.csv"), pd.read_csv(f"{data_dir}/unfixed_variants.csv")]).query("Phase_Variant==1")
    
    # exclude variants in regions that are promoters, or transcriptional signals
    # df_variants = df_variants.query("GENE not in @non_coding_regions").reset_index(drop=True)
            
    df_variants = df_variants.merge(pids_for_analysis[['SampleID', 'Sampling_Week', 'Paired_Sample_Num']], how='inner', on='SampleID')

    # get all samples and variants so that we can add in zeros if needed
    all_indels = df_variants.sort_values('POS')['variant'].unique()
    all_samples = pids_for_analysis.pid.unique() 
        
    sample_1_variants = df_variants.query("Paired_Sample_Num==1").pivot(index='pid', columns='variant', values='AF').fillna(0)

    # add in any missing indels
    sample_1_variants = sample_1_variants.merge(pd.DataFrame(columns = list(set(all_indels) - set(sample_1_variants.columns)), index=sample_1_variants.index.values).fillna(0), left_index=True, right_index=True)

    # add in any missing samples
    sample_1_variants = pd.concat([sample_1_variants, pd.DataFrame(columns = all_indels, index=list(set(all_samples) - set(sample_1_variants.index.values))).fillna(0)])
    
    # ensure same row and column ordering
    sample_1_variants = sample_1_variants.loc[all_samples, all_indels]
        
    if binarize_fixed_variants:
        # replace fixed variant AFs with 1
        sample_1_variants[sample_1_variants >= fixed_thresh] = 1
        
    # these shouldn't be there because we kept only unfixed variants with AF >= 0.5, but check anyway
    sample_1_variants[sample_1_variants < absent_thresh] = 0
    
    if sample_2:
        sample_2_variants = df_variants.query("Paired_Sample_Num==2").pivot(index='pid', columns='variant', values='AF').fillna(0)
    
        sample_2_variants = sample_2_variants.merge(pd.DataFrame(columns = list(set(all_indels) - set(sample_2_variants.columns)), index=sample_2_variants.index.values).fillna(0), left_index=True, right_index=True)

        sample_2_variants = pd.concat([sample_2_variants, pd.DataFrame(columns = all_indels, index=list(set(all_samples) - set(sample_2_variants.index.values))).fillna(0)])

        sample_2_variants = sample_2_variants.loc[all_samples, all_indels]

        assert sample_1_variants.shape[1] == sample_2_variants.shape[1]

        if binarize_fixed_variants:
            sample_2_variants[sample_2_variants >= fixed_thresh] = 1

        sample_2_variants[sample_2_variants < absent_thresh] = 0
    
    # remove variants that are fixed in every sample (probably some issue in variant calling :()
    variants_fixed_high_proportion = []

    for col in sample_1_variants.columns:

        if len(sample_1_variants.loc[sample_1_variants[col] >= 0.9]) / len(sample_1_variants) >= 0.90:
            variants_fixed_high_proportion.append(col)

    print(f"{sample_1_variants.shape[0]} samples with {sample_1_variants.shape[1] - len(variants_fixed_high_proportion)} variants")

    if sample_2:
        return df_variants, sample_1_variants.drop(variants_fixed_high_proportion, axis=1).astype(float), sample_2_variants.drop(variants_fixed_high_proportion, axis=1).astype(float)
    else:
        return df_variants, sample_1_variants.drop(variants_fixed_high_proportion, axis=1).astype(float)