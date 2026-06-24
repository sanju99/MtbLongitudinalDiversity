import numpy as np
import pandas as pd
import os, glob, argparse, re
from Bio import Seq, SeqIO

parser = argparse.ArgumentParser()

# use edirect_env environment
# parser.add_argument("-i", "--input", dest='input_file', type=str, required=True, help='Input CSV or TSV file of IDs to search. Must be a single column with no header')
parser.add_argument("-i", "--input", dest='assembly_dir', type=str, required=True, help='Directory containing assemblies')
parser.add_argument("-I", "--input2", dest='LR_SR_data', type=str, required=True, help='CSV file mapping long read and short read data from the same participant/timepoint')
parser.add_argument("-o", "--output", dest='output_file', type=str, required=True, help='Output CSV file for the assembly statistics')

cmd_line_args = parser.parse_args()

assembly_dir = cmd_line_args.assembly_dir
LR_SR_data = cmd_line_args.LR_SR_data
output_file = cmd_line_args.output_file

df_LR_SR = pd.read_csv(LR_SR_data)

hybrid_asms = glob.glob(f"{assembly_dir}/*/PB/Flye_Assembly/assembly.fasta")

print(f"found {len(hybrid_asms)} assemblies")

df_hybridASM_QC = pd.DataFrame(columns = ['Original_ID', 'numContigs', 'circContigLength', 'Flye_Cov', 'Flye_Coll2014', 'Flye_PP_Coll2014', 'NumPilonChanges', 'ASM'])

for i, fName in enumerate(hybrid_asms):
    
    fasta_file = [(seq.id, seq.seq) for seq in SeqIO.parse(fName, "fasta")]
    
    assembly_stats = pd.read_csv(f"{os.path.dirname(fName)}/assembly_info.txt", sep='\t')
    
    # complete contigs
    complete_circular_contigs = assembly_stats.loc[(assembly_stats['circ.']=='Y') & (assembly_stats['length'] >= 4000000) & (assembly_stats['repeat']=='N')]
        
    if len(complete_circular_contigs) == 0:
        circContigLength = np.nan
        flye_coverage = np.nan
        
    elif len(complete_circular_contigs) == 1:
        circContigLength = complete_circular_contigs['length'].values[0]
        flye_coverage = complete_circular_contigs['cov.'].values[0]
     
    # shouldn't happen
    else:
        raise ValueError(f"{fName} has more than 1 circular contig")
        
#     lengths_dist = [len(seq[1]) for seq in fasta_file]
    
#     if len(lengths_dist) > 1:
#         second_longest = np.sort(lengths_dist)[-2]
#     else:
#         second_longest = np.max(lengths_dist)
        
    try:
        match = re.search(r"S\d{4}-..", fName)
        sample = match.group()
    except:
        match = re.search(r"MFS-\d{2,3}", fName)
        sample = match.group()
    
    try:
        flye_lineage_call = pd.read_csv(f"{assembly_dir}/{sample}/LineageCalling/LineageCall_FlyeI3Asm/{sample}.AsmToRef.FlyeI3Asm.lineage_call.tsv", sep='\t').coll2014.str.replace('lineage', '').values[0]
        pilon_polished_lineage_call = pd.read_csv(f"{assembly_dir}/{sample}/LineageCalling/LineageCall_FlyeI3AsmPP/{sample}.AsmToRef.FlyeI3AsmPP.lineage_call.tsv", sep='\t').coll2014.str.replace('lineage', '').values[0]

        pilon_changes_file = f"{assembly_dir}/{sample}/FlyeAssembly_I3_PilonPolishing/pilon_IllPE_Polishing_I3_Asm_ChangeSNPsINDELsOnly/{sample}.Flye.I3Asm.PilonPolished.changes"

        with open(pilon_changes_file, "r") as file:
            num_changes = len(file.readlines())
    except:
        flye_lineage_call = ''
        pilon_polished_lineage_call = ''
        num_changes = np.nan
        
#     if num_SNP_changes > 0:
#         df_pilon_changes = pd.read_csv(pilon_changes_file, sep='\t', header=None)
#         df_pilon_changes = df_pilon_changes[0].str.split(' ', expand=True)
#         df_pilon_changes.columns = ['Orig_POS', 'New_POS', 'REF', 'ALT']
        
#         num_SNP_changes = len(df_pilon_changes.query("REF.str.len() == ALT.str.len()"))

    final_ASM_file = f"{assembly_dir}/{sample}/FlyeAssembly_I3_PilonPolishing/pilon_IllPE_Polishing_I3_Asm_ChangeSNPsINDELsOnly/{sample}.Flye.I3Asm.PilonPolished.fasta"
    
    if not os.path.isfile(final_ASM_file):
        final_ASM_file = np.nan

    df_hybridASM_QC.loc[i, :] = [sample, 
                                 len(assembly_stats), 
                                 circContigLength,
                                 flye_coverage,
                                 flye_lineage_call, 
                                 pilon_polished_lineage_call,
                                 num_changes,
                                 final_ASM_file
                                ]
    

if len(df_hybridASM_QC.query("Original_ID.str.startswith('MFS')")) > 0:
    
    df_hybridASM_QC.rename(columns={'Original_ID': 'PacBio_ID'}, inplace=True)
    df_hybridASM_QC = df_hybridASM_QC.merge(df_LR_SR[['Original_ID', 'PacBio_ID', 'Illumina_ID', 'Illumina_F2', 'Illumina_Coll2014', 'PacBio_FQ_PATH']].drop_duplicates(), on='PacBio_ID')
    
    print(len(df_hybridASM_QC), df_hybridASM_QC.Original_ID.nunique())
    df_hybridASM_QC['pid'] = df_hybridASM_QC['Original_ID'].str.split('-').str[0].str.replace('S', 'T')
    
    df_hybridASM_QC.set_index('pid').reset_index().sort_values(['pid', 'Original_ID']).to_csv(output_file, index=False)
    
else:
    df_hybridASM_QC = df_hybridASM_QC.merge(df_LR_SR[['Original_ID', 'PacBio_ID', 'Illumina_ID', 'Illumina_F2', 'Illumina_Coll2014', 'PacBio_FQ_PATH']].drop_duplicates(), on='Original_ID')
    
    print(len(df_hybridASM_QC), df_hybridASM_QC.Original_ID.nunique())

    df_hybridASM_QC['pid'] = df_hybridASM_QC['Original_ID'].str.split('-').str[0].str.replace('S', 'T')

    # these were the PacBio IDs of samples that had 2 PacBio runs done from the same sample. These were the less good runs
    df_hybridASM_QC.query("PacBio_ID not in ['MFS-212', 'MFS-44', 'MFS-214', 'MFS-136']").set_index('pid').reset_index().sort_values(['pid', 'Original_ID']).to_csv(output_file, index=False)