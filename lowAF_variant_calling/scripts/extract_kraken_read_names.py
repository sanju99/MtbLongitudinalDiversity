import numpy as np
import pandas as pd
import glob, os, argparse, time
from Bio import Seq, SeqIO, Entrez

parser = argparse.ArgumentParser()

# "/n/data1/hms/dbmi/farhat/rollingDB/TRUST/clinical_data/20240826_raw_data.csv"
parser.add_argument("-t", "--taxid", dest='taxid', type=int, required=True, help='Taxid for which to get reads')
parser.add_argument("-d", "--database", dest='db_name', type=str, required=True, help='Database name from which to get taxonomy')
parser.add_argument("-i", "--input", dest="in_fName", type=str, required=True, help='Kraken classifications file')
parser.add_argument("-o", "--output", dest='out_fName', type=str, required=True, help='File of read names to keep')
parser.add_argument("--include-children", dest='include_children', action='store_true', help='Include children taxa of the given taxid')
parser.add_argument("--include-parents", dest='include_parents', action='store_true', help='Include parent taxa of the given taxid')

cmd_line_args = parser.parse_args()
taxid = cmd_line_args.taxid
db_name = cmd_line_args.db_name
include_children = cmd_line_args.include_children
include_parents = cmd_line_args.include_parents

in_fName = cmd_line_args.in_fName
out_fName = cmd_line_args.out_fName


def load_nodes_dmp(nodes_path):
    parent_map = {}
    child_map = {}
    with open(nodes_path, 'r') as f:
        for line in f:
            parts = [x.strip() for x in line.split('|')]
            child_taxid, parent_taxid = int(parts[0]), int(parts[1])
            parent_map[child_taxid] = parent_taxid
            if parent_taxid not in child_map:
                child_map[parent_taxid] = []
            child_map[parent_taxid].append(child_taxid)
    return parent_map, child_map
    

def get_parent_taxids(taxid, parent_map):
    ancestors = []
    while taxid != 1:  # 1 is the root (cellular organisms)
        taxid = parent_map[taxid]
        ancestors.append(taxid)
    return ancestors


def get_child_taxids(taxid, child_map):
    descendants = []
    stack = [taxid]
    while stack:
        current = stack.pop()
        children = child_map.get(current, [])
        descendants.extend(children)
        stack.extend(children)
    return descendants


# use the taxonomy map to get child and parent taxids of the argument taxid
parent_map, child_map = load_nodes_dmp(f"{db_name}/taxonomy/nodes.dmp")

# use the functions to get all taxids
if include_parents:
    parent_taxids = get_parent_taxids(taxid, parent_map)
else:
    parent_taxids = []

if include_children:
    child_taxids = get_child_taxids(taxid, child_map)
else:
    child_taxids = []

# use the functionss to get all taxids
parent_taxids = get_parent_taxids(taxid, parent_map)
child_taxids = get_child_taxids(taxid, child_map)

df_kraken_classifications = pd.read_csv(in_fName, sep='\t', header=None)
df_kraken_classifications.columns = ['Classified', 'ReadName', 'TaxID', 'Length', 'LCA_kmers']

# keep the argument taxid and children
select_taxids = [taxid] + child_taxids + parent_taxids
keep_reads = df_kraken_classifications.query("TaxID in @select_taxids")['ReadName']

print(f"Keeping {len(keep_reads)}/{len(df_kraken_classifications)} ({np.round(len(keep_reads)/len(df_kraken_classifications)*100)}%) reads")

keep_reads.to_csv(out_fName, sep='\t', header=None, index=False)

# gzip the kraken classifications file
df_kraken_classifications.to_csv(in_fName + ".csv.gz", compression='gzip', index=False)