import os, glob
import numpy as np
import pandas as pd

# define some paths to make the path names more readable
sample_out_dir = f"{output_dir}/{{sample_ID}}"

scripts_dir = config["scripts_dir"]
references_dir = config["references_dir"]

conda_directory = config['conda_dir']
primary_directory = os.getcwd()

sample_H37Rv_ref_dir = f"/n/data1/hms/dbmi/farhat/Sanjana/TRUST_lowAF/{{sample_ID}}"

run_unicycler:
    input:
    
    output:
    
    params:
    
    shell:
        """
        """
        
        
rule 