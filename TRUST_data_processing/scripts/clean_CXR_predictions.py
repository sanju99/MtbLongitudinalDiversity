import numpy as np
import pandas as pd
import argparse, os, glob, warnings, sys
warnings.filterwarnings('ignore')
sys.path.append(os.path.dirname(__file__))
from data_utils import *
from epi_utils import *

parser = argparse.ArgumentParser()

parser.add_argument("-i", dest='input_dir', type=str, required=True, help='Directory with predictions')
parser.add_argument("-o", dest='output_file', type=str, required=True, help='Output file to save the annotated predictions')

cmd_line_args = parser.parse_args()

input_dir = cmd_line_args.input_dir
output_file = cmd_line_args.output_file


################################### THIS SCRIPT READS IN PLI AND TIMIKA PREDICTIONS AND ADDS ANNOTATION FOR OUTLIER AND BASELINE VS. FOLLOW-UP ###################################


df_PLI_pred = pd.read_csv(f"{input_dir}/pli_regression_predictions.csv").rename(columns={'predicted_label': 'predicted_PLI'})
df_Timika_pred = pd.read_csv(f"{input_dir}/timika_regression_predictions.csv").rename(columns={'predicted_label': 'predicted_Timika'})
df_outlier_annot = pd.read_csv(f"{input_dir}/outlier_detection_results_all_dicoms.csv")

df_outlier_annot['patientID_view'] = df_outlier_annot['Filename'].transform(lambda x: os.path.basename(x).split('.dcm')[0])

df_combined_pred = df_PLI_pred.merge(df_Timika_pred, on='patientID_view').merge(df_outlier_annot, on='patientID_view', how='inner')

# separate pid and view. View can have multiple suffixes, so string join them
df_combined_pred['pid'] = df_combined_pred['patientID_view'].str.split('_').str[0]
df_combined_pred['view'] = df_combined_pred['patientID_view'].str.split('_').str[1:].transform(lambda x: '_'.join(x))

# 1 = BL, 2 = FU. Check what 3 is, but it's probably additional
df_combined_pred['Time'] = df_combined_pred['view'].str[0].map({'1': 'BL', '2': 'FU', '3': 'ADDL'})

df_combined_pred.to_csv(output_file, index=False)

# drop outliers and keep only baseline samples
df_combined_pred_baseline = df_combined_pred.query("Status=='Normal' & Time=='BL'")

if 'TRUST' in input_dir:
    
    # looked at these manually
    df_combined_pred_baseline.loc[df_combined_pred_baseline['patientID_view']=='T0395_1B', 'view'] = '1C'
    df_combined_pred_baseline.loc[df_combined_pred_baseline['patientID_view'].isin(['T0128_1C', 'T0137_1C', 'T0043_1C', 'T0134_1C']), 'view'] = '1B'

    pids_no_PA = set(df_combined_pred_baseline.query("view.str.contains('C')").pid) - set(df_combined_pred_baseline.query("view.str.contains('B')").pid)
    print(f"{len(pids_no_PA)} pids don't have a PA view but have an AP view")

    df_combined_pred_baseline = pd.concat([df_combined_pred_baseline.query("view.str.contains('|'.join(['B', 'D', 'E']))"),
                                           df_combined_pred_baseline.query("pid in @pids_no_PA")
                                          ])

    # T0025_1B_2 is truncated. T0025_1B seems fine
    df_combined_pred_baseline = df_combined_pred_baseline.query("patientID_view != 'T0025_1B_2'")

    df_combined_pred_baseline['Diff'] = df_combined_pred_baseline.groupby('pid')['predicted_PLI'].transform(lambda x: np.max(x) - np.min(x))

    print(f"Maximum difference between PLI predictions for the same pid: {df_combined_pred_baseline.Diff.max()}%")

elif 'TOTAL' in input_dir:
    
    # H0055_1C is a PA view
    df_combined_pred_baseline.loc[df_combined_pred_baseline['patientID_view']=='H0055_1C', 'view'] = '1B'

    pids_no_PA = set(df_combined_pred_baseline.query("view.str.contains('C')").pid) - set(df_combined_pred_baseline.query("view.str.contains('B')").pid)
    print(f"{len(pids_no_PA)} pids don't have a PA view but have an AP view")

    # the two AP view look okay, so keep them to retain patients
    df_combined_pred_baseline = pd.concat([df_combined_pred_baseline.query("view.str.contains('|'.join(['B', 'D', 'E']))"),
                                           df_combined_pred_baseline.query("pid in @pids_no_PA")
                                          ])

    # P0695_1B_2 is truncated, so keep P0695_1B_1
    df_combined_pred_baseline = df_combined_pred_baseline.query("patientID_view != 'P0695_1B_2'")

    # H0053_1B_1 and H0053_1B_2 look the same to me, so just keep the first. But also the difference is only 5.5 pp
    df_combined_pred_baseline['Diff'] = df_combined_pred_baseline.groupby('pid')['predicted_PLI'].transform(lambda x: np.max(x) - np.min(x))
    
    print(f"Maximum difference between PLI predictions for the same pid: {df_combined_pred_baseline.Diff.max()}%")
    
else:
    raise ValueError(f"Neither 'TRUST' nor 'TOTAL' are in {input_dir}")
    
    
print(f"{df_combined_pred_baseline.pid.nunique()}/{df_combined_pred.pid.nunique()} pids have non-outlier baseline predictions")

df_combined_pred_baseline.to_csv(f"{output_file.replace('.csv', '_baseline.csv')}", index=False)