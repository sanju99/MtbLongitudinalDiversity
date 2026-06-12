import os, glob, pydicom, argparse
import numpy as np
import pandas as pd

parser = argparse.ArgumentParser()

# Add a required string argument for the config file
parser.add_argument("-i", dest='input_file', type=str, required=True)

cmd_line_args = parser.parse_args()
input_file = cmd_line_args.input_file

df = pd.read_csv(input_file)
print(len(df))


def compare_two_dicoms(fName1, fName2):
    '''
    Returns 1 if they are identical. 0 if not. Returns two separate values for headers being identical and image data being identical (pixels)
    '''
    
    d1 = pydicom.dcmread(fName1)
    d2 = pydicom.dcmread(fName2)

    same_pixels = np.array_equal(d1.pixel_array, d2.pixel_array)

    # d1 == d2 checks headers and same_pixels checks that they have all the same pixels
    return int(d1 == d2), int(same_pixels)



for i, row in df.iterrows():
    
    df.loc[i, ['SameHeader', 'SamePixels']] = compare_two_dicoms(row['fName1'], row['fName2'])

    if i % 100 == 0:
        df.to_csv(input_file, index=False)
        print(i)

df.to_csv(input_file, index=False)