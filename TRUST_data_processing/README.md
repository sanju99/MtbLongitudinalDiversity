



```bash
python3 -u scripts/combine_patient_WGS_data.py -i raw_data/TRUST_DATA_2025-05-12_1129.cleaned.wide.csv -I raw_data/WGS_data.csv -o processed_data/20250520_combined_patient_WGS_data.csv
```

```bash
python3 -u scripts/process_TRUST_data_for_analysis.py -i processed_data/20250520_combined_patient_WGS_data.csv
```

TOTAL/CXR/dicom/H0002_1A.dcm failed. It could not be read.
TRUST/CXR/dicom/T0336_1C.dcm failed. It could not be read.