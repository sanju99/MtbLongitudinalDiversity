# Low Frequency Variant Calling and Error Model

The bioinformatics pipeline to process FASTQ files and perform variant calling with freebayes can be called from `run_smk.sh`.

If aligning short-read FASTQs to the H37Rv reference genome, use the rules in `rules/rules.smk`. If aligning short-read FASTQs to personal reference genomes, use the rules in `rules/personal_asm.smk`.

Both sets of rules call `variantDetector/01_write_lowAF_BED.py` and `variantDetector/03_get_H37Rv_alignment_stats.py`. Only `rules/personal_asm.smk` runs `variantDetector/02_combine_lowAF_variants.py` to combine low frequency variant calls determined from aligning to H37Rv and a personal reference genome in order to compare them.

Once you have the resulting low frequency variant calls, substitution variants should be processed by `variantDetector/04_prep_variant_tables_for_model.py` before running the error model. 

The data used in the manuscript and the individual variable means and standard deviations are in `variantDetector/unmixed_only`. The logistic regression model with which to get predictions on new data is `variantDetector/unmixed_only/penalty_none/logistic_model.pkl`, which was fit on the data using the script `variantDetector/05_logistic_model.py`.

Indel variants should be processed and identified using `variantDetector/06_get_unfixed_indels.py` and `variantDetector/07_get_fixed_indels.py`.
