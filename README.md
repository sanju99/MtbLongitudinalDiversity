# MtbLongitudinalDiversity

Analysis of longitudinal changes in <i>Mycobacterium tuberculosis (Mtb)</i> genetic diversity during the course of tuberculosis infection and treatment in the TRUST cohort.

Diversity is assessed using <i>Mtb</i> lineages and F2 scores.

```bash
cd /home/sak0914/MtbLongitudinalDiversity/scripts

python3 concatenate_SNPs_freebayes.py -i ../data/freebayes_VCF_fNames.txt -start 0 -end 4411532 -o ../data/all_samples_SNP_concatenate.fasta -sense POS --AF-thresh 0.95

python3 drop_constant_ambiguous_sites_FASTA.py -i ../data/all_samples_SNP_concatenate.fasta -o ../data/all_samples_SNP_concatenate_filtered.fasta
```

https://iqtree.github.io/doc/Frequently-Asked-Questions

https://iqtree.github.io/doc/Tutorial

```bash
# -B is ultrafast bootstrap, -b is standard non-parametric bootstrap. Non-parameric bootstrap is taking forever
# MFP means to run modelfinder to find the best model
iqtree -s "SNP_concatenate_filtered.fasta" -m MFP+ASC -abayes -nt AUTO #-B 1000 -nt AUTO
```