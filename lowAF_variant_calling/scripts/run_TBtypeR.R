# install.packages("BiocManager")
# Install all required Bioconductor packages
# BiocManager::install(c(
#   "GenomicRanges",
#   "treeio",
#   "SeqVarTools",
#   "IRanges",
#   "Rsamtools",
#   "rtracklayer",
#   "ggtree",
#   "SeqArray",
#   "BSgenome",
#   "Biostrings"
# ))

# remotes::install_github('bahlolab/TBtypeR')

library(TBtypeR)
library(tidyverse)

# replace with path to your VCF file
vcf_filename <- '~/S0224_01_dedup_freebayes_AFCor_tbtyper.vcf'

tbtype_result <- 
  # generate TBtypeR results
  tbtype(vcf = vcf_filename) %>% 
  # filter TBtypeR results
  filter_tbtype(max_phylotypes = 3) %>%
  # unnest data so there is 1 row per identified Mtb strain in each sample
  unnest_mixtures()

tbtype_result %>% 
  select(sample_id, n_phy, mix_phylotype, mix_prop) %>% 
  knitr::kable()

tbtype_result %>% 
  ggplot(aes(x = sample_id,
             y = mix_prop,
             fill = mix_phylotype)) +
  geom_col() +
  coord_flip() +
  labs(x = 'Sample ID',
       y = 'Minor Strain Fraction (%)', 
       fill = 'Sublineage') +
  theme(text = element_text(size = 6))