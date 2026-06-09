library(tidyverse)
library(TBtypeR)
library(purrr)
library(dplyr)
library(tidyr)

# the discrepant samples
# samples <- c('S0030-01', 'S0201-01', 'S0209-01', 'S0214-01', 'S0220-01',
#        'S0223-01', 'S0089-01', 'S0150-01', 'S0157-01', 'S0170-01',
#        'S0180-01', 'S0184-01', 'S0188-01')

# the discrepant samples between DS and culture
# samples <- c('S0030-01')#, 'S0089-01', 'S0096-01', 'S0150-01', 'S0157-01', 'S0170-01',
       # 'S0180-01', 'S0184-01', 'S0188-01', 'S0201-01', 'S0209-01',
       # 'S0214-01', 'S0220-01', 'S0223-01', 'S0225-01', 'S0248-01',
       # 'S0252-01', 'S0262-01', 'S0265-01', 'S0340-01', 'S0377-01_K064756',
       # 'S0401-01')

# culture samples with 3 lineages. Incrase mix_phylotypes to see if any of those change
# samples <- c('MFS-168', 'MFS-317', 'MFS-696')

# DS samples with low coverage
samples <- c('S0004-m5', 'S0018-01', 'S0029-01', 'S0032-01', 'S0064-01',
       'S0103-01', 'S0115-01', 'S0189-01', 'S0194-01', 'S0222-01',
       'S0229-01', 'S0247-01', 'S0261-01', 'S0267-01', 'S0269-04',
       'S0272-01', 'S0274-08', 'S0278-06', 'S0279-06_K064731', 'S0279-06',
       'S0280-06', 'S0281-01', 'S0288-01', 'S0291-09', 'S0298-01',
       'S0301-01', 'S0307-01', 'S0314-m5', 'S0325-01', 'S0326-01',
       'S0334-01B', 'S0334-01', 'S0338-01', 'S0340-01', 'S0341-01',
       'S0348-01', 'S0351-01', 'S0370-01', 'S0372-01', 'S0377-01B',
       'S0387-01', 'S0391-01', 'S0401-01B', 'S0475-01', 'S0476-01')

vcf_files <- file.path(
 "/n/data1/hms/dbmi/farhat/Sanjana/TRUST_directSputum",
 samples,
 "TBtypeR",
 paste0(samples, ".filtered.vcf.gz")
)

# vcf_files <- Sys.glob("/n/data1/hms/dbmi/farhat/Sanjana/TRUST_directSputum/**/TBtypeR/*filtered.vcf.gz") # [1:10]

# vcf_files <- Sys.glob("/n/data1/hms/dbmi/farhat/Sanjana/TRUST_lowAF/**/TBtypeR/*filtered.vcf.gz") # [1:10]

print(length(vcf_files))

tbtype_result_all <-
  map_df(vcf_files, function(vcf_file) {
    # tbtype(vcf = vcf_file, min_median_depth = 10, min_depth_fold = 5, min_mix_prop = 0.01) %>% 
    tbtype(vcf = vcf_file, min_median_depth = 5, min_depth_fold = 2.5, min_mix_prop = 0.01) %>% 
      filter_tbtype(max_phylotypes = 5) %>% # keep only the top 3 phylotypes.
      unnest_mixtures() %>% 
      mutate(sample_id = basename(vcf_file))   # add sample name
  })

tbtype_result_all %>%
  select(sample_id, n_phy, mix_phylotype, mix_prop) %>%
  knitr::kable()

# save
# write.csv(tbtype_result_all, file="~/MtbLongitudinalDiversity/direct_sputum/culture_TBtypeR_results_5phylotypes.csv", row.names=FALSE)
# write.csv(tbtype_result_all, file="~/MtbLongitudinalDiversity/direct_sputum/TBtypeR_results_discrepant.csv", row.names=FALSE)
# write.csv(tbtype_result_all, file="~/MtbLongitudinalDiversity/direct_sputum/TBtypeR_results.csv", row.names=FALSE)
write.csv(tbtype_result_all, file="~/MtbLongitudinalDiversity/direct_sputum/rerun/TBtypeR_results_lowCov.csv", row.names=FALSE)