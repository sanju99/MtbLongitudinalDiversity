library(tidyverse)
library(TBtypeR)
library(purrr)
library(dplyr)
library(tidyr)

# vcf_files <- c("/n/data1/hms/dbmi/farhat/Sanjana/TRUST_lowAF/MFS-4/TBtypeR/MFS-4.vcf.gz",
#                "/n/data1/hms/dbmi/farhat/Sanjana/TRUST_lowAF/MFS-414/TBtypeR/MFS-414.vcf.gz",
#                "/n/data1/hms/dbmi/farhat/Sanjana/TRUST_lowAF/MFS-866/TBtypeR/MFS-866.vcf.gz"
#                )

# samples <- c('S0004-m5', 'S0018-01', 'S0029-01', 'S0032-01', 'S0064-01',
#             'S0103-01', 'S0115-01', 'S0189-01', 'S0194-01', 'S0222-01',
#              'S0229-01', 'S0247-01', 'S0261-01', 'S0267-01', 'S0269-04',
#              'S0272-01', 'S0274-08', 'S0278-06', 'S0279-06_K064731', 'S0279-06',
#              'S0280-06', 'S0281-01', 'S0288-01', 'S0291-09', 'S0298-01',
#              'S0301-01', 'S0307-01', 'S0314-m5', 'S0325-01', 'S0326-01',
#              'S0334-01B', 'S0334-01', 'S0338-01', 'S0340-01', 'S0341-01',
#              'S0348-01', 'S0351-01', 'S0370-01', 'S0372-01', 'S0377-01B',
#              'S0387-01', 'S0391-01', 'S0401-01B', 'S0475-01', 'S0476-01'
# )

samples <- c("S0030-01", "S0089-01")

vcf_files <- file.path(
  "/n/data1/hms/dbmi/farhat/Sanjana/TRUST_directSputum",
  samples,
  "TBtypeR",
  # paste0(samples, ".filtered.vcf.gz")
  paste0(samples, ".vcf.gz")
)

# vcf_files <- Sys.glob("/n/data1/hms/dbmi/farhat/Sanjana/TRUST_directSputum/**/TBtypeR/*filtered.vcf.gz") # [1:10]

vcf_files <- c("/n/data1/hms/dbmi/farhat/Sanjana/TRUST_mixed_infection_HP/S0305-01/TBtypeR/haplotype_1.vcf.gz",
               "/n/data1/hms/dbmi/farhat/Sanjana/TRUST_mixed_infection_HP/S0305-01/TBtypeR/haplotype_2.vcf.gz"
              )

print(length(vcf_files))

tbtype_result_all <-
  map_df(vcf_files, function(vcf_file) {
    tbtype(vcf = vcf_file, min_median_depth = 10, min_depth_fold = 5, min_mix_prop = 0.01) %>% 
    # tbtype(vcf = vcf_file, min_median_depth = 5, min_depth_fold = 2.5, min_mix_prop = 0.01) %>% 
      filter_tbtype(max_phylotypes = 3) %>% # keep only the top 3 phylotypes.
      unnest_mixtures() %>% 
      mutate(sample_id = basename(vcf_file))   # add sample name
  })

tbtype_result_all %>%
  select(sample_id, n_phy, mix_phylotype, mix_prop) %>%
  knitr::kable()

# save
write.csv(tbtype_result_all, file="~/MtbLongitudinalDiversity/direct_sputum/TBtypeR_results_lowCov.csv", row.names=FALSE)
# write.csv(tbtype_result_all, file="~/MtbLongitudinalDiversity/direct_sputum/TBtypeR_results_masked_regions_lowCov.csv", row.names=FALSE)


