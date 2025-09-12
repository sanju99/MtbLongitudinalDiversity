# install.packages('coda')
# install.packages('phytools')
# install.packages("remotes")
# remotes::install_github("donkeyshot/phybreak")

library(ape)
library(phybreak)
library(phytools)

# make phybreakdata object with function phybreak

# Read aligned FASTA file
seqs <- read.dna("~/MtbLongitudinalDiversity/transmission/L4_samples/SNP_concatenate_filtered.fasta", format = "fasta")

# Check sequence names
rownames(seqs)

df_sampling_dates <- read.csv("~/MtbLongitudinalDiversity/transmission/L4_samples/dates.csv")

# keep only those samples in the FASTA file
df_sampling_dates <- df_sampling_dates[df_sampling_dates$SampleID %in% rownames(seqs), ]

# make sure the data column is a date object
df_sampling_dates$SampleDate <- as.Date(df_sampling_dates$SampleDate)

# Dates must be numeric, so make them relative to time 0, which is the reference date
t0 <- min(df_sampling_dates$SampleDate)

# Sampling times in days since t0
df_sampling_dates$SampleDate_numeric <- as.numeric(df_sampling_dates$SampleDate - t0)

# Make a named vector for phybreak. setNames assigns names to an object. So you assign the sample IDs to each of the numeric dates
sampling_times <- setNames(df_sampling_dates$SampleDate_numeric, df_sampling_dates$SampleID)

# generate a parsimony tree to initialize with
# parsimony_tree <- pratchet(seqs)

# root on oldest sample
# first_sample <- names(sampling_times)[sampling_times == 0]
# parsimony_tree <- root(parsimony_tree, outgroup=first_sample, resolve.root=TRUE)

# plot(parsimony_tree)

L4_dataset <- phybreakdata(sequences = seqs, 
                           sample.times = sampling_times,
                           # sim.tree = parsimony_tree
                           )

# create phybreak object
# use 0.5 SNPs/genome/year. Transformed to days and the smaller SNP dataset, divide by 365 and number of sites in the MSA
mu <- 0.5 / ncol(as.matrix(seqs)) / 365

pb_results <- phybreak(L4_dataset, 
                       times = sampling_times, 
                       mu = mu,
                       gen.shape = 2, # shape param of gamma distribution of gen.mean
                       gen.mean = 365, # expected time between infection and infecting someone else (in days)
                       sample.shape = 2, # shape param of gamma distribution of sample.mean
                       sample.mean = 180, # expected time between infection and sampling (in days)
                       wh.model = 1, # effective size = 0, meaning that there is no within-host diversity 
                       wh.slope = 1,
                       est.gen.mean = TRUE, 
                       prior.mean.gen.mean = 1, 
                       prior.mean.gen.sd = Inf,
                       est.sample.mean = TRUE, 
                       prior.mean.sample.mean = 1,
                       prior.mean.sample.sd = Inf, 
                       est.wh.slope = TRUE, 
                       prior.wh.shape = 3,
                       prior.wh.mean = 1
                       # use.tree = TRUE
)

# exclude burn-in phase with burnin_phybreak
pb_results <- burnin_phybreak(pb_results, ncycles = 10000)

# saveRDS(pb_results, file = "~/MtbLongitudinalDiversity/transmission/L4_samples/burn_in_results_all.rds")

post_results <- sample_phybreak(pb_results, nsample = 50000)

# saveRDS(post_results, file = "~/MtbLongitudinalDiversity/transmission/L4_samples/posterior_results_all.rds")

transmission_tree <- transtree(post_results, method="mtcc")

write.csv(transmission_tree, "~/MtbLongitudinalDiversity/transmission/L4_samples/pilon_inferred_tree.csv")