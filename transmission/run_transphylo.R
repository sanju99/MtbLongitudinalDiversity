# remotes::install_github('xavierdidelot/TransPhylo')
# install.packages('treedater')

library(TransPhylo)
library(ape)
library(treedater)
library(lubridate)
library(phytools)

# Read your Newick tree
tree <- read.tree("~/MtbLongitudinalDiversity/data/trees/L4_samples/SNP_concatenate_filtered.fasta.treefile")

# seqs <- read.dna("~/MtbLongitudinalDiversity/transmission/L4_samples/genome_SNPs.fasta", format = "fasta")
seqs <- read.dna("~/MtbLongitudinalDiversity/transmission/L4_samples/SNP_concatenate_filtered.fasta", format = "fasta")
keep_names <- rownames(seqs)

# Drop all others
pruned_tree <- drop.tip(tree, setdiff(tree$tip.label, keep_names))

# Save the subset tree
write.tree(pruned_tree, file = "~/MtbLongitudinalDiversity/transmission/L4_samples/SNP_concatenate_filtered.nwk")

# Create a dated tree. The above is undated
tree <- read.tree("~/MtbLongitudinalDiversity/transmission/L4_samples/SNP_concatenate_filtered.nwk")
df_dates <- read.csv("~/MtbLongitudinalDiversity/transmission/L4_samples/dates.csv", stringsAsFactors = FALSE)

df_dates$Sample_Date <- as.Date(df_dates$Sample_Date)

# convert to decimal format
df_dates$decimal_year <- decimal_date(df_dates$Sample_Date)

# named vector of dates in decimal years
dates <- setNames(df_dates$decimal_year, df_dates$SampleID)

#tree_rooted <- midpoint.root(tree)

mu <- 1e-7 #0.5 / ncol(as.matrix(seqs))

# scale factor = total time / total branch length from root to tip
total_time <- max(dates) - min(dates)
total_branch_length <- max(node.depth.edgelength(tree))
tree$edge.length <- tree$edge.length * (total_time / total_branch_length)

# Use rooted tree (tree_rooted). Provide dates as numeric vector (in years).
td <- dater(tre = tree,
             sts = dates,
             s = ncol(as.matrix(seqs)),                # number of sites in the alignment used to build tree
             clock = "strict",
             omega0 = c(mu),
             maxit = 100
             )

dated_tree <- td$intree

# from Treetime
# dated_tree <- read.nexus("~/MtbLongitudinalDiversity/transmission/L4_samples/treetime_results/timetree.nexus")

# add epsilon because can't have 0s
# dated_tree$edge.length[dated_tree$edge.length == 0] <- 1e-8

# ptree <- ptreeFromPhylo(dated_tree, dateLastSample = max(dates))
# plot(ptree)

# plot(dated_tree)
# axisPhylo(backward = FALSE)

ptree<-ptreeFromPhylo(dated_tree, dateLastSample=max(dates))
plot(ptree)

res<-inferTTree(ptree,mcmcIterations=10000,w.shape=180,w.scale=1,dateT=2025)
plot(res)
