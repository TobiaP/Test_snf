## Original code by Alessandro Zandonà (https://github.com/AleZandona/INF)
## Major edits, code improvements and multiview extension by Marco Chierici <chierici@fbk.eu>.
##
## Requires R >= 3.2.3
suppressPackageStartupMessages(library(argparse))
library(cvTools)
library(doParallel)
library(TunePareto)
library(igraph)
library(data.table)
library(lubridate)

load_data <- function(filename)
{
	#use fread for performance
	df <- fread(filename, data.table=FALSE)
	rownames(df) <- df[, 1]
	df <- df[, -1]
	df
}

load_labels <- function(filename)
{
	lab <- read.table(filename, as.is=TRUE, sep='\t')
	lab[[1]]
}
# -------------------

parser <- ArgumentParser(description="Perform a Similarity Network Fusion analysis on two or more datasets [samples by features]. NB: Same sample ordering across the datasets is required!")
parser$add_argument("--data", type="character", help="Omic layers datasets [samples by features]", nargs="+")
parser$add_argument("--lab", type="character", help="one column: labels associated to samples; NO HEADER")
parser$add_argument("--outf", type="character", help="Output file")
parser$add_argument("--scriptDir", type="character", help="Directory with R files necessary to SNF")
parser$add_argument("--clust", type="character", choices=c('spectral', 'fastgreedy'), help="Clustering method on fused graph")
parser$add_argument("--clustinfo", action="store_true", help="Should the number of clusters be equal to the number of classes? [default: FALSE]")
parser$add_argument("--threads", type="integer", default=4, help="Number of threads for rSNF [default = %default]")
args <- parser$parse_args()

# Read input parameters
outFile <- args$outf
clustMethod <- args$clust
clustInfo <- args$clustinfo
threads <- args$threads
labFile = args$lab

print(clustInfo)
# load R scripts
file_names <- as.list(dir(path=args$scriptDir, pattern="*", full.names=TRUE))
lpack <- lapply(file_names,source,.GlobalEnv)

# load files
dataF <- lapply(args$data, load_data)
lab <- load_labels(labFile)

# nrow check
stopifnot(length(unique( c(sapply(dataF, nrow), length(lab)) )) == 1)
# sample names check
tmp <- as.data.frame(sapply(dataF, rownames))
stopifnot(length(unique(as.list(tmp))) == 1)

# data normalization (mean 0, std 1)
print(paste(now(), "-- Data normalization"))
dataL <- lapply(dataF, standardNormalization)

# Calculate pairwise distance between samples
print(paste(now(), "-- Pairwise distances"))
distL <- lapply(dataL, function(x) (dist2(as.matrix(x), as.matrix(x))))

distK# Parameters tuning (K, alpha)
t0 <- now()
print(paste(t0, "-- Parameter tuning"))
opt_par <- snf_tuning(distL, lab=lab, clm=clustMethod, infocl=clustInfo)
K_opt <- opt_par[[1]]
alpha_opt <- opt_par[[2]]
t1 <- now()
print(paste("Done:", time_length(interval(t0, t1), "minute"), "minutes elapsed."))
print("Optimal parameters:")
print(paste0("K_opt = ", K_opt, "; alpha_opt = ", alpha_opt))

# Similarity graphs
print(paste(now(), "-- Similarity graphs"))
affinityL <- lapply(distL, function(x) affinityMatrix(x, K=K_opt, alpha_opt))

# Fuse the graphs
print(paste(now(), "-- Graph fusion"))
W = SNF(affinityL, K=K_opt)
# Rescale fused graph
W_sc <- W/max(W)
colnames(W_sc) <- rownames(dataF[[1]])
  
# Write fused graph
outfused <- gsub('.txt', '_similarity_mat_fused.txt', outFile)
write.table(cbind(Samples=colnames(W_sc), W_sc), file=outfused, quote=FALSE, sep='\t', row.names=FALSE, col.names=TRUE)

if (clustMethod=="spectral"){
  if (clustInfo){
    # Impose number of clusters (based on true samples labels)
    nclust <- length(unique(lab))
    group <-  spectralClustering(W, nclust)
  } else {
    print(paste(now(), "-- Estimating number of clusters"))
    nclust <- estimateNumberOfClustersGivenGraph(W)[[1]]
    # Spectral clustering
    print(paste(now(), "-- Spectral clustering"))
    group <-  spectralClustering(W, nclust)
  }
    
} else if (clustMethod=="fastgreedy"){
  # Rescale fused graph, so to apply community detection 
  W_sc <- W/max(W)
  # Graph from similarity matrix
  g <- graph.adjacency(W_sc, weighted = TRUE, diag=FALSE, mode='undirected')
  # Community detection
  m  <- cluster_fast_greedy(g)
  if (clustInfo){
    # Impose number of clusters (based on true samples labels)
    nclust <- length(unique(lab))
    group <- cutree(as.hclust(m), nclust)
  group} else {
    group <- m$membership
  }
}

# Goodness of clustering
# The closer SNFNMI to 0, the less similar the inferred clusters are to the real ones
SNFNMI_allfeats <-  calNMI(group, lab)

# Write out SMI score computed by using all features
outNMI <- gsub('.txt', '_NMI_score.txt', outFile)
write.table(SNFNMI_allfeats, file=outNMI, quote=FALSE, col.names=FALSE, row.names=FALSE)

### For a posteriori features ranking, build affinity matrix on one feature at a time ###
t0 <- now()
print(paste(t0, "-- Testing importance of each feature..."))
print(paste("[I] Running on", threads, "threads"))
SNFNMI_L <- lapply(dataL, function(dataset) {
    cl <- makeCluster(threads, outfile=gsub('.txt', '_MultiCoreLogging.txt', outFile))
    registerDoParallel(cl)
    toExport <- c("K_opt", "alpha_opt", "clustMethod", "clustInfo", "group", "file_names")
    SNFNMI <- foreach(d1_onefeat=dataset, .combine=c, .verbose=FALSE, .export=toExport) %dopar% {
        # reload SNF scripts
        lpack <- lapply(file_names, source, .GlobalEnv)
        # Calculate pairwise distance between samples
        dist_d1 <- dist2(as.matrix(d1_onefeat), as.matrix(d1_onefeat))
        
        # Similarity graphs
        W1 <- affinityMatrix(dist_d1, K=K_opt, alpha_opt) 
        
        if (clustMethod=="spectral"){
            if (clustInfo){
                # Impose number of clusters (based on true samples labels)
                nclust <- length(unique(lab))
                group_fi <- spectralClustering(W1, nclust)
            } else {
                nclust <- estimateNumberOfClustersGivenGraph(W1)[[1]]
                # Spectral clustering
                group_fi <- spectralClustering(W1, nclust)
            }
        } else if (clustMethod=="fastgreedy"){
            # Rescale fused graph, so to apply community detection 
            W1_sc <- W1/max(W1)
            # Graph from similarity matrix
            g <- graph.adjacency(W1_sc, weighted=TRUE, diag=FALSE, mode='undirected')
            # Community detection
            m  <- cluster_fast_greedy(g)
            if (clustInfo){
                # Impose number of clusters (based on true samples labels)
                nclust <- length(unique(lab))
                group_fi <- cutree(as.hclust(m), nclust)
            } else {
                group_fi <- m$membership
            }
        }
        
        # Goodness of clustering
        # The closer SNFNMI to 0, the less similar the inferred clusters are to the real ones
        TMP <- calNMI(group_fi, group)
        TMP
    }
    stopCluster(cl)
    SNFNMI
}
)
t1 <- now()
print(paste("Done:", time_length(interval(t0, t1), "minute"), "minutes elapsed."))

# Associate feature name to respective SNFNMI score
# [MC] not-so-elegant way to apply names to all SNFNMI_L objects
for(i in 1:length(SNFNMI_L)) {
    names(SNFNMI_L[[i]]) <- colnames(dataL[[i]])
}

# Combine data type 1 & 2 results
SNFNMI_integr <- unlist(SNFNMI_L)

# Sort (decreasing) features based on SNFNMI score
idx_snfnmi <- order(SNFNMI_integr, decreasing=TRUE)
SNFNMI_integr_sort <- SNFNMI_integr[idx_snfnmi]

# Convert list to a matrix for output file format purposes
ranked_list <- as.matrix(SNFNMI_integr_sort)
colnames(ranked_list) <- c("SNFNMI_score")

# Number of communities found
ncommun <- length(unique(group))
ml <- vector('numeric')
for (i in c(1:ncommun)){
  ml[i] <- length(which(group==i))
}
# Largest community
max_comm <- max(ml)
# Matrix to save communities (one per row )
commun_mat <- matrix("NaN", nrow=ncommun, ncol=max_comm+1)

for (i in c(1:ncommun)){
  commun_mat[i,0:ml[i]+1] <- c(paste("Community_",i,sep=''), which(group==i)-1)
}

# Write table with communities
outcomm <- gsub('.txt', '_communities.txt', outFile)
write.table(commun_mat, file=outcomm, sep='\t', row.names=FALSE, col.names=FALSE, quote=FALSE)

# Save workspace
outF_rdata <- gsub('.txt', '.RData', outFile)
save.image(outF_rdata)

# Write output file
write.table(cbind(Features=rownames(ranked_list), ranked_list), file=outFile, quote=FALSE, sep='\t', row.names=FALSE, col.names=TRUE)

# Write median NMI score for all possible combinations of parameters K and alpha
out_med_NMI <- gsub('.txt', '_median_NMI_params.txt', outFile)
med_NMI_params <- opt_par[[3]]
write.table(cbind(Parameters=rownames(med_NMI_params), med_NMI_params), file=out_med_NMI, quote=FALSE, sep='\t', row.names=FALSE, col.names=TRUE)
