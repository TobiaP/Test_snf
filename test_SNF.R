## Original code by Alessandro Zandonà (https://github.com/AleZandona/INF)
## Major edits, code improvements and multiview extension by Marco Chierici <chierici@fbk.eu>.
##
## Requires R >= 3.2.3
suppressPackageStartupMessages(library(argparse))
library(cvTools)
library(foreach)
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

#distK# Parameters tuning (K, alpha)

#t0 <- now()
#print(paste(t0, "-- Parameter tuning"))
#opt_par <- snf_tuning(distL, lab=lab, clm=clustMethod, infocl=clustInfo)
#K_opt <- opt_par[[1]]
#alpha_opt <- opt_par[[2]]
#t1 <- now()
#print(paste("Done:", time_length(interval(t0, t1), "minute"), "minutes elapsed."))
#print("Optimal parameters:")
#print(paste0("K_opt = ", K_opt, "; alpha_opt = ", alpha_opt))

K_opt <- 20
alpha_opt <- 0.5

# Similarity graphs
print(paste(now(), "-- Similarity graphs"))
affinityL <- lapply(distL, function(x) affinityMatrix(x, K=K_opt, alpha_opt))
i <- 0
for (mat in affinityL){
	matfile <- gsub('.txt', paste("_", toString(i),'_mat.txt',sep=""), outFile)
	write.table( mat, file=matfile, quote=FALSE, sep='\t', row.names=TRUE, col.names=TRUE)
	i <- i+1
}


# Fuse the graphs
print(paste(now(), "-- Graph fusion"))
W = SNF(affinityL, K=K_opt)
write.table(cbind(Samples=colnames(W), W), file=outFile, quote=FALSE, sep='\t', row.names=TRUE, col.names=TRUE)

# Rescale fused graph
W_sc <- W/max(W)
colnames(W_sc) <- rownames(dataF[[1]])
  
# Write fused graph
outfused <- gsub('.txt', '_similarity_mat_fused.txt', outFile)
write.table(cbind(Samples=colnames(W_sc), W_sc), file=outfused, quote=FALSE, sep='\t', row.names=FALSE, col.names=TRUE)

