#!/bin/bash
# Example script for the INF pipeline

CORES=8

OUTFOLDER="C:\Users\tpavo\Desktop\Prova\results"
DATAFOLDER="C:\Users\tpavo\Desktop\Tirocinio"
DATASET=tcga_aml
LAYER1=gene
LAYER2=meth
LAYER3=mirna
TARGET=OS
MODEL=randomForest
N_SPLITS_START=0
N_SPLITS_END=10
SNAKEFILE=Test_SNF_snake

RANDOM_LABELS=false

for (( i=$N_SPLITS_START; i<$N_SPLITS_END; i++ ))
do
	snakemake -s $SNAKEFILE --cores $CORES --config datafolder=$DATAFOLDER outfolder=$OUTFOLDER dataset=$DATASET target=$TARGET layer1=$LAYER1 layer2=$LAYER2 layer3=$LAYER3 model=$MODEL random=$RANDOM_LABELS split_id=$i -r -p
done