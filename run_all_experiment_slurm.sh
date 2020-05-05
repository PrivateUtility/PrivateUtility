#!/bin/bash -l


#for R in 1 2 3 4 5
#do 
#    for E in 1000 500 100 50 10 5 1 0.5 0.1 0.05 0.01

#D='purchase_100'
#M=2
#'cifar_100' 'cifar_20' 'purchase_100' 'purchase_50' 'purchase_20' 'purchase_10' 'purchase_2' 'netflix_100' 'netflix_50' 'netflix_20' 'netflix_10' 'netflix_2'
for D in 'synthetic_200'
do
    for M in 20 21 22 23
    do
        for R in 1 2 3 4 5
        do 
            for E in 1.0
            do 
                sbatch --export D=$D,R=$R,E=$E,M=$M --job-name=$E.$D.$M.$R run_all_experiment.jobscript
            done
        done
    done
done
