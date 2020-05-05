#!/bin/bash -l

for Y in 1 2 3 4 5
do 
    for X in 1000 500 100 50 10 5 1 0.5 0.1 0.05 0.01
    do 
        python run_all_experiment.py netflix_100 $X 1 -1 $Y &
    done
done

sleep 20

for Y in 6 7 8 9 10
do 
    for X in 1000 500 100 50 10 5 1 0.5 0.1 0.05 0.01
    do 
        python run_all_experiment.py netflix_100 $X 1 -1 $Y &
    done
done

sleep 20

for Y in 1 2 3 4 5
do 
    for X in 1000 500 100 50 10 5 1 0.5 0.1 0.05 0.01
    do 
        python run_all_experiment.py netflix_50 $X 1 -1 $Y &
    done
done

sleep 20

for Y in 6 7 8 9 10
do 
    for X in 1000 500 100 50 10 5 1 0.5 0.1 0.05 0.01
    do 
        python run_all_experiment.py netflix_50 $X 1 -1 $Y &
    done
done

#sleep 30m

# for Y in 1 2 3 4 5 6 7 8 9 10
# do 
#     for X in 1000 500 100 50 10 5 1 0.5 0.1 0.05 0.01
#     do 

#sleep 30m

# for Y in 1 2 3 4 5
# do 
#     for X in 1000 500 100 50 10 5 1 0.5 0.1 0.05 0.01
#     do 
#         python run_all_experiment.py purchase_20 $X 1 -1 $Y &
#     done
# done

# sleep 60m

# for Y in 1 2 3 4 5
# do 
#     for X in 1000 500 100 50 10 5 1 0.5 0.1 0.05 0.01
#     do 
#         python run_all_experiment.py purchase_50 $X 1 -1 $Y &
#     done
# done

# sleep 20m

# for Y in 6 7 8 9 10
# do 
#     for X in 1000 500 100 50 10 5 1 0.5 0.1 0.05 0.01
#     do 
#         python run_all_experiment.py purchase_50 $X 1 -1 $Y &
#     done
# done

# sleep 20m

# for Y in 1 2 3 4 5
# do 
#     for X in 1000 500 100 50 10 5 1 0.5 0.1 0.05 0.01
#     do 
#         python run_all_experiment.py purchase_100 $X 1 -1 $Y &
#     done
# done

# sleep 20m

# for Y in 6 7 8 9 10
# do 
#     for X in 1000 500 100 50 10 5 1 0.5 0.1 0.05 0.01
#     do 
#         python run_all_experiment.py purchase_100 $X 1 -1 $Y &
#     done
# done

# sleep 20m

# for Y in 1 2 3 4 5 
# do 
#     for X in 1000 500 100 50 10 5 1 0.5 0.1 0.05 0.01
#     do 
#         python run_all_experiment.py cifar_100 $X 1 -1 $Y &
#     done
# done

# sleep 20m

# for Y in 6 7 8 9 10
# do 
#     for X in 1000 500 100 50 10 5 1 0.5 0.1 0.05 0.01
#     do 
#         python run_all_experiment.py cifar_100 $X 1 -1 $Y &
#     done
# done
#run_dprndf_experiment.py
#5000 1000 500 100 50 10 5 1 0.5 0.1 0.05
#1000 500 100 50 10 5 1 0.5 0.1 0.05 0.01

# python run_all_experiment.py netflix_100 1000 0