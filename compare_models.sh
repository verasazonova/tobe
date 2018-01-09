#!/usr/bin/env bash

if [ ! -d results ]; then
    mkdir results
fi

if [ ! -d logs ]; then
    mkdir logs
fi

# Training
echo "Training"
for N in 2 5 10 20; do
    echo $N
    python tobe/main.py --model models/train/weights_$N.hdf5 --train -n 50 --filenames resources/processed_corpus_$N.txt --logs logs/log_$N.csv
done

# Loss, scores for training
echo "Plotting loss"
python tobe/plot_results.py -f logs/log_2.csv logs/log_5.csv logs/log_10.csv logs/log_20.csv

# Evaluating
echo "Evaluating"
echo "avg,epochs,P,R,F1" > results/results.txt
for N in 2 5 10 20; do
    echo $N
    python tobe/main.py --model models/train/weights_$N.hdf5 --evaluate --filenames resources/processed_corpus_$N.txt > results/result_$N.txt
    cat results/result_$N.txt | grep -e "^---," >> results/results.txt
done

# Plot context dependence
echo "Plotting metrics"
python tobe/plot_results.py -f results/results.txt --metric
