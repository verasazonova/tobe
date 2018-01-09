#!/usr/bin/env bash

echo "avg,epochs,P,R,F1" > results.txt
for N in 2 5 10 20; do
    echo $N
    python tobe/main.py --model models/by_loss/weights_$N.hdf5 --evaluate --filename resources/processed_corpus_$N.txt > result_$N.txt
    cat result_$N.txt | grep -e "---," >> results.txt
done