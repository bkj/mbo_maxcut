#!/bin/bash

# run.sh

# ---------------------------------------------------------
# Small examples (from paper, < 10 seconds)

mkdir -p _data
mkdir -p _results/AS
wget --header "Authorization:$TOKEN" https://hiveprogram.com/data/_v1/mbo_maxcut/AS.tar.gz
tar -xzvf AS.tar.gz
mv AS _data/

time python mbo.py --inpath _data/AS/AS1.graph --outpath _results/AS/AS1 --plot
time python mbo.py --inpath _data/AS/AS2.graph --outpath _results/AS/AS2 --plot
time python mbo.py --inpath _data/AS/AS3.graph --outpath _results/AS/AS3 --plot

# ---------------------------------------------------------
# Medium examples

# Texas road network (< 1 minute)
mkdir -p {_data,_results}/tx
wget --header "Authorization:$TOKEN" https://hiveprogram.com/data/_v1/mbo_maxcut/tx.txt.gz
gunzip tx.txt.gz
mv tx.txt _data/tx

time python mbo.py --inpath _data/tx/tx.txt --outpath _results/tx/

# Wikipedia SNAP
mkdir -p {_data,_results}/wiki
wget --header "Authorization:$TOKEN" https://hiveprogram.com/data/_v1/mbo_maxcut/wiki.txt.gz
gunzip wiki.txt.gz
mv wiki.txt _data/wiki

time python mbo.py --inpath _data/wiki/wiki.txt --outpath _results/wiki/ --n-runs 1


