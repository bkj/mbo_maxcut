#!/bin/bash

# run.sh

function prep {
    cat $1 | cut -d$'\t' -f1,2 > tmp && mv tmp $1
}

for i in $(seq 1 8); do
    mkdir -p {_data,_results}/AS
    wget http://www.cs.uoi.gr/~tsap/teaching/InformationNetworks/data-code/AS$i.graph
    mv AS$i.graph _data/AS
    prep _data/AS/AS$i.graph
done

python mbo.py --inpath _data/AS/AS1.graph --plot > _results/AS/AS1.graph
python mbo.py --inpath _data/AS/AS2.graph --plot > _results/AS/AS2.graph
python mbo.py --inpath _data/AS/AS3.graph --plot > _results/AS/AS3.graph
# ...