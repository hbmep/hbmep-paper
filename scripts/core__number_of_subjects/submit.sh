#! /bin/bash

STEP=200
START=0
END=1800

for (( COUNTER=START; COUNTER<=END; COUNTER+=STEP )); do
    sbatch -c 64 --mem=258GB core.sh $COUNTER $((COUNTER+200))
done
