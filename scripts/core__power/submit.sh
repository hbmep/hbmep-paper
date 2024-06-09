#! /bin/bash

STEP=50
START=0
END=$((2000-$STEP))

for (( COUNTER=START; COUNTER<=END; COUNTER+=STEP )); do
    sbatch -c 32 --mem=128GB core.sh $COUNTER $((COUNTER+STEP))
done
