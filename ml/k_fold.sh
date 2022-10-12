#!/bin/bash
subsets="RYB20 RYB21 GRO20 GRO21 AMO18"
substitution=""
for valid in $subsets
do
    train=${subsets/"$valid "/"$substitution"}
    train=${train/" $valid"/"$substitution"}
    train=${train//" "/","}
    test="None"
    jupyter nbconvert ml/main.ipynb --to python
    ipython ml/main.py $train $valid $test
done