#!/bin/bash

params=(0.2 0.4 0.6 0.8)
for param in "${params[@]}"
do 
    echo "param: $param is running"
    python train_net_2gpus_stedge.py --sharp_weight $param
    echo "param value: $param is running"
done