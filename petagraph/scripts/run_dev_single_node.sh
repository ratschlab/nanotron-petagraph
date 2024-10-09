#!/bin/bash

# random master port in the range 20000 - 30000
export MASTER_PORT=$((20000 + RANDOM % 10000))
export MASTER_ADDR=$(hostname)
export CUDA_DEVICE_MAX_CONNECTIONS=1         # required by nanotron
 
# Ensure package is installed
echo "===== Installing package ====="
pip install -e ./

# Set OMP threads
export OMP_NUM_THREADS=32
 
# Torchrun main script
echo "===== Running torchrun ====="
torchrun \
    --node-rank=0 \
    --master-addr=${MASTER_ADDR} \
    --master-port=${MASTER_PORT} \
    --nnodes=1 \
    --nproc-per-node=4 \
    petagraph/run_train.py \
    --config-file ./petagraph/configs/config_petagraph_dev.yaml
