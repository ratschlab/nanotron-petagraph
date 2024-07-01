#!/bin/bash
#SBATCH --job-name=petagraph      # create a short name for your job
#SBATCH --nodes=8                # total number of nodes
#SBATCH --ntasks-per-node=1      # total number of tasks per node
#SBATCH --gpus-per-task=4
#SBATCH --time=12:00:00
#SBATCH --output=/users/burgerm/petagraph/logs/slurm/wgs_fungi_%x_%j.log
 
# Initialization.
set -x
cat $0

# random master port in the range 20000 - 30000
export MASTER_PORT=$((20000 + RANDOM % 10000))
export MASTER_ADDR=$(hostname)
export CUDA_DEVICE_MAX_CONNECTIONS=1         # required by nanotron
 
# Run main script.
srun -ul --environment=petagraph_python_env bash -c "
  # Change cwd and run the main training script.
  cd /users/burgerm/petagraph/nanotron-petagraph
  pip install -e ./   # Only required the first time.

  export OMP_NUM_THREADS=32
 
  TORCHRUN_ARGS=\"
   --node-rank=\${SLURM_PROCID} \
   --master-addr=\${MASTER_ADDR} \
   --master-port=\${MASTER_PORT} \
   --nnodes=\${SLURM_NNODES} \
   --nproc-per-node=\${SLURM_GPUS_PER_TASK}
  \"

  echo \"Running with node rank \${SLURM_PROCID}\"
  echo \"Running with master addr \${MASTER_ADDR}\"
  echo \"Running with master port \${MASTER_PORT}\"
  echo \"Running with nnodes \${SLURM_NNODES}\"
 
  torchrun \${TORCHRUN_ARGS} petagraph/run_train.py --config-file petagraph/configs/config_petagraph_multi_node.yaml
"