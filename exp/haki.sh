#!/bin/bash

#OAR -n neuropt-haki
#OAR -O ./logs/OAR_%jobid%.out
#OAR -E ./logs/OAR_%jobid%.err

#roarrrrr -p neowise
#roarrrrr -t exotic
#roarrrrr -l walltime=12:00:00
#roarrrrr -l host=1,walltime=11:59:00
# Check if a batch number was provided
config_file=$1
number_batches=$2
current_batch=$3

echo "[haki] Starting job on $(hostname) at $(date)"
module load conda
conda activate haki

module load gcc/13.2.0_gcc-10.4.0

ulimit -n 65536

NUM_THREADS=$(nproc)
NUMBA_NUM_THREADS=$NUM_THREADS
OMP_NUM_THREADS=$NUM_THREADS
MKL_NUM_THREADS=$NUM_THREADS
OPENBLAS_NUM_THREADS=$NUM_THREADS

export NUM_THREADS
export NUMBA_NUM_THREADS
export OMP_NUM_THREADS
export MKL_NUM_THREADS
export OPENBLAS_NUM_THREADS

echo "[haki] Using $NUM_THREADS threads"

# Run the Python script with the specified configuration file
echo "Running Python script with configuration files"

echo "***************************************"
echo "Running with config ${config_file}, ${current_batch} of ${number_batches}"
echo "***************************************"
echo ""
python run_exp.py $config_file $number_batches $current_batch
