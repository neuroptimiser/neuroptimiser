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
# **************************************
# Experiment 1: linlin-1 (Apr. 18, 2025)
#rm -r exconf/autoconfs

#python gen_configs.py instances 1-15 -i exconf/prelim_01.yaml
#parallel --jobs 5 python run_exp.py ::: exconf/autoconfs/*.yaml
#parallel --jobs 5 --bar --line-buffer --tag python run_exp.py ::: exconf/autoconfs/*.yaml

# **************************************
# Experiment 2: linlin-1 (Apr. 19, 2025) -> dim=2,5, instance=year:2009
#number_batches=10
#for ((i = 1; i <= number_batches; i++)); do
#    echo "Running experiment ${i} of ${number_batches} with config: exconf/prelim_01.yaml"
#    python run_exp.py exconf/prelim_01.yaml $number_batches $i
#    echo "Job ${i} finished at $(date)"
#done
# *** it only did 3 of 10

# **************************************
# Experiment 3: linlin-1 (Apr. 20, 2025) -> dim=2,5, instance=year:2009 (this instance is wrong, but cocoex get default, i.e., many instances)
#echo "Running with ${current_batch} of ${number_batches}"
#python run_exp.py exconf/prelim_01.yaml $number_batches $current_batch

# **************************************
# Experiment 4: izhizh-1 (Apr. 22, 2025) -> dim=2,5, instance=year:2009 (this instance is wrong,...)
#echo "Running with ${current_batch} of ${number_batches}"
#python run_exp.py exconf/prelim_02.yaml $number_batches $current_batch

# **************************************
# Experiment 5: linlin-2 and izhizh-2(Apr. 29, 2025) -> first with everything
#echo "Running with ${current_batch} of ${number_batches}"
#python run_exp.py exconf/prelim_11.yaml $number_batches $current_batch
#python run_exp.py exconf/prelim_12.yaml $number_batches $current_batch

# **************************************
# Experiment 6: linlin-2 and izhizh-2 (Apr. 30, 2025) -> missing 10, 20, 40
#echo "Running with ${current_batch} of ${number_batches}"
#python run_exp.py exconf/prelim_11.yaml $number_batches $current_batch
#python run_exp.py exconf/prelim_12.yaml $number_batches $current_batch
#python run_exp.py exconf/prelim_11_1.yaml $number_batches $current_batch
#python run_exp.py exconf/prelim_12_1.yaml $number_batches $current_batch

# **************************************
# Experiment 7: linlin-2 and izhizh-2 (May. 5, 2025)
echo "***************************************"
echo "Running with config ${config_file}, ${current_batch} of ${number_batches}"
echo "***************************************"
python run_exp.py $config_file $number_batches $current_batch
