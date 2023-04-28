#!/bin/bash

#SBATCH --job-name=mmvae    # create a short name for your job
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=10       # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --partition=gpu          # Name of the partition
#SBATCH --gres=gpu:rtx6000:1     # GPU nodes are only available in gpu partition
#SBATCH --mem=15G                # Total memory allocated
#SBATCH --hint=multithread       # we get logical cores (threads) not physical (cores)
#SBATCH --time=20:00:00          # total run time limit (HH:MM:SS)
#SBATCH --output=%x_%j.out   # output file name

echo "### Running $SLURM_JOB_NAME ###"

cd ${SLURM_SUBMIT_DIR}

module purge
module load cuda/11.4.0

# Set your conda environment
source /home/$USER/.bashrc
# tensorflow environment shloud bre created previously
source activate mmvae_repro
# python bin/make-mnist-svhn-idx.py
# python -u src/main.py --model mnist_svhn --obj dreg --K 30 --learn-prior --looser --epochs 30 --batch-size 128 --latent-dim 20 --seed 8
python -u src/report/analyse_ms.py --save-dir experiments/mnist-svhn/2023-04-21T15:23:54.1423561y8r4gz2
# python src/report/calculate_likelihoods.py --save-dir experiments/mnist-svhn/2023-04-02T15:15:43.155828radthud4 --iwae-samples 1000