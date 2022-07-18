#!/bin/bash
#SBATCH --job-name=bekVSall # nom du job
#SBATCH --partition=gpu_p2 # partition
#SBATCH --qos=qos_gpu-t4 # QoS
#SBATCH --output=bekVSall_%j.out    # fichier de sortie (%j = job ID)
#SBATCH --error=bekVSall_%j.err # fichier d’erreur (%j = job ID)
#SBATCH --time=48:00:00 # temps maximal d’allocation "(HH:MM:SS)"
#SBATCH --nodes=1 # reserver 1 nœud
#SBATCH --ntasks=1 # reserver 8 taches (ou processus MPI)
#SBATCH --gres=gpu:1
# reserver 8 GPU
#SBATCH --cpus-per-task=2
#SBATCH --hint=nomultithread # desactiver l’hyperthreading
#SBATCH -A izg@v100
#SBATCH --mail-user=cartel.gouabou@lis-lab.fr
#SBATCH --mail-type=FAIL

module purge # nettoyer les modules herites par defaut

module load pytorch-gpu/py3/1.11.0 
ulimit -n 4096
set -x # activer l’echo des commandes

cd $WORK/journal


srun python -u train_isic_bin.py --task bekVSall 
 



