#!/bin/bash
#SBATCH --job-name=conf # nom du job
#SBATCH --partition=gpu_p2 # partition
#SBATCH --qos=qos_gpu-t4 # QoS
#SBATCH --output=conf_%j.out    # fichier de sortie (%j = job ID)
#SBATCH --error=conf_%j.err # fichier d’erreur (%j = job ID)
#SBATCH --time=24:00:00 # temps maximal d’allocation "(HH:MM:SS)"
#SBATCH --nodes=1 # reserver 1 nœud
#SBATCH --ntasks=1 # reserver 8 taches (ou processus MPI)
#SBATCH --gres=gpu:3
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

cd $WORK/journal/heat_maps


srun python -u confident_hm.py  --prev_id 66 
 



