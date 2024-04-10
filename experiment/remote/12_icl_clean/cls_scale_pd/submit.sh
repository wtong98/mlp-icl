#!/bin/bash
#SBATCH -c 8
#SBATCH -t 1-12:00
#SBATCH -p pehlevan_gpu,seas_gpu,gpu
# #SBATCH -p kempner
#SBATCH --gres=gpu:1
#SBATCH --mem=32000
#SBATCH -o log.%A.%a.out
#SBATCH -e log.%A.%a.err
#SBATCH --array=1-8
#SBATCH --mail-type=END
#SBATCH --mail-user=wtong@g.harvard.edu
#SBATCH --account=pehlevan_lab

source ../../../../../venv_haystack/bin/activate
python run.py ${SLURM_ARRAY_TASK_ID}
