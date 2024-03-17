#!/bin/bash
#SBATCH -c 16
#SBATCH -t 1-00:00:00
#SBATCH -p pehlevan_gpu,seas_gpu,gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=32000
#SBATCH -o log.%A.%a.out
#SBATCH -e log.%A.%a.err
#SBATCH --array=1-5
#SBATCH --mail-type=END
#SBATCH --mail-user=wtong@g.harvard.edu
# #SBATCH --account=kempner_pehlevan_lab
#SBATCH --account=pehlevan_lab

source ../../../../../venv_haystack/bin/activate
python run.py

