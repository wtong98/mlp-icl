#!/bin/bash
#SBATCH -c 8
#SBATCH -t 1-00:00:00
#SBATCH -p seas_gpu,kempner,pehlevan_gpu,gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=64000
#SBATCH -o log.%A.%a.out
#SBATCH -e log.%A.%a.err
#SBATCH --array=1-3
#SBATCH --mail-type=END
#SBATCH --mail-user=wtong@g.harvard.edu
#SBATCH --account=kempner_pehlevan_lab

source ../../../../../venv_haystack/bin/activate
python run.py

