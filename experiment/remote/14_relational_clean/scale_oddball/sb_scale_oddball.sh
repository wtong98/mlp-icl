#!/bin/bash
#SBATCH -c 8
#SBATCH -t 3:00:00
#SBATCH -p kempner
#SBATCH --gres=gpu:1
#SBATCH --mem=16000
#SBATCH -o log.%A.%a.out
#SBATCH -e log.%A.%a.err
#SBATCH --array=1-5
#SBATCH --mail-type=END
#SBATCH --mail-user=wtong@g.harvard.edu
#SBATCH --account=kempner_pehlevan_lab

source ../../../../../venv_haystack/bin/activate
python run.py

