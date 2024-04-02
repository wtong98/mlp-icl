#!/bin/bash
#SBATCH -c 8
#SBATCH -t 2-00:00:00
#SBATCH -p kempner
#SBATCH --gres=gpu:1
#SBATCH --mem=64000
#SBATCH -o run.%j.out
#SBATCH -e run.%j.err
#SBATCH --mail-type=END
#SBATCH --mail-user=wtong@g.harvard.edu
#SBATCH --account=kempner_grads

source ../../../../venv_haystack/bin/activate
python run.py

