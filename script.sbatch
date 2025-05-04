#!/bin/sh
#SBATCH --job-name=recsys
#SBATCH -M ukko
#SBATCH -o logs/slurm/%x_%j.out
#SBATCH -e logs/slurm/%x_%j.err
#SBATCH -c 4
#SBATCH -t 2-00:00:00
#SBATCH --mem=64G
#SBATCH -p gpu
#SBATCH --constraint=v100
#SBATCH --gres=gpu:1
#SBATCH --chdir=/home/hoanghu/recsys/

module purge
module load Mamba cuDNN

date


/home/hoanghu/foodwaste-prediction/.venv/bin/python -W ignore -m src.whole_pos_forecast.finetune -r phy



date


