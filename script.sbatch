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

date

cd /home/hoanghu/recsys/

module load cuDNN

UV_VERSION="0.6.3"

# we have to re-define the cache directory otherwise it writes files to
# ~/.cache and fills home quota
# it would make sense to use one cache for all projects or your entire research
# group but don't put it in your home
# just for the sake of demonstration it is now placed in ${USERWORK} but
# project space would be better


# if uv does not exist, download it
if [ ! -e uv ]; then
    wget -q https://github.com/astral-sh/uv/releases/download/${UV_VERSION}/uv-x86_64-unknown-linux-musl.tar.gz -O - | tar xz --strip-components=1 -C . uv-x86_64-unknown-linux-musl/uv
fi

if [ ! -d venv ]; then
    ./uv venv venv
fi

# activate the environment
source venv/bin/activate

# synchronize environment with requirements.txt
# in other words: install dependencies into the venv
UV_PROJECT_ENVIRONMENT=venv uv pip sync requirements.txt


UV_PROJECT_ENVIRONMENT=venv uv run python run.py

echo
echo

date

