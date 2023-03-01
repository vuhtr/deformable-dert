#!/bin/bash
#SBATCH --job-name=aic-5         # create a short name for your job
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=2        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem-per-cpu=16G         # memory per cpu-core (4G is default)
#SBATCH --time=2-00:00:00          # total run time limit (HH:MM:SS)
#SBATCH --mail-type=begin        # send email when job begins
#SBATCH --mail-type=end          # send email when job ends
#SBATCH --mail-user=<YourNetID>@princeton.edu
#Number of GPUs, this can be in the format of "gpu:[1-4]", or "gpu:K80:[1-4] with the type included
#SBATCH --gres=gpu:2
##SBATCH --nodelist=selab3
#SBATCH -oslurm.out
#SBATCH -eslurm.err

nvidia-smi
module purge
module load anaconda3-2021.05-gcc-9.3.0-r6itwa7
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/x86_64-linux-gnu/
#module load CUDA/10.2.89-GCC-8.3.0
#module load zlib/1.2.11
conda init bash 
conda create --name deformable_dert python=3.8
source activate deformable_dert

# pip install gdown
pip install -r requirements.txt

# weights
# gdown https://drive.google.com/uc?id=15I03A7hNTpwuLNdfuEmW9_taZMNVssEp

nvidia-smi

# CUDA operators
cd ./models/ops
sh ./make.sh
python test.py
cd ../../

# Train
python -u main.py \
    --output_dir exps/exp0 \
    --with_box_refine --two_stage \
    --resume ./r50_deformable_detr_plus_iterative_bbox_refinement_plus_plus_two_stage-checkpoint.pth \
    --coco_path ./5k_coco \
    --num_classes 8 \
    --epochs 50 \
    --batch_size 4

conda deactivate
