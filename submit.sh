#!/bin/bash

#SBATCH --job-name=stft_tr
#SBATCH --partition=4090v2,4090,a10
#SBATCH --nodes=1 
#SBATCH --gres=gpu:1              
#SBATCH --ntasks-per-node=4         
#SBATCH --mem=32G                 
#SBATCH --cpus-per-task=8           
#SBATCH --qos=qlong                
#SBATCH --time=3-00:00:00                        
#SBATCH --output="./slurm_logs/slurm-%j.out" 
#SBATCH --error="./slurm_logs/slurm-%j.err"    

echo "Slurm 任务开始时间: $(date)"
echo "当前工作目录: $(pwd)"
echo "当前节点主机名: $(hostname)"
echo "分配的 GPU 数量: $CUDA_VISIBLE_DEVICES" 

source /exp/xuanyu.wang/espnet_20250624/tools/miniconda/bin/activate espnet
cd /exp/xuanyu.wang/espnet_20250624/stft

python -u train.py > exp/train.log 2>&1

echo "Slurm 任务结束时间: $(date)"