srun -N 1 --cpus-per-task=16 -p bigdata_alg --gres=gpu:0
srun -N 1 --cpus-per-task=16 -p bigdata_alg -x SH-IDC1-10-140-24-12 --gres=gpu:0 
srun -N 1 --cpus-per-task=16 -p bigdata_alg -x SH-IDC1-10-140-24-[12,111,75,78,65,20] --gres=gpu:0 

# generate layout
# random
python random_generator.py --n 1000000 --json-file /mnt/petrelfs/zhaozhiyuan/layout/LACE/datasets/publaynet-max25/raw/publaynet/val.json --filter 0.1

# bestfit
python bestfit_generator.py --n 500000 --json-file /mnt/petrelfs/zhaozhiyuan/layout/LACE/datasets/publaynet-max25/raw/publaynet/val.json --output-file output-bestfit_layout-part2.json
jq -s 'flatten' output-bestfit_layout-part*.json > output-bestfit_layout.json

# diffusion
torchrun --nproc_per_node 8 batch_generate.py \
--amount 1000000 \
--load_model model/multiple/multiple,maxtoken=25,lr=1e-4/pos_multiple_epoch1400_ckpt.pt
jq -s 'flatten' output-diffusion-rank*.json > output-diffusion.json
python diffusion_generator.py --json-file output-diffusion.json

# remove duplicate
python remove_duplicate.py --json-file output-bestfit_layout-part1.json --output-file output-bestfit_layout-part1-deduplicated.json

# render
python render.py --json-file output-bestfit_layout.json --set bestfit
python render.py --json-file output-diffusion_processed.json --set diffusion
python render.py --json-file output-random_layout.json --set random

python render.py --json-file output-bestfit_layout.json --set bestfitv2
python render.py --json-file output-diffusion_processed.json --set diffusionv2

# delete
/mnt/petrelfs/zhaozhiyuan/sensesync --dryrun --deleteSrc cp bestfitv2/ ./srcSync/

# count
/mnt/petrelfs/zhaozhiyuan/sensesync --dryrun cp s3://X2B9QEQ012TRZN39YVGA:8CybumhrcYPPt1SqAZPB09txzuJFxrmDbuT5ycMg@layout_pretrain_data_shdd.10.140.14.204:80/bestfit_layout/images/ ./srcSync/
/mnt/petrelfs/zhaozhiyuan/sensesync --dryrun cp s3://X2B9QEQ012TRZN39YVGA:8CybumhrcYPPt1SqAZPB09txzuJFxrmDbuT5ycMg@layout_pretrain_data_shdd.10.140.14.204:80/diffusion_layout/images/ ./srcSync/
/mnt/petrelfs/zhaozhiyuan/sensesync --dryrun cp s3://X2B9QEQ012TRZN39YVGA:8CybumhrcYPPt1SqAZPB09txzuJFxrmDbuT5ycMg@layout_pretrain_data_shdd.10.140.14.204:80/random_layout/images/ ./srcSync/

# copy data
/mnt/petrelfs/zhaozhiyuan/sensesync cp s3://X2B9QEQ012TRZN39YVGA:8CybumhrcYPPt1SqAZPB09txzuJFxrmDbuT5ycMg@layout_pretrain_data_shdd.10.140.14.204:80/bestfit_layout/ ./
/mnt/petrelfs/zhaozhiyuan/sensesync cp s3://X2B9QEQ012TRZN39YVGA:8CybumhrcYPPt1SqAZPB09txzuJFxrmDbuT5ycMg@layout_pretrain_data_shdd.10.140.14.204:80/diffusion_layout/ ./
/mnt/petrelfs/zhaozhiyuan/sensesync cp s3://X2B9QEQ012TRZN39YVGA:8CybumhrcYPPt1SqAZPB09txzuJFxrmDbuT5ycMg@layout_pretrain_data_shdd.10.140.14.204:80/random_layout/ ./