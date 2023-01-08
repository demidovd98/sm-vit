# Running

For training and evaluation, please, follow the below-mentioned respective instructions.

NOTE: In case you have multiple CUDA versions installed, please, make sure to initialise the appropriate system CUDA version before running any command.
```bash
# <xx.x> - CUDA version number
module load cuda-xx.x 
```

<hr />


## Stanford Dogs

### Train + Test:

```bash
python3 -W ignore -m torch.distributed.launch --nproc_per_node 1 train.py --name cub --dataset CUB --img_size 400 --train_batch_size 16 --eval_batch_size 8 --learning_rate 0.03 --num_steps 40000 --sm_vit --coeff_max 0.3 --fp16 --low_memory --data_root '<your_dataset_path>'
```

### Test only:


<hr />


## CUB-200-2011

### Train + Test:

```bash
python3 -W ignore -m torch.distributed.launch --nproc_per_node 1 train.py --name cub --dataset CUB --img_size 400 --train_batch_size 16 --eval_batch_size 8 --learning_rate 0.03 --num_steps 40000 --sm_vit --coeff_max 0.25 --fp16 --low_memory --data_root '<your_dataset_path>'
```

### Test only:


<hr />


## NABirds

### Train + Test:

```bash
python3 -W ignore -m torch.distributed.launch --nproc_per_node 1 train.py --name cub --dataset CUB --img_size 400 --train_batch_size 16 --eval_batch_size 8 --learning_rate 0.03 --num_steps 40000 --sm_vit --coeff_max 0.25 --fp16 --low_memory --data_root '<your_dataset_path>'
```

### Test only:

