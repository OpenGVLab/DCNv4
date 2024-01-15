python -m torch.distributed.launch --nproc_per_node 1 --master_port 12345 main.py --eval \
--cfg configs/flash_intern_image_l_22k_384.yaml  --data-path /path/to/imagenet1k
