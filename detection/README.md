# FlashInternImage for Object Detection

This folder contains the implementation of the FlashInternImage for object detection. 

Our detection code is developed on top of [MMDetection v2.28.1](https://github.com/open-mmlab/mmdetection/tree/v2.28.1).


## Usage

### Install

- Clone this repo:

```bash
git clone https://github.com/OpenGVLab/DCNv4.git
cd DCNv4
```

- Create a conda virtual environment and activate it:

```bash
conda create -n dcnv4 python=3.7 -y
conda activate dcnv4
```

- Install `CUDA>=10.2` with `cudnn>=7` following
  the [official installation instructions](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html)
- Install `PyTorch>=1.10.0` and `torchvision>=0.9.0` with `CUDA>=10.2`:

For examples, to install torch==1.11 with CUDA==11.3:
```bash
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113  -f https://download.pytorch.org/whl/torch_stable.html
```

- Install `timm==0.6.11` and `mmcv-full==1.5.0`:

```bash
pip install -U openmim
mim install mmcv-full==1.5.0
pip install timm==0.6.11 mmdet==2.28.1
```

- Install other requirements:

```bash
pip install opencv-python termcolor yacs pyyaml scipy
```

- Install DCNv4
```bash
pip install DCNv4
```


### Data Preparation

Prepare COCO according to the guidelines in [MMDetection v2.28.1](https://github.com/open-mmlab/mmdetection/resolve/master/docs/en/1_exist_data_model.md).


### Evaluation

To evaluate our `FlashInternImage` on COCO val, run:

```bash
sh dist_test.sh <config-file> <checkpoint> <gpu-num> --eval bbox segm
```

For example, to evaluate the `FlashInternImage-T` with a single GPU:

```bash
python test.py configs/coco/mask_rcnn_flash_intern_image_t_fpn_1x_coco.py checkpoint_dir/det/mask_rcnn_flash_internimage_t_fpn_1x_coco.pth --eval bbox segm
```

For example, to evaluate the `FlashInternImage-B` with a single node with 8 GPUs:

```bash
sh dist_test.sh configs/coco/mask_rcnn_flash_intern_image_b_fpn_1x_coco.py checkpoint_dir/det/mask_rcnn_flash_internimage_b_fpn_1x_coco.py 8 --eval bbox segm
```

### Training on COCO

To train an `FlashInternImage` on COCO, run:

```bash
sh dist_train.sh <config-file> <gpu-num>
```

For example, to train `FlashInternImage-T` with 8 GPU on 1 node, run:

```bash
sh dist_train.sh configs/coco/mask_rcnn_flash_intern_image_t_fpn_1x_coco.py 8
```

