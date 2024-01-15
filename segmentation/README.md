# FlashInternImage for Semantic Segmentation

This folder contains the implementation of the InternImage for semantic segmentation. 

Our segmentation code is developed on top of [MMSegmentation v0.27.0](https://github.com/open-mmlab/mmsegmentation/tree/v0.27.0).

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

For examples, to install torch==1.11 with CUDA==11.3 and nvcc:
```bash
conda install pytorch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0 cudatoolkit=11.3 -c pytorch -y
conda install -c conda-forge cudatoolkit-dev=11.3 -y # to install nvcc
```

- Install other requirements:

  note: conda opencv will break torchvision as not to support GPU, so we need to install opencv using pip. 	  

```bash
conda install -c conda-forge termcolor yacs pyyaml scipy pip -y
pip install opencv-python
```

- Install `timm` and `mmcv-full` and `mmsegmentation':

```bash
pip install -U openmim
mim install mmcv-full==1.5.0
mim install mmsegmentation==0.27.0
pip install timm==0.6.11 mmdet==2.28.1
```

- Install DCNv4
```bash
pip install DCNv4
```

### Data Preparation

Prepare datasets according to the [guidelines](https://github.com/open-mmlab/mmsegmentation/blob/master/docs/en/dataset_prepare.md#prepare-datasets) in MMSegmentation.


### Evaluation

To evaluate our `FlashInternImage` on ADE20K val, run:

```bash
sh dist_test.sh <config-file> <checkpoint> <gpu-num> --eval mIoU
```
You can download checkpoint files from [here](https://huggingface.co/OpenGVLab/DCNv4). Then place it to segmentation/checkpoint_dir/seg.

For example, to evaluate the `FlashInternImage-T` with a single GPU:

```bash
python test.py configs/ade20k/upernet_flash_internimage_t_512_160k_ade20k.py checkpoint_dir/seg/upernet_flash_internimage_t_512_160k_ade20k.pth --eval mIoU
```

For example, to evaluate the `FlashInternImage-B` with a single node with 8 GPUs:

```bash
sh dist_test.sh configs/ade20k/upernet_flash_internimage_b_512_160k_ade20k.py checkpoint_dir/seg/upernet_flash_internimage_b_512_160k_ade20k.pth 8 --eval mIoU
```

### Training

To train an `FlashInternImage` on ADE20K, run:

```bash
sh dist_train.sh <config-file> <gpu-num>
```

For example, to train `FlashInternImage-T` with 8 GPU on 1 node (total batch size 16), run:

```bash
sh dist_train.sh configs/ade20k/upernet_flash_internimage_t_512_160k_ade20k.py 8
```
