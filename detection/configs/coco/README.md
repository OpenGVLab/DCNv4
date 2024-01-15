# COCO

## Introduction

Introduced by Lin et al. in [Microsoft COCO: Common Objects in Context](https://arxiv.org/pdf/1405.0312v3.pdf)

The MS COCO (Microsoft Common Objects in Context) dataset is a large-scale object detection, segmentation, key-point detection, and captioning dataset. The dataset consists of 328K images.

Splits: The first version of MS COCO dataset was released in 2014. It contains 164K images split into training (83K), validation (41K) and test (41K) sets. In 2015 additional test set of 81K images was released, including all the previous test images and 40K new images.

Based on community feedback, in 2017 the training/validation split was changed from 83K/41K to 118K/5K. The new split uses the same images and annotations. The 2017 test set is a subset of 41K images of the 2015 test set. Additionally, the 2017 release contains a new unannotated dataset of 123K images.


## Model Zoo

### Mask R-CNN + FlashInternImage


|    backbone    |  schd | box mAP | mask mAP |Config | Download | 
| :-----------------: |  :---------: | :-----:  |:------: | :-----: | :---: |
| FlashInternImage-T  |          1x      |  48.0   |   43.1    |  [config](./mask_rcnn_flash_intern_image_t_fpn_1x_coco.py) | [ckpt](https://huggingface.co/OpenGVLab/DCNv4/resolve/main/mask_rcnn_flash_internimage_t_fpn_1x_coco.pth) \| [log](https://huggingface.co/OpenGVLab/DCNv4/resolve/main/mask_rcnn_flash_internimage_t_fpn_1x_coco.log) |
| FlashInternImage-T  |          3x      |  49.5   |   44.0     | [config](././mask_rcnn_flash_intern_image_t_fpn_3x_coco.py) | [ckpt](https://huggingface.co/OpenGVLab/DCNv4/resolve/main/mask_rcnn_flash_internimage_t_fpn_3x_coco.pth) \| [log](https://huggingface.co/OpenGVLab/DCNv4/resolve/main/mask_rcnn_flash_internimage_t_fpn_3x_coco.log) |
| FlashInternImage-S  |          1x      |  49.2   |   44.0    |  [config](./mask_rcnn_flash_intern_image_s_fpn_1x_coco.py) | [ckpt](https://huggingface.co/OpenGVLab/DCNv4/resolve/main/mask_rcnn_flash_internimage_s_fpn_1x_coco.pth) \| [log](https://huggingface.co/OpenGVLab/DCNv4/resolve/main/mask_rcnn_flash_internimage_s_fpn_1x_coco.log) |
| FlashInternImage-S  |          3x      |  50.5   |   44.9   | [config](./mask_rcnn_flash_intern_image_s_fpn_3x_coco.py) | [ckpt](https://huggingface.co/OpenGVLab/DCNv4/resolve/main/mask_rcnn_flash_internimage_s_fpn_3x_coco.pth) \| [log](https://huggingface.co/OpenGVLab/DCNv4/resolve/main/mask_rcnn_flash_internimage_s_fpn_3x_coco.log) |
| FlashInternImage-B  |          1x      |  50.1   |   44.5  | [config](./mask_rcnn_flash_intern_image_b_fpn_1x_coco.py) | [ckpt](https://huggingface.co/OpenGVLab/DCNv4/resolve/main/mask_rcnn_flash_internimage_b_fpn_1x_coco.pth) \| [log](https://huggingface.co/OpenGVLab/DCNv4/resolve/main/mask_rcnn_flash_internimage_b_fpn_1x_coco.log) |
| FlashInternImage-B  |          3x      |  50.6   |   45.4  |  [config](./mask_rcnn_flash_intern_image_b_fpn_3x_coco.py)| [ckpt](https://huggingface.co/OpenGVLab/DCNv4/resolve/main/mask_rcnn_flash_internimage_b_fpn_3x_coco.pth) \| [log](https://huggingface.co/OpenGVLab/DCNv4/resolve/main/mask_rcnn_flash_internimage_b_fpn_3x_coco.log) |

- Training speed is measured with A100 GPUs using current code and may be faster than the speed in logs.
- Some logs are our recent newly trained ones. There might be slight differences between the results in logs and our paper.
- Please set `with_cp=True` to save memory if you meet `out-of-memory` issues.

### Cascade Mask R-CNN + FlashInternImage

|    backbone    |         schd | box mAP | mask mAP | Config | Download |
| :------------: |  :---------: | :-----: | :------: | :---: | :---: |
| FlashInternImage-L  |        1x      |  55.6   |   48.2     | [config](./cascade_flash_intern_image_l_fpn_1x_coco.py) | [ckpt](https://huggingface.co/OpenGVLab/DCNv4/resolve/main/cascade_flash_internimage_l_fpn_1x_coco.pth) \| [log](https://huggingface.co/OpenGVLab/DCNv4/resolve/main/cascade_flash_internimage_l_fpn_1x_coco.log)
| FlashInternImage-L  |        3x      |  56.7   |   48.9    | [config](./cascade_flash_intern_image_l_fpn_3x_coco.py) | [ckpt](https://huggingface.co/OpenGVLab/DCNv4/resolve/main/cascade_flash_internimage_l_fpn_3x_coco.pth)  |

- Training speed is measured with A100 GPUs using current code and may be faster than the speed in logs.
- Some logs are our recent newly trained ones. There might be slight differences between the results in logs and our paper.
- Please set `with_cp=True` to save memory if you meet `out-of-memory` issues.


### DINO + FlashInternImage
|    backbone    |  lr type     | pretrain    |       schd | box mAP  | Config | Download |
| :------------: |  :---------: |:---------: | :---------: | :-----: |  :---: | :-----: 
| FlashInternImage-T  | layer-wise lr    | ImageNet-1K  |     1x      |  54.7   |     [config](./dino_4scale_flash_internimage_t_1x_coco.py)     | [ckpt](https://huggingface.co/OpenGVLab/DCNv4/resolve/main/dino_4scale_flash_internimage_t_1x_coco.pth) \| [log](https://huggingface.co/OpenGVLab/DCNv4/resolve/main/dino_4scale_flash_internimage_t_1x_coco.json) |
| FlashInternImage-S  | layer-wise lr    | ImageNet-1K  |     1x      |  55.3   |    [config](./dino_4scale_flash_internimage_s_1x_coco.py)     | [ckpt](https://huggingface.co/OpenGVLab/DCNv4/resolve/main/dino_4scale_flash_internimage_s_1x_coco.pth) \| [log](https://huggingface.co/OpenGVLab/DCNv4/resolve/main/dino_4scale_flash_internimage_s_1x_coco.log) |
| FlashInternImage-B  | layer-wise lr    | ImageNet-1K  |     1x      |  56.0     | [config](./dino_4scale_flash_internimage_b_1x_coco.py)     | [ckpt](https://huggingface.co/OpenGVLab/DCNv4/resolve/main/dino_4scale_flash_internimage_b_1x_coco.pth) \| [log](https://huggingface.co/OpenGVLab/DCNv4/resolve/main/dino_4scale_flash_internimage_b_1x_coco.log) |
| FlashInternImage-L  | 0.1x backbone lr | ImageNet-22K |     1x      |  58.8     |  [config](./dino_4scale_flash_internimage_l_1x_coco.py) | [ckpt](https://huggingface.co/OpenGVLab/DCNv4/resolve/main/dino_4scale_flash_internimage_l_1x_coco.pth) \| [log](https://huggingface.co/OpenGVLab/DCNv4/resolve/main/dino_4scale_flash_internimage_l_1x_coco.log) |

