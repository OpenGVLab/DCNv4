

# [DCNv4](https://arxiv.org/pdf/2401.06197.pdf)


## News
- `Jan 15, 2024`: 🚀 Compared with InternImage, the new FlashInternImage powered with DCNv4 has faster inference speed, faster convergence, and better performance!!!
- `Jan 15, 2024`: 🚀 "DCNv4" is released！


## Introduction
We introduce Deformable Convolution v4 (DCNv4), a highly efficient and effective operator designed for a broad spectrum of vision applications. DCNv4 addresses the limitations of its predecessor, DCNv3, with two key enhancements: 1. removing softmax normalization in spatial aggregation to enhance its dynamic property and expressive power and 2. optimizing memory access to minimize redundant operations for speedup. These improvements result in a significantly faster convergence compared to DCNv3 and a substantial increase in processing speed, with DCNv4 achieving more than three times the forward speed.
DCNv4 demonstrates exceptional performance across various tasks, including image classification, instance and semantic segmentation, and notably, image generation. 
When integrated into generative models like U-Net in the latent diffusion model, DCNv4 outperforms its baseline, underscoring its possibility to enhance generative models.
In practical applications, replacing DCNv3 with DCNv4 in the InternImage model to create FlashInternImage results in up to 80\% speed increase and further performance improvement without further modifications.
The advancements in speed and efficiency of DCNv4, combined with its robust performance across diverse vision tasks, show its potential as a foundational building block for future vision models.

## Released Models



<details>
<summary> ImageNet Image Classification </summary>
<br>
<div>

|      name      |   pretrain   | resolution | acc@1 | #param | download                                                                              |
| :------------: | :----------: | :--------: | :---: | :----: | :---------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| FlashInternImage-T  | ImageNet-1K  |  224x224   | 83.6  |  30M   |        [ckpt](https://huggingface.co/OpenGVLab/DCNv4/resolve/main/flash_intern_image_t_1k_224.pth) \| [cfg](classification/configs/flash_intern_image_t_1k_224.yaml)       |
| FlashInternImage-S  | ImageNet-1K  |  224x224   | 84.4  |  50M   |        [ckpt](https://huggingface.co/OpenGVLab/DCNv4/resolve/main/flash_intern_image_s_1k_224.pth) \| [cfg](classification/configs/flash_intern_image_s_1k_224.yaml)       |
| FlashInternImage-B  | ImageNet-1K  |  224x224   | 84.9  |  97M   |      [ckpt](https://huggingface.co/OpenGVLab/DCNv4/resolve/main/flash_intern_image_b_1k_224.pth) \| [cfg](classification/configs/flash_intern_image_b_1k_224.yaml)       |
| FlashInternImage-L  | ImageNet-22K |  384x384   | 88.1  |  223M  | [ckpt](https://huggingface.co/OpenGVLab/DCNv4/resolve/main/flash_internimage_l_22kto1k_384.pth) \| [cfg](classification/configs/flash_intern_image_l_22kto1k_384.yaml)  |
</div>

</details>

<details>
<summary> COCO Object Detection and Instance Segmentation </summary>
<br>
<div>

|    backbone   |method |  schd | box mAP | mask mAP |Config | Download | 
| :-----------------:| :----------:  |  :---------: | :-----:  |:------: | :-----: | :---: |
| FlashInternImage-T  |Mask-RCNN|          1x      |  48.0   |   43.1    |  [config](./detection/configs/coco/mask_rcnn_flash_intern_image_t_fpn_1x_coco.py) | [ckpt](https://huggingface.co/OpenGVLab/DCNv4/resolve/main/mask_rcnn_flash_internimage_t_fpn_1x_coco.pth) \| [log](https://huggingface.co/OpenGVLab/DCNv4/resolve/main/mask_rcnn_flash_internimage_t_fpn_1x_coco.log) |
| FlashInternImage-T  |Mask-RCNN |          3x      |  49.5   |   44.0     | [config](./detection/configs/coco/mask_rcnn_flash_intern_image_t_fpn_3x_coco.py) | [ckpt](https://huggingface.co/OpenGVLab/DCNv4/resolve/main/mask_rcnn_flash_internimage_t_fpn_3x_coco.pth) \| [log](https://huggingface.co/OpenGVLab/DCNv4/resolve/main/mask_rcnn_flash_internimage_t_fpn_3x_coco.log) |
| FlashInternImage-S   |Mask-RCNN|          1x      |  49.2   |   44.0    |  [config](./detection/configs/coco/mask_rcnn_flash_intern_image_s_fpn_1x_coco.py) | [ckpt](https://huggingface.co/OpenGVLab/DCNv4/resolve/main/mask_rcnn_flash_internimage_s_fpn_1x_coco.pth) \| [log](https://huggingface.co/OpenGVLab/DCNv4/resolve/main/mask_rcnn_flash_internimage_s_fpn_1x_coco.log) |
| FlashInternImage-S  |Mask-RCNN |          3x      |  50.5   |   44.9   | [config](./detection/configs/coco/mask_rcnn_flash_intern_image_s_fpn_3x_coco.py) | [ckpt](https://huggingface.co/OpenGVLab/DCNv4/resolve/main/mask_rcnn_flash_internimage_s_fpn_3x_coco.pth) \| [log](https://huggingface.co/OpenGVLab/DCNv4/resolve/main/mask_rcnn_flash_internimage_s_fpn_3x_coco.log) |
| FlashInternImage-B  |Mask-RCNN |          1x      |  50.1   |   44.5  | [config](./detection/configs/coco/mask_rcnn_flash_intern_image_b_fpn_1x_coco.py) | [ckpt](https://huggingface.co/OpenGVLab/DCNv4/resolve/main/mask_rcnn_flash_internimage_b_fpn_1x_coco.pth) \| [log](https://huggingface.co/OpenGVLab/DCNv4/resolve/main/mask_rcnn_flash_internimage_b_fpn_1x_coco.log) |
| FlashInternImage-B   |Mask-RCNN|          3x      |  50.6   |   45.4  |  [config](./detection/configs/coco/mask_rcnn_flash_intern_image_b_fpn_3x_coco.py)| [ckpt](https://huggingface.co/OpenGVLab/DCNv4/resolve/main/mask_rcnn_flash_internimage_b_fpn_3x_coco.pth) \| [log](https://huggingface.co/OpenGVLab/DCNv4/resolve/main/mask_rcnn_flash_internimage_b_fpn_3x_coco.log) |

|    backbone    |   method|      schd | box mAP | mask mAP | Config | Download |
| :------------:|  :---------: |  :---------: | :-----: | :------: | :---: | :---: |
| FlashInternImage-L |Cascade Mask R-CNN |        1x      |  55.6   |   48.2     | [config](./detection/configs/coco/cascade_flash_intern_image_l_fpn_1x_coco.py) | [ckpt](https://huggingface.co/OpenGVLab/DCNv4/resolve/main/cascade_flash_internimage_l_fpn_1x_coco.pth) \| [log](https://huggingface.co/OpenGVLab/DCNv4/resolve/main/cascade_flash_internimage_l_fpn_1x_coco.log)
| FlashInternImage-L |Cascade Mask R-CNN |        3x      |  56.7   |   48.9    | [config](./detection/configs/coco/cascade_flash_intern_image_l_fpn_3x_coco.py) | [ckpt](https://huggingface.co/OpenGVLab/DCNv4/resolve/main/cascade_flash_internimage_l_fpn_3x_coco.pth)  |

|    backbone   |method |  lr type     | pretrain    |       schd | box mAP  | Config | Download |
| :------------: |  :---------: |  :---------: |:---------: | :---------: | :-----: |  :---: | :-----: |
| FlashInternImage-T  |DINO| layer-wise lr    | ImageNet-1K  |     1x      |  54.7   |     [config](./detection/configs/coco/dino_4scale_flash_internimage_t_1x_coco.py)     | [ckpt](https://huggingface.co/OpenGVLab/DCNv4/resolve/main/dino_4scale_flash_internimage_t_1x_coco.pth) \| [log](https://huggingface.co/OpenGVLab/DCNv4/resolve/main/dino_4scale_flash_internimage_t_1x_coco.json) |
| FlashInternImage-S  |DINO | layer-wise lr    | ImageNet-1K  |     1x      |  55.3   |    [config](./detection/configs/coco/dino_4scale_flash_internimage_s_1x_coco.py)     | [ckpt](https://huggingface.co/OpenGVLab/DCNv4/resolve/main/dino_4scale_flash_internimage_s_1x_coco.pth) \| [log](https://huggingface.co/OpenGVLab/DCNv4/resolve/main/dino_4scale_flash_internimage_s_1x_coco.log) |
| FlashInternImage-B   |DINO| layer-wise lr    | ImageNet-1K  |     1x      |  56.0     | [config](./detection/configs/coco/dino_4scale_flash_internimage_b_1x_coco.py)     | [ckpt](https://huggingface.co/OpenGVLab/DCNv4/resolve/main/dino_4scale_flash_internimage_b_1x_coco.pth) \| [log](https://huggingface.co/OpenGVLab/DCNv4/resolve/main/dino_4scale_flash_internimage_b_1x_coco.log) |
| FlashInternImage-L  |DINO | 0.1x backbone lr | ImageNet-22K |     1x      |  58.8     |  [config](./detection/configs/coco/dino_4scale_flash_internimage_l_1x_coco.py) | [ckpt](https://huggingface.co/OpenGVLab/DCNv4/resolve/main/dino_4scale_flash_internimage_l_1x_coco.pth) \| [log](https://huggingface.co/OpenGVLab/DCNv4/resolve/main/dino_4scale_flash_internimage_l_1x_coco.log) |


</div>

</details>


<details>
<summary> ADE20K Semantic Segmentation </summary>
<br>
<div>

| backbone      |method | resolution | mIoU (ss/ms) | Config | Download            |
|:--------------:|:----------:|:----------:|:-----------:|:-----------:|:----------:
| FlashInternImage-T|UperNet  | 512x512    | 49.3 / 50.3   | [config](./segmentation/configs/ade20k/upernet_flash_internimage_t_512_160k_ade20k.py) | [ckpt](https://huggingface.co/OpenGVLab/DCNv4/resolve/main/upernet_flash_internimage_t_512_160k_ade20k.pth) \| [log](https://huggingface.co/OpenGVLab/DCNv4/resolve/main/upernet_flash_internimage_t_512_160k_ade20k.log)   | 
| FlashInternImage-S |UperNet   | 512x512    | 50.6 / 51.6     | [config](./segmentation/configs/ade20k/upernet_flash_internimage_s_512_160k_ade20k.py)  | [ckpt](https://huggingface.co/OpenGVLab/DCNv4/resolve/main/upernet_flash_internimage_s_512_160k_ade20k.pth) \| [log](https://huggingface.co/OpenGVLab/DCNv4/resolve/main/upernet_flash_internimage_s_512_160k_ade20k.log)  | 
| FlashInternImage-B |UperNet   | 512x512    | 52.0 / 52.6       | [config](./segmentation/configs/ade20k/upernet_flash_internimage_b_512_160k_ade20k.py) | [ckpt](https://huggingface.co/OpenGVLab/DCNv4/resolve/main/upernet_flash_internimage_b_512_160k_ade20k.pth) \| [log](https://huggingface.co/OpenGVLab/DCNv4/resolve/main/upernet_flash_internimage_s_512_160k_ade20k.log)  | 
| FlashInternImage-L  |UperNet  | 640x640    | 55.6 / 56.0    | [config](./segmentation/configs/ade20k/upernet_flash_internimage_l_640_160k_ade20k.py)| [ckpt](https://huggingface.co/OpenGVLab/DCNv4/resolve/main/upernet_flash_internimage_l_640_160k_ade20k.pth) \| [log](https://huggingface.co/OpenGVLab/DCNv4/resolve/main/upernet_flash_internimage_l_640_160k_ade20k.log)  | 


| backbone      |method | resolution | mIoU (ss) | Config | Download            |
|:--------------:|:----------:|:----------:|:-----------:|:-----------:|:----------:
| FlashInternImage-T  |Mask2Former| 512x512    | 51.2   | [config](./segmentation/configs/ade20k/mask2former_flash_internimage_t_512_160k_ade20k_ss.py) | [ckpt](https://huggingface.co/OpenGVLab/DCNv4/resolve/main/mask2former_flash_internimage_t_512_160k_ade20k_ss.pth) \| [log](https://huggingface.co/OpenGVLab/DCNv4/resolve/main/mask2former_flash_internimage_t_512_160k_ade20k_ss.log)   | 
| FlashInternImage-S   |Mask2Former| 640x640    | 52.6     | [config](./segmentation/configs/ade20k/mask2former_flash_internimage_s_640_160k_ade20k_ss.py)  | [ckpt](https://huggingface.co/OpenGVLab/DCNv4/resolve/main/mask2former_flash_internimage_s_640_160k_ade20k_ss.pth) \| [log](https://huggingface.co/OpenGVLab/DCNv4/resolve/main/mask2former_flash_internimage_s_640_160k_ade20k_ss.log)  | 
| FlashInternImage-B   |Mask2Former| 640x640    |  53.4       | [config](./segmentation/configs/ade20k/mask2former_flash_internimage_b_640_160k_ade20k_ss.py) | [ckpt](https://huggingface.co/OpenGVLab/DCNv4/resolve/main/mask2former_flash_internimage_b_640_160k_ade20k_ss.pth) \| [log](https://huggingface.co/OpenGVLab/DCNv4/resolve/main/mask2former_flash_internimage_b_640_160k_ade20k_ss.log)  | 
| FlashInternImage-L   |Mask2Former| 640x640    | 56.7     | [config](./segmentation/configs/ade20k/mask2former_flash_internimage_l_640_160k_ade20k_ss.py)| [ckpt](https://huggingface.co/OpenGVLab/DCNv4/resolve/main/mask2former_flash_internimage_l_640_160k_ade20k_ss.pth) \| [log](https://huggingface.co/OpenGVLab/DCNv4/resolve/main/mask2former_flash_internimage_l_640_160k_ade20k_ss.log)  | 



</div>

</details>

## Citations

If this work is helpful for your research, please consider citing the following BibTeX entry.

```bibtex

@article{xiong2024efficient,
      title={Efficient Deformable ConvNets: Rethinking Dynamic and Sparse Operator for Vision Applications}, 
      author={Yuwen Xiong and Zhiqi Li and Yuntao Chen and Feng Wang and Xizhou Zhu and Jiapeng Luo and Wenhai Wang and Tong Lu and Hongsheng Li and Yu Qiao and Lewei Lu and Jie Zhou and Jifeng Dai},
      journal={arXiv preprint arXiv:2401.06197},
      year={2024}
}

@article{wang2022internimage,
  title={InternImage: Exploring Large-Scale Vision Foundation Models with Deformable Convolutions},
  author={Wang, Wenhai and Dai, Jifeng and Chen, Zhe and Huang, Zhenhang and Li, Zhiqi and Zhu, Xizhou and Hu, Xiaowei and Lu, Tong and Lu, Lewei and Li, Hongsheng and others},
  journal={arXiv preprint arXiv:2211.05778},
  year={2022}
}

@inproceedings{zhu2022uni,
  title={Uni-perceiver: Pre-training unified architecture for generic perception for zero-shot and few-shot tasks},
  author={Zhu, Xizhou and Zhu, Jinguo and Li, Hao and Wu, Xiaoshi and Li, Hongsheng and Wang, Xiaohua and Dai, Jifeng},
  booktitle={CVPR},
  pages={16804--16815},
  year={2022}
}

@article{zhu2022uni,
  title={Uni-perceiver-moe: Learning sparse generalist models with conditional moes},
  author={Zhu, Jinguo and Zhu, Xizhou and Wang, Wenhai and Wang, Xiaohua and Li, Hongsheng and Wang, Xiaogang and Dai, Jifeng},
  journal={arXiv preprint arXiv:2206.04674},
  year={2022}
}

@article{li2022uni,
  title={Uni-Perceiver v2: A Generalist Model for Large-Scale Vision and Vision-Language Tasks},
  author={Li, Hao and Zhu, Jinguo and Jiang, Xiaohu and Zhu, Xizhou and Li, Hongsheng and Yuan, Chun and Wang, Xiaohua and Qiao, Yu and Wang, Xiaogang and Wang, Wenhai and others},
  journal={arXiv preprint arXiv:2211.09808},
  year={2022}
}

@article{yang2022bevformer,
  title={BEVFormer v2: Adapting Modern Image Backbones to Bird's-Eye-View Recognition via Perspective Supervision},
  author={Yang, Chenyu and Chen, Yuntao and Tian, Hao and Tao, Chenxin and Zhu, Xizhou and Zhang, Zhaoxiang and Huang, Gao and Li, Hongyang and Qiao, Yu and Lu, Lewei and others},
  journal={arXiv preprint arXiv:2211.10439},
  year={2022}
}

@article{su2022towards,
  title={Towards All-in-one Pre-training via Maximizing Multi-modal Mutual Information},
  author={Su, Weijie and Zhu, Xizhou and Tao, Chenxin and Lu, Lewei and Li, Bin and Huang, Gao and Qiao, Yu and Wang, Xiaogang and Zhou, Jie and Dai, Jifeng},
  journal={arXiv preprint arXiv:2211.09807},
  year={2022}
}

@inproceedings{li2022bevformer,
  title={Bevformer: Learning bird’s-eye-view representation from multi-camera images via spatiotemporal transformers},
  author={Li, Zhiqi and Wang, Wenhai and Li, Hongyang and Xie, Enze and Sima, Chonghao and Lu, Tong and Qiao, Yu and Dai, Jifeng},
  booktitle={ECCV},
  pages={1--18},
  year={2022},
}
```
