# ADE20K

Introduced by Zhou et al. in [Scene Parsing Through ADE20K Dataset](https://paperswithcode.com/paper/scene-parsing-through-ade20k-dataset).

The ADE20K semantic segmentation dataset contains more than 20K scene-centric images exhaustively annotated with pixel-level objects and object parts labels. There are totally 150 semantic categories, which include stuffs like sky, road, grass, and discrete objects like person, car, bed.


## Model Zoo

### UperNet + InternImage


| backbone       | resolution | mIoU (ss/ms) | Config | Download            |
|:--------------:|:----------:|:-----------:|:-----------:|:----------:
| FlashInternImage-T  | 512x512    | 49.3 / 50.3   | [config](./upernet_flash_internimage_t_512_160k_ade20k.py) | [ckpt](https://huggingface.co/OpenGVLab/DCNv4/resolve/main/upernet_flash_internimage_t_512_160k_ade20k.pth) \| [log](https://huggingface.co/OpenGVLab/DCNv4/resolve/main/upernet_flash_internimage_t_512_160k_ade20k.log)   | 
| FlashInternImage-S  | 512x512    | 50.6 / 51.6     | [config](./upernet_flash_internimage_s_512_160k_ade20k.py)  | [ckpt](https://huggingface.co/OpenGVLab/DCNv4/resolve/main/upernet_flash_internimage_s_512_160k_ade20k.pth) \| [log](https://huggingface.co/OpenGVLab/DCNv4/resolve/main/upernet_flash_internimage_s_512_160k_ade20k.log)  | 
| FlashInternImage-B  | 512x512    | 52.0 / 52.6       | [config](./upernet_flash_internimage_b_512_160k_ade20k.py) | [ckpt](https://huggingface.co/OpenGVLab/DCNv4/resolve/main/upernet_flash_internimage_b_512_160k_ade20k.pth) \| [log](https://huggingface.co/OpenGVLab/DCNv4/resolve/main/upernet_flash_internimage_s_512_160k_ade20k.log)  | 
| FlashInternImage-L  | 640x640    | 55.6 / 56.0    | [config](./upernet_flash_internimage_l_640_160k_ade20k.py)| [ckpt](https://huggingface.co/OpenGVLab/DCNv4/resolve/main/upernet_flash_internimage_l_640_160k_ade20k.pth) \| [log](https://huggingface.co/OpenGVLab/DCNv4/resolve/main/upernet_flash_internimage_l_640_160k_ade20k.log)  | 

- Training speed is measured with A100 GPU.
- Please set `with_cp=True` to save memory if you meet `out-of-memory` issues.
- The logs are our recent newly trained ones. There are slight differences between the results in logs and our paper.


### Mask2Former + InternImage

| backbone       | resolution | mIoU (ss) | Config | Download            |
|:--------------:|:----------:|:-----------:|:-----------:|:----------:
| FlashInternImage-T  | 512x512    | 51.2   | [config](./mask2former_flash_internimage_t_512_160k_ade20k_ss.py) | [ckpt](https://huggingface.co/OpenGVLab/DCNv4/resolve/main/mask2former_flash_internimage_t_512_160k_ade20k_ss.pth) \| [log](https://huggingface.co/OpenGVLab/DCNv4/resolve/main/mask2former_flash_internimage_t_512_160k_ade20k_ss.log)   | 
| FlashInternImage-S  | 640x640    | 52.2     | [config](./mask2former_flash_internimage_s_640_160k_ade20k_ss.py)  | [ckpt](https://huggingface.co/OpenGVLab/DCNv4/resolve/main/mask2former_flash_internimage_s_640_160k_ade20k_ss.pth) \| [log](https://huggingface.co/OpenGVLab/DCNv4/resolve/main/mask2former_flash_internimage_s_640_160k_ade20k_ss.log)  | 
| FlashInternImage-B  | 640x640    |  53.4       | [config](./mask2former_flash_internimage_b_640_160k_ade20k_ss.py) | [ckpt](https://huggingface.co/OpenGVLab/DCNv4/resolve/main/mask2former_flash_internimage_b_640_160k_ade20k_ss.pth) \| [log](https://huggingface.co/OpenGVLab/DCNv4/resolve/main/mask2former_flash_internimage_b_640_160k_ade20k_ss.log)  | 
| FlashInternImage-L  | 640x640    | 56.7     | [config](./mask2former_flash_internimage_l_640_160k_ade20k_ss.py)| [ckpt](https://huggingface.co/OpenGVLab/DCNv4/resolve/main/mask2former_flash_internimage_l_640_160k_ade20k_ss.pth) \| [log](https://huggingface.co/OpenGVLab/DCNv4/resolve/main/mask2former_flash_internimage_l_640_160k_ade20k_ss.log)  | 

