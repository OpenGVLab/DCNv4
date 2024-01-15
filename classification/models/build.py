# --------------------------------------------------------
# DCNv4
# Copyright (c) 2024 OpenGVLab
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------


from .intern_image import InternImage
from .flash_intern_image import FlashInternImage

def build_model(config):
    model_type = config.MODEL.TYPE
    if model_type == 'intern_image':
        model = InternImage(
            core_op=config.MODEL.INTERN_IMAGE.CORE_OP,
            num_classes=config.MODEL.NUM_CLASSES,
            channels=config.MODEL.INTERN_IMAGE.CHANNELS,
            depths=config.MODEL.INTERN_IMAGE.DEPTHS,
            groups=config.MODEL.INTERN_IMAGE.GROUPS,
            layer_scale=config.MODEL.INTERN_IMAGE.LAYER_SCALE,
            offset_scale=config.MODEL.INTERN_IMAGE.OFFSET_SCALE,
            post_norm=config.MODEL.INTERN_IMAGE.POST_NORM,
            mlp_ratio=config.MODEL.INTERN_IMAGE.MLP_RATIO,
            with_cp=config.TRAIN.USE_CHECKPOINT,
            drop_path_rate=config.MODEL.DROP_PATH_RATE,
            res_post_norm=config.MODEL.INTERN_IMAGE.RES_POST_NORM, # for InternImage-H/G
            dw_kernel_size=config.MODEL.INTERN_IMAGE.DW_KERNEL_SIZE, # for InternImage-H/G
            use_clip_projector=config.MODEL.INTERN_IMAGE.USE_CLIP_PROJECTOR, # for InternImage-H/G
            level2_post_norm=config.MODEL.INTERN_IMAGE.LEVEL2_POST_NORM, # for InternImage-H/G
            level2_post_norm_block_ids=config.MODEL.INTERN_IMAGE.LEVEL2_POST_NORM_BLOCK_IDS, # for InternImage-H/G
            center_feature_scale=config.MODEL.INTERN_IMAGE.CENTER_FEATURE_SCALE # for InternImage-H/G
        )
    elif model_type == 'flash_intern_image':
        model = FlashInternImage(
            core_op=config.MODEL.FLASH_INTERN_IMAGE.CORE_OP,
            num_classes=config.MODEL.NUM_CLASSES,
            channels=config.MODEL.FLASH_INTERN_IMAGE.CHANNELS,
            depths=config.MODEL.FLASH_INTERN_IMAGE.DEPTHS,
            groups=config.MODEL.FLASH_INTERN_IMAGE.GROUPS,
            layer_scale=config.MODEL.FLASH_INTERN_IMAGE.LAYER_SCALE,
            offset_scale=config.MODEL.FLASH_INTERN_IMAGE.OFFSET_SCALE,
            post_norm=config.MODEL.FLASH_INTERN_IMAGE.POST_NORM,
            mlp_ratio=config.MODEL.FLASH_INTERN_IMAGE.MLP_RATIO,
            with_cp=config.TRAIN.USE_CHECKPOINT,
            drop_path_rate=config.MODEL.DROP_PATH_RATE,
            mlp_fc2_bias=config.MODEL.FLASH_INTERN_IMAGE.MLP_FC2_BIAS,
            dcn_output_bias=config.MODEL.FLASH_INTERN_IMAGE.DCN_OUTPUT_BIAS,
            res_post_norm=config.MODEL.FLASH_INTERN_IMAGE.RES_POST_NORM, # for InternImage-H/G
            dw_kernel_size=config.MODEL.FLASH_INTERN_IMAGE.DW_KERNEL_SIZE,
            use_clip_projector=config.MODEL.FLASH_INTERN_IMAGE.USE_CLIP_PROJECTOR, # for InternImage-H/G
            level2_post_norm=config.MODEL.FLASH_INTERN_IMAGE.LEVEL2_POST_NORM, # for InternImage-H/G
            level2_post_norm_block_ids=config.MODEL.FLASH_INTERN_IMAGE.LEVEL2_POST_NORM_BLOCK_IDS, # for InternImage-H/G
            center_feature_scale=config.MODEL.FLASH_INTERN_IMAGE.CENTER_FEATURE_SCALE # for InternImage-H/G
        )
    else:
        raise NotImplementedError(f"Unkown model: {model_type}")

    return model
