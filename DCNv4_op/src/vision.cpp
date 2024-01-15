/*!
**************************************************************************************************
* Deformable DETR
* Copyright (c) 2020 SenseTime. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 [see LICENSE for details]
**************************************************************************************************
* Modified from
*https://github.com/chengdazhi/Deformable-Convolution-V2-PyTorch/tree/pytorch_1.0.0
**************************************************************************************************
*/

#include "dcnv4.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("flash_deform_attn_forward", &flash_deform_attn_forward,
        "flash_deform_attn_forward");
  m.def("flash_deform_attn_backward", &flash_deform_attn_backward,
        "flash_deform_attn_backward");
  m.def("dcnv4_forward", &dcnv4_forward, "dcnv4_forward");
  m.def("dcnv4_backward", &dcnv4_backward, "dcnv4_backward");
}
