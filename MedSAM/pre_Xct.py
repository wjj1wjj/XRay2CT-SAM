from segment_anything.build_sam import _build_sam_zm
from segment_anything.modeling.image_encoder import ImageEncoderViT
import torch
import os
import argparse
from segment_anything import sam_model_registry
from skimage import io, transform
import torch.nn.functional as F
import numpy as np
from functools import partial
import cfg_file
args = cfg_file.parse_args()



device = torch.device(args.device)

medsam_model = _build_sam_zm(args,
                             encoder_embed_dim=768,
                             encoder_depth=12,
                             encoder_num_heads=12,
                             encoder_global_attn_indexes=[2, 5, 8, 11],
                             checkpoint=args.checkpoint,
                             )

encoder = ImageEncoderViT(args,
                          depth=12,
                          embed_dim=768,
                          img_size=1024,
                          mlp_ratio=4,
                          norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),
                          num_heads=12,
                          patch_size=16,
                          qkv_bias=True,
                          use_rel_pos=True,
                          global_attn_indexes=[2, 5, 8, 11],
                          window_size=14,
                          out_chans=256,
                          )

encoder_state_dict={}
for k, v in medsam_model.state_dict().items():
    print(k)
    if k.startswith('image_encoder.'):
        new_key = k.replace('image_encoder.', '')
        encoder_state_dict[new_key] = v