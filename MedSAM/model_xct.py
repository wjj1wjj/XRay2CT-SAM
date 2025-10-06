import numpy as np
import math
import torch
from torch import nn
from torch.nn import init
from collections import OrderedDict
from torch.nn import functional as F
import math 
from segment_anything.build_sam import _build_sam_zm
from segment_anything.modeling.image_encoder import ImageEncoderViT
import os
import argparse
from segment_anything import sam_model_registry
from skimage import io, transform
import torch.nn.functional as F
from functools import partial
from segment_anything.modeling.swin_trans import *
from segment_anything.utils import *
from segment_anything.modeling.blocks_ada.block import *

import cfg_file
args = cfg_file.parse_args()
class BM_encoder(nn.Module):
    def __init__(self,img_size=128,
                 checkpoint="MedSAM/work_dir/MedSAM/SAM/sam_vit_b_01ec64.pth", 
                 sam_model = _build_sam_zm(args=args),
                 model=ImageEncoderViT(args),
                 ):
        super().__init__()
        self.img_size = img_size
        self.checkpoint = checkpoint
        self.sam ,self.para_list= sam_model
        self.encoder = model
        self.encoder_state_dict={}
        keys,new_para=[],[]
        for k, v in self.sam.state_dict().items():
            if k.startswith('image_encoder.'):
                new_key = k.replace('image_encoder.', '')
                keys.append(new_key)
                self.encoder_state_dict[new_key] = v
        self.encoder.load_state_dict(self.encoder_state_dict,strict=False)

        for k,v in self.para_list.items():
            if k.startswith('image_encoder.'):
                temp = k.replace('image_encoder.', '')
                new_para.append(temp)
        for name, param in self.encoder.named_parameters():
            if name in new_para:
                param.requires_grad = False
            else:
                param.requires_grad = True
    def forward(self,x):
        out = self.encoder(x)
        return out

# Convert the feature map from 2D to 3D
class TransformBlock(nn.Module):
    def __init__(self, in_ch, k):
        super(TransformBlock,self).__init__()
        self.layer1 = nn.ConvTranspose3d(in_ch,in_ch,(k,1,1),1,0)
        self.layer2 = nn.BatchNorm3d(in_ch)
        self.layer3 = nn.ReLU(inplace = True)
    def forward(self,x):
        result = torch.unsqueeze(x, dim=2)
        result = self.layer1(result)
        result = self.layer2(result)
        result = self.layer3(result)
        return result

class BasicTranspose3d(nn.Module):
    def __init__(self, in_ch,out_ch, kernel=4, stride=2, padding=1):
        super(BasicTranspose3d,self).__init__()
        self.layer1 = nn.ConvTranspose3d(in_ch,out_ch,kernel,stride,padding)
        self.layer2 = nn.BatchNorm3d(out_ch)
        self.layer3 = nn.ReLU(inplace = True)
    def forward(self,x):
        result = self.layer1(x)
        result = self.layer2(result)
        result = self.layer3(result)
        return result
#Up 
class BasicTranspose2d(nn.Module):
    def __init__(self, in_ch,out_ch, kernel=4, stride=2, padding=1):
        super(BasicTranspose2d,self).__init__()
        self.layer1 = nn.ConvTranspose2d(in_ch,out_ch,kernel,stride,padding)
        self.layer2 = nn.BatchNorm2d(out_ch)
        self.layer3 = nn.ReLU(inplace = True)
    def forward(self,x):
        result = self.layer1(x)
        result = self.layer2(result)
        result = self.layer3(result)
        return result

# Multi-scale feature extraction
class NewInceptionBlock(nn.Module):
    def __init__(self, in_ch, out_ch1, out_ch2, out_ch3, stride=1):
        super(NewInceptionBlock,self).__init__()
        self.top1 = nn.Sequential(
            nn.Conv3d(in_ch,out_ch1,1,stride,padding=0,bias=False),
            nn.BatchNorm3d(out_ch1),
            nn.ReLU(inplace = True)  
        )
        self.down1 = nn.Sequential(
            nn.Conv3d(out_ch1,out_ch2,3,stride,padding=1,bias=False),
            nn.BatchNorm3d(out_ch2),
            nn.ReLU(inplace = True)  
        )
        self.down2 = nn.Sequential(
            nn.Conv3d(out_ch1,out_ch3,5,stride,padding=2,bias=False),
            nn.BatchNorm3d(out_ch3),
            nn.ReLU(inplace = True)  
        )
    def forward(self,x):
        out = self.top1(x)
        temp1 = self.down1(out)
        temp2 = self.down2(out)
        result = torch.cat((temp1,temp2),dim=1)
        return result

class NewInceptionBlock2d(nn.Module):
    def __init__(self, in_ch, out_ch1, out_ch2, out_ch3, stride=1):
        super(NewInceptionBlock2d,self).__init__()
        self.top1 = nn.Sequential(
            nn.Conv2d(in_ch,out_ch1,1,stride,padding=0,bias=False),
            nn.BatchNorm2d(out_ch1),
            nn.ReLU(inplace = True)  
        )
        self.down1 = nn.Sequential(
            nn.Conv2d(out_ch1,out_ch2,3,stride,padding=1,bias=False),
            nn.BatchNorm2d(out_ch2),
            nn.ReLU(inplace = True)  
        )
        self.down2 = nn.Sequential(
            nn.Conv2d(out_ch1,out_ch3,5,stride,padding=2,bias=False),
            nn.BatchNorm2d(out_ch3),
            nn.ReLU(inplace = True)  
        )
    def forward(self,x):
        out = self.top1(x)
        temp1 = self.down1(out)
        temp2 = self.down2(out)
        result = torch.cat((temp1,temp2),dim=1)
        return result

class transform_layer(nn.Module):
    def __init__(self, in_ch,out_ch, kernel=4, stride=2, padding=1):
        super(BasicTranspose3d,self).__init__()
        self.layer1 = nn.ConvTranspose3d(in_ch,out_ch,kernel,stride,padding)
        self.layer2 = nn.BatchNorm3d(out_ch)
        self.layer3 = nn.ReLU(inplace = True)
    def forward(self,x):
        result = self.layer1(x)
        result = self.layer2(result)
        result = self.layer3(result)
        return result

class Agent_Attention_3D(nn.Module):
    """Agent Attention block."""

    def __init__(
        self,
        dim: int,
        feature_size:int,
        num_heads: int = 8,
        agent_num: int = 64,
        qkv_bias: bool = True,
        use_rel_pos: bool = False,
        rel_pos_zero_init: bool = True,
        input_size: Optional[Tuple[int, int]] = None,
    ) -> None:
        super().__init__()
        self.feature_size = feature_size
        self.num_heads = num_heads if dim>num_heads else dim
        head_dim = dim // self.num_heads
        self.scale = head_dim**-0.5
        self.agent_num = agent_num

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)#proj
        self.softmax = nn.Softmax(dim=-1)
        
        pool_size = round(math.pow(agent_num,(1/3)))
        self.pool = nn.AdaptiveAvgPool3d(output_size=(pool_size, pool_size,pool_size))
        self.dwc = nn.Conv3d(dim,dim,kernel_size=3,stride=1,padding=1)

        self.use_rel_pos = use_rel_pos
        if self.use_rel_pos:
            assert (
                input_size is not None
            ), "Input size must be provided if using relative positional encoding."
            # initialize relative positional embeddings
            self.rel_h = nn.Parameter(torch.zeros(2 * input_size[0] - 1, head_dim))
            self.rel_w = nn.Parameter(torch.zeros(2 * input_size[1] - 1, head_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape 
        head_dim = C // self.num_heads
        H,W,K = self.feature_size,self.feature_size,self.feature_size
        
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.reshape(3, B*self.num_heads, N, -1).unbind(0)
        
        A = self.pool(q.reshape(B, H, W, K, C).permute(0, 4, 1, 2, 3)).reshape(B,C,-1).permute(0,2,1)
        A = A.reshape(B, self.agent_num, self.num_heads, head_dim).permute(0, 2, 1, 3).reshape(B*self.num_heads,self.agent_num,head_dim)
        
        
        agent_attn = self.softmax((A * self.scale) @ k.transpose(-2, -1))
        agent_v = agent_attn @ v
        
        q_attn = self.softmax((q * self.scale) @ A.transpose(-2, -1))
        x = q_attn @ agent_v
        
        x = x.transpose(1, 2).reshape(B, N, C)
        v = v.transpose(1, 2).reshape(B,C,H,W,K)
        x=x+self.dwc(v).permute(0, 2, 3,4, 1).reshape(B,N,C)
        x = self.proj(x)

        return x

class Multi_Task_Learning_1(nn.Module):
    """Multi-head Attention block with relative position embeddings."""

    def __init__(
        self,
        dim: int,
        feature_size:int,
        num_heads: int = 8,
        qkv_bias: bool = True,
        use_rel_pos: bool = False,
        rel_pos_zero_init: bool = True,
        input_size: Optional[Tuple[int, int]] = None,
    ) -> None:
        super().__init__()
        self.feature_size = feature_size
        self.num_heads = num_heads if dim>num_heads else dim
        head_dim = dim // self.num_heads
        self.scale = head_dim**-0.5
        self.scale_channel = (self.feature_size**2 // self.num_heads)**-0.5

        self.qkv_x = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.qkv_y = nn.Linear(dim, dim * 3, bias=qkv_bias)
        
        self.qkv_x_channel = nn.Linear(self.feature_size**2, (self.feature_size**2) * 3, bias=qkv_bias)
        self.qkv_y_channel = nn.Linear(self.feature_size**2, (self.feature_size**2) * 3, bias=qkv_bias)
        self.proj_x_channel = nn.Linear(self.feature_size**2, self.feature_size**2,bias=qkv_bias)
        self.proj_y_channel = nn.Linear(self.feature_size**2, self.feature_size**2,bias=qkv_bias)
        
        self.add_x = nn.Linear(dim, dim,bias=qkv_bias)
        self.add_y = nn.Linear(dim, dim,bias=qkv_bias)
        
        self.use_rel_pos = use_rel_pos
        if self.use_rel_pos:
            assert (
                input_size is not None
            ), "Input size must be provided if using relative positional encoding."
            # initialize relative positional embeddings
            self.rel_h = nn.Parameter(torch.zeros(2 * input_size[0] - 1, head_dim))
            self.rel_w = nn.Parameter(torch.zeros(2 * input_size[1] - 1, head_dim))
            
    def forward(self, x: torch.Tensor,y: torch.Tensor) -> torch.Tensor:
        B, C,H,W = x.shape
        N=H*W
        x_=x.permute(0,2,3,1).reshape(B,N,C)
        y_=y.permute(0,2,3,1).reshape(B,N,C)
        qkv_x = self.qkv_x(x_).reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        qkv_y = self.qkv_y(y_).reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        
        x_channel=x.reshape(B,C,N)
        y_channel=y.reshape(B,C,N)
        qkv_x_channel = self.qkv_x_channel(x_channel).reshape(B, C, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        qkv_y_channel = self.qkv_y_channel(y_channel).reshape(B, C, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q_x_channel, k_x_channel, v_x_channel = qkv_x_channel.reshape(3, B * self.num_heads, C, -1).unbind(0)
        q_y_channel, k_y_channel, v_y_channel = qkv_y_channel.reshape(3, B * self.num_heads, C, -1).unbind(0)
        
        q_x, k_x, v_x = qkv_x.reshape(3, B * self.num_heads, N, -1).unbind(0)
        q_y, k_y, v_y = qkv_y.reshape(3, B * self.num_heads, N, -1).unbind(0)

        attn_x_channel = (q_x_channel * self.scale_channel) @ k_y_channel.transpose(-2, -1)
        attn_y_channel = (q_y_channel * self.scale_channel) @ k_x_channel.transpose(-2, -1)

        attn_x=(q_x * self.scale) @ k_x.transpose(-2, -1)
        attn_y = (q_y * self.scale) @ k_y.transpose(-2, -1)
        attn_x=attn_x.softmax(dim=-1)
        attn_y=attn_y.softmax(dim=-1)
        x_self = (attn_x@v_x).reshape(B, N, -1)
        y_self = (attn_y@v_y).reshape(B, N, -1)
        x_self = self.add_x(x_self)
        y_self = self.add_y(y_self)

        attn_x_channel = attn_x_channel.softmax(dim=-1)
        attn_y_channel = attn_y_channel.softmax(dim=-1)
        
        x_channel = (attn_x_channel @ v_y_channel).reshape(B, C, -1) 
        y_channel = (attn_y_channel @ v_x_channel).reshape(B, C, -1) 
                     
        x_channel = self.proj_x_channel(x_channel)
        y_channel = self.proj_y_channel(y_channel)
        
        x_channel=x_channel.reshape(B,C,H,W)
        y_channel=y_channel.reshape(B,C,H,W)
        
        x_self=x_self.reshape(B,H,W,C).permute(0,3,1,2)
        y_self=y_self.reshape(B,H,W,C).permute(0,3,1,2)
        
        x=x_self+x_channel
        y=y_self+y_channel
        
        return x,y
  
class SAM_XCT(nn.Module):
    def __init__(self,in_channels=3, gain=0.02, init_type='standard'):
        super(SAM_XCT,self).__init__()
        self.take_afa = Multi_Task_Learning_1(256,8)
        
        self.encoder1 = BM_encoder(img_size=128,checkpoint="MedSAM/work_dir/MedSAM/SAM/sam_vit_b_01ec64.pth")
        self.encoder2 = BM_encoder(img_size=128,checkpoint="MedSAM/work_dir/MedSAM/SAM/sam_vit_b_01ec64.pth")
        
        #decoder        
        self.mid_layer1 = TransformBlock(256,8)
        self.mid_layers1 = TransformBlock(256,8)
        self.mid_layer2 = TransformBlock(128,16)
        self.mid_layers2 = TransformBlock(128,16)
        self.mid_layer3 = TransformBlock(64,32)
        self.mid_layers3 = TransformBlock(64,32)
        self.mid_layer4 = TransformBlock(32,64)
        self.mid_layers4 = TransformBlock(32,64)
        
        ## Agent Attention Module
        self.gloabal_attention_1 = Agent_Attention_3D(dim=256,feature_size=8)
        self.gloabal_attention_2 = Agent_Attention_3D(dim=256,feature_size=8)

        #HFE-D
        self.left_basic4 = BasicTranspose2d(256,128) #16
        self.left_layer4 = NewInceptionBlock2d(128,80,68,60)
        self.left_basic3 = BasicTranspose2d(128,64) #32
        self.left_layer3 = NewInceptionBlock2d(64,20,34,30)
        self.left_basic2 = BasicTranspose2d(64,32) #64
        self.left_layer2 = NewInceptionBlock2d(32,10,18,14)
        
        self.left_basics4 = BasicTranspose2d(256,128)
        self.left_layers4 = NewInceptionBlock2d(128,80,68,60)
        self.left_basics3 = BasicTranspose2d(128,64)
        self.left_layers3 = NewInceptionBlock2d(64,20,34,30)
        self.left_basics2 = BasicTranspose2d(64,32)
        self.left_layers2 = NewInceptionBlock2d(32,10,18,14)
        
        
        self.right_basic4 = BasicTranspose3d(256,128)
        self.right_layer4 = NewInceptionBlock(128,80,68,60)
        self.right_basic3 = BasicTranspose3d(256,64)
        self.right_layer3 = NewInceptionBlock(64,20,34,30)
        self.right_basic2 = BasicTranspose3d(128,32)
        self.right_layer2 = NewInceptionBlock(32,10,18,14)
        self.right_basic1 = BasicTranspose3d(64,16)
        self.right_layer1 = NewInceptionBlock(16,8,10,6)
        
        self.right_basics4 = BasicTranspose3d(256,128)
        self.right_layers4 = NewInceptionBlock(128,80,68,60)
        self.right_basics3 = BasicTranspose3d(256,64)
        self.right_layers3 = NewInceptionBlock(64,20,34,30)
        self.right_basics2 = BasicTranspose3d(128,32)
        self.right_layers2 = NewInceptionBlock(32,10,18,14)
        self.right_basics1 = BasicTranspose3d(64,16)
        self.right_layers1 = NewInceptionBlock(16,8,10,6)
        
        
        self.right_basic0 = BasicTranspose3d(64,1)
        self.right_basics0 = BasicTranspose3d(64,2)
        self.conv1_1 = nn.Conv3d(in_channels=2,out_channels=2,kernel_size=1,stride=1,padding=0)
        
        self.urim1 = nn.Softmax(dim=1)
        self.urim2 = nn.Softmax(dim=1)
        
    def forward(self,x):    
        feature_p1 = self.encoder1(x)
        feature_p2 = self.encoder2(x)
        
        feature_1,feature_2 = self.take_afa(feature_p1,feature_p2)
        
        # encoder FPN 
        fpn_1 = self.left_basic4(feature_1)
        fpn_11 = self.left_layer4(fpn_1)#16
        fpn_2 = self.left_basic3(fpn_11)
        fpn_22 = self.left_layer3(fpn_2)#32
        fpn_3 = self.left_basic2(fpn_22)
        fpn_33 = self.left_layer2(fpn_3)#64
        
        fpn_s1 = self.left_basics4(feature_2)
        fpn_s11 = self.left_layers4(fpn_s1)
        fpn_s2 = self.left_basics3(fpn_s11)
        fpn_s22 = self.left_layers3(fpn_s2)
        fpn_s3 = self.left_basics2(fpn_s22)
        fpn_s33 = self.left_layers2(fpn_s3)
        
        skip_connection_1 = self.mid_layer2(fpn_11)
        skip_connection_2 = self.mid_layer3(fpn_22)
        skip_connection_3 = self.mid_layer4(fpn_33)
        
        skip_connection_s1 = self.mid_layers2(fpn_s11)
        skip_connection_s2 = self.mid_layers3(fpn_s22)
        skip_connection_s3 = self.mid_layers4(fpn_s33)
        
        recon_1 = self.mid_layer1(feature_1)
        seg_1 = self.mid_layers1(feature_2)

        B,C,H,W,D =recon_1.shape
        recon_1=recon_1.reshape(B,H*W*D,C)
        seg_1 = seg_1.reshape(B,H*W*D,C)
        recon_1 = self.gloabal_attention_1(recon_1)
        seg_1 = self.gloabal_attention_2(seg_1)
        recon_1=recon_1.view(B,C,H,W,D)
        seg_1 = seg_1.view(B,C,H,W,D)
        
        #Reconstruction
        recon_2 = self.right_basic4(recon_1)
        recon_22 = self.right_layer4(recon_2)
        #Segmentation
        seg_2 = self.right_basics4(seg_1)
        seg_22=self.right_layers4(seg_2)
        
        recon_3 = self.right_basic3(torch.cat((skip_connection_1,recon_22),dim=1))
        recon_33 = self.right_layer3(recon_3)
        
        seg_3 = self.right_basics3(torch.cat((skip_connection_s1,seg_22),dim=1))
        seg_33=self.right_layers3(seg_3)
        
        recon_4 = self.right_basic2(torch.cat((skip_connection_2,recon_33),dim=1))
        recon_44 = self.right_layer2(recon_4)
        
        seg_4 = self.right_basics2(torch.cat((skip_connection_s2,seg_33),dim=1))
        seg_44=self.right_layers2(seg_4)
        
        recon_out = self.right_basic0(torch.cat((skip_connection_3,recon_44),dim=1))
        seg_5 = self.right_basics0(torch.cat((skip_connection_s3,seg_44),dim=1))
        
        seg_out = self.urim2(seg_5)
        
        return recon_out,seg_out      

