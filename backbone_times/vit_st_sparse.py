# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# Copyright 2020 Ross Wightman
# Modified Model definition

import torch
import torch.nn as nn
from functools import partial
import math
import warnings
import torch.nn.functional as F
import numpy as np

from backbone_times.vit_utils import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from backbone_times.helpers import load_pretrained
from backbone_times.vit_utils import DropPath, to_2tuple, trunc_normal_
from backbone_times.Spationtemporal import SpatioTemporalFeatureExtractor,creat_spationtemporal,DEC,target_distribution,SpatioTemporalDEC,Autoencoder
from backbone_times.clip import ClipObjectAugmenter

#from .build import MODEL_REGISTRY
from torch import einsum
from einops import rearrange, reduce, repeat

def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': None,
        'crop_pct': .9, 'interpolation': 'bicubic',
        'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD,
        'first_conv': 'patch_embed.proj', 'classifier': 'head',
        **kwargs
    }


default_cfgs = {
    'vit_base_patch16_224': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_base_p16_224-80ecf9dd.pth',
        mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5),
    ),
    'vit_base_patch16_224_mix': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_base_p16_224-80ecf9dd.pth',
        mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5),
    ),
}

class MotionGuidedTemporalModule(nn.Module):
    """
    修复维度问题并增强的时空分离模块
    输入: [batch, T, D] 帧级特征
    输出: [batch, D] 时间特征 (D=768)
    """
    def __init__(self, dim, reduction_ratio=4):
        super().__init__()
        self.dim = dim
        self.saliency_dim = dim // reduction_ratio  # 显著性特征维度
        self.content_dim = dim - self.saliency_dim  # 内容特征维度
        
        # 运动显著性提取器 - 使用因果卷积保持时间顺序
        self.motion_extractor = nn.Sequential(
            nn.Conv1d(dim, dim//2, kernel_size=3, padding=1, dilation=1),
            nn.GELU(),
            nn.Conv1d(dim//2, dim//2, kernel_size=3, padding=2, dilation=2),
            nn.GELU(),
            nn.Conv1d(dim//2, self.saliency_dim, kernel_size=3, padding=4, dilation=4),
            nn.Sigmoid()
        )
        
        # 内容特征提取器 - 修复维度问题
        self.content_extractor = nn.Sequential(
            nn.Conv1d(dim, self.content_dim, kernel_size=1),
            nn.GroupNorm(8, self.content_dim),  # 使用GroupNorm替代LayerNorm
            nn.GELU()
        )
        
        # 运动-内容协调器 - 使用门控机制
        self.motion_content_coordinator = nn.Sequential(
            nn.Linear(self.saliency_dim, self.content_dim),
            nn.GELU(),
            nn.Linear(self.content_dim, self.content_dim),
            nn.Sigmoid()
        )
        
        # 时间聚合器 - 使用多头注意力
        self.temporal_aggregator = nn.Sequential(
            nn.MultiheadAttention(self.dim, 8, batch_first=True),
            nn.LayerNorm(self.dim),
            nn.Linear(self.dim, dim),
            nn.LayerNorm(dim)
        )

    def forward(self, temporal_feats):
        """
        temporal_feats: [B, T, D] (如 [16, 8, 768])
        返回: [B, D]
        """
        B, T, D = temporal_feats.shape
        
        # 转置为通道优先 [B, D, T]
        feats_t = temporal_feats.transpose(1, 2)
        
        # 1. 提取运动显著性特征 [B, saliency_dim, T]
        motion_saliency = self.motion_extractor(feats_t)
        
        # 2. 提取内容特征 [B, content_dim, T] - 修复维度问题
        content_feats = self.content_extractor(feats_t)
        
        # 3. 转置为时间优先 [B, T, content_dim]
        content_feats = content_feats.transpose(1, 2)
        
        # 4. 协调运动与内容
        motion_guidance = self.motion_content_coordinator(
            motion_saliency.transpose(1, 2)  # [B, T, saliency_dim]
        )  # [B, T, content_dim]
        
        # 5. 应用运动引导到内容特征
        guided_content = content_feats * motion_guidance  # [B, T, content_dim]
        
        # 6. 拼接运动特征和引导后的内容特征
        motion_feats = motion_saliency.transpose(1, 2)  # [B, T, saliency_dim]
        combined_feats = torch.cat([motion_feats, guided_content], dim=-1)  # [B, T, D]
        
        # 7. 时间聚合 (使用注意力机制)
        aggregated, _ = self.temporal_aggregator[0](combined_feats, combined_feats, combined_feats)
        aggregated = self.temporal_aggregator[1](aggregated)
        aggregated = aggregated.mean(dim=1)  # [B, D]
        aggregated = self.temporal_aggregator[2](aggregated)
        return self.temporal_aggregator[3](aggregated)  # [B, D]

class SpatioTemporalSeparator(nn.Module):
    """
    增强的时空分离模块
    输入: [batch, 1 + T*H*W, dim]
    输出: 时间特征 [batch, 768], 空间特征 [batch, 768]
    """
    def __init__(self, dim, num_heads, num_frames):
        super().__init__()
        self.dim = dim
        self.T = num_frames
        
        # 时间特征提取模块
        self.temporal_module = MotionGuidedTemporalModule(dim)
        
        # 空间特征提取模块 - 使用多头注意力
        self.spatial_attention = nn.MultiheadAttention(
            embed_dim=dim, 
            num_heads=num_heads,
            batch_first=True
        )
        
        # 空间特征增强器
        self.spatial_enhancer = nn.Sequential(
            nn.Linear(dim, dim*2),
            nn.GELU(),
            nn.Linear(dim*2, dim),
            nn.LayerNorm(dim)
        )
        
        # 时空分离约束 - 使用对比学习
        self.separation_constraint = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, dim//8)
        )

    def forward(self, x, H, W):
        """
        x: [B, 1 + T*H*W, D]
        返回: temporal_feature [B, D], spatial_feature [B, D]
        """
        B, N, D = x.shape
        T = self.T
        
        # 分离CLS token和时空特征
        cls_token = x[:, 0:1, :]  # [B, 1, D]
        st_feats = x[:, 1:, :]    # [B, T*H*W, D]
        
        # 重组为时空特征 [B, T, H*W, D]
        st_feats = st_feats.view(B, T, H*W, D)
        
        # ===== 时间特征提取 =====
        # 1. 空间平均得到帧级特征 [B, T, D]
        frame_features = st_feats.mean(dim=2)
        
        # 2. 使用运动引导模块提取时间特征
        temporal_feature = self.temporal_module(frame_features)  # [B, D]
        
        # ===== 空间特征提取 =====
        # 1. 选择最具代表性的帧（最小运动）
        with torch.no_grad():
            # 计算帧间差异作为运动估计
            frame_diff = torch.mean(
                torch.abs(frame_features[:, 1:] - frame_features[:, :-1]),
                dim=-1
            )  # [B, T-1]
            # 添加首帧的虚拟差异
            frame_stability = torch.cat([
                torch.zeros(B, 1, device=frame_features.device),
                frame_diff
            ], dim=1)  # [B, T]
            min_motion_idx = frame_stability.argmin(dim=1)  # [B]
        
        # 2. 提取关键帧特征
        key_frame_feats = st_feats[torch.arange(B), min_motion_idx]  # [B, H*W, D]
        
        # 3. 添加CLS token并增强空间一致性
        spatial_input = torch.cat([cls_token, key_frame_feats], dim=1)
        spatial_feats, _ = self.spatial_attention(
            spatial_input, spatial_input, spatial_input
        )
        
        # 4. 获取空间特征 (CLS token)
        spatial_feature = spatial_feats[:, 0, :]  # [B, D]
        
        # 5. 空间特征增强
        spatial_feature = self.spatial_enhancer(spatial_feature)  # [B, D]
        
        # ===== 时空分离约束 =====
        loss_sparse = self.apply_separation_constraint(temporal_feature, spatial_feature)
        
        return temporal_feature, spatial_feature,loss_sparse
    
    def apply_separation_constraint(self, temporal, spatial):
        """
        应用时空分离约束 - 对比学习方式
        """
        # 1. 投影到低维空间
        temporal_proj = self.separation_constraint(temporal)  # [B, D//8]
        spatial_proj = self.separation_constraint(spatial)    # [B, D//8]
        
        # 2. 归一化
        temporal_proj = F.normalize(temporal_proj, p=2, dim=1)
        spatial_proj = F.normalize(spatial_proj, p=2, dim=1)
        
        # 3. 计算相似度矩阵
        similarity_matrix = torch.mm(temporal_proj, spatial_proj.t())  # [B, B]
        
        # 4. 创建对比标签
        labels = torch.arange(similarity_matrix.size(0), device=temporal.device)
        
        temperature = 0.1
        logits = similarity_matrix / temperature
        loss_t = F.cross_entropy(logits, labels)
        loss_s = F.cross_entropy(logits.t(), labels)

        loss_sparse =  0.5 * (loss_t + loss_s)

        return loss_sparse


class SpatioTemporalCooperation(nn.Module):
    """
    时空协同模块 - 融合空间和时间特征并进行协同预测
    输入: 空间特征 [batch, D], 时间特征 [batch, D]
    输出: 协同预测 [batch, num_classes], 协同损失
    """
    def __init__(self, feature_dim, num_classes, num_heads=8, fusion_dim=1024):
        super().__init__()
        self.feature_dim = feature_dim
        self.num_classes = num_classes
        
        # 特征融合模块
        self.fusion = nn.Sequential(
            nn.Linear(feature_dim * 2, fusion_dim),
            nn.GELU(),
            nn.LayerNorm(fusion_dim),
            nn.Dropout(0.1)
        )
        
        # 协同注意力机制
        self.spatial_attention = nn.MultiheadAttention(
            embed_dim=fusion_dim, 
            num_heads=num_heads,
            batch_first=True
        )
        self.temporal_attention = nn.MultiheadAttention(
            embed_dim=fusion_dim, 
            num_heads=num_heads,
            batch_first=True
        )
        
        # 协同预测器
        self.predictor = nn.Sequential(
            nn.Linear(fusion_dim * 2, fusion_dim),
            nn.ReLU(),
            nn.LayerNorm(fusion_dim),
            nn.Linear(fusion_dim, num_classes)
        )
        
        # 协同约束模块
        self.constraint = nn.Sequential(
            nn.Linear(fusion_dim, feature_dim),
            nn.ReLU(),
            nn.LayerNorm(feature_dim)
        )

    def forward(self, spatial_feats, temporal_feats, labels=None):
        """
        spatial_feats: [B, D] 空间特征
        temporal_feats: [B, D] 时间特征
        labels: [B] 真实标签 (可选，用于训练)
        返回: 预测logits, 协同损失
        """
        B = spatial_feats.size(0)
        
        # 1. 特征融合
        combined = torch.cat([spatial_feats, temporal_feats], dim=1)
        fused_feats = self.fusion(combined)  # [B, fusion_dim]
        
        # 2. 协同注意力机制
        # 空间特征作为query，时间特征作为key/value
        spatial_as_query = fused_feats.unsqueeze(1)  # [B, 1, fusion_dim]
        temporal_as_kv = fused_feats.unsqueeze(1)    # [B, 1, fusion_dim]
        
        # 空间关注时间
        spatial_attended, _ = self.spatial_attention(
            spatial_as_query, temporal_as_kv, temporal_as_kv
        )
        
        # 时间关注空间
        temporal_attended, _ = self.temporal_attention(
            temporal_as_kv, spatial_as_query, spatial_as_query
        )
        
        spatial_enhanced = spatial_attended.squeeze(1)  # [B, fusion_dim]
        temporal_enhanced = temporal_attended.squeeze(1)  # [B, fusion_dim]
        cooperative_feats = torch.cat([spatial_enhanced, temporal_enhanced], dim=1)  # [B, fusion_dim*2]
        
        logits = self.predictor(cooperative_feats)  # [B, num_classes]
        
        return logits
        

class ResidualAttentionModule(nn.Module):
    def __init__(self, dim=768, temp_init=0.5):
        super().__init__()
        self.temperature = nn.Parameter(torch.tensor(temp_init))  # 可学习温度
        self.gate = nn.Sequential(  # 抑制过多注意力的 learnable gate
            nn.Linear(dim, dim),
            nn.Sigmoid()
        )

    def forward(self, t, s, st):
        # t, s, st: [B, dim]
        residual = ((t + s) / 2 - st).abs()       # [B, dim]
        importance = residual.mean(dim=0)         # [dim]
        
        # 通道注意力分布：越大代表该通道残差信息越强
        weights = (importance / (importance.sum() + 1e-6)) / self.temperature
        weights = F.softmax(weights, dim=0)       # [dim]

        # gating 控制每个通道增强幅度（类似 squeeze-and-excite）
        gate_weights = self.gate(st)              # [B, dim]

        # 通道增强：加入 gated residual
        enhanced = st + gate_weights * residual * weights  # [B, dim]
        return enhanced
        

class EnhancedTemporalModule(nn.Module):
    def __init__(self, dim, num_heads=8, T=8, use_multiscale=False):
        super().__init__()
        self.T = T
        self.use_multiscale = use_multiscale

        # Positional Encoding
        self.temporal_pos = nn.Parameter(torch.randn(T, dim))

        # Global Attention
        self.global_attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, batch_first=True)

        if use_multiscale:
            self.local_attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, batch_first=True)

        # Attention Pooling: learnable query
        self.query_token = nn.Parameter(torch.randn(1, 1, dim))
        self.attn_pool = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, batch_first=True)

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., with_qkv=True):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.with_qkv = with_qkv
        if self.with_qkv:
           self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
           self.proj = nn.Linear(dim, dim)
           self.proj_drop = nn.Dropout(proj_drop)
        self.attn_drop = nn.Dropout(attn_drop)

    def forward(self, x):
        B, N, C = x.shape
        if self.with_qkv:
           qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
           q, k, v = qkv[0], qkv[1], qkv[2]
        else:
           qkv = x.reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
           q, k, v  = qkv, qkv, qkv

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        if self.with_qkv:
           x = self.proj(x)
           x = self.proj_drop(x)
        return x

class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0.1, act_layer=nn.GELU, norm_layer=nn.LayerNorm, attention_type='divided_space_time'):
        super().__init__()
        self.attention_type = attention_type
        assert(attention_type in ['divided_space_time', 'space_only','joint_space_time'])

        self.norm1 = norm_layer(dim)
        self.attn = Attention(
           dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        ## Temporal Attention Parameters
        if self.attention_type == 'divided_space_time':
            self.temporal_norm1 = norm_layer(dim)
            self.temporal_attn = Attention(
              dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
            self.temporal_fc = nn.Linear(dim, dim)
            
        self.temporal_module = EnhancedTemporalModule(dim=768, T=8, use_multiscale=True)

        ## drop path
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)


    def forward(self, x, B, T, W):
        num_spatial_tokens = (x.size(1) - 1) // T
        H = num_spatial_tokens // W

        if self.attention_type in ['space_only', 'joint_space_time']:
            x = x + self.drop_path(self.attn(self.norm1(x)))
            x = x + self.drop_path(self.mlp(self.norm2(x)))
            return x
        elif self.attention_type == 'divided_space_time':
            ## Temporal
            xt = x[:,1:,:]
            xt = rearrange(xt, 'b (h w t) m -> (b h w) t m',b=B,h=H,w=W,t=T)
            res_temporal = self.drop_path(self.temporal_attn(self.temporal_norm1(xt)))
            res_temporal = rearrange(res_temporal, '(b h w) t m -> b (h w t) m',b=B,h=H,w=W,t=T)
            res_temporal = self.temporal_fc(res_temporal)
            xt = x[:,1:,:] + res_temporal
            # xt 已经过 Temporal Attention 处理
            temporal_feature = torch.mean(xt, dim=1)  # [batch, 768]
            # temporal_feature, xt = self.temporal_module(x[:,1:,:], B, H, W)

            ## Spatial
            init_cls_token = x[:,0,:].unsqueeze(1)
            cls_token = init_cls_token.repeat(1, T, 1)
            cls_token = rearrange(cls_token, 'b t m -> (b t) m',b=B,t=T).unsqueeze(1)
            xs = xt
            xs = rearrange(xs, 'b (h w t) m -> (b t) (h w) m',b=B,h=H,w=W,t=T)
            xs = torch.cat((cls_token, xs), 1)
            res_spatial = self.drop_path(self.attn(self.norm1(xs)))
            

            ### Taking care of CLS token
            cls_token = res_spatial[:,0,:]
            cls_token = rearrange(cls_token, '(b t) m -> b t m',b=B,t=T)
            cls_token = torch.mean(cls_token,1,True) ## averaging for every frame
            spatial_feature = torch.mean(cls_token, dim=1)
            
            res_spatial = res_spatial[:,1:,:]
            res_spatial = rearrange(res_spatial, '(b t) (h w) m -> b (h w t) m',b=B,h=H,w=W,t=T)
            res = res_spatial
            x = xt
            
            ## Mlp
            x = torch.cat((init_cls_token, x), 1) + torch.cat((cls_token, res), 1)
            x = x + self.drop_path(self.mlp(self.norm2(x)))
            return x,xt


class SimpleHighValueAttention(nn.Module):
    def __init__(self, alpha=1.5):
        super().__init__()
        self.alpha = alpha

    def forward(self, x):
        """
        x: Tensor of shape [batch_size, dim]
        """
        x_min = x.min(dim=-1, keepdim=True)[0]
        x_max = x.max(dim=-1, keepdim=True)[0]
        x_norm = (x - x_min) / (x_max - x_min + 1e-6)
        return x_norm ** self.alpha
    
class HighValueEnhancedAttention(nn.Module):
    def __init__(self, dim=768, alpha=2.0):
        super().__init__()
        self.alpha = alpha
        self.attn_fc = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        # 1. Min-Max 归一化（对每个样本独立处理）
        x_min = x.min(dim=-1, keepdim=True)[0]
        x_max = x.max(dim=-1, keepdim=True)[0]
        x_norm = (x - x_min) / (x_max - x_min + 1e-6)

        # 2. 放大高值（幂次变换）
        x_amplified = x_norm ** self.alpha

        # 3. Learnable dimension attention
        attn_weights = self.attn_fc(x)  # shape: [batch, dim]
        out = x_amplified * attn_weights  # 元素级加权
        return out
   
class LogitAttention(nn.Module):
    def __init__(self, init_power=2.0, max_power=5.0, alpha=0.5, drop_prob=0.3):
        super().__init__()
        self.register_buffer("current_power", torch.tensor(init_power))
        self.max_power = max_power
        self.alpha = alpha
        self.drop_prob = drop_prob

    def update_power(self, current_epoch, total_epochs):
        ratio = min(1.0, current_epoch / float(total_epochs))
        new_power = 1.0 + (self.max_power - 1.0) * ratio
        self.current_power.fill_(new_power)

    def forward(self, logits):
        if self.training and torch.rand(1).item() < self.drop_prob:
            return logits  # skip attention
        
        power = self.current_power.item()
        attn = torch.softmax(logits * power, dim=1)
        sharpened = logits * attn
        return self.alpha * logits + (1 - self.alpha) * sharpened
    
    
class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, C, T, H, W = x.shape
        x = rearrange(x, 'b c t h w -> (b t) c h w')
        x = self.proj(x)
        W = x.size(-1)
        x = x.flatten(2).transpose(1, 2)
        return x, T, W



class VisionTransformer(nn.Module):
    """ Vision Transformere
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0.1, hybrid_backbone=None, norm_layer=nn.LayerNorm, num_frames=8, attention_type='divided_space_time', dropout=0.):
        super().__init__()
        self.attention_type = attention_type
        self.depth = depth

        self.dropout = nn.Dropout(dropout)
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        ## Positional Embeddings
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches+1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)
        if self.attention_type != 'space_only':
            self.time_embed = nn.Parameter(torch.zeros(1, num_frames, embed_dim))
            self.time_drop = nn.Dropout(p=drop_rate)

        self.autoencoder = Autoencoder()
        self.spation= SpatioTemporalFeatureExtractor(768)

        ## Attention Blocks
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, self.depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, attention_type=self.attention_type)
            for i in range(self.depth)])
        self.norm = norm_layer(embed_dim)
    
        # Classifier head
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        self.logit_attn = LogitAttention(init_power=3.0, max_power=5.0, alpha=0.5, drop_prob=0.3)
        self.st_attn = HighValueEnhancedAttention(dim=768, alpha=2.0)
        self.temporal_module = EnhancedTemporalModule(dim=embed_dim, T=num_frames, use_multiscale=True)
        self.residual_attention = ResidualAttentionModule(dim=768)

        trunc_normal_(self.pos_embed, std=.02)
        trunc_normal_(self.cls_token, std=.02)
        self.apply(self._init_weights)

        ## initialization of temporal attention weights
        if self.attention_type == 'divided_space_time':
            i = 0
            for m in self.blocks.modules():
                m_str = str(m)
                if 'Block' in m_str:
                    if i > 0:
                      nn.init.constant_(m.temporal_fc.weight, 0)
                      nn.init.constant_(m.temporal_fc.bias, 0)
                    i += 1

        #time_space divided
        self.separator = SpatioTemporalSeparator(
        dim=embed_dim,
        num_heads=8,
        num_frames=num_frames
        )

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token', 'time_embed'}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x,mask=None):
        B = x.shape[0]
        if mask is not None:
            mask = mask[0:B//2]
            mask = mask.reshape(-1,mask.size(-1))
            mask1 = mask.bool().unsqueeze(-1)
            mask2 = ~mask1
            strong1 = x[0:B//2]
            strong2 = x[B//2:B]
            strong1,T,W = self.patch_embed(strong1)
            strong2,_,_ = self.patch_embed(strong2)
            B = (B//2)
            x1 = strong1 * mask1 + strong2 * mask2
            x = x1
        else:
            x, T, W = self.patch_embed(x)
        
        cls_tokens = self.cls_token.expand(x.size(0), -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        ## resizing the positional embeddings in case they don't match the input at inference
        if x.size(1) != self.pos_embed.size(1):
            pos_embed = self.pos_embed
            cls_pos_embed = pos_embed[0,0,:].unsqueeze(0).unsqueeze(1)
            other_pos_embed = pos_embed[0,1:,:].unsqueeze(0).transpose(1, 2)
            P = int(other_pos_embed.size(2) ** 0.5)
            H = x.size(1) // W
            other_pos_embed = other_pos_embed.reshape(1, x.size(2), P, P)
            new_pos_embed = F.interpolate(other_pos_embed, size=(H, W), mode='nearest')
            new_pos_embed = new_pos_embed.flatten(2)
            new_pos_embed = new_pos_embed.transpose(1, 2)
            new_pos_embed = torch.cat((cls_pos_embed, new_pos_embed), 1)
            x = x + new_pos_embed
        else:
            x = x + self.pos_embed
        x = self.pos_drop(x)

        ## Time Embeddings
        if self.attention_type != 'space_only':
            cls_tokens = x[:B, 0, :].unsqueeze(1)
            x = x[:,1:]
            x = rearrange(x, '(b t) n m -> (b n) t m',b=B,t=T)

            ## Resizing time embeddings in case they don't match
            if T != self.time_embed.size(1):
                time_embed = self.time_embed.transpose(1, 2)
                new_time_embed = F.interpolate(time_embed, size=(T), mode='nearest')
                new_time_embed = new_time_embed.transpose(1, 2)
                x = x + new_time_embed
            else:
                x = x + self.time_embed
            x = self.time_drop(x)
            x = rearrange(x, '(b n) t m -> b (n t) m',b=B,t=T) # spatial_time embedding;
            x = torch.cat((cls_tokens, x), dim=1)
        for blk in self.blocks:
            x,xt = blk(x, B, T, W)


        temporal_feature, spatial_feature, loss_sparse = self.separator(
            torch.cat([x[:,0:1], xt], dim=1), W, W
        )
       
        ### Predictions for space-only baseline
        if self.attention_type == 'space_only':
            x = rearrange(x, '(b t) n m -> b t n m',b=B,t=T)
            x = torch.mean(x, 1) # averaging predictions for every frame

        x = self.norm(x)
        
        return x[:, 0],temporal_feature,spatial_feature

    def forward(self, x,mask):
        proj,temporal_feature,spatial_feature = self.forward_features(x,mask)
        # proj = self.residual_attention(temporal_feature, spatial_feature, proj)
        cls_x = self.head(proj)
        # cls_x = self.logit_attn(cls_x)
       
        return proj,cls_x,temporal_feature,spatial_feature
    
    def get_head(self,x):
        cls_x = self.head(x)
        return cls_x

def _conv_filter(state_dict, patch_size=16):
    """ convert patch embedding weight from manual patchify + linear proj to conv"""
    out_dict = {}
    for k, v in state_dict.items():
        if 'patch_embed.proj.weight' in k:
            if v.shape[-1] != patch_size:
                patch_size = v.shape[-1]
            v = v.reshape((v.shape[0], 3, patch_size, patch_size))
        out_dict[k] = v
    return out_dict

#@MODEL_REGISTRY.register()
class vit_base_patch16_224(nn.Module):
    def __init__(self, cfg, **kwargs):
        super(vit_base_patch16_224, self).__init__()
        self.pretrained=True
        patch_size = 16
        self.model = VisionTransformer(img_size=cfg.DATA.TRAIN_CROP_SIZE, num_classes=cfg.MODEL.NUM_CLASSES, patch_size=patch_size, embed_dim=768, depth=8, num_heads=12, mlp_ratio=4, qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1, num_frames=cfg.DATA.NUM_FRAMES, attention_type=cfg.TIMESFORMER.ATTENTION_TYPE, **kwargs)

        self.attention_type = cfg.TIMESFORMER.ATTENTION_TYPE
        self.model.default_cfg = default_cfgs['vit_base_patch16_224']
        self.num_patches = (cfg.DATA.TRAIN_CROP_SIZE // patch_size) * (cfg.DATA.TRAIN_CROP_SIZE // patch_size)
        pretrained_model= None
        pretrained_model=cfg.TIMESFORMER.PRETRAINED_MODEL
       
        if self.pretrained:
            load_pretrained(self.model, num_classes=self.model.num_classes, in_chans=kwargs.get('in_chans', 3), filter_fn=_conv_filter, img_size=cfg.DATA.TRAIN_CROP_SIZE, num_patches=self.num_patches, attention_type=self.attention_type, pretrained_model=pretrained_model)

    def forward(self, x,mask=None):
        x = self.model(x,mask)
        return x
    
    def get_head(self,x):
        #x [b,768]
        cls_x = self.model.get_head(x)
        return cls_x
    
class vit_base_patch16_224_mix(nn.Module):
    def __init__(self, cfg, **kwargs):
        super(vit_base_patch16_224_mix, self).__init__()
        self.pretrained=True
        patch_size = 16
        self.model = VisionTransformer(img_size=cfg.DATA.TRAIN_CROP_SIZE, num_classes=cfg.MODEL.NUM_CLASSES, patch_size=patch_size, embed_dim=768, depth=8, num_heads=12, mlp_ratio=4, qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1, num_frames=cfg.DATA.NUM_FRAMES, attention_type=cfg.TIMESFORMER.ATTENTION_TYPE, **kwargs)

        self.attention_type = cfg.TIMESFORMER.ATTENTION_TYPE
        self.model.default_cfg = default_cfgs['vit_base_patch16_224_mix']
        self.num_patches = (cfg.DATA.TRAIN_CROP_SIZE // patch_size) * (cfg.DATA.TRAIN_CROP_SIZE // patch_size)
        pretrained_model= None
        pretrained_model=cfg.TIMESFORMER.PRETRAINED_MODEL
       
        if self.pretrained:
            load_pretrained(self.model, num_classes=self.model.num_classes, in_chans=kwargs.get('in_chans', 3), filter_fn=_conv_filter, img_size=cfg.DATA.TRAIN_CROP_SIZE, num_patches=self.num_patches, attention_type=self.attention_type, pretrained_model=pretrained_model)

    def forward(self, x,mask=None):
        x = self.model(x,mask)
        return x
    
#@MODEL_REGISTRY.register()
class TimeSformer(nn.Module):
    def __init__(self, img_size=224, patch_size=16, num_classes=400, num_frames=8, attention_type='divided_space_time',  pretrained_model='', **kwargs):
        super(TimeSformer, self).__init__()
        self.pretrained=True
        self.model = VisionTransformer(img_size=img_size, num_classes=num_classes, patch_size=patch_size, embed_dim=768, depth=8, num_heads=12, mlp_ratio=4, qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1, num_frames=num_frames, attention_type=attention_type, **kwargs)

        self.attention_type = attention_type
        self.model.default_cfg = default_cfgs['vit_base_patch'+str(patch_size)+'_224_mix']
        self.num_patches = (img_size // patch_size) * (img_size // patch_size)
        if self.pretrained:
            load_pretrained(self.model, num_classes=self.model.num_classes, in_chans=kwargs.get('in_chans', 3), filter_fn=_conv_filter, img_size=img_size, num_frames=num_frames, num_patches=self.num_patches, attention_type=self.attention_type, pretrained_model=pretrained_model)
    def forward(self, x):
        x = self.model(x)
        return x
    
    
