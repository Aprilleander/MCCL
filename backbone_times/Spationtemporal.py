import torch
import torch.nn as nn
from functools import partial
import math
import warnings
import torch.nn.functional as F
import numpy as np
from einops import rearrange


def extract_dynamic_temporal_feature(res_temporal, cls_token, B, H, W, T):
    """
    时间特征提取：结合原始序列、反向序列和CLS token
    """
    # 重排为 [B, T, H, W, C]
    temporal_4d = rearrange(res_temporal, 'b (h w t) c -> b t h w c', h=H, w=W, t=T)
    
    # 1. 创建时间反向播放序列
    reversed_temporal = torch.flip(temporal_4d, dims=[1])  # 沿时间维度反转
    
    # 2. 合并原始序列和反向序列 (8+8=16帧)
    combined_temporal = torch.cat([temporal_4d, reversed_temporal], dim=1)  # [B, 16, H, W, C]
    
    # 3. 简单的全局池化得到时间特征
    temporal_feature = torch.mean(combined_temporal, dim=(1, 2, 3))  # [B, C]
    
    # 4. 加入CLS token信息（与空间特征一致）
    # 假设cls_token形状为[B, 1, C]，与空间特征提取保持一致
    global_cls = cls_token.squeeze(1)  # [B, C]
    temporal_feature = temporal_feature +0.3*global_cls
    
    return temporal_feature

    
# def extract_dynamic_temporal_feature(res_temporal, B, H, W, T):
#     """
#     时间特征增强：选择变化最激烈的4帧，缩减时间维度并增强动作特征
#     """
#     # 重排为 [B, T, H, W, C]
#     temporal_4d = rearrange(res_temporal, 'b (h w t) c -> b t h w c', h=H, w=W, t=T)
    
#     # 1. 计算每帧的变化强度（与前一帧的差异）
#     if T > 1:
#         # 计算相邻帧差异
#         frame_changes = []
#         for t in range(1, T):
#             # 计算帧间差异的幅度
#             change_magnitude = torch.mean(
#                 torch.abs(temporal_4d[:, t] - temporal_4d[:, t-1]), 
#                 dim=(1, 2, 3)  # 在H,W,C维度平均
#             )  # [B]
#             frame_changes.append(change_magnitude)
        
#         frame_changes = torch.stack(frame_changes, dim=1)  # [B, T-1]
        
#         # 2. 选择变化最激烈的4个时刻（对应变化后的帧）
#         top_k = min(4, T-1)
#         _, top_indices = torch.topk(frame_changes, k=top_k, dim=1)  # [B, 4]
        
#         # 3. 构建缩减后的时间序列（只保留关键帧）
#         reduced_temporal = []
#         for i in range(B):
#             # 选择变化最激烈的4帧（选择变化发生后的帧）
#             key_frames = []
#             for idx in top_indices[i]:
#                 frame_idx = min(idx + 1, T-1)  # 选择变化后的帧
#                 key_frames.append(temporal_4d[i, frame_idx])
            
#             key_frames = torch.stack(key_frames)  # [4, H, W, C]
#             reduced_temporal.append(key_frames)
        
#         reduced_temporal = torch.stack(reduced_temporal)  # [B, 4, H, W, C]
        
#         # 4. 全局池化得到特征
#         temporal_feature = torch.mean(reduced_temporal, dim=(1, 2, 3))  # [B, C]
        
#     else:
#         # 如果只有1帧，直接使用
#         temporal_feature = torch.mean(temporal_4d, dim=(1, 2, 3))  # [B, C]
    
#     return temporal_feature

# def extract_dynamic_temporal_feature(res_temporal, B, H, W, T):
#     """
#     简洁的时间特征增强：复制变化激烈的帧来增强动作特征
#     """
#     # 重排为 [B, T, H, W, C]
#     temporal_4d = rearrange(res_temporal, 'b (h w t) c -> b t h w c', h=H, w=W, t=T)
    
#     # 1. 计算帧间变化强度
#     if T > 1:
#         # 计算相邻帧差异
#         frame_diffs = []
#         for t in range(1, T):
#             diff = torch.mean(torch.abs(temporal_4d[:, t] - temporal_4d[:, t-1]), dim=(1, 2, 3))  # [B]
#             frame_diffs.append(diff)
        
#         frame_diffs = torch.stack(frame_diffs, dim=1)  # [B, T-1]
        
#         # 找到变化最激烈的4个时刻
#         top_k = min(4, T-1)
#         _, top_indices = torch.topk(frame_diffs, k=top_k, dim=1)  # [B, 4]
        
#         # 2. 复制变化激烈的帧（增强动作片段）
#         enhanced_frames = []
#         for i in range(B):
#             # 原始8帧
#             original_sequence = temporal_4d[i]  # [T, H, W, C]
            
#             # 复制变化激烈的帧（选择这些时刻的下一帧，因为变化发生在两帧之间）
#             copied_frames = []
#             for idx in top_indices[i]:
#                 # 复制变化发生后的那一帧
#                 frame_idx = min(idx + 1, T-1)  # 确保不越界
#                 copied_frames.append(original_sequence[frame_idx])
            
#             copied_frames = torch.stack(copied_frames)  # [4, H, W, C]
            
#             # 组合：复制的4帧 + 原始8帧 = 12帧
#             enhanced_sequence = torch.cat([copied_frames, original_sequence], dim=0)  # [12, H, W, C]
#             enhanced_frames.append(enhanced_sequence)
        
#         enhanced_temporal = torch.stack(enhanced_frames)  # [B, 12, H, W, C]
        
#         # 3. 简单的全局池化得到特征
#         temporal_feature = torch.mean(enhanced_temporal, dim=(1, 2, 3))  # [B, C]
        
#     else:
#         # 如果只有1帧，直接使用原始特征
#         temporal_feature = torch.mean(temporal_4d, dim=(1, 2, 3))  # [B, C]
    
#     return temporal_feature

# def extract_dynamic_temporal_feature(res_temporal, B, H, W, T):
#     """
#     提取强调动态变化的时间特征 [B, C]
#     """
#     # 重排为 [B, H, W, T, C]
#     temporal_3d = rearrange(res_temporal, 'b (h w t) c -> b h w t c', h=H, w=W, t=T)
    
#     # 1. 时间动态特征 - 帧间变化
#     if T > 1:
#         # 计算相邻帧差异（运动特征）
#         motion_features = temporal_3d[:, :, :, 1:, :] - temporal_3d[:, :, :, :-1, :]  # [B, H, W, T-1, C]
#         motion_energy = torch.mean(torch.norm(motion_features, dim=-1), dim=(1, 2, 3))  # [B]
#     else:
#         motion_energy = torch.zeros(B, device=res_temporal.device)
    
#     # 2. 时间注意力分布 - 重要时间点
#     temporal_global = torch.mean(temporal_3d, dim=(1, 2))  # [B, T, C]
    
#     # 时间重要性权重（基于特征强度）
#     temporal_importance = torch.softmax(torch.norm(temporal_global, dim=-1), dim=-1)  # [B, T]
    
#     # 3. 关键时间点特征
#     # 选择注意力权重最高的几帧
#     top_k = min(3, T)  # 选择最重要的3帧或所有帧
#     top_weights, top_indices = torch.topk(temporal_importance, k=top_k, dim=-1)  # [B, top_k]
    
#     key_temporal_features = []
#     for i in range(B):
#         key_frames = temporal_global[i, top_indices[i]]  # [top_k, C]
#         weighted_key_frames = key_frames * top_weights[i].unsqueeze(1)  # 加权
#         key_temporal_features.append(torch.mean(weighted_key_frames, dim=0))
    
#     key_temporal = torch.stack(key_temporal_features)  # [B, C]
    
#     # 4. 时间变化模式特征
#     temporal_variation = torch.std(temporal_global, dim=1)  # [B, C] - 时间维度变化
    
#     # 5. 组合动态时间特征 [B, C]
#     # 使用线性层将多个特征融合到C维度
#     temporal_feature = key_temporal + 0.5 * temporal_variation
#     temporal_feature = temporal_feature + motion_energy.unsqueeze(1) * 0.1  # 加入运动能量
    
    
#     return temporal_feature


def extract_keyframe_spatial_feature(res_spatial, cls_token, B, H, W, T):
    """
    提取基于关键帧的空间特征 [B, C]
    cls_token: [B, 1, C] - 单个CLS token（不是时序的）
    """
    # 重排空间特征 [B, T, H, W, C]
    spatial_4d = rearrange(res_spatial, 'b (h w t) c -> b t h w c', h=H, w=W, t=T)
    
    # 1. 基于空间特征选择关键帧（因为cls_token没有时序信息）
    # 计算每帧的空间特征重要性（平均激活强度）
    frame_importance = torch.mean(torch.norm(spatial_4d, dim=-1), dim=(-2, -1))  # [B, T]
    frame_weights = torch.softmax(frame_importance, dim=-1)  # [B, T]
    
    # 选择最重要的关键帧
    top_k_frames = min(2, T)  # 选择最重要的2帧
    top_weights, top_indices = torch.topk(frame_weights, k=top_k_frames, dim=-1)
    
    # 2. 关键帧空间特征提取
    keyframe_spatial_features = []
    
    for i in range(B):
        batch_keyframes = []
        for j in range(top_k_frames):
            frame_idx = top_indices[i, j]
            frame_weight = top_weights[i, j]
            
            # 提取该帧的空间特征
            frame_spatial = spatial_4d[i, frame_idx]  # [H, W, C]
            
            # 空间注意力热点（高响应区域）
            spatial_attention_map = torch.mean(torch.abs(frame_spatial), dim=-1)  # [H, W]
            
            # 选择空间注意力最高的区域
            hot_spots = extract_spatial_hotspots(frame_spatial, spatial_attention_map)
            weighted_hot_spots = hot_spots * frame_weight
            batch_keyframes.append(weighted_hot_spots)
        
        # 聚合关键帧特征
        if batch_keyframes:
            keyframe_feature = torch.mean(torch.stack(batch_keyframes), dim=0)
        else:
            keyframe_feature = torch.mean(spatial_4d[i].view(-1, spatial_4d.shape[-1]), dim=0)
        
        keyframe_spatial_features.append(keyframe_feature)
    
    keyframe_features = torch.stack(keyframe_spatial_features)  # [B, C]
    
    spatial_consistency = extract_spatial_consistency(spatial_4d)  # [B, C]
    
    # 4. 组合关键帧空间特征 [B, C]
    spatial_feature = keyframe_features + 0.3*spatial_consistency
    
    # 加入全局CLS token信息（直接使用cls_token，因为只有1个）
    global_cls = cls_token.squeeze(1)  # [B, C] - 移除第1维
    spatial_feature = spatial_feature + 0.3*global_cls
    
    return spatial_feature


def extract_spatial_hotspots(frame_spatial, attention_map, num_spots=5):
    """
    提取空间注意力热点区域特征
    """
    H, W, C = frame_spatial.shape
    attention_flat = attention_map.view(-1)
    
    # 选择注意力最高的几个点
    top_attention, top_indices = torch.topk(attention_flat, k=min(num_spots, H*W))
    
    # 提取对应位置的特征
    hotspot_features = []
    for idx in top_indices:
        h_idx = idx // W
        w_idx = idx % W
        hotspot_features.append(frame_spatial[h_idx, w_idx])
    
    if hotspot_features:
        return torch.mean(torch.stack(hotspot_features), dim=0)
    else:
        return torch.mean(frame_spatial.view(-1, C), dim=0)

def extract_spatial_consistency(spatial_4d):
    """
    提取空间结构一致性特征
    """
    B, T, H, W, C = spatial_4d.shape
    
    if T > 1:
        # 计算帧间空间结构相似性
        spatial_avg = torch.mean(spatial_4d, dim=(2, 3))  # [B, T, C]
        spatial_correlation = F.cosine_similarity(
            spatial_avg[:, :-1], spatial_avg[:, 1:], dim=-1
        )  # [B, T-1]
        consistency = torch.mean(spatial_correlation, dim=1)  # [B]
    else:
        consistency = torch.ones(B, device=spatial_4d.device)
    
    # 将一致性分数扩展到C维度
    consistency = consistency.unsqueeze(1).repeat(1, C)  # [B, C]
    return consistency