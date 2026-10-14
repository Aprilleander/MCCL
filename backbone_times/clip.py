import torch
import torch.nn as nn
import clip
from PIL import Image
import openai
import requests
from io import BytesIO

class ClipObjectAugmenter(nn.Module):
    """
    使用CLIP模型增强空间特征 - 提取代表性帧中的物体信息
    输入: 视频帧序列
    输出: 物体感知的空间特征增强
    """
    def __init__(self, clip_model_name="ViT-B/32", top_k=5, cache_dir="clip_embeddings"):
        super().__init__()
        self.top_k = top_k
        self.cache_dir = cache_dir
        
        # 加载CLIP模型
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.clip_model, self.preprocess = clip.load(clip_model_name, device=self.device)
        
        # 冻结CLIP模型参数
        for param in self.clip_model.parameters():
            param.requires_grad = False
            
        # 创建物体类别文本嵌入缓存
        self.object_embeddings = self._init_object_embeddings()
        
        # 物体信息融合模块
        self.fusion_module = nn.Sequential(
            nn.Linear(512 * 2, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.LayerNorm(512)
        )

    def _init_object_embeddings(self):
        """初始化常见物体类别的文本嵌入"""
        # 使用COCO数据集80类常见物体
        coco_classes = [
            "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", 
            "truck", "boat", "traffic light", "fire hydrant", "stop sign", 
            "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", 
            "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", 
            "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", 
            "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", 
            "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork", 
            "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", 
            "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", 
            "couch", "potted plant", "bed", "dining table", "toilet", "tv", 
            "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", 
            "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", 
            "scissors", "teddy bear", "hair drier", "toothbrush"
        ]
        
        # 为每个类别创建文本提示
        text_descriptions = [f"a photo of a {obj}" for obj in coco_classes]
        
        # 获取文本嵌入
        text_inputs = torch.cat([clip.tokenize(desc) for desc in text_descriptions]).to(self.device)
        with torch.no_grad():
            text_embeddings = self.clip_model.encode_text(text_inputs)
        
        # 归一化嵌入
        text_embeddings /= text_embeddings.norm(dim=-1, keepdim=True)
        
        return {
            "classes": coco_classes,
            "embeddings": text_embeddings,
            "descriptions": text_descriptions
        }

    def detect_objects(self, frame):
        """
        检测帧中的主要物体
        frame: PIL.Image 图像
        返回: (物体列表, 置信度, 物体嵌入)
        """
        # 预处理图像
        image = self.preprocess(frame).unsqueeze(0).to(self.device)
        
        # 获取图像特征
        with torch.no_grad():
            image_features = self.clip_model.encode_image(image)
            image_features /= image_features.norm(dim=-1, keepdim=True)
        
        # 计算与物体文本的相似度
        similarity = (100.0 * image_features @ self.object_embeddings["embeddings"].T).softmax(dim=-1)
        values, indices = similarity[0].topk(self.top_k)
        
        # 提取检测到的物体信息
        detected_objects = []
        confidences = []
        object_embeddings = []
        
        for i in range(self.top_k):
            idx = indices[i].item()
            detected_objects.append(self.object_embeddings["classes"][idx])
            confidences.append(values[i].item())
            object_embeddings.append(self.object_embeddings["embeddings"][idx])
        
        return detected_objects, confidences, torch.stack(object_embeddings)

    def generate_contextual_description(self, objects, confidences):
        """使用GPT生成上下文感知的描述"""
        # 构造提示
        prompt = "Describe the scene concisely with the following objects: "
        obj_list = ", ".join([f"{obj} ({conf:.2f})" for obj, conf in zip(objects, confidences)])
        prompt += obj_list + ". Description:"
        
        # 调用OpenAI API (需要设置API密钥)
        openai.api_key = "YOUR_OPENAI_API_KEY"
        response = openai.Completion.create(
            engine="text-davinci-003",
            prompt=prompt,
            max_tokens=50,
            n=1,
            stop=None,
            temperature=0.7
        )
        
        return response.choices[0].text.strip()

    def forward(self, frame_tensor, spatial_feature):
        """
        frame_tensor: 代表性帧张量 [C, H, W]
        spatial_feature: 原始空间特征 [D]
        返回: 增强的空间特征 [D]
        """
        # 转换张量为PIL图像
        frame = self.tensor_to_pil(frame_tensor)
        
        # 检测物体
        detected_objects, confidences, object_embeddings = self.detect_objects(frame)
        
        # 生成上下文描述 (可选)
        # scene_description = self.generate_contextual_description(detected_objects, confidences)
        
        # 计算加权物体嵌入
        weights = torch.tensor(confidences).to(self.device).unsqueeze(1)
        weighted_embeddings = (object_embeddings * weights).sum(dim=0)
        weighted_embeddings /= weighted_embeddings.norm()
        
        # 融合物体信息到空间特征
        spatial_feature = spatial_feature.to(self.device)
        combined = torch.cat([spatial_feature, weighted_embeddings], dim=0)
        augmented_feature = self.fusion_module(combined)
        
        return augmented_feature, detected_objects

    def tensor_to_pil(self, tensor):
        """将张量转换为PIL图像"""
        # 假设输入张量范围为[0,1]且为CHW格式
        tensor = tensor.cpu().detach()
        if tensor.max() > 1.0:
            tensor = tensor / 255.0
        tensor = tensor.clamp(0, 1)
        array = tensor.permute(1, 2, 0).numpy() * 255
        return Image.fromarray(array.astype('uint8'))

# 在视频ViT中集成
class EnhancedViT(nn.Module):
    def __init__(self, vit_model, clip_augmenter=None):
        super().__init__()
        self.vit = vit_model
        self.clip_augmenter = clip_augmenter or ClipObjectAugmenter()
        
        # 空间特征适配器
        self.spatial_adapter = nn.Sequential(
            nn.Linear(512, 768),
            nn.ReLU(),
            nn.Linear(768, 768),
            nn.LayerNorm(768)
        )

    def forward(self, x, mask=None):
        # 原始ViT前向传播
        x, temporal_feature, spatial_feature = self.vit.forward_features(x, mask)
        
        # 获取代表性帧 (这里使用第一帧作为示例)
        representative_frame = x[:, 0, :, :, :]  # [B, C, H, W]
        
        # 应用CLIP物体增强
        augmented_spatial_features = []
        detected_objects_list = []
        
        for i in range(representative_frame.size(0)):
            frame = representative_frame[i]
            aug_feature, detected_objects = self.clip_augmenter(frame, spatial_feature[i])
            augmented_spatial_features.append(aug_feature)
            detected_objects_list.append(detected_objects)
        
        augmented_spatial = torch.stack(augmented_spatial_features)
        
        # 调整维度匹配原始空间特征
        spatial_feature = self.spatial_adapter(augmented_spatial)
        
        return x, temporal_feature, spatial_feature, detected_objects_list

# 使用示例
if __name__ == "__main__":
    # 创建伪视频数据 (batch, frames, channels, height, width)
    batch_size = 2
    num_frames = 8
    channels = 3
    height, width = 224, 224
    video_data = torch.randn(batch_size, num_frames, channels, height, width)
    
    # 初始化ViT模型 (这里使用伪模型)
    class DummyViT(nn.Module):
        def forward_features(self, x, mask=None):
            B, T, C, H, W = x.shape
            # 返回伪特征
            return torch.randn(B, T, C, H, W), \
                   torch.randn(B, 768), \
                   torch.randn(B, 768)
    
    # 创建增强的ViT模型
    vit = DummyViT()
    enhanced_vit = EnhancedViT(vit)
    
    # 前向传播
    output, temporal_feat, spatial_feat, detected_objects = enhanced_vit(video_data)
    
    print("输出形状:", output.shape)
    print("时间特征形状:", temporal_feat.shape)
    print("空间特征形状:", spatial_feat.shape)
    print("检测到的物体:", detected_objects)