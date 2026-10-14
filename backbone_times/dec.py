import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from sklearn.cluster import KMeans
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

# 3D CNN 特征提取器
class CNNFeatureExtractor(nn.Module):
    def __init__(self, embedding_dim=128):
        super(CNNFeatureExtractor, self).__init__()
        self.conv3d = nn.Sequential(
            nn.Conv3d(3, 64, kernel_size=(3, 3, 3), stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool3d((1, 2, 2)),  # 时间维度不变，空间降维
            nn.Conv3d(64, 128, kernel_size=(3, 3, 3), stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool3d((2, 2, 2)),  # 时间和空间降维
            nn.Conv3d(128, 256, kernel_size=(3, 3, 3), stride=1, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool3d((1, 1, 1))  # 压缩成固定大小
        )
        self.fc = nn.Linear(256, embedding_dim)  # 降维到 embedding_dim

    def forward(self, x):
        x = self.conv3d(x)  # [batch, 256, 1, 1, 1]
        x = x.view(x.size(0), -1)  # [batch, 256]
        x = self.fc(x)  # [batch, embedding_dim]
        return x

# DEC 目标分布计算
def target_distribution(q):
    weight = q ** 2 / q.sum(0)
    return (weight.T / weight.sum(1)).T  # 归一化

# DEC 聚类模型
class DEC(nn.Module):
    def __init__(self, feature_extractor, n_clusters=10):
        super(DEC, self).__init__()
        self.feature_extractor = feature_extractor  # CNN 提取特征
        self.cluster_layer = nn.Parameter(torch.Tensor(n_clusters, feature_extractor.fc.out_features))  # 聚类中心
        self.n_clusters = n_clusters
        nn.init.xavier_normal_(self.cluster_layer.data)  # 初始化聚类中心

    def forward(self, x):
        z = self.feature_extractor(x)  # 获取嵌入特征 [,768]
        dist = torch.sum((z.unsqueeze(1) - self.cluster_layer) ** 2, dim=2)
        q = 1.0 / (1.0 + dist) #[16,10] soft assignment
        q = (q.T / q.sum(1)).T  # 归一化
        return q, z

# 训练 DEC
def train_DEC(model, dataloader, n_clusters=10, max_epochs=100, lr=0.001, update_interval=10):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # 初始化聚类中心
    print("Initializing cluster centers with K-Means...")
    feature_list = []
    for batch in dataloader:
        batch = batch[0].to(device)
        _, features = model(batch)  # 提取特征
        feature_list.append(features.detach().cpu().numpy())

    features = np.concatenate(feature_list, axis=0)
    kmeans = KMeans(n_clusters=n_clusters, n_init=20)
    y_pred = kmeans.fit_predict(features)
    model.cluster_layer.data.copy_(torch.tensor(kmeans.cluster_centers_, dtype=torch.float32, device=device))
    
    # 训练
    for epoch in range(max_epochs):
        for i, batch in enumerate(dataloader):
            batch = batch[0].to(device)
            q, _ = model(batch)  # 获取聚类分布
            p = target_distribution(q).detach()  # 计算目标分布
            loss = F.kl_div(q.log(), p, reduction='batchmean')  # KL 散度损失
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if i % update_interval == 0:
                print(f"Epoch {epoch+1}/{max_epochs}, Batch {i}, Loss: {loss.item()}")

    print("Training Complete.")
    return model

video_data = torch.randn(1024, 3, 8, 224, 224)  # 1024 个视频样本
dataloader = DataLoader(TensorDataset(video_data), batch_size=16, shuffle=True)

# 定义模型
feature_extractor = CNNFeatureExtractor(embedding_dim=128)
dec_model = DEC(feature_extractor, n_clusters=10)

# 训练 DEC
trained_model = train_DEC(dec_model, dataloader)
