import torch
import torch.nn.functional as F
import torch
import torch.nn.functional as F
from collections import Counter


def entropy_stable_clustering(similarity_matrix, min_clusters=2, max_clusters=4, 
                            max_iter=100, temperature=1.0, learning_rate=0.1):
    """
    基于熵稳定性的聚类算法
    
    参数:
        similarity_matrix: [n, n] 相似度矩阵
        min_clusters: 最小聚类数
        max_clusters: 最大聚类数
        max_iter: 最大迭代次数
        temperature: 熵调节温度参数
        learning_rate: 学习率
    
    返回:
        cluster_assignments: 每个样本的聚类标签
    """
    n = similarity_matrix.shape[0]
    
    # 初始化聚类分配（软分配）
    cluster_probs = initialize_cluster_probs(n, min_clusters, max_clusters)
    
    best_cluster_probs = None
    best_energy = float('inf')
    
    for k in range(min_clusters, max_clusters + 1):
        # 对每个可能的聚类数量进行优化
        current_probs = cluster_probs[:, :k].clone()
        
        for iteration in range(max_iter):
            # 计算能量函数（损失）
            energy = compute_energy(current_probs, similarity_matrix, temperature)
            
            # 梯度下降更新
            updated_probs = update_cluster_probs(
                current_probs, similarity_matrix, temperature, learning_rate
            )
            
            # 检查收敛
            if torch.norm(updated_probs - current_probs) < 1e-6:
                break
                
            current_probs = updated_probs
            
        # 记录最佳结果
        final_energy = compute_energy(current_probs, similarity_matrix, temperature)
        if final_energy < best_energy:
            best_energy = final_energy
            best_cluster_probs = current_probs
    
    # 将软分配转换为硬分配
    cluster_assignments = torch.argmax(best_cluster_probs, dim=1)
    
    return cluster_assignments

def initialize_cluster_probs(n, min_k, max_k):
    """初始化聚类概率"""
    max_k_to_use = max_k
    probs = torch.rand(n, max_k_to_use)
    probs = F.softmax(probs, dim=1)
    return probs

def compute_energy(probs, similarity_matrix, temperature):
    """
    计算能量函数，包含内部紧凑度和熵稳定性
    """
    n, k = probs.shape
    
    # 1. 内部紧凑度项：鼓励相似样本分配到同一聚类
    compactness_term = compute_compactness_term(probs, similarity_matrix)
    
    # 2. 熵稳定性项：鼓励清晰的分配（低熵）
    entropy_term = compute_entropy_term(probs, temperature)
    
    # 3. 平衡项：避免单个聚类过大
    balance_term = compute_balance_term(probs)
    
    # 总能量（损失）
    energy = compactness_term - entropy_term + balance_term
    
    return energy

def compute_compactness_term(probs, similarity_matrix):
    """计算内部紧凑度项"""
    n, k = probs.shape
    
    # 计算聚类间的连接强度
    compactness = 0
    for i in range(n):
        for j in range(i+1, n):
            # 样本i和j分配到同一聚类的概率
            same_cluster_prob = torch.sum(probs[i] * probs[j])
            # 加权相似度
            compactness += similarity_matrix[i, j] * same_cluster_prob
    
    return -compactness  # 负号因为我们希望最大化紧凑度

def compute_entropy_term(probs, temperature):
    """计算熵稳定性项"""
    n, k = probs.shape
    
    # 计算每个样本的分配熵
    sample_entropies = -torch.sum(probs * torch.log(probs + 1e-8), dim=1)
    avg_sample_entropy = torch.mean(sample_entropies)
    
    # 计算聚类分布的熵
    cluster_sizes = torch.sum(probs, dim=0)
    cluster_dist = cluster_sizes / n
    cluster_entropy = -torch.sum(cluster_dist * torch.log(cluster_dist + 1e-8))
    
    # 总熵项（我们希望最小化熵，所以用负号）
    entropy_term = - (avg_sample_entropy + cluster_entropy) / temperature
    
    return entropy_term

def compute_balance_term(probs):
    """计算平衡项，避免单个聚类过大"""
    n, k = probs.shape
    cluster_sizes = torch.sum(probs, dim=0)
    
    # 计算聚类大小的方差
    balance_term = torch.var(cluster_sizes)
    
    return balance_term

def update_cluster_probs(probs, similarity_matrix, temperature, learning_rate=0.1):
    """更新聚类概率"""
    n, k = probs.shape
    
    # 计算梯度
    with torch.enable_grad():
        probs_requires_grad = probs.clone().requires_grad_(True)
        
        energy = compute_energy(probs_requires_grad, similarity_matrix, temperature)
        energy.backward()
        
        gradient = probs_requires_grad.grad
        
    # 梯度下降更新
    updated_probs = probs - learning_rate * gradient
    
    # 投影到概率单纯形
    updated_probs = F.softmax(updated_probs, dim=1)
    
    return updated_probs

# 备选方案：基于图割的稳态聚类方法
def graph_cut_steady_clustering(similarity_matrix, min_clusters=2, max_clusters=4):
    """
    基于图割和模块度的稳态聚类
    """
    n = similarity_matrix.shape[0]
    
    best_partition = None
    best_modularity = -float('inf')
    
    for k in range(min_clusters, max_clusters + 1):
        # 使用谱聚类初始化
        initial_partition = spectral_clustering(similarity_matrix, k)
        
        # 优化分区
        partition = optimize_partition(initial_partition, similarity_matrix)
        
        # 计算模块度
        modularity = compute_modularity(partition, similarity_matrix)
        
        if modularity > best_modularity:
            best_modularity = modularity
            best_partition = partition
    
    return best_partition

def spectral_clustering(similarity_matrix, k):
    """谱聚类"""
    n = similarity_matrix.shape[0]
    
    # 计算拉普拉斯矩阵
    degree_matrix = torch.diag(torch.sum(similarity_matrix, dim=1))
    laplacian = degree_matrix - similarity_matrix
    
    # 计算特征值和特征向量
    eigenvalues, eigenvectors = torch.linalg.eigh(laplacian)
    
    # 选择前k个最小特征值对应的特征向量
    indices = torch.argsort(eigenvalues)[:k]
    selected_eigenvectors = eigenvectors[:, indices]
    
    # 对特征向量进行k-means聚类
    from sklearn.cluster import KMeans
    kmeans = KMeans(n_clusters=k, random_state=42)
    cluster_labels = kmeans.fit_predict(selected_eigenvectors.numpy())
    
    return torch.tensor(cluster_labels)

def optimize_partition(partition, similarity_matrix, max_iter=50):
    """优化分区以提高模块度"""
    n = similarity_matrix.shape[0]
    current_partition = partition.clone()
    current_modularity = compute_modularity(current_partition, similarity_matrix)
    
    for iteration in range(max_iter):
        improved = False
        
        # 尝试移动每个节点
        for i in range(n):
            original_cluster = current_partition[i]
            best_cluster = original_cluster
            best_delta = 0
            
            # 尝试将节点i移动到其他聚类
            for candidate_cluster in torch.unique(current_partition):
                if candidate_cluster == original_cluster:
                    continue
                
                # 临时移动节点
                temp_partition = current_partition.clone()
                temp_partition[i] = candidate_cluster
                
                # 计算模块度变化
                new_modularity = compute_modularity(temp_partition, similarity_matrix)
                delta = new_modularity - current_modularity
                
                if delta > best_delta:
                    best_delta = delta
                    best_cluster = candidate_cluster
            
            # 如果找到改进，则应用
            if best_delta > 0:
                current_partition[i] = best_cluster
                current_modularity += best_delta
                improved = True
        
        if not improved:
            break
    
    return current_partition

def compute_modularity(partition, similarity_matrix):
    """计算模块度"""
    n = similarity_matrix.shape[0]
    clusters = torch.unique(partition)
    k = len(clusters)
    
    # 计算总连接强度
    total_strength = torch.sum(similarity_matrix)
    
    modularity = 0
    for cluster in clusters:
        # 找到属于该聚类的节点
        cluster_nodes = (partition == cluster).nonzero(as_tuple=True)[0]
        
        # 计算聚类内连接强度
        intra_strength = 0
        for i in cluster_nodes:
            for j in cluster_nodes:
                intra_strength += similarity_matrix[i, j]
        
        # 计算该聚类的总连接强度
        cluster_strength = 0
        for i in cluster_nodes:
            cluster_strength += torch.sum(similarity_matrix[i])
        
        # 计算模块度贡献
        modularity += intra_strength / total_strength - (cluster_strength / total_strength) ** 2
    
    return modularity

def steady_clustering(u_feats,u_targets,pse_labels, mask_old_new=None):
    """
    使用稳态聚类细化伪标签
    
    参数:
        u_feats: 无标签特征 [n, 768]
        pse_labels: 初始伪标签 [n]
        mask_old_new: 可选，旧新样本掩码 [n]
    
    返回:
        refined_pse_labels: 细化后的伪标签
        class_cluster_results: 每个类别的聚类结果字典
        clustering_quality: 聚类质量分析结果
    """
    # 如果没有提供mask_old_new，则创建一个全为True的掩码
    if mask_old_new is None:
        mask_old_new = torch.ones_like(pse_labels, dtype=torch.bool)
    
    # 获取所有偶数伪标签（0, 2, 4, 6, ...）
    even_pse_labels = torch.unique(pse_labels)
    even_pse_labels = even_pse_labels[even_pse_labels % 2 == 0]
    
    # 存储每个类别的聚类结果
    class_cluster_results = {}
    results = []
    
    # 对每个类别进行稳态聚类
    for class_label in even_pse_labels:
        # 创建类别掩码，标记哪些样本属于当前类别
        class_mask = (pse_labels == class_label) & mask_old_new
        
        # 获取当前类别的样本索引
        class_indices = torch.where(class_mask)[0]
        
        # 获取当前类别的样本特征
        class_samples = u_feats[class_mask]
        class_true_labels = u_targets[class_mask]
        
        # 如果样本数量足够，进行稳态聚类
        if len(class_samples) > 1:
            # 计算样本间的相似度矩阵
            features_norm = F.normalize(class_samples, p=2, dim=1)
            similarity_matrix = torch.matmul(features_norm, features_norm.T)
            
            # 使用稳态聚类算法
            subcluster_labels = steady_state_clustering_wrapper(class_samples)
            unique_subclusters = torch.unique(subcluster_labels)
            
            for sub_label in unique_subclusters:
                # 获取当前子类的样本
                subcluster_mask = (subcluster_labels == sub_label)
                subcluster_true_labels = class_true_labels[subcluster_mask]
                
                # 统计真实标签分布
                label_counts = Counter(subcluster_true_labels.tolist())
                total_samples = len(subcluster_true_labels)
                
                # 找到占比最大的真实标签
                if label_counts:
                    majority_label, majority_count = label_counts.most_common(1)[0]
                    majority_ratio = majority_count / total_samples
                    
                    # 创建结果字典
                    result = {
                        '原始类别': class_label,
                        '子类ID': f"{class_label}_{sub_label}",
                        '样本数量': total_samples,
                        '主要真实标签': majority_label,
                        '主要标签数量': majority_count,
                        '纯度': majority_ratio,
                        '纯度评估': '高' if majority_ratio >= 0.7 else ('中等' if majority_ratio >= 0.5 else '低')
                    }
                    results.append(result)

    df = pd.DataFrame(results)
    # 按原始类别和纯度排序
    df = df.sort_values(['原始类别', '纯度'], ascending=[True, False])
    
    # 保存到Excel
    df.to_excel(save_path, index=False)
    
    print(f"纯度分析结果已保存到: {save_path}")
    print(f"共分析了 {len(df)} 个子类")
    print(f"总体平均纯度: {df['纯度'].mean():.4f}")
    
    # 生成新的伪标签（将子聚类结果映射回原始索引）
    refined_pse_labels = pse_labels.clone()
    
    for class_label, result in class_cluster_results.items():
        class_mask = result['class_mask']
        subcluster_labels = result['subcluster_labels']
        
        # 为每个子聚类分配新的标签（例如，原始标签 * 10 + 子聚类标签）
        new_labels = class_label * 10 + subcluster_labels
        
        # 更新伪标签
        refined_pse_labels[class_mask] = new_labels
    
    return refined_pse_labels, class_cluster_results, clustering_quality

def steady_state_clustering_wrapper(class_samples):
    """
    稳态聚类包装函数
    """
    n = class_samples.shape[0]
    
    if n <= 1:
        return torch.zeros(n, dtype=torch.long)
    
    # 计算相似度矩阵
    features_norm = F.normalize(class_samples, p=2, dim=1)
    similarity_matrix = torch.matmul(features_norm, features_norm.T)
    
    # 使用熵稳定性聚类
    cluster_assignments = entropy_stable_clustering(
        similarity_matrix, 
        min_clusters=2, 
        max_clusters=4
    )
    
    return cluster_assignments

def analyze_clustering_quality(class_cluster_results):
    """分析聚类结果的质量"""
    quality_results = {}
    
    for class_label, result in class_cluster_results.items():
        similarity_matrix = result['similarity_matrix']
        subcluster_labels = result['subcluster_labels']
        
        # 计算聚类内平均相似度
        intra_cluster_similarity = compute_intra_cluster_similarity(
            similarity_matrix, subcluster_labels
        )
        
        # 计算聚类间平均相似度
        inter_cluster_similarity = compute_inter_cluster_similarity(
            similarity_matrix, subcluster_labels
        )
        
        # 计算轮廓系数
        silhouette_score = compute_silhouette_score(
            similarity_matrix, subcluster_labels
        )
        
        quality_results[class_label] = {
            'intra_cluster_similarity': intra_cluster_similarity,
            'inter_cluster_similarity': inter_cluster_similarity,
            'silhouette_score': silhouette_score,
            'num_subclusters': result['num_subclusters']
        }
    
    return quality_results

def compute_intra_cluster_similarity(similarity_matrix, labels):
    """计算聚类内平均相似度"""
    unique_labels = torch.unique(labels)
    total_intra_sim = 0
    total_pairs = 0
    
    for label in unique_labels:
        # 获取当前聚类的样本索引
        cluster_indices = torch.where(labels == label)[0]
        n_cluster = len(cluster_indices)
        
        if n_cluster > 1:
            # 计算聚类内所有样本对的相似度
            for i in range(n_cluster):
                for j in range(i+1, n_cluster):
                    idx_i = cluster_indices[i]
                    idx_j = cluster_indices[j]
                    total_intra_sim += similarity_matrix[idx_i, idx_j]
                    total_pairs += 1
    
    return total_intra_sim / total_pairs if total_pairs > 0 else 0

def compute_inter_cluster_similarity(similarity_matrix, labels):
    """计算聚类间平均相似度"""
    unique_labels = torch.unique(labels)
    total_inter_sim = 0
    total_pairs = 0
    
    for i, label1 in enumerate(unique_labels):
        for label2 in unique_labels[i+1:]:
            # 获取两个聚类的样本索引
            cluster1_indices = torch.where(labels == label1)[0]
            cluster2_indices = torch.where(labels == label2)[0]
            
            # 计算两个聚类间所有样本对的相似度
            for idx1 in cluster1_indices:
                for idx2 in cluster2_indices:
                    total_inter_sim += similarity_matrix[idx1, idx2]
                    total_pairs += 1
    
    return total_inter_sim / total_pairs if total_pairs > 0 else 0

def compute_silhouette_score(similarity_matrix, labels):
    """计算轮廓系数（基于相似度）"""
    n = similarity_matrix.shape[0]
    unique_labels = torch.unique(labels)
    
    if len(unique_labels) <= 1:
        return 0.0
    
    silhouette_scores = []
    
    for i in range(n):
        # 当前样本的标签
        current_label = labels[i]
        
        # 计算与同聚类其他样本的平均相似度（a_i）
        same_cluster_indices = torch.where(labels == current_label)[0]
        same_cluster_indices = same_cluster_indices[same_cluster_indices != i]  # 排除自身
        
        if len(same_cluster_indices) > 0:
            a_i = torch.mean(similarity_matrix[i, same_cluster_indices])
        else:
            a_i = 0
        
        # 计算与其他聚类样本的平均相似度，取最大值（b_i）
        b_i = -float('inf')
        
        for other_label in unique_labels:
            if other_label == current_label:
                continue
                
            other_cluster_indices = torch.where(labels == other_label)[0]
            if len(other_cluster_indices) > 0:
                avg_sim = torch.mean(similarity_matrix[i, other_cluster_indices])
                if avg_sim > b_i:
                    b_i = avg_sim
        
        # 如果b_i仍然是负无穷，说明只有一个聚类
        if b_i == -float('inf'):
            s_i = 0
        else:
            s_i = (b_i - a_i) / max(a_i, b_i) if max(a_i, b_i) > 0 else 0
        
        silhouette_scores.append(s_i)
    
    return torch.mean(torch.tensor(silhouette_scores))

# 打印聚类结果的辅助函数
def print_clustering_results(refined_pse_labels, class_cluster_results, clustering_quality):
    """打印聚类结果"""
    print("稳态聚类细化结果:")
    print("=" * 50)
    
    # 打印每个类别的聚类情况
    for class_label, result in class_cluster_results.items():
        unique_subclusters, counts = torch.unique(result['subcluster_labels'], return_counts=True)
        print(f"类别 {class_label}:")
        print(f"  样本数量: {len(result['class_samples'])}")
        print(f"  子类数量: {result['num_subclusters']}")
        for i, (sub_label, count) in enumerate(zip(unique_subclusters, counts)):
            print(f"    子类 {sub_label}: {count} 个样本")
    
    # 打印聚类质量
    print("\n聚类质量分析:")
    print("=" * 50)
    for class_label, quality in clustering_quality.items():
        print(f"类别 {class_label}:")
        print(f"  聚类内平均相似度: {quality['intra_cluster_similarity']:.4f}")
        print(f"  聚类间平均相似度: {quality['inter_cluster_similarity']:.4f}")
        print(f"  轮廓系数: {quality['silhouette_score']:.4f}")
    
    # 打印伪标签统计
    original_unique = torch.unique(pse_labels)
    refined_unique = torch.unique(refined_pse_labels)
    print(f"\n伪标签统计:")
    print(f"  原始伪标签类别数: {len(original_unique)}")
    print(f"  细化后伪标签类别数: {len(refined_unique)}")

if __name__ == "__main__":
    # 假设您已经有了以下数据
    # u_feats = ...  # [9935, 768]
    # pse_labels = ...  # [9935]
    # mask_old_new = ...  # [9935]
    
    # 调用函数进行伪标签细化
    refined_pse_labels, class_cluster_results, clustering_quality = steady_clustering(
        u_feats, pse_labels, mask_old_new
    )
    print_clustering_results(refined_pse_labels, class_cluster_results, clustering_quality)