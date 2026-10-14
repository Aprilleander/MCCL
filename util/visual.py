import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import euclidean_distances
import seaborn as sns
from matplotlib import cm
import matplotlib.patches as mpatches
import torch
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.preprocessing import StandardScaler
import seaborn as sns
from matplotlib import cm
import matplotlib.patches as mpatches
import torch
import os


def visualize_all_prototypes(samples, true_labels, pred_labels, prototypes,n_samples=50, save_path=None):
    """
    专门分析prototype分布的可视化函数
    """
    
    # 确保所有输入都是NumPy数组
    if torch.is_tensor(samples):
        samples = samples.cpu().numpy()
    if torch.is_tensor(true_labels):
        true_labels = true_labels.cpu().numpy()
    if torch.is_tensor(pred_labels):
        pred_labels = pred_labels.cpu().numpy()
    if torch.is_tensor(prototypes):
        prototypes = prototypes.cpu().numpy()
    
    # 数据标准化
    scaler = StandardScaler()
    samples_scaled = scaler.fit_transform(samples)
    prototypes_scaled = scaler.transform(prototypes)
    
    # 合并样本和prototype
    all_data = np.vstack([samples_scaled, prototypes_scaled])
    all_labels = np.hstack([pred_labels, np.arange(len(prototypes)) + 100])  # 给prototype特殊标签
    
    print(f"数据形状: 样本={samples.shape}, prototype={prototypes.shape}")
    print(f"合并后数据形状: {all_data.shape}")
    
    # 创建可视化
    fig, axes = plt.subplots(2, 3, figsize=(24, 16))
    
    # 方法1: 标准t-SNE
    print("进行标准t-SNE...")
    try:
        tsne_standard = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
        embeddings_standard = tsne_standard.fit_transform(all_data)
        
        # 分离样本和prototype的嵌入
        samples_2d_standard = embeddings_standard[:-len(prototypes), :]
        prototypes_2d_standard = embeddings_standard[-len(prototypes):, :]
        
        # 子图1: 标准t-SNE
        ax1 = axes[0, 0]
        # 绘制样本（透明度较低，避免遮挡）
        ax1.scatter(samples_2d_standard[:, 0], samples_2d_standard[:, 1], 
                   c=pred_labels, cmap='tab20', alpha=0.3, s=5)
        # 绘制prototype（突出显示）
        for i in range(len(prototypes)):
            ax1.scatter(prototypes_2d_standard[i, 0], prototypes_2d_standard[i, 1], 
                       c='red', marker='*', s=300, edgecolors='black', linewidth=2)
            ax1.annotate(f'P{i}', (prototypes_2d_standard[i, 0], prototypes_2d_standard[i, 1]),
                        xytext=(8, 8), textcoords='offset points', fontsize=12, 
                        fontweight='bold', color='darkred')
        
        ax1.set_title('标准t-SNE (perplexity=30)')
        ax1.set_xlabel('t-SNE 维度 1')
        ax1.set_ylabel('t-SNE 维度 2')
        ax1.grid(True, alpha=0.3)
    except Exception as e:
        print(f"标准t-SNE失败: {e}")
        axes[0, 0].text(0.5, 0.5, 't-SNE失败', ha='center', va='center', transform=axes[0, 0].transAxes)
    
    # 方法2: 高perplexity的t-SNE（更好的全局结构）
    print("进行高perplexity t-SNE...")
    try:
        tsne_high_perp = TSNE(n_components=2, random_state=42, perplexity=100, n_iter=1500)
        embeddings_high_perp = tsne_high_perp.fit_transform(all_data)
        
        samples_2d_high_perp = embeddings_high_perp[:-len(prototypes), :]
        prototypes_2d_high_perp = embeddings_high_perp[-len(prototypes):, :]
        
        # 子图2: 高perplexity t-SNE
        ax2 = axes[0, 1]
        ax2.scatter(samples_2d_high_perp[:, 0], samples_2d_high_perp[:, 1], 
                   c=pred_labels, cmap='tab20', alpha=0.3, s=5)
        for i in range(len(prototypes)):
            ax2.scatter(prototypes_2d_high_perp[i, 0], prototypes_2d_high_perp[i, 1], 
                       c='red', marker='*', s=300, edgecolors='black', linewidth=2)
            ax2.annotate(f'P{i}', (prototypes_2d_high_perp[i, 0], prototypes_2d_high_perp[i, 1]),
                        xytext=(8, 8), textcoords='offset points', fontsize=12, 
                        fontweight='bold', color='darkred')
        
        ax2.set_title('高perplexity t-SNE (perplexity=100)\n更好的全局结构')
        ax2.set_xlabel('t-SNE 维度 1')
        ax2.set_ylabel('t-SNE 维度 2')
        ax2.grid(True, alpha=0.3)
    except Exception as e:
        print(f"高perplexity t-SNE失败: {e}")
        axes[0, 1].text(0.5, 0.5, '高perplexity t-SNE失败', ha='center', va='center', transform=axes[0, 1].transAxes)
    
    # 方法3: PCA（线性方法，保持全局结构）
    print("进行PCA...")
    pca = PCA(n_components=2, random_state=42)
    embeddings_pca = pca.fit_transform(all_data)
    
    samples_2d_pca = embeddings_pca[:-len(prototypes), :]
    prototypes_2d_pca = embeddings_pca[-len(prototypes):, :]
    
    # 子图3: PCA
    ax3 = axes[0, 2]
    ax3.scatter(samples_2d_pca[:, 0], samples_2d_pca[:, 1], 
               c=pred_labels, cmap='tab20', alpha=0.3, s=5)
    for i in range(len(prototypes)):
        ax3.scatter(prototypes_2d_pca[i, 0], prototypes_2d_pca[i, 1], 
                   c='red', marker='*', s=300, edgecolors='black', linewidth=2)
        ax3.annotate(f'P{i}', (prototypes_2d_pca[i, 0], prototypes_2d_pca[i, 1]),
                    xytext=(8, 8), textcoords='offset points', fontsize=12, 
                    fontweight='bold', color='darkred')
    
    ax3.set_title('PCA (保持全局结构)')
    ax3.set_xlabel('PC1 (方差: {:.2f}%)'.format(pca.explained_variance_ratio_[0]*100))
    ax3.set_ylabel('PC2 (方差: {:.2f}%)'.format(pca.explained_variance_ratio_[1]*100))
    ax3.grid(True, alpha=0.3)
    
    # 方法4: 仅prototype的t-SNE
    print("进行仅prototype的t-SNE...")
    try:
        tsne_prototype_only = TSNE(n_components=2, random_state=42, perplexity=15, n_iter=1000)
        prototypes_2d_only = tsne_prototype_only.fit_transform(prototypes_scaled)
        
        # 子图4: 仅prototype的t-SNE
        ax4 = axes[1, 0]
        for i in range(len(prototypes)):
            ax4.scatter(prototypes_2d_only[i, 0], prototypes_2d_only[i, 1], 
                       c='red', marker='*', s=400, edgecolors='black', linewidth=2)
            ax4.annotate(f'P{i}', (prototypes_2d_only[i, 0], prototypes_2d_only[i, 1]),
                        xytext=(10, 10), textcoords='offset points', fontsize=14, 
                        fontweight='bold', color='darkred')
        
        ax4.set_title('仅Prototype的t-SNE\n(无样本干扰)')
        ax4.set_xlabel('t-SNE 维度 1')
        ax4.set_ylabel('t-SNE 维度 2')
        ax4.grid(True, alpha=0.3)
        
        # 计算prototype间的距离并显示
        prototype_distances = euclidean_distances(prototypes_2d_only)
        avg_distance = np.mean(prototype_distances)
        ax4.text(0.02, 0.98, f'平均距离: {avg_distance:.2f}', 
                transform=ax4.transAxes, fontsize=12, 
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8))
    except Exception as e:
        print(f"仅prototype的t-SNE失败: {e}")
        axes[1, 0].text(0.5, 0.5, '仅prototype t-SNE失败', ha='center', va='center', transform=axes[1, 0].transAxes)
    
    # 方法5: 高维空间中prototype的距离分布
    ax5 = axes[1, 1]
    # 计算高维空间中prototype之间的距离
    prototype_high_dim_distances = euclidean_distances(prototypes_scaled)
    
    # 绘制距离分布直方图
    distances_flat = prototype_high_dim_distances[np.triu_indices_from(prototype_high_dim_distances, k=1)]
    ax5.hist(distances_flat, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
    ax5.axvline(np.mean(distances_flat), color='red', linestyle='--', linewidth=2, 
                label=f'平均距离: {np.mean(distances_flat):.2f}')
    ax5.set_xlabel('Prototype间距离')
    ax5.set_ylabel('频次')
    ax5.set_title('高维空间中Prototype间距离分布')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # 方法6: prototype与样本的平均距离
    ax6 = axes[1, 2]
    # 计算每个prototype与所有样本的平均距离
    avg_distances_to_samples = []
    for i in range(len(prototypes)):
        distances = np.linalg.norm(samples_scaled - prototypes_scaled[i], axis=1)
        avg_distances_to_samples.append(np.mean(distances))
    
    ax6.bar(range(len(prototypes)), avg_distances_to_samples, color='lightgreen', alpha=0.7)
    ax6.axhline(np.mean(avg_distances_to_samples), color='red', linestyle='--', linewidth=2,
                label=f'总平均: {np.mean(avg_distances_to_samples):.2f}')
    ax6.set_xlabel('Prototype索引')
    ax6.set_ylabel('与样本的平均距离')
    ax6.set_title('每个Prototype与所有样本的平均距离')
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # 保存图片
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"图片已保存到: {save_path}")
    
    plt.show()
    
    # 打印详细分析
    print("\n=== Prototype分布分析 ===")
    print(f"高维空间中prototype间平均距离: {np.mean(distances_flat):.4f}")
    print(f"高维空间中prototype间最小距离: {np.min(distances_flat):.4f}")
    print(f"高维空间中prototype间最大距离: {np.max(distances_flat):.4f}")
    print(f"Prototype与样本的平均距离: {np.mean(avg_distances_to_samples):.4f}")
    
    # 分析prototype是否真的集中
    if hasattr('prototypes_2d_only', 'var'):
        prototype_2d_std = np.std(prototypes_2d_only, axis=0)
        print(f"仅prototype t-SNE中，prototype在2D空间的标准差: {prototype_2d_std}")
    
    return {
        'prototypes_standard': prototypes_2d_standard if 'prototypes_2d_standard' in locals() else None,
        'prototypes_high_perp': prototypes_2d_high_perp if 'prototypes_2d_high_perp' in locals() else None,
        'prototypes_pca': prototypes_2d_pca,
        'prototypes_only': prototypes_2d_only if 'prototypes_2d_only' in locals() else None,
        'high_dim_distances': distances_flat,
        'avg_distances_to_samples': avg_distances_to_samples
    }
    

def visualize_prototype_distances(samples, true_labels, prototypes, prototype_idx, n_samples=200, save_path='./pca4.png'):
    """
    可视化指定prototype与样本的距离关系并保存图片
    
    参数:
    - samples: [N, 768] 样本特征
    - true_labels: [N] 真实标签
    - prototypes: [50, 768] 所有prototype特征
    - prototype_idx: 要可视化的prototype索引
    - n_samples: 要选择的样本数量
    - save_path: 图片保存路径，如果为None则不保存
    """
    
    # 确保所有输入都是NumPy数组
    if torch.is_tensor(samples):
        samples = samples.cpu().numpy()
    if torch.is_tensor(true_labels):
        true_labels = true_labels.cpu().numpy()
    if torch.is_tensor(prototypes):
        prototypes = prototypes.cpu().numpy()
    
    # 确保n_samples不超过可用样本数
    actual_n_samples = min(n_samples, len(samples))
    
    # 1. 计算所有样本与指定prototype的距离
    prototype = prototypes[prototype_idx]
    distances_to_prototype = np.linalg.norm(samples - prototype, axis=1)
    
    # 2. 选择距离最近的n_samples个样本
    nearest_indices = np.argsort(distances_to_prototype)[:actual_n_samples]
    selected_samples = samples[nearest_indices]
    selected_true_labels = true_labels[nearest_indices]
    selected_distances = distances_to_prototype[nearest_indices]
    
    # 3. 计算选定样本之间的距离矩阵
    sample_distances = euclidean_distances(selected_samples)
    
    # 4. 创建可视化
    fig = plt.figure(figsize=(20, 6))
    
    # 子图1: 样本与prototype的距离分布
    plt.subplot(1, 3, 1)
    colors = ['blue' if label == prototype_idx else 'red' for label in selected_true_labels]
    plt.scatter(range(len(selected_distances)), selected_distances, c=colors, alpha=0.7)
    plt.xlabel('样本索引 (按距离排序)')
    plt.ylabel('到Prototype的距离')
    plt.title(f'Prototype {prototype_idx} - 样本距离分布\n(蓝色: 同类, 红色: 异类)')
    plt.grid(True, alpha=0.3)
    
    # 添加图例
    same_class_patch = mpatches.Patch(color='blue', label=f'真实类别 {prototype_idx}')
    diff_class_patch = mpatches.Patch(color='red', label='其他类别')
    plt.legend(handles=[same_class_patch, diff_class_patch])
    
    # 子图2: 样本间距离的热力图
    plt.subplot(1, 3, 2)
    
    # 按真实标签排序以便更好观察
    sort_indices = np.argsort(selected_true_labels)
    sorted_distances = sample_distances[sort_indices][:, sort_indices]
    sorted_labels = selected_true_labels[sort_indices]
    
    # 创建掩码来突出显示类别边界
    unique_labels = np.unique(sorted_labels)
    mask = np.zeros_like(sorted_distances, dtype=bool)
    
    # 在热力图中添加分隔线
    current_idx = 0
    for label in unique_labels:
        count = np.sum((sorted_labels == label).astype(int))
        if current_idx > 0:
            mask[current_idx-1:current_idx+1, current_idx-1:current_idx+1] = True
        current_idx += count
    
    sns.heatmap(sorted_distances, 
                mask=mask, 
                cmap='viridis', 
                cbar_kws={'label': '样本间距离'},
                xticklabels=False, 
                yticklabels=False)
    plt.title(f'Prototype {prototype_idx} - 样本间距离热力图')
    
    # 子图3: 所有样本的降维可视化
    plt.subplot(1, 3, 3)
    
    # 使用所有样本进行降维
    all_samples = samples
    all_true_labels = true_labels
    
    # 安全地设置perplexity
    safe_n_samples = len(all_samples)
    perplexity = min(30, safe_n_samples - 1)
    
    # 根据样本数量选择合适的降维方法
    if safe_n_samples <= 30:
        # 样本太少，使用PCA
        pca = PCA(n_components=2, random_state=42)
        all_samples_2d = pca.fit_transform(all_samples)
        method_name = "PCA"
        
        # 计算prototype在PCA空间的位置
        prototype_2d = pca.transform(prototype.reshape(1, -1))
    else:
        # 样本足够多，使用t-SNE
        try:
            tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity)
            
            # 将prototype与所有样本一起降维
            combined_samples = np.vstack([all_samples, prototype.reshape(1, -1)])
            combined_2d = tsne.fit_transform(combined_samples)
            all_samples_2d = combined_2d[:-1, :]
            prototype_2d = combined_2d[-1, :].reshape(1, -1)
            method_name = "t-SNE"
        except Exception as e:
            print(f"t-SNE失败: {e}，使用PCA替代")
            # 如果t-SNE失败，使用PCA
            pca = PCA(n_components=2, random_state=42)
            all_samples_2d = pca.fit_transform(all_samples)
            prototype_2d = pca.transform(prototype.reshape(1, -1))
            method_name = "PCA"
    
    # 为不同类别创建颜色映射
    unique_true_labels = np.unique(all_true_labels)
    # 使用tab20色彩映射，如果类别超过20个，使用viridis
    if len(unique_true_labels) <= 20:
        color_map = cm.get_cmap('tab20', len(unique_true_labels))
    else:
        color_map = cm.get_cmap('viridis', len(unique_true_labels))
    
    # 绘制所有样本的散点图
    for i, label in enumerate(unique_true_labels):
        mask = all_true_labels == label
        color = color_map(i)
        # 特殊标记目标类别
        if label == prototype_idx:
            label_text = f'目标类别 {label}'
            # 使用更大的点标记目标类别
            plt.scatter(all_samples_2d[mask, 0], all_samples_2d[mask, 1], 
                       c=[color], label=label_text, alpha=0.7, s=80, marker='s')
        else:
            label_text = f'类别 {label}'
            plt.scatter(all_samples_2d[mask, 0], all_samples_2d[mask, 1], 
                       c=[color], label=label_text, alpha=0.7, s=50)
    
    # 标记prototype位置
    plt.scatter(prototype_2d[0, 0], prototype_2d[0, 1], 
               c='black', marker='*', s=400, label=f'Prototype {prototype_idx}', 
               edgecolors='yellow', linewidth=3)
    
    # 标记选定的最近样本
    selected_mask = np.zeros(len(all_samples), dtype=bool)
    selected_mask[nearest_indices] = True
    plt.scatter(all_samples_2d[selected_mask, 0], all_samples_2d[selected_mask, 1], 
               facecolors='none', edgecolors='black', s=60, 
               linewidth=1, alpha=0.8, label='最近样本')
    
    plt.xlabel(f'{method_name} 维度 1')
    plt.ylabel(f'{method_name} 维度 2')
    plt.title(f'Prototype {prototype_idx} - 所有样本{method_name}可视化\n(共{len(all_samples)}个样本)')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # 保存图片
    if save_path:
        # 确保目录存在
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"图片已保存到: {save_path}")
    
    plt.show()
    
    # 5. 打印统计信息
    same_class_count = np.sum((selected_true_labels == prototype_idx).astype(int))
    different_class_count = actual_n_samples - same_class_count
    
    print(f"=== Prototype {prototype_idx} 统计信息 ===")
    print(f"总样本数: {actual_n_samples}")
    print(f"同类样本数: {same_class_count} ({same_class_count/actual_n_samples*100:.1f}%)")
    print(f"异类样本数: {different_class_count} ({different_class_count/actual_n_samples*100:.1f}%)")
    print(f"平均距离: {np.mean(selected_distances):.4f}")
    print(f"最小距离: {np.min(selected_distances):.4f}")
    print(f"最大距离: {np.max(selected_distances):.4f}")
    print(f"使用的降维方法: {method_name}")
    print(f"所有样本中的类别: {unique_true_labels}")
    
    return {
        'selected_indices': nearest_indices,
        'selected_distances': selected_distances,
        'sample_distances': sample_distances,
        'embeddings': all_samples_2d,
        'method': method_name
    }

def compare_multiple_prototypes(samples, true_labels, predicted_labels, prototypes, prototype_indices, n_samples=100):
    """
    比较多个prototype的距离分布
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.ravel()
    
    for i, proto_idx in enumerate(prototype_indices[:4]):
        # 计算距离
        prototype = prototypes[proto_idx]
        distances = np.linalg.norm(samples - prototype, axis=1)
        nearest_indices = np.argsort(distances)[:n_samples]
        selected_distances = distances[nearest_indices]
        selected_true_labels = true_labels[nearest_indices]
        
        # 创建颜色编码
        colors = ['blue' if label == proto_idx else 'red' for label in selected_true_labels]
        
        axes[i].scatter(range(len(selected_distances)), selected_distances, c=colors, alpha=0.7)
        axes[i].set_xlabel('样本索引')
        axes[i].set_ylabel('距离')
        axes[i].set_title(f'Prototype {proto_idx}\n(蓝色: 同类, 红色: 异类)')
        axes[i].grid(True, alpha=0.3)
        
        # 添加统计信息
        same_class_ratio = np.mean(np.array(selected_true_labels) == proto_idx)
        axes[i].text(0.05, 0.95, f'同类比例: {same_class_ratio:.2%}', 
                    transform=axes[i].transAxes, bbox=dict(boxstyle="round", facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.show()

# 使用示例
if __name__ == "__main__":
    # 假设你有以下数据（请替换为你的实际数据）
    # samples = np.random.randn(9935, 768)  # 你的样本数据
    # true_labels = np.random.randint(0, 50, 9935)  # 真实标签
    # predicted_labels = np.random.randint(0, 50, 9935)  # 预测标签
    # prototypes = np.random.randn(50, 768)  # prototype数据
    
    # 可视化单个prototype
    result = visualize_prototype_distances(
        samples, true_labels, predicted_labels, prototypes, 
        prototype_idx=0, n_samples=300
    )
    
    # 比较多个prototype
    compare_multiple_prototypes(
        samples, true_labels, predicted_labels, prototypes,
        prototype_indices=[0, 1, 2, 3], n_samples=100
    )