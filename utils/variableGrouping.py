import numpy as np
from scipy.stats import pearsonr
from sklearn.metrics import mutual_info_score
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
import pandas as pd

def downsample_data(data, max_samples=5000):
    """
    数据下采样平滑
    """
    n_samples, n_vars = data.shape
    
    if n_samples <= max_samples:
        return data
    
    # 简单均匀下采样
    step = n_samples // max_samples
    downsampled_indices = np.arange(0, n_samples, step)
    downsampled_data = data[downsampled_indices]
    
    # 可选：添加滑动平均平滑
    if len(downsampled_data) > 100:
        window_size = min(10, len(downsampled_data) // 10)
        smoothed_data = np.zeros_like(downsampled_data)
        for i in range(n_vars):
            smoothed_data[:, i] = np.convolve(
                downsampled_data[:, i], 
                np.ones(window_size)/window_size, 
                mode='same'
            )
        return smoothed_data
    
    return downsampled_data


def compute_similarity_matrix( data, args):
    """
    计算综合相似度矩阵
    """
    n_samples, n_vars = data.shape
    
    # 初始化相似度矩阵
    pearson_matrix = np.zeros((n_vars, n_vars))
    mi_matrix = np.zeros((n_vars, n_vars))
    
    # 计算每对变量的相似度
    for i in range(n_vars):
        for j in range(i, n_vars):  # 利用对称性
            if i == j:
                pearson_matrix[i, j] = 1.0
                mi_matrix[i, j] = 1.0
            else:
                # 计算皮尔逊相关系数
                pearson_corr, _ = pearsonr(data[:, i], data[:, j])
                pearson_matrix[i, j] = abs(pearson_corr)  # 取绝对值，关注相关性强度
                pearson_matrix[j, i] = abs(pearson_corr)
                
                # 计算互信息（需要离散化）
                mi = compute_mutual_info(data[:, i], data[:, j])
                mi_matrix[i, j] = mi
                mi_matrix[j, i] = mi
    
    # 归一化互信息矩阵
    mi_max = np.max(mi_matrix)
    if mi_max > 0:
        mi_matrix_normalized = mi_matrix / mi_max
    else:
        mi_matrix_normalized = mi_matrix
    
    # 综合相似度矩阵（α为皮尔逊系数所占比重）
    alpha = getattr(args, 'similarity_alpha', 0.6)
    similarity_matrix = (
        alpha * (pearson_matrix + 1) / 2 +  # 皮尔逊从[-1,1]映射到[0,1]
        (1 - alpha) * mi_matrix_normalized
    )
    
    print(f"皮尔逊矩阵范围: [{np.min(pearson_matrix):.3f}, {np.max(pearson_matrix):.3f}]")
    print(f"互信息矩阵范围: [{np.min(mi_matrix):.3f}, {np.max(mi_matrix):.3f}]")
    print(f"综合相似度矩阵范围: [{np.min(similarity_matrix):.3f}, {np.max(similarity_matrix):.3f}]")
    
    return similarity_matrix

def compute_mutual_info( x, y,  bins='auto'):
    """
    计算两个连续变量的互信息
    """
    
    try:
        # 使用更鲁棒的离散化方法
        if bins == 'auto':
            # 基于数据特征的自适应分箱
            bins = min(50, len(x) // 10)  # 避免过多或过少的分箱
        
        # 使用分位数分箱，对异常值更鲁棒
        x_discrete = pd.cut(x, bins=bins, labels=False, duplicates='drop')
        y_discrete = pd.cut(y, bins=bins, labels=False, duplicates='drop')
        
        # 移除NaN值
        mask = ~(np.isnan(x_discrete) | np.isnan(y_discrete))
        x_clean = x_discrete[mask]
        y_clean = y_discrete[mask]
        
        if len(x_clean) == 0:
            return 0.0
            
        mi = mutual_info_score(x_clean, y_clean)
        return mi
        
    except Exception as e:
        print(f"互信息计算错误: {e}")
        return 0.0

def sparsify_similarity_matrix( similarity_matrix):
    """
    稀疏化相似度矩阵
    """
    # 计算分位数阈值
    eta = np.percentile(similarity_matrix, 70)  # 保留前30%的连接
    #eta = np.median(similarity_matrix)
    print(f"相似度阈值η(分位数): {eta:.3f}")
    
    # 稀疏化处理
    sparse_matrix = np.where(similarity_matrix > eta, similarity_matrix, 0)
    
    # 对角线置为0（避免自相关影响聚类）
    np.fill_diagonal(sparse_matrix, 0)
    
    return sparse_matrix

def hierarchical_clustering( sparse_matrix, n_vars, target_idx,args):
    """
    层次化聚类分组
    """
    # 初始化：每个变量一个簇
    clusters = [{i} for i in range(n_vars)]
    
    # 聚类参数（可从args获取或使用默认值）
    theta = args.cluster_theta   # 簇内最小相似度阈值
    gamma = args.cluster_gamma # 簇间最大相似度阈值  
    kappa = args.cluster_kappa   # 最大簇大小
    delta = args.cluster_delta # 动态调整步长
    max_iterations = args.cluster_max_iter
    
    print(f"聚类参数: θ={theta}, γ={gamma}, κ={kappa}, δ={delta}")
    
    iteration = 0
    while iteration < max_iterations:
        iteration += 1
        
        # 计算当前聚类状态
        intra_similarities = compute_intra_cluster_similarities(clusters, sparse_matrix)
        inter_similarities = compute_inter_cluster_similarities(clusters, sparse_matrix)
        
        # 检查终止条件
        condition1 = all(sim >= theta for sim in intra_similarities)  # 所有簇内相似度≥θ
        condition2 = all(sim < gamma for sim in inter_similarities)   # 所有簇间相似度<γ
        
        if condition1 or condition2:
            print(f"聚类完成，迭代次数: {iteration}")
            break
        
        # 寻找最相似的两个簇进行合并
        max_sim = -1
        merge_i, merge_j = -1, -1
        
        for i in range(len(clusters)):
            for j in range(i + 1, len(clusters)):
                sim = cluster_similarity(clusters[i], clusters[j], sparse_matrix)
                if sim > max_sim:
                    max_sim = sim
                    merge_i, merge_j = i, j
        
        if max_sim <= 0:  # 没有可合并的簇
            break
            
        # 合并簇
        new_cluster = clusters[merge_i].union(clusters[merge_j])
        clusters.pop(merge_j)
        clusters.pop(merge_i)
        clusters.append(new_cluster)
        
        # 检查并分裂过大的簇
        clusters = split_large_clusters(clusters, sparse_matrix, kappa, theta)
        
        # 动态调整阈值（如果迭代次数较多）
        if iteration % 10 == 0:
            theta = max(0.3, theta - delta)  # 逐步降低θ
            gamma = min(0.7, gamma + delta)  # 逐步提高γ
            print(f"动态调整阈值: θ={theta:.3f}, γ={gamma:.3f}")
    
    # 转换为分组列表
    groups = [list(cluster) for cluster in clusters]
    
    # 确保目标变量单独一组（根据你的研究设计）
    final_groups = ensure_target_separate(groups, target_idx)
    
    return final_groups

def compute_intra_cluster_similarities( clusters, sparse_matrix):
    """计算每个簇内的最小相似度"""
    intra_sims = []
    for cluster in clusters:
        if len(cluster) <= 1:
            intra_sims.append(0.0)  # 单变量簇设置一个很小的值，确保不满足聚类终止条件
        else:
            min_sim = float('inf')
            variables = list(cluster)
            for i in range(len(variables)):
                for j in range(i + 1, len(variables)):
                    sim = sparse_matrix[variables[i], variables[j]]
                    if sim > 0 and sim < min_sim:
                        min_sim = sim
            intra_sims.append(min_sim if min_sim != float('inf') else 0)
    return intra_sims

def compute_inter_cluster_similarities( clusters, sparse_matrix):
    """计算所有簇间的最大相似度"""
    inter_sims = []
    for i in range(len(clusters)):
        for j in range(i + 1, len(clusters)):
            sim = cluster_similarity(clusters[i], clusters[j], sparse_matrix)
            inter_sims.append(sim)
    return inter_sims

def cluster_similarity( cluster1, cluster2, sparse_matrix):
    """计算两个簇间的相似度（最大相关性得分）"""
    max_sim = 0
    for var1 in cluster1:
        for var2 in cluster2:
            sim = sparse_matrix[var1, var2]
            if sim > max_sim:
                max_sim = sim
    return max_sim

def split_large_clusters( clusters, sparse_matrix, kappa, theta):
    """分裂过大的簇"""
    new_clusters = []
    for cluster in clusters:
        if len(cluster) > kappa:
            # 对过大簇进行二次聚类
            sub_clusters = split_cluster(cluster, sparse_matrix, theta)
            new_clusters.extend(sub_clusters)
        else:
            new_clusters.append(cluster)
    return new_clusters

def split_cluster( cluster, sparse_matrix, theta):
    """分裂单个过大的簇"""
    variables = list(cluster)
    if len(variables) <= 1:
        return [cluster]
    
    # 提取子相似度矩阵
    sub_matrix = sparse_matrix[np.ix_(variables, variables)]
    
    # 使用层次聚类进行分裂
    from scipy.cluster.hierarchy import linkage, fcluster
    
    # 转换为距离矩阵（1 - 相似度）
    distance_matrix = 1 - sub_matrix
    np.fill_diagonal(distance_matrix, 0)
    
    # 层次聚类
    Z = linkage(squareform(distance_matrix), method='complete')
    
    # 尝试不同的聚类数来找到合适的分裂
    for k in range(2, len(variables)):
        labels = fcluster(Z, k, criterion='maxclust')
        sub_clusters = []
        for label in set(labels):
            sub_vars = [variables[i] for i in range(len(variables)) if labels[i] == label]
            sub_clusters.append(set(sub_vars))
        
        # 检查分裂后的簇内相似度
        valid_split = True
        for sub_cluster in sub_clusters:
            if len(sub_cluster) > 1:
                min_sim = compute_intra_cluster_similarities([sub_cluster], sparse_matrix)[0]
                if min_sim < theta:
                    valid_split = False
                    break
        
        if valid_split:
            return sub_clusters
    
    # 如果无法有效分裂，返回原始簇
    return [cluster]

def ensure_target_separate( groups, target_idx):
    """确保目标变量单独一组（根据你的研究设计）"""
    # 找到包含目标变量的组
    target_group_idx = -1
    for i, group in enumerate(groups):
        if target_idx in group:
            target_group_idx = i
            break
    
    if target_group_idx >= 0:
        target_group = groups[target_group_idx]
        if len(target_group) > 1:
            # 将目标变量单独分出
            new_groups = []
            for i, group in enumerate(groups):
                if i == target_group_idx:
                    # 保留目标变量，其他变量形成新组
                    other_vars = [var for var in group if var != target_idx]
                    if other_vars:  # 如果还有其他变量
                        new_groups.append(other_vars)
                    new_groups.append([target_idx])  # 目标变量单独一组
                else:
                    new_groups.append(group)
            return new_groups
    
    return groups

