import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.metrics import mutual_info_score
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
import torch

class Exp_forecasting(object):
    def __init__(self, args, setting, folder_path, plot_dir):
        self.args = args  
        self.setting = setting
        # ... 其他初始化代码保持不变 ...
        
        # 模型初始化
        self._init_models()

    def _get_groups(self, data_set, target_idx):
        """
        基于皮尔逊相关系数和互信息的变量自适应分组算法
        返回分组列表，如 [[0,1,2], [3,4], [5]]
        """
        print("=== 开始变量自适应分组 ===")
        
        # 获取数据（不包括时间戳列）
        data = data_set.data_x  # [样本数, 变量数]
        n_samples, n_vars = data.shape
        print(f"数据形状: {data.shape}, 目标变量索引: {target_idx}")
        
        # 1. 下采样平滑（如果数据量太大）
        downsampled_data = self._downsample_data(data)
        T_prime, n_vars_down = downsampled_data.shape
        print(f"下采样后数据形状: {downsampled_data.shape}")
        
        # 2. 计算综合相似度矩阵
        similarity_matrix = self._compute_similarity_matrix(downsampled_data, target_idx)
        
        # 3. 稀疏化处理
        sparse_matrix = self._sparsify_similarity_matrix(similarity_matrix)
        
        # 4. 层次化聚类
        groups = self._hierarchical_clustering(sparse_matrix, n_vars, target_idx)
        
        print(f"最终分组结果: {groups}")
        print("=== 变量自适应分组完成 ===")
        return groups



    def _downsample_data(self, data, max_samples=5000):
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

    def _compute_similarity_matrix(self, data, target_idx):
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
                    mi = self._compute_mutual_info(data[:, i], data[:, j])
                    mi_matrix[i, j] = mi
                    mi_matrix[j, i] = mi
        
        # 归一化互信息矩阵
        mi_max = np.max(mi_matrix)
        if mi_max > 0:
            mi_matrix_normalized = mi_matrix / mi_max
        else:
            mi_matrix_normalized = mi_matrix
        
        # 综合相似度矩阵（α=0.5，可调整）
        alpha = getattr(self.args, 'similarity_alpha', 0.5)
        similarity_matrix = (
            alpha * (pearson_matrix + 1) / 2 +  # 皮尔逊从[-1,1]映射到[0,1]
            (1 - alpha) * mi_matrix_normalized
        )
        
        print(f"皮尔逊矩阵范围: [{np.min(pearson_matrix):.3f}, {np.max(pearson_matrix):.3f}]")
        print(f"互信息矩阵范围: [{np.min(mi_matrix):.3f}, {np.max(mi_matrix):.3f}]")
        print(f"综合相似度矩阵范围: [{np.min(similarity_matrix):.3f}, {np.max(similarity_matrix):.3f}]")
        
        return similarity_matrix

    def _compute_mutual_info(self, x, y, bins=20):
        """
        计算两个连续变量的互信息
        """
        # 离散化连续变量
        x_discrete = np.digitize(x, np.histogram_bin_edges(x, bins=bins))
        y_discrete = np.digitize(y, np.histogram_bin_edges(y, bins=bins))
        
        # 计算互信息
        mi = mutual_info_score(x_discrete, y_discrete)
        return mi

    def _sparsify_similarity_matrix(self, similarity_matrix):
        """
        稀疏化相似度矩阵
        """
        # 计算中位数阈值
        eta = np.median(similarity_matrix)
        print(f"相似度阈值η(中位数): {eta:.3f}")
        
        # 稀疏化处理
        sparse_matrix = np.where(similarity_matrix > eta, similarity_matrix, 0)
        
        # 对角线置为0（避免自相关影响聚类）
        np.fill_diagonal(sparse_matrix, 0)
        
        return sparse_matrix

    def _hierarchical_clustering(self, sparse_matrix, n_vars, target_idx):
        """
        层次化聚类分组
        """
        # 初始化：每个变量一个簇
        clusters = [{i} for i in range(n_vars)]
        
        # 聚类参数（可从args获取或使用默认值）
        theta = getattr(self.args, 'cluster_theta', 0.6)    # 簇内最小相似度阈值
        gamma = getattr(self.args, 'cluster_gamma', 0.3)    # 簇间最大相似度阈值  
        kappa = getattr(self.args, 'cluster_kappa', 5)      # 最大簇大小
        delta = getattr(self.args, 'cluster_delta', 0.05)   # 动态调整步长
        max_iterations = getattr(self.args, 'cluster_max_iter', 100)
        
        print(f"聚类参数: θ={theta}, γ={gamma}, κ={kappa}, δ={delta}")
        
        iteration = 0
        while iteration < max_iterations:
            iteration += 1
            
            # 计算当前聚类状态
            intra_similarities = self._compute_intra_cluster_similarities(clusters, sparse_matrix)
            inter_similarities = self._compute_inter_cluster_similarities(clusters, sparse_matrix)
            
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
                    sim = self._cluster_similarity(clusters[i], clusters[j], sparse_matrix)
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
            clusters = self._split_large_clusters(clusters, sparse_matrix, kappa, theta)
            
            # 动态调整阈值（如果迭代次数较多）
            if iteration % 10 == 0:
                theta = max(0.3, theta - delta)  # 逐步降低θ
                gamma = min(0.7, gamma + delta)  # 逐步提高γ
                print(f"动态调整阈值: θ={theta:.3f}, γ={gamma:.3f}")
        
        # 转换为分组列表
        groups = [list(cluster) for cluster in clusters]
        
        # 确保目标变量单独一组（根据你的研究设计）
        final_groups = self._ensure_target_separate(groups, target_idx)
        
        return final_groups

    def _compute_intra_cluster_similarities(self, clusters, sparse_matrix):
        """计算每个簇内的最小相似度"""
        intra_sims = []
        for cluster in clusters:
            if len(cluster) <= 1:
                intra_sims.append(1.0)  # 单变量簇相似度为1
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

    def _compute_inter_cluster_similarities(self, clusters, sparse_matrix):
        """计算所有簇间的最大相似度"""
        inter_sims = []
        for i in range(len(clusters)):
            for j in range(i + 1, len(clusters)):
                sim = self._cluster_similarity(clusters[i], clusters[j], sparse_matrix)
                inter_sims.append(sim)
        return inter_sims

    def _cluster_similarity(self, cluster1, cluster2, sparse_matrix):
        """计算两个簇间的相似度（最大相关性得分）"""
        max_sim = 0
        for var1 in cluster1:
            for var2 in cluster2:
                sim = sparse_matrix[var1, var2]
                if sim > max_sim:
                    max_sim = sim
        return max_sim

    def _split_large_clusters(self, clusters, sparse_matrix, kappa, theta):
        """分裂过大的簇"""
        new_clusters = []
        for cluster in clusters:
            if len(cluster) > kappa:
                # 对过大簇进行二次聚类
                sub_clusters = self._split_cluster(cluster, sparse_matrix, theta)
                new_clusters.extend(sub_clusters)
            else:
                new_clusters.append(cluster)
        return new_clusters

    def _split_cluster(self, cluster, sparse_matrix, theta):
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
                    min_sim = self._compute_intra_cluster_similarities([sub_cluster], sparse_matrix)[0]
                    if min_sim < theta:
                        valid_split = False
                        break
            
            if valid_split:
                return sub_clusters
        
        # 如果无法有效分裂，返回原始簇
        return [cluster]

    def _ensure_target_separate(self, groups, target_idx):
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