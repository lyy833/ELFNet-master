from torch.utils.data import DataLoader
from data_process.custom_dataset import CustomDataset
from torch.utils.data.dataloader import default_collate
import math 
import numpy as np
import time
# 选择每个簇的最中心点
def find_central_point(cluster_indices, features):
    # 计算簇内每个样本到其他样本的距离之和
    distances_sum = []
    for index in cluster_indices:
        distance_sum = np.sum(np.linalg.norm(features[index] - features[cluster_indices], axis=1))
        distances_sum.append(distance_sum)
    
    # 找到距离之和最小的样本索引
    central_index = cluster_indices[np.argmin(distances_sum)]
    return central_index


def cluster_and_create_loader(features, data_set, batch_size,numClusters,anchor='random'):
    from sklearn.cluster import KMeans
    from torch.utils.data import DataLoader, Dataset
    """
    使用KMeans聚类生成新的数据加载器。
    :param features: 特征数组
    :param data_set: 原始数据集
    :param batch_size: 每个批次的大小
    :return: 新的数据加载器
    """
    num_clusters = min(numClusters, len(features))  # 聚类的数量可以调整
    
    t = time.time()
    print("===============Starting K-Means Clustering...=============")
    kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(features)
    
    # 创建聚类标签
    cluster_labels = kmeans.labels_
    
    # 选择每个簇的代表样本
    unique_labels = np.unique(cluster_labels)
    ##随机选择：简单快速，但可能不是最优选择。
    if anchor == 'random':
        selected_indices = [np.random.choice(np.where(cluster_labels == label)[0]) for label in unique_labels]
    elif anchor == 'center': ##中心点选择：更准确，但可能耗时较长。
        selected_indices = [find_central_point(np.where(cluster_labels == label)[0], features) for label in unique_labels]
    else:
        raise ValueError("Invalid anchor type. Choose 'random' or 'center'.")
    
    print(f"Total Clustering Time: {time.time() - t:.2f}s")
    
    # 创建新的数据集和数据加载器
    # 创建新的数据集和数据加载器
    class NewCustomDataset(Dataset):
        def __init__(self, data_set, selected_indices):
            self.data_set = data_set
            self.selected_indices = selected_indices
            self.data_x = data_set.data_x
            self.targetidx = data_set.targetidx
        def __getitem__(self, index):
            original_index = self.selected_indices[index]
            return self.data_set.__getitem__(original_index)

        def __len__(self):
            return len(self.selected_indices)

        def get_max_iterations(self, batch_size):
            return math.ceil(len(self) / batch_size)

    new_data_set = NewCustomDataset(data_set, selected_indices)
    new_loader = DataLoader(new_data_set, batch_size=batch_size, shuffle=True)

    return new_data_set, new_loader


def data_provider(args, flag, cluster_data=False, pretrain_stage=False):
    """
    根据提供的参数（args）和标志（flag），返回数据集和数据加载器。
    支持 single 和 one2many 两种预训练模式

    :param args: 参数
    :param flag: 数据标志 'train', 'val', 'test'
    :plot_dir: 数据增强可视化结果路径
    :param cluster_data: 是否基于聚类进行样本选择
    :param pretrain_stage: 是否为预训练阶段
    """
    shuffle_flag = False if flag == 'test' else True
    drop_last = True
    batch_size = args.batch_size
    
    # 确定数据路径
    if pretrain_stage and args.pretrain_mode == 'one2many':
        # one2many预训练：使用预训练数据集
        data_path = args.pretrain_data_path
    else:
        # single模式或 One-to-Many微调阶段：使用目标数据集
        data_path = args.data_path # 注意这个设置不影响 many2one模式预训练数据集list
    
    # 统一使用CustomDataset，传入确定的数据路径
    data_set = CustomDataset(
        args=args,
        flag=flag,
        data_path=data_path , # 关键：动态传入数据路径
        pretrain_stage=pretrain_stage
    )
    
    # 打印详细信息用于调试
    stage_info = 'pretrain' if pretrain_stage else 'finetune'
    print(f"{flag} | mode: {args.pretrain_mode} | stage: {stage_info} | dataset: {data_path} | length: {len(data_set)}")

    # 训练数据加载且需要样本聚类（尽在预训练阶段考虑使用样本聚类）
    if flag == 'train' and cluster_data:
        original_train_loader = DataLoader(
            data_set,
            batch_size=1, # 注意这里由于需要聚类先把 batch size 设置为 1
            shuffle=False,
            num_workers=args.num_workers,
            drop_last=drop_last
        )
        features = data_set.extract_features()
        new_data_set, new_train_loader = cluster_and_create_loader(features, data_set, batch_size, args.numClusters, args.anchor)
        return new_data_set, new_train_loader
    
    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        num_workers=args.num_workers,
        drop_last=drop_last,
        collate_fn=default_collate
    )
    
    return data_set, data_loader