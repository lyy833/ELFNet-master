from torch.utils.data import DataLoader
from data_process.custom_dataset import CustomDataset
from torch.utils.data.dataloader import default_collate


def data_provider(args, flag, pretrain_stage=False):
    """
    根据提供的参数（args）和标志（flag），返回数据集和数据加载器。
    支持 single 和 one2many 两种预训练模式

    :param args: 参数
    :param flag: 数据标志 'train', 'val', 'test'
    :plot_dir: 数据增强可视化结果路径
    :param pretrain_stage: 是否为预训练阶段
    """
    shuffle_flag = False if flag == 'test' else True
    drop_last = True
    batch_size = args.batch_size
    
    # 确定数据路径
    if pretrain_stage and args.training_mode == 'one2many':
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
    print(f"{flag} | mode: {args.training_mode} | stage: {stage_info} | dataset: {data_path} | length: {len(data_set)}")

    
    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        num_workers=args.num_workers,
        drop_last=drop_last,
        collate_fn=default_collate
    )
    
    return data_set, data_loader