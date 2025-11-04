# coding = utf-8
import argparse
from exp_forecasting import Exp_forecasting
import os

def main(args):

    if args.model_path is not None:   # model_path: 已保存模型第二阶段的微调继续训练
      setting = (args.model_path).split('/')[-1]  # 使用split('/')分割字符串，然后取最后一个元素
    
    else:
      if args.compare is not None: # baseline对比模型实验或者消融实验
        setting = f"{args.compare}_{os.path.splitext(args.data_path.split('/')[-1])[0]}_seq_{args.seq_len}_pred_{args.pred_len}_stride_{args.stride}_cluster_{args.numClusters}_{args.anchor}_{args.pretrain_mode}"
      else:
        setting = f"ELFNet_{os.path.splitext(args.data_path.split('/')[-1])[0]}_seq_{args.seq_len}_pred_{args.pred_len}_stride_{args.stride}_cluster_{args.numClusters}_{args.anchor}_{args.pretrain_mode}"
    
    exp = Exp_forecasting(args,setting)

    print(f"Training {setting}")
    if args.compare is None or args.compare == 'ELFNet_depthwise'or args.compare == 'ELFNet_no_disentanglement' or args.compare == 'ELFNet_no_contrastive' or args.compare == 'ELFNet_dilution':
      training_time_stage1, training_time_stage2, total_training_time = exp._train_ELFNet() #ELFNet或4个消融模型
    else:
      total_training_time = exp._train_compare()

    print(f"Testing {setting}")
    exp.test(setting, test=1)

    # 直接保存在根目录下
    result_folder = "./"
    # 构建保存时间的文件路径
    file_path = os.path.join(result_folder, 'traning_time.txt')
    # 以追加模式打开文件并写入训练时间
    with open(file_path, 'a') as file:
        if args.compare is None or args.compare == 'ELFNet_depthwise'or args.compare == 'ELFNet_no_disentanglement' or args.compare == 'ELFNet_no_contrastive' or args.compare == 'ELFNet_dilution':
          file.write(f"{setting}: {training_time_stage1:.2f}, {training_time_stage2:.2f}, {total_training_time:.2f}\n")
        else:
          file.write(f"{setting}: {total_training_time:.2f}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run time series prediction experiments with causal discovery')
    
    # 数据集设置
    ## 预训练模式选择
    parser.add_argument('--pretrain_mode', type=str, default='one2many', choices=['single', 'one2many'],help='预训练模式: single-单数据集任务, one2many-一对多跨数据集任务')
    ## one2many模式下的预训练数据集路径，此模式指 “单数据集预训练+其它数据集独立微调与测试”
    #parser.add_argument('--pretrain_data_path', type=str, default=None,help='预训练数据集路径，用于one2many模式')
    parser.add_argument('--pretrain_data_path', type=str, default='datasets_copy/Mathematical_Modeling_Competition.csv',help='预训练数据集路径，用于one2many模式')
    ## data_path : single 模式下的数据集路径
    ### datasets/Mathematical_Modeling_Competition.csv
    ### datasets/Australia_Load&Price.csv
    ### datasets/XJ_Photovoltaic.csv
    parser.add_argument('--data_path', type=str, default='datasets/Australia_Load&Price.csv', help='single 模式下的唯一数据集路径')
    parser.add_argument('--root_path', type=str,default='./', help='Root path to the dataset')

    # 数据预处理相关
    parser.add_argument('--freq', type=str, default='d',choices=['t','h','d'], help='Frequency for time features(可选：[分钟t,小时h,天d])')
    parser.add_argument('--scale', type=str, default='True', help='Whether to perform data standardization')
    parser.add_argument('--num_augment', type=int, default=4, help='The number of augmented data samples')
    parser.add_argument('--cluster', type=str, default='False', help='Whether to use clustering to decrease samples')
    parser.add_argument('--numClusters', type=int, default=None, help='Number of clusters to use for clustering')
    parser.add_argument('--anchor', type=str, default=None, help='Center for clustering, optional choices: [random, center]')
    
    # basic config
    parser.add_argument('--alpha', type=float, default=0.05, help='Weighting hyperparameter for loss function')
    parser.add_argument('--plot', action='store_true', help='Whether to plot ')
    parser.add_argument('--compare', type=str, default=None, help='Compare Model to use (e.g., TimesNet)') 
    parser.add_argument('--log_interval',type = int, default=5, help='Log interval for training')
    
    # forecasting task
    parser.add_argument('--seq_len', type=int, default=96, help='Sequence length')
    parser.add_argument('--pred_len', type=int, default=48, help='Prediction length')
    parser.add_argument('--stride', type=int, default=1, help='Stride for sliding window')
    parser.add_argument('--target_idx', type=int, default=5, help='Target feature')
    
    # train settiings and optimization
    parser.add_argument('--temperature', type=float, default=0.7, help='temperature scaling factor')
    parser.add_argument('--optimizername', type=str, default='Adam', help='The type of optimizer')
    parser.add_argument('--train_epochs1', type=int, default=2, help='Number of epochs for pretrain ELFNet using contrasitive learning')
    parser.add_argument('--train_epochs2', type=int, default=2, help='Number of epochs for pretrain ELFNet using supervised learning')
    parser.add_argument('--epochs', type=int, default=4, help='Number of total epochs for training compare model')
    parser.add_argument('--lr', type=float, default=0.01, help='Learning rate')
    parser.add_argument('--patience', type=int, default=10, help='early stopping patience')
    parser.add_argument('--threshold', type=float, default=0.05, help='stable early stopping threshold')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training')
    parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate, options: [type1, type2, cosine]')
    #parser.add_argument('--model_path', type=str, default='./test_results/ELFNet_Mathematical_Modeling_Competition_seq_96_pred_48_stride_1_cluster_None_None', help='Saved pretrained model path for further finetune')
    parser.add_argument('--model_path', type=str, default=None, help='Saved pretrained model path for further finetune')
    
    # GPU
    parser.add_argument('--num_workers', type=int, default=0, help='Number of workers for data loading')
    parser.add_argument('--use_gpu', type=bool, default=False, help='Whether to use GPU')
    parser.add_argument('--gpu', type=int, default=0, help='GPU device id')
    parser.add_argument('--use_multi_gpu', type=bool, default=False, help='Whether to use multiple GPUs')
    parser.add_argument('--devices', type=str, default='cpu', help='Device ids for multiple GPUs')
    
    # model define
    parser.add_argument('--depth', type=int, default=8, help='Number of hidden layers in the FeatureExtractor')
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
    parser.add_argument('--kernels', type=int, nargs='+', default=[1, 2, 4, 8, 16, 32, 64, 128], help='The kernel sizes used in the mixture of AR expert layers')
    parser.add_argument('--kernel_size', type=int, default=4, help='The kernel sizes used in the FeatureExtractor and the FeatureReducer')
    parser.add_argument('--hidden_dims', type=int,  default=64, help='The hidden layers dimensions used in the feature extractor')
    parser.add_argument('--reduce_hidden_dims', type=int, nargs='+', default=[128 ,64,32], help='The hidden layers dimensions used in the FeatureReducer')
    parser.add_argument('--repr_dims', type=int, default=320, help='The representation dimension ')
    parser.add_argument('--c_out', type=int, default=1, help='final output size')
    
    parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
    parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
    parser.add_argument('--d_ff', type=int, default=2048, help='dimension of fcn')
    parser.add_argument('--distil', action='store_false',
                        help='whether to use distilling in encoder, using this argument means not using distilling',
                        default=True)
    parser.add_argument('--factor', type=int, default=1, help='attn factor')
    parser.add_argument('--output_attention', action='store_true', help='whether to output attention in ecoder')
    parser.add_argument('--n_heads', type=int, default=8, help='The number of heads in the attention layers')
    parser.add_argument('--activation', type=str, default='gelu', help='activation')
    parser.add_argument('--num_kernels', type=int, default=6, help='for Inception')
    parser.add_argument('--enc_in', type=int, default=6, help='encoder input size')
    parser.add_argument('--dec_in', type=int, default=1, help='decoder input size')
    parser.add_argument('--embed', type=str, default='timeF',
                        help='time features encoding, options:[timeF, fixed, learned]')
    parser.add_argument('--dilation_c', type=int, default=4, help='Dilation coefficient in the ADDSTCN and MixedChannelConvEncoder, recommended to be equal to kernel size (default: 4)')
    parser.add_argument('--seg_len', type=int, default=12,
                        help='the length of segmen-wise iteration of SegRNN')
    parser.add_argument('--moving_avg', type=int, default=25, help='window size of moving average')
 
    
    
    parser.add_argument('--seed', type=int, default=2, help="Randomization seed")
    parser.add_argument('--inverse', action='store_true', help='inverse output data', default=False)

    # MixedChannelConv Machanism
    parser.add_argument("--groups", type=str, default='[[0,1,2],[3],[4],[5]]', help='Comma-separated list of lists defining column groups')  # [[0,1,2],[3],[4],[5]]  or [[0,1],[2],[3],[4,5,6],[7]]
                        
    
    args = parser.parse_args()
    main(args)
