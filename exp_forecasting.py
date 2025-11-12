import os
import time
import numpy as np
import torch
import torch.nn as nn
from utils.tools import IterationEarlyStopping, adjust_learning_rate,visual
from utils.metrics import metric
from models.ELFNet import ELFNet
import ast
import torch.optim as optim
from models.TimesNet import  TimesNet
from models.ADDSTCN import ADDSTCN
from models.Informer import Informer
from models.SegRNN import SegRNN
from models.DLinear import DLinear
from models.ELFNet_ablation import *
from models.ELFNet import  ELFNet
from data_process.data_provider import data_provider
import warnings
from utils.variableGrouping import *

warnings.filterwarnings('ignore')

class Exp_forecasting(object):
    def __init__(self, args, setting,folder_path,plot_dir):
        
        self.args = args  
        self.setting = setting
        # Initialize data based on pretraining mode
        self._init_data()
        
        self.criterion = self._select_criterion()
        self.early_stopping = IterationEarlyStopping(patience_epochs=args.patience_epochs,patience_iterations=args.patience_iterations,min_iterations=args.min_iterations,verbose=True,delta=args.improved_delta)  # Set early stopping mechanism
        self.device = self._acquire_device()
        
        # Define and create the path to save the results of this task
        self.folder_path = folder_path
        self.plot_dir = plot_dir

        # Model initialization
        self._init_models()

    
    def _init_data(self):
        """数据初始化，变量自适应分组在此阶段完成"""
        # 聚类样本，只考虑在数据规模较大、计算更复杂的自监督预训练阶段使用该策略（如果需要）
        if self.args.cluster == 'True':
            self.pretrain_data, self.pretrain_loader = self._get_data(flag='train', cluster_data=True, pretrain_stage=True)
        else:
            self.pretrain_data, self.pretrain_loader = self._get_data(flag='train', pretrain_stage=True)
        self.pretrain_groups = ast.literal_eval(self.args.pretrain_groups) if hasattr(self.args, 'pretrain_groups') and self.args.pretrain_groups else self._get_groups(self.pretrain_data,self.args.pretrain_target_idx)
        self.finetune_data, self.finetune_loader = self._get_data(flag='train', pretrain_stage=False)
        #self.finetune_input_channel = self.finetune_data.data_x.shape[1] 
        self.finetune_groups = ast.literal_eval(self.args.finetune_groups) if hasattr(self.args, 'finetune_groups') and self.args.finetune_groups else self._get_groups(self.finetune_data,self.args.finetune_target_idx)
        print(f"预训练数据集变量分组(下标): {self.pretrain_groups}")
        print(f"微调数据集变量分组(下标): {self.finetune_groups}")

    def _init_models(self):
        """Initialize models based on model used"""   
        if self.args.model_used in ['ELFNet', 'ELFNet_depthwise', 'ELFNet_no_disentanglement', 'ELFNet_no_contrastive', 'ELFNet_dilution']:
            self._init_ELFNet_family()
        else:
            self._init_baseline_model()


    def _init_ELFNet_family(self):
        """Initialization ELFNet family model """
        if self.args.model_used =='ELFNet' : # Complete version of ELFNet
            print('Standard VG-HCS ELFNet...')
            if self.args.pretrained_model_path is None:
                self.model = ELFNet(self.args, device=self.device, stage2=False).to(self.device)
            else:
                self.model = ELFNet(self.args, device=self.device, stage2=True).to(self.device)
        
        # 除ELFNet_no_contrastive以外的消融模型具备跨数据集实验能力
        elif self.args.model_used == 'ELFNet_depthwise':
            print('ELFNet without VG-HCS in ablation experiment (CI)...')
            self.model = ELFNet_depthwise(self.args,device=self.device).to(self.device)
        elif self.args.model_used == 'ELFNet_dilution':
            print('ELFNet without VG-HCS in ablation experiment (CD)...')
            self.model = ELFNet_Dilation(self.args,self.pretrain_input_channels, device=self.device).to(self.device)
        elif self.args.model_used == 'ELFNet_no_disentanglement':
            print('ELFNet without disentanglement in ablation experiment...')
            self.model = ELFNet_no_disentanglement(self.args,self.pretrain_input_channels, device=self.device).to(self.device)
        # ELFNet_no_contrastive为单阶段训练，仅仅支持 single 模式
        elif self.args.model_used == 'ELFNet_no_contrastive':
            print('ELFNet without contrastive learning in ablation experiment...')
            self.model = ELFNet_no_contrastive(self.args, device=self.device).to(self.device)
        else:
            print('Please input the correct model name')
        
        if self.args.use_multi_gpu and self.args.use_gpu:
            self.model = nn.DataParallel(self.model, device_ids=self.args.device_ids)
        
        self.optimizer = getattr(optim, self.args.optimizername)(self.model.parameters(), lr=self.args.lr)
        
        
    def _init_baseline_model(self):
        """Initialize baseline model"""
        CompareModel_dict = {
            'TimesNet': TimesNet,
            'ADDSTCN': ADDSTCN,
            'Informer': Informer,
            'SegRNN': SegRNN,
            'DLinear': DLinear
        }
        
        self.CompareModel = CompareModel_dict[self.args.model_used](self.args).float().to(self.device)

        if self.args.use_multi_gpu and self.args.use_gpu:
            self.CompareModel = nn.DataParallel(self.CompareModel, device_ids=self.args.device_ids)
        
        self.optimizer = getattr(optim, self.args.optimizername)(self.CompareModel.parameters(), lr=self.args.lr)


    def _acquire_device(self):
        if self.args.use_gpu:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(self.args.gpu) if not self.args.use_multi_gpu else self.args.devices
            device = torch.device('cuda:{}'.format(self.args.gpu))
            print('Use GPU: cuda:{}'.format(self.args.gpu))
        else:
            device = torch.device('cpu')
            print('Use CPU')
        return device

    def _get_data(self, flag,cluster_data=False,pretrain_stage=False):
        data_set, data_loader = data_provider(self.args, flag,cluster_data,pretrain_stage)
        return data_set, data_loader

    def _select_criterion(self):
        criterion = torch.nn.MSELoss()
        return criterion


    def _train_compare(self):
        epochs = self.args.epochs
        optimizer = self.optimizer
        early_stopping = self.early_stopping
        
        model_path = os.path.join(self.folder_path, 'trained_compare_model')
        if not os.path.exists(model_path):
                os.makedirs(model_path)
        trained_model_path = os.path.join(model_path,f"{self.args.model_used}.pth")
        
        t = time.time()
        print('=============Starting to train model_used Model==============')
        self.CompareModel.train()
        losses = []
        global_iteration = 0
        for epoch in range(epochs):
            train_loss = []
            # 是否聚类以减少样本
            if self.args.cluster=='True':
                train_loader=self.cluster_train_loader
            else:
                train_loader=self.train_loader

            for iteration, (batch_x, batch_x_mark, batch_y, batch_y_mark) in enumerate(train_loader):
                optimizer.zero_grad()
                batch_x = batch_x.transpose(1, 2).float().to(self.device)
                batch_y = batch_y.float().unsqueeze(2).to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()

                outputs = self.CompareModel(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                outputs = outputs[:, -self.args.pred_len:, -1:]
                batch_y = batch_y[:, -self.args.pred_len:, -1:].to(self.device)
                
                loss = self.criterion(outputs, batch_y)
                train_loss.append(loss.item())
                loss.backward()
                optimizer.step()
                
                losses.append(loss.item())
                
                global_iteration += 1
                # 更新损失窗口
                early_stopping.update_loss_window(loss.item())
                # 每100个iteration检查一次早停
                if global_iteration % 100 == 0:
                    # 计算最近100个iteration的平均损失
                    recent_avg_loss = np.mean(train_loss[-100:]) if len(train_loss) >= 100 else np.mean(train_loss)
                    # iteration级别早停判断
                    early_stopping(recent_avg_loss, self.CompareModel, trained_model_path,is_iteration=True, current_iteration=global_iteration)
                    if early_stopping.is_loss_stable(threshold=0.001): # 额外检查损失是否稳定
                        print(f"损失已趋于稳定，考虑早停")
                        early_stopping.early_stop = True
                    if early_stopping.early_stop:
                        print("iteration级别训练早停")
                        break
                             
                if (iteration+1) % self.args.log_interval==0:
                    print(f"Iter: {iteration+1}, Train Loss: {loss:.7f}")
           
            train_loss = np.average(train_loss)
            print(f"Epoch: {epoch+1}, Train Loss: {train_loss:.7f}")

            vali_data, vali_loader = self._get_data(flag='val')
            vali_loss = self.vali(vali_loader)
            print(f"Vali Loss: {vali_loss:.7f}")
            
            print(f"Training Time  until now: {time.time() - t:.2f}s")

            early_stopping(vali_loss, self.CompareModel, trained_model_path,is_iteration=False,current_iteration=global_iteration)
            
            if self.args.plot_loss: # 可视化训练损失
                self._plot_losses(losses, f"{self.args.model_used}_compare")
            
            if early_stopping.early_stop :
                break

            adjust_learning_rate(optimizer, epoch + 1, self.args)
        
        total_training_time = time.time() - t
        print(f"Total Training Time: {total_training_time:.2f}s, 总迭代次数: {global_iteration}")

        return total_training_time,trained_model_path
    

    def _train_ELFNet_family(self):  
        """ ELFNet及其消融模型 训练流程，支持one2many 和 single 两种模式"""
        print(f"Pretrain mode:{self.args.pretrain_mode}")
        if self.args.model_used == 'ELFNet_no_contrastive' or self.args.pretrained_model_path is not None:
            training_time_stage1 = 0
            training_time_stage2,model_path = self.train_stage2()  
        else:
            training_time_stage1 = self.train_stage1()
            training_time_stage2,model_path = self.train_stage2()
        
        total_training_time = training_time_stage1 + training_time_stage2   
        return training_time_stage1, training_time_stage2, total_training_time,model_path

    def train_stage1(self):
        epochs = self.args.train_epochs1
        optimizer = self.optimizer
        early_stopping = self.early_stopping
        model_path = os.path.join(self.folder_path, 'pretrained_ELFNet_family')
        if not os.path.exists(model_path):
                os.makedirs(model_path)
        pretrained_model_path = os.path.join(model_path,f"{self.args.model_used}.pth")
        if self.args.pretrained_model_path == None: # 用于第二阶段训练加载这个模型
            self.args.pretrained_model_path = pretrained_model_path
        # 初始化输入投影层
        self.model._init_input_projections(self.pretrain_data.data_x.shape[1], self.args.hidden_dims)

        t = time.time()
        print(f"=======Starting to train {self.args.model_used}: stage1=======")
        self.model.train()
        losses = []

        global_iteration = 0
        plot_augment_flag = True # 用于仅仅在第一次前向传播时可视化增强数据
        for epoch in range(epochs):
            train_loss = []
            for iteration, (batch_x, _,batch_y,_) in enumerate(self.pretrain_loader): # batch_x: tensor (b, seq_len, c)   batch_x: (b,pred_len,1)
                optimizer.zero_grad()
                loss = self.model.compute_loss(batch_x.to(self.device).transpose(1,2), batch_y.to(self.device),self.plot_dir,self.pretrain_groups,plot_augment_flag)
                if plot_augment_flag:
                    plot_augment_flag = False
                train_loss.append(loss.item())
                loss.backward()
                optimizer.step()
                losses.append(loss.item())
                global_iteration += 1
                # 更新损失窗口
                early_stopping.update_loss_window(loss.item())
                # 每100个iteration检查一次早停
                if global_iteration % 100 == 0:
                    # 计算最近100个iteration的平均损失
                    recent_avg_loss = np.mean(train_loss[-100:]) if len(train_loss) >= 100 else np.mean(train_loss)
                    # iteration级别早停判断
                    early_stopping(recent_avg_loss, self.model, pretrained_model_path, is_iteration=True, current_iteration=global_iteration)
                    if early_stopping.is_loss_stable(threshold=0.001): # 额外检查损失是否稳定
                        print(f"损失已趋于稳定，考虑早停")
                        early_stopping.early_stop = True
                    if early_stopping.early_stop:
                        print("iteration级别训练早停")
                        break # 跳出当前iteration
                if (iteration+1) % self.args.log_interval==0:
                    print(f"Iter: {iteration+1}, Train Loss in Stage1 : {loss:.7f}")
                
            
            train_loss = np.average(train_loss)
            print(f"Epoch: {epoch+1}, Train Loss in stage1: {train_loss:.7f}")

            if self.args.plot_loss:
                self._plot_losses(losses, "stage1")
            
            print(f"Training time in stage1 until now: {time.time() - t:.2f}s")
           
            if early_stopping.early_stop : 
                break

            early_stopping(train_loss, self.model, pretrained_model_path,is_iteration=False,current_iteration=global_iteration)
            if early_stopping.early_stop:
                print("epoch级别训练早停")
                break

            adjust_learning_rate(optimizer, epoch + 1, self.args)
        
        training_time = time.time() - t
        print(f"Total training Time in Stage1: {training_time:.2f}s")
        
        return training_time
    
    def train_stage2(self):
        epochs = self.args.train_epochs2
        optimizer = self.optimizer
        model_path = os.path.join(self.folder_path, f"finetuned_ELFNet_family/{self.args.model_used}")
        finetuned_model_path = os.path.join(model_path,f"{os.path.splitext(self.args.data_path.split('/')[-1])[0]}.pth")
        if not os.path.exists(model_path):
                os.makedirs(model_path)
        self.early_stopping = IterationEarlyStopping(patience_epochs=self.args.patience_epochs,patience_iterations=self.args.patience_iterations,min_iterations=self.args.min_iterations,verbose=True,delta=self.args.improved_delta)
        early_stopping = self.early_stopping

        
        self.model._init_input_projections(self.finetune_data.data_x.shape[1],self.args.hidden_dims)
        
        t = time.time()
        
        print('加载预训练模型进行权重迁移以进一步微调')
        pretrained_state_dict = torch.load(self.args.pretrained_model_path)

        #self.model.load_state_dict(torch.load(self.args.pretrained_model_path ))
    
        print(f"====Starting to train {self.args.model_used}: stage2====")
        self.model.train() 

        self.model.stage2 = True
        
        losses = []
        
        global_iteration = 0
        for epoch in range(epochs):
            train_loss = []
            for iteration, (batch_x, _,batch_y,_) in enumerate(self.finetune_loader): 
                optimizer.zero_grad()

                #batch_x = batch_x.transpose(1, 2)
                batch_x = batch_x.float().to(self.device)
                batch_y = (batch_y.float().to(self.device))
                outputs = self.model(batch_x,self.finetune_groups,pretrained_state_dict)
                
                loss = self.criterion(outputs, batch_y)
                train_loss.append(loss.item())
                loss.backward()
                optimizer.step()
                losses.append(loss.item())

                global_iteration += 1
                # 更新损失窗口
                early_stopping.update_loss_window(loss.item())
                # 每100个iteration检查一次早停
                if global_iteration % 100 == 0:
                    # 计算最近100个iteration的平均损失
                    recent_avg_loss = np.mean(train_loss[-100:]) if len(train_loss) >= 100 else np.mean(train_loss)
                    # iteration级别早停判断
                    early_stopping(recent_avg_loss, self.model,finetuned_model_path, is_iteration=True, current_iteration=global_iteration)
                    if early_stopping.is_loss_stable(threshold=0.001): # 额外检查损失是否稳定
                        print(f"损失已趋于稳定，考虑早停")
                        early_stopping.early_stop = True
                    if early_stopping.early_stop:
                        print("iteration级别训练早停")
                        break

                if (iteration+1) % self.args.log_interval==0:
                    print(f"Iter: {iteration+1}, Train Loss in Stage2: {loss:.7f}")
            
            if self.args.plot_loss:    
                self._plot_losses(losses,  f"{os.path.splitext(self.args.data_path.split('/')[-1])[0]}_stage2")

            train_loss = np.average(train_loss)
            print(f"Epoch: {epoch+1}, Train Loss in Stage2: {train_loss:.7f}")

            vali_data, vali_loader = self._get_data(flag='val')
            vali_loss = self.vali(vali_loader)
            print(f"Vali Loss: {vali_loss:.7f}")
            print(f"Training Time in Stage2 until now: {time.time() - t:.2f}s")
            
            early_stopping(vali_loss, self.model, finetuned_model_path,is_iteration=False,current_iteration=global_iteration)
            if early_stopping.early_stop :
                break

            adjust_learning_rate(optimizer, epoch + 1, self.args)
        
        training_time = time.time() - t
        print(f"Total training time in stage2: {training_time:.2f}s")
        
        return training_time,finetuned_model_path
        
        
    def vali(self, vali_loader):
        total_loss = []
        with torch.no_grad():
            for batch_x, batch_x_mark,batch_y,batch_y_mark in vali_loader:
                batch_x = (batch_x.float().to(self.device)) # (b,seq_len,c)
                batch_y = batch_y.float().to(self.device)
                batch_y = batch_y[:, -self.args.pred_len:, :].to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                if self.args.model_used in ['ELFNet','ELFNet_depthwise','ELFNet_no_disentanglement','ELFNet_no_contrastive','ELFNet_no_contrastive','ELFNet_dilution'] :
                    self.model.eval()
                    outputs = self.model(batch_x,self.finetune_groups)
                    self.model.train()    
                else:
                    self.CompareModel.eval()
                    # decoder input
                    dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                    #dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                    outputs = self.CompareModel(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                    self.CompareModel.train()
                
                outputs = outputs[:, -self.args.pred_len:, :]
                loss = self.criterion(outputs, batch_y)
                total_loss.append(loss.item())
        total_loss = np.average(total_loss)
        return total_loss


    def test(self,model_path, setting, test):
        test_data, test_loader = self._get_data(flag='test')
        if test:
            print('Loading model')
            if self.args.model_used in ['ELFNet','ELFNet_depthwise','ELFNet_no_disentanglement','ELFNet_no_contrastive','ELFNet_no_contrastive','ELFNet_dilution'] :
                self.model.load_state_dict(torch.load(model_path))
            else:
                self.CompareModel.load_state_dict(torch.load(model_path))
        preds, trues = [], []
        
        # derive folder_path from provided model_path (use model file name as prefix + "_test_results")
        model_dir = os.path.dirname(os.path.abspath(model_path))
        folder_path = os.path.join(model_dir, f"{os.path.splitext(self.args.data_path.split('/')[-1])[0]}_test_visual")
        
        if not os.path.exists(folder_path):
            os.makedirs(folder_path, exist_ok=True)
        with torch.no_grad():
            for i,(batch_x, batch_x_mark,  batch_y,batch_y_mark )in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device) 
                batch_y = batch_y.float().to(self.device) 
                batch_y = batch_y[:, -self.args.pred_len:, :].to(self.device)
                if self.args.model_used in ['ELFNet','ELFNet_depthwise','ELFNet_no_disentanglement','ELFNet_no_contrastive','ELFNet_no_contrastive','ELFNet_dilution'] :
                    self.model.eval()
                    outputs = self.model(batch_x,self.finetune_groups)   
                else:
                    self.CompareModel.eval()
                    batch_x_mark = batch_x_mark.float().to(self.device)
                    batch_y_mark = batch_y_mark.float().to(self.device) 
                    # decoder input
                    dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                    #dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                    
                    outputs = self.CompareModel(batch_x, batch_x_mark, dec_inp,  batch_y_mark)
                outputs = outputs[:, -self.args.pred_len:, :]
                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()

                if test_data.scale and self.args.inverse:
                    shape = outputs.shape
                    outputs = test_data.inverse_transform(outputs.squeeze(0)).reshape(shape)
                    batch_y = test_data.inverse_transform(batch_y.squeeze(0)).reshape(shape)
                
                preds.append(outputs)
                trues.append(batch_y)

                if i % 4 == 0:
                    input = batch_x.detach().cpu().numpy()
                    if test_data.scale and self.args.inverse:
                        shape = input.shape
                        input = test_data.inverse_transform(input.squeeze(0)).reshape(shape)
                    gt = np.concatenate((input[0, :, -1], batch_y[0, :, -1]), axis=0)
                    pd = np.concatenate((input[0, :, -1], outputs[0, :, -1]), axis=0)
                    visual(gt, pd, os.path.join(folder_path, str(i) + '.pdf'))
        
        # 先将list转换为numpy
        np_preds = np.array(preds)
        np_trues =  np.array(trues) 

        preds = np_preds.reshape(-1, np_preds.shape[-2], np_preds.shape[-1])
        trues = np_trues.reshape(-1, np_trues.shape[-2],np_trues.shape[-1])

        mae, mse, rmse, mape, mspe = metric(preds, trues)
        print('mae:{}, mse:{}, rmse:{}, mape:{}, mspe:{}'.format(mae, mse, rmse, mape, mspe))
        f = open("result_forecast.txt", 'a')
        f.write(setting + "  \n")
        f.write('mae:{}, mse:{}, rmse:{}, mape:{}, mspe:{}'.format(mae, mse, rmse, mape, mspe))
        f.write('\n')
        f.write('\n')
        f.close()
        
        return 

    def _plot_losses(self, losses, phase):
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10, 5))
        plt.plot(losses, label='Loss')
        plt.xlabel('Iteration')
        plt.ylabel('Loss')
        plt.title(f'{phase.capitalize()} Loss')
        plt.legend()
        plt.savefig(os.path.join(self.plot_dir, f'{phase}_losses.png'))
        plt.close()

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
        downsampled_data = downsample_data(data)
        print(f"下采样后数据形状: {downsampled_data.shape}")
        
        # 2. 计算综合相似度矩阵
        similarity_matrix = compute_similarity_matrix(downsampled_data, self.args)
        
        # 3. 稀疏化处理
        sparse_matrix = sparsify_similarity_matrix(similarity_matrix)
        
        # 4. 层次化聚类
        groups = hierarchical_clustering(sparse_matrix, n_vars, target_idx,self.args)
        
        print(f"最终分组结果: {groups}")
        print("=== 变量自适应分组完成 ===")
        return groups



