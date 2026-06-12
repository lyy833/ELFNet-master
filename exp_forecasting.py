# coding = utf-8
import os
import time
import numpy as np
import torch
from utils.tools import IterationEarlyStopping, adjust_learning_rate,visual
from utils.metrics import metric
from models.ELFNet import ELFNet
import ast
import torch.optim as optim
from models.TimesNet import  TimesNet
from models.ADDSTCN import ADDSTCN
from models.Informer import Informer
from models.SegRNN import SegRNN
from models.TS2Vec import TS2Vec
from models.DLinear import DLinear
from models.ELFNet_ablation import *
from models.ELFNet import  ELFNet,ELFNet_supervised
from models.PatchTST import PatchTST_SS
from models.CoST import CoST
from models.TimeMAE import TimeMAE
from models.PatchTST_supervised import PatchTST_SU
from data_process.data_provider import data_provider
import warnings
from utils.variableGrouping import *

warnings.filterwarnings('ignore')

class Exp_forecasting(object):
    def __init__(self, args, setting,folder_path,plot_dir):
        
        self.args = args  
        self.setting = setting
        
        self._init_data() # Initialize data based on training mode
        
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
        self.pretrain_data, self.pretrain_loader = self._get_data(flag='train', pretrain_stage=True)
        self.pretrain_groups = ast.literal_eval(self.args.pretrain_groups) if hasattr(self.args, 'pretrain_groups') and self.args.pretrain_groups else self._get_groups(self.pretrain_data,self.args.pretrain_target_idx)
        self.finetune_data, self.finetune_loader = self._get_data(flag='train', pretrain_stage=False)
        #self.finetune_input_channel = self.finetune_data.data_x.shape[1] 
        self.finetune_groups = ast.literal_eval(self.args.finetune_groups) if hasattr(self.args, 'finetune_groups') and self.args.finetune_groups else self._get_groups(self.finetune_data,self.args.finetune_target_idx)
        print(f"预训练数据集变量分组(下标): {self.pretrain_groups}")
        print(f"微调数据集变量分组(下标): {self.finetune_groups}")

    def _init_models(self):
        """Initialize models based on model used"""   
        if self.args.model_used in ['ELFNet', 'ELFNet_wo_TS', 'ELFNet_supervised','ELFNet_ablation_augmentor','ELFNet_common_TS','ELFNet_supervised_pretrain','ELFNet_single_band_SRD','ELFNet_wo_CGU']:
            self._init_ELFNet_family()
        else:
            self._init_baseline_model()


    def _init_ELFNet_family(self):
        """Initialization ELFNet family model """
        if self.args.model_used =='ELFNet' : # Complete version of ELFNet
            print('Standard ELFNet (Self-supervised version)...')
            if not self.args.finetune_pretrained_model and not self.args.test_finetuned_model:
                self.model = ELFNet(self.args, device=self.device, stage=1).to(self.device)
                self.model.augmentor.initialize_from_data(self.pretrain_data.data_x,self.pretrain_data.data_y)
            else:
                self.model = ELFNet(self.args, device=self.device, stage=2).to(self.device)
                if self.args.test_finetuned_model:
                    self.model.stage = 3
                    self.model._init_input_projections(self.finetune_data.data_x.shape[1],self.args.hidden_dims)
                    self.model._init_feature_extractor(self.finetune_groups)
        
        # 消融模型
        elif self.args.model_used =='ELFNet_ablation_augmentor': 
            print('Standard ELFNet (Self-supervised version)...')
            if not self.args.finetune_pretrained_model and not self.args.test_finetuned_model:
                self.model = ELFNet_ablation_augmentor(self.args, self.args.wo_augmentor,device=self.device, stage=1).to(self.device)
                self.model.augmentor.initialize_from_data(self.pretrain_data.data_x,self.pretrain_data.data_y)
            else:
                self.model = ELFNet_ablation_augmentor(self.args, self.args.wo_augmentor,device=self.device, stage=2).to(self.device)
                if self.args.test_finetuned_model:
                    self.model.stage = 3
                    self.model._init_input_projections(self.finetune_data.data_x.shape[1],self.args.hidden_dims)
                    self.model._init_feature_extractor(self.finetune_groups)
        elif self.args.model_used == 'ELFNet_wo_CGU':
            print('ELFNet without CoupledGatingUnit (Self-supervised, two-stage)...')
            if not self.args.finetune_pretrained_model and not self.args.test_finetuned_model:
                self.model = ELFNet_wo_CGU(self.args, device=self.device, stage=1).to(self.device)
                self.model.augmentor.initialize_from_data(self.pretrain_data.data_x, self.pretrain_data.data_y)
            else:
                self.model = ELFNet_wo_CGU(self.args, device=self.device, stage=2).to(self.device)
                if self.args.test_finetuned_model:
                    self.model.stage = 3
                    self.model._init_input_projections(self.finetune_data.data_x.shape[1], self.args.hidden_dims)
                    self.model._init_feature_extractor(self.finetune_groups)
        elif self.args.model_used == 'ELFNet_wo_TS':
            print('ELFNet without disentanglement (Supervised version))...')
            self.model = ELFNet_wo_TS(self.args,device=self.device).to(self.device)
            if self.args.test_finetuned_model:
                self.model._init_input_projections(self.finetune_data.data_x.shape[1],self.args.hidden_dims)
                self.model._init_feature_extractor(self.finetune_groups)
        elif self.args.model_used == 'ELFNet_common_TS':
            self.model = ELFNet_common_TS(self.args,device=self.device).to(self.device)
            if self.args.test_finetuned_model:
                self.model._init_input_projections(self.finetune_data.data_x.shape[1],self.args.hidden_dims)
                self.model._init_feature_extractor(self.finetune_groups)
        elif self.args.model_used == 'ELFNet_supervised':
            print('Standard ELFNet (Supervised version)...')
            self.model = ELFNet_supervised(self.args, device=self.device).to(self.device)
            if self.args.test_finetuned_model:
                self.model._init_input_projections(self.finetune_data.data_x.shape[1],self.args.hidden_dims)
                self.model._init_feature_extractor(self.finetune_groups)
        elif self.args.model_used == 'ELFNet_single_band_SRD':
            print('ELFNet single-band SRD ablation (Supervised version)...')
            self.model = ELFNet_single_band_SRD(self.args, device=self.device).to(self.device)
            if self.args.test_finetuned_model:
                self.model._init_input_projections(self.finetune_data.data_x.shape[1], self.args.hidden_dims)
                self.model._init_feature_extractor(self.finetune_groups)
        elif self.args.model_used == 'ELFNet_supervised_pretrain':
            print('ELFNet supervised pretraining version (two-stage)...')
            if not self.args.finetune_pretrained_model and not self.args.test_finetuned_model:
                self.model = ELFNet_supervised_pretrain(self.args, device=self.device, stage=1).to(self.device)
            else:
                self.model = ELFNet_supervised_pretrain(self.args, device=self.device, stage=2).to(self.device)
                if self.args.test_finetuned_model:
                    self.model.stage = 3
                    self.model._init_input_projections(self.finetune_data.data_x.shape[1], self.args.hidden_dims)
                    self.model._init_feature_extractor(self.finetune_groups)
        else:
            print('Please input the correct model name')
        
        self.optimizer = getattr(optim, self.args.optimizername)(self.model.parameters(), lr=self.args.lr)
        
        
    def _init_baseline_model(self):
        """Initialize baseline model"""
        CompareModel_dict = {
            'TimesNet': TimesNet,
            'ADDSTCN': ADDSTCN,
            'Informer': Informer,
            'SegRNN': SegRNN,
            'DLinear': DLinear,
            'TS2Vec':TS2Vec,
            'PatchTST_SS':PatchTST_SS,
            'CoST': CoST,
            'TimeMAE':TimeMAE,
            'PatchTST_SU':PatchTST_SU
        }
        
        self.CompareModel = CompareModel_dict[self.args.model_used](self.args).float().to(self.device)

        
        self.optimizer = getattr(optim, self.args.optimizername)(self.CompareModel.parameters(), lr=self.args.lr)


    def _acquire_device(self):
        """设置训练设备，支持CPU或单GPU"""
        if self.args.use_gpu:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(self.args.gpu) 
            device = torch.device('cuda:{}'.format(self.args.gpu))
            print('Use GPU: cuda:{}'.format(self.args.gpu))
        else:
            device = torch.device('cpu')
            print('Use CPU')
        return device

    def _get_data(self, flag,pretrain_stage=False):
        """调用data_provider函数获得data_set, data_loader，训练、验证、测试数据集都是通过它设置不同参数来获得"""
        data_set, data_loader = data_provider(self.args, flag,pretrain_stage)
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
        
        if self.args.finetune_pretrained_model:
            trained_model_dict = torch.load(trained_model_path)
            self.CompareModel.load_state_dict(trained_model_dict)

        t = time.time()
        print(f"=======Starting to train {self.args.model_used}=======")
        self.CompareModel.train()
        losses = []
        global_iteration = 0
        for epoch in range(epochs):
            train_loss = []

            for iteration, (batch_x, batch_x_mark, batch_y, batch_y_mark) in enumerate(self.finetune_loader):
                optimizer.zero_grad()
                batch_x = batch_x.transpose(1, 2).float().to(self.device)
                batch_y = batch_y.float().to(self.device)
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
    
    def _train_ss_compare(self):
        if self.args.finetune_pretrained_model:
            training_time_stage1 = 0
        else:
            training_time_stage1 = self._train_stage1_ss()
        
        training_time_stage2, model_path = self._train_stage2_ss()
        total_training_time = training_time_stage1 + training_time_stage2
        return training_time_stage1, training_time_stage2, total_training_time,model_path

    def _train_ELFNet_family(self):  
        """ ELFNet及其消融模型训练流程，支持one2many和single两种模式"""
        print(f"Training mode:{self.args.training_mode}")
        if self.args.model_used in ['ELFNet_supervised','ELFNet_wo_TS','ELFNet_common_TS','ELFNet_single_band_SRD'] or self.args.finetune_pretrained_model:
            training_time_stage1 = 0
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
        # 初始化输入投影层
        self.model._init_input_projections(self.pretrain_data.data_x.shape[1], self.args.hidden_dims)
        
        pretrained_model_path = os.path.join(model_path,f"{self.args.model_used}.pth")
        if self.args.pretrained_model_path == None: 
            self.args.pretrained_model_path = pretrained_model_path
            self.model._init_feature_extractor(self.pretrain_groups)
        else: # 继续接着预训练已有pretrain_model
            self.model._init_feature_extractor(self.pretrain_groups)
            pretrained_model_state_dict = torch.load(self.args.pretrained_model_path)
            self.model.load_state_dict(pretrained_model_state_dict)
         
        t = time.time()
        print(f"=======Starting to train {self.args.model_used}: stage1=======")
        self.model.train()
        losses = []

        global_iteration = 0
        plot_augment_flag = True # 用于仅仅在第一次前向传播时可视化增强数据
        for epoch in range(epochs):
            train_loss = []
            for iteration, (batch_x, _, batch_y, _) in enumerate(self.pretrain_loader): # batch_x: tensor (b, seq_len, c)
                optimizer.zero_grad()
                if self.args.model_used == 'ELFNet_supervised_pretrain':
                    # 有监督预训练：直接使用 batch_y 计算 MSE 损失
                    batch_x = batch_x.to(self.device)
                    batch_y = batch_y.to(self.device)
                    loss = self.model.compute_loss(batch_x, batch_y)
                else:
                    # 自监督预训练：对比学习损失
                    batch_x = batch_x.transpose(1, 2).to(self.device)
                    loss = self.model.compute_loss(batch_x, self.plot_dir, plot_augment_flag)
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

        t = time.time()
        
        if not self.args.model_used in ['ELFNet_supervised','ELFNet_wo_TS','ELFNet_common_TS','ELFNet_single_band_SRD']:
            print('加载预训练模型进行权重迁移以进一步微调')
            pretrained_model_state_dict = torch.load(self.args.pretrained_model_path)
            self.model._init_feature_extractor(self.pretrain_groups)
            #print(self.model.feature_extractor)
            if self.args.finetune_pretrained_model:
                self.model._init_input_projections(self.pretrain_data.data_x.shape[1],self.args.hidden_dims)
                self.model.stage = 2
            self.model.load_state_dict(pretrained_model_state_dict)
        else:
            pretrained_model_state_dict = None

            
        self.model._init_input_projections(self.finetune_data.data_x.shape[1],self.args.hidden_dims)
        self.model._init_feature_extractor(self.finetune_groups)
        
        print(f"====Starting to train {self.args.model_used}: stage {self.model.stage}====")
        self.model.train() 

        self.model.stage = 2  
        losses = []
        
        global_iteration = 0
        for epoch in range(epochs):
            train_loss = []
            for iteration, (batch_x, _,batch_y,_) in enumerate(self.finetune_loader): 
                optimizer.zero_grad()

                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                # during the per-batch training we do not need to re-run transfer (already done above)
                outputs = self.model(batch_x,pretrained_model_state_dict)
                
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
        print(f"Total training time in stage: {training_time:.2f}s")
        
        return training_time,finetuned_model_path
        
    def _train_stage1_ss(self):
        """自监督对比模型预训练阶段"""
        epochs = self.args.train_epochs1
        optimizer = self.optimizer
        early_stopping = self.early_stopping
        model_path = os.path.join(self.folder_path, 'pretrained_ss_model')
        if not os.path.exists(model_path):
                os.makedirs(model_path)

        pretrained_model_path = os.path.join(model_path,f"{self.args.model_used}.pth")
        if self.args.pretrained_model_path == None: 
            self.args.pretrained_model_path = pretrained_model_path
        else: # 继续接着预训练已有pretrain_model
            self.CompareModel.load(self.args.pretrained_model_path)
         
        # 训练
        t = time.time()
        print(f"=======Starting to train {self.args.model_used}: stage1=======")
        self.CompareModel.train()
        losses = []
        global_iteration = 0
        for epoch in range(epochs):
            train_loss = []
            for iteration, (batch_x, _,_,_) in enumerate(self.pretrain_loader): # batch_x: tensor (b, seq_len, c)   batch_x: (b,pred_len,1)
                optimizer.zero_grad()
                batch_x = batch_x.to(self.device)
                loss = self.CompareModel.compute_loss(batch_x)
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
                    early_stopping(recent_avg_loss, self.CompareModel, pretrained_model_path, is_iteration=True, current_iteration=global_iteration)
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

            early_stopping(train_loss, self.CompareModel, pretrained_model_path,is_iteration=False,current_iteration=global_iteration)
            if early_stopping.early_stop:
                print("epoch级别训练早停")
                break

            adjust_learning_rate(optimizer, epoch + 1, self.args)
        
        training_time = time.time() - t
        print(f"Total training Time in Stage1: {training_time:.2f}s")
        
        return training_time

    def _train_stage2_ss(self):
        """自监督对比模型微调阶段"""
        epochs = self.args.train_epochs2
        optimizer = self.optimizer
        model_path = os.path.join(self.folder_path, f"finetuned_ss_model/{self.args.model_used}")
        finetuned_model_path = os.path.join(model_path,f"{os.path.splitext(self.args.data_path.split('/')[-1])[0]}.pth")
        if not os.path.exists(model_path):
                os.makedirs(model_path)
        self.early_stopping = IterationEarlyStopping(patience_epochs=self.args.patience_epochs,patience_iterations=self.args.patience_iterations,min_iterations=self.args.min_iterations,verbose=True,delta=self.args.improved_delta)
        early_stopping = self.early_stopping

        t = time.time()
        
        print('加载预训练模型进一步微调')
        
        self.CompareModel.stage = 2
        self.CompareModel.load(self.args.pretrained_model_path)
        
        print(f"====Starting to train {self.args.model_used}: stage {self.CompareModel.stage}====")
        self.CompareModel.train() 

        self.CompareModel.stage = 2  
        losses = []
        
        global_iteration = 0
        for epoch in range(epochs):
            train_loss = []
            for iteration, (batch_x, _,batch_y,_) in enumerate(self.finetune_loader): 
                optimizer.zero_grad()

                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                outputs = self.CompareModel(batch_x)
                
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
                    early_stopping(recent_avg_loss, self.CompareModel,finetuned_model_path, is_iteration=True, current_iteration=global_iteration)
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
            
            early_stopping(vali_loss, self.CompareModel, finetuned_model_path,is_iteration=False,current_iteration=global_iteration)
            if early_stopping.early_stop :
                break

            adjust_learning_rate(optimizer, epoch + 1, self.args)
        
        training_time = time.time() - t
        print(f"Total training time in stage: {training_time:.2f}s")
        
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

                if self.args.model_used in ['ELFNet','ELFNet_wo_TS','ELFNet_supervised','ELFNet_ablation_augmentor','ELFNet_common_TS','ELFNet_supervised_pretrain','ELFNet_single_band_SRD','ELFNet_wo_CGU'] :
                    self.model.eval()
                    outputs = self.model(batch_x)
                    self.model.train()
                else:
                    self.CompareModel.eval()
                    # decoder input
                    dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                    #dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                    outputs = self.CompareModel(batch_x.transpose(1,2), batch_x_mark, dec_inp, batch_y_mark)
                    self.CompareModel.train()
                
                outputs = outputs[:, -self.args.pred_len:, :]
                loss = self.criterion(outputs, batch_y)
                total_loss.append(loss.item())
        total_loss = np.average(total_loss)
        return total_loss


    def test(self,model_path, setting, test):
        test_data, test_loader = self._get_data(flag='test')

        if hasattr(self.args, 'finetune_freq'):
            freq_mapping = {'H': 1, 'T': 1/60, 'D': 24}
            time_step = freq_mapping.get(self.args.finetune_freq, 1)
            period_mapping = {'H': 24, 'T': 96, 'D': 7}  # 小时、15分钟、天
            seasonal_period = period_mapping.get(self.args.finetune_freq, 24)

        if test:
            print('Loading model')
            if self.args.model_used in ['ELFNet','ELFNet_wo_TS','ELFNet_supervised','ELFNet_ablation_augmentor','ELFNet_common_TS','ELFNet_supervised_pretrain','ELFNet_single_band_SRD','ELFNet_wo_CGU'] :
                self.model.load_state_dict(torch.load(model_path))
                self.model.stage = 3
            else:
                self.CompareModel.load_state_dict(torch.load(model_path))
        preds, trues = [], []
        inference_times = []
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
                
                # 记录推理开始时间
                start_time = time.time()        
                
                if self.args.model_used in ['ELFNet','ELFNet_wo_TS','ELFNet_supervised','ELFNet_ablation_augmentor','ELFNet_common_TS','ELFNet_supervised_pretrain','ELFNet_single_band_SRD','ELFNet_wo_CGU'] :
                    self.model.eval()
                    outputs = self.model(batch_x)   
                else:
                    self.CompareModel.eval()
                    batch_x_mark = batch_x_mark.float().to(self.device)
                    batch_y_mark = batch_y_mark.float().to(self.device) 
                    # decoder input
                    dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                    #dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                    
                    outputs = self.CompareModel(batch_x.transpose(1,2), batch_x_mark, dec_inp,  batch_y_mark)
                    
                # 记录推理结束时间
                end_time = time.time()
                inference_times.append(end_time - start_time)
                outputs = outputs[:, -self.args.pred_len:, :]
                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()

                if test_data.scale and self.args.inverse:
                    shape = outputs.shape
                    outputs = test_data.inverse_transform(outputs,self.args.finetune_target_idx)
                    batch_y = test_data.inverse_transform(batch_y,self.args.finetune_target_idx)
                    # 后处理 inverse后的预测值，确保物理合理性
                    ## 对于真实值为0的点，如果预测值小于0，可以强制为0
                    pred_mask = (batch_y == 0) & (outputs < 0)
                    outputs[pred_mask] = 0
                
                preds.append(outputs)
                trues.append(batch_y)

                if i % 4 == 0 and self.args.plot_test:
                    # 转换为numpy
                    input_np = batch_x.detach().cpu().numpy()  # [batch_size, seq_len, n_features]
                    
                    if test_data.scale and self.args.inverse:
                        # 只提取输入中的目标变量列并反标准化
                        input_target = input_np[:, :, test_data.targetidx:test_data.targetidx+1]  # [batch_size, seq_len, 1]
                        input_target_denorm = test_data.inverse_transform(input_target,self.args.finetune_target_idx)
                        # 取第一个样本
                        input_target_0 = input_target_denorm[0, :, 0]  # [seq_len]
                    else:
                        # 直接取目标变量列
                        input_target_0 = input_np[0, :, test_data.targetidx]  # [seq_len]
                    
                    outputs_0 = outputs[0, :, 0]  # [pred_len]
                    batch_y_0 = batch_y[0, :, 0]  # [pred_len]
                    
                    # 连接历史部分和未来部分
                    gt = np.concatenate((input_target_0, batch_y_0), axis=0)
                    pd = np.concatenate((input_target_0, outputs_0), axis=0)
                    visual(gt, pd, os.path.join(folder_path, str(i) + '.pdf'))
        
        # 先将list转换为numpy
        np_preds = np.array(preds)
        np_trues =  np.array(trues) 

        preds = np_preds.reshape(-1, np_preds.shape[-2], np_preds.shape[-1])
        trues = np_trues.reshape(-1, np_trues.shape[-2],np_trues.shape[-1])

        # 计算所有指标
        metrics = metric(preds, trues,seasonal_period,time_step)
        # 计算效率指标
        avg_inference_time = np.mean(inference_times) * 1000  # 转换为毫秒
        
        # 计算模型参数量
        if self.args.model_used in ['ELFNet','ELFNet_wo_TS','ELFNet_supervised','ELFNet_ablation_augmentor','ELFNet_common_TS','ELFNet_supervised_pretrain','ELFNet_single_band_SRD','ELFNet_wo_CGU'] :
            total_params = sum(p.numel() for p in self.model.parameters())
        else:
            total_params = sum(p.numel() for p in self.CompareModel.parameters())
        
        metrics['Total_Params'] = total_params
        metrics['Params_M'] = total_params / 1e6  # 百万为单位
        

        # 输出所有指标
        print("\n" + "="*80)
        print("Comprehensive evaluation metrics")
        print("="*80)

        # 分组输出指标
        print(f"\n1. Basic metrics (scale = {self.args.scale}, inverse = {self.args.inverse}):")
        print(f"   MAE: {metrics['MAE']:.4f}")
        print(f"   RMSE: {metrics['RMSE']:.4f}")
        print(f"   NMAE: {metrics['NMAE']:.4f}")
        print(f"   NRMSE: {metrics['NRMSE']:.4f}")
        print(f"   CV_RMSE: {metrics['CV_RMSE']:.4f}")
        print(f"   MAPE: {metrics['MAPE']:.4f}")
        print(f"   R2: {metrics['R2']:.4f}")
        print(f"   MASE: {metrics['MASE']:.4f}")
        
        print("\n2. Peak-related metrics:")
        print(f"   Peak Absolute Error: {metrics['Peak_Abs_Error']:.4f},Normalized Peak Absolute Error: {metrics['Normalized_Peak_Abs_Error']:.4f},Peak Relative Error: {metrics['Peak_Rel_Error']:.4f}")
        print(f"   Peak Time Shift: {metrics['Peak_Time_Shift']:.2f} time steps")
        
        print("\n3. Correlation metrics:")
        print(f"   Correlation Coefficient: {metrics['Correlation']:.4f}")
        
        print("\n4. Efficiency metrics:")
        print(f"   Average Inference Time: {avg_inference_time:.2f} ms")
        print(f"   Total Parameters: {metrics['Total_Params']:,}")
        print(f"   Parameters (M): {metrics['Params_M']:.2f}M")
        print("="*80)

        # 保存结果到文件
        f = open("result_forecast.txt", 'a')
        if self.args.model_used in ['ELFNet','ELFNet_wo_TS','ELFNet_supervised','ELFNet_ablation_augmentor','ELFNet_common_TS','ELFNet_supervised_pretrain','ELFNet_single_band_SRD','ELFNet_wo_CGU']:
            f.write(f"{self.args.model_used}_{setting}_{os.path.splitext(self.args.data_path.split('/')[-1])[0]}\n")
        else:
            f.write(f"{self.args.model_used}_{setting}\n")
        
        f.write(f"BASIC METRICS (scale = {self.args.scale}, inverse = {self.args.inverse}):\n")
        f.write(f"MAE: {metrics['MAE']:.4f}, RMSE: {metrics['RMSE']:.4f}\n")
        f.write(f"NMAE: {metrics['NMAE']:.4f},  NRMSE: {metrics['NRMSE']:.4f}\n")
        f.write(f"CV_RMSE: {metrics['CV_RMSE']:.4f},MAPE: {metrics['MAPE']:.4f}\n")
        f.write(f"R2: {metrics['R2']:.4f}, MASE: {metrics['MASE']:.4f}\n")
        
        f.write("\nPEAK METRICS:\n")
        f.write(f"Peak Abs Error: {metrics['Peak_Abs_Error']:.4f},Normalized Peak Abs Error: {metrics['Normalized_Peak_Abs_Error']:.4f}, Peak Rel Error: {metrics['Peak_Rel_Error']:.4f}, Peak Time Shift: {metrics['Peak_Time_Shift']:.2f}\n")
        
        f.write(f"\nCorrelation: {metrics['Correlation']:.4f}\n")
        f.write(f"\nEFFICIENCY:\n")
        f.write(f"\n Average Inference Time: {avg_inference_time:.2f} ms\n")
        f.write(f"\nTotal Parameters: {metrics['Total_Params']:,}\n")
        f.write(f"\nParameters (M): {metrics['Params_M']:.2f}M\n")
        f.write('='*80 + '\n\n')
        f.close()
        
        return avg_inference_time

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



