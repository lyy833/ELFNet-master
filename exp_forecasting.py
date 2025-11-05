import os
import time
import numpy as np
import torch
import torch.nn as nn
from utils.tools import IterationEarlyStopping, adjust_learning_rate,visual
from utils.metrics import metric
from ELFNet.ELFNet import ELFNet
import ast
import torch.optim as optim
from models.TimesNet import  TimesNet
from models.ADDSTCN import ADDSTCN
from models.Informer import Informer
from models.SegRNN import SegRNN
from models.DLinear import DLinear
from models import ELFNet_depthwise,ELFNet_no_disentanglement ,ELFNet_no_contrastive
from ELFNet.ELFNet import  ELFNet_Base,ELFNet,load_pretrained_weights
from data_process.data_provider import data_provider
import warnings

warnings.filterwarnings('ignore')

class Exp_forecasting(object):
    def __init__(self, args, setting,folder_path,plot_dir):
        
        self.args = args  
        self.setting = setting
        # Initialize data based on pretraining mode
        if args.pretrain_mode == 'one2many':
            # Cross-dataset mode 
            self.pretrain_data, self.pretrain_loader = self._get_data(flag='train', pretrain_stage=True)
            self.finetune_data, self.finetune_loader = self._get_data(flag='train', pretrain_stage=False) # initialize pretraining and fine-tuning data separately
            self.max_iterations = max(self.pretrain_data.get_max_iterations(self.args.batch_size),self.finetune_data.get_max_iterations(self.args.batch_size))
            self.pretrain_input_channels = self.pretrain_data.data_x.shape[1]
            self.finetune_input_channels = self.finetune_data.data_x.shape[1]
            print(f"Pretraining dataset channels: {self.pretrain_input_channels}")
            print(f"Fine-tuning dataset channels: {self.finetune_input_channels}")
        else:
            # self.train_data is an instance of CustomDataset, self.train_loader is an instance of DataLoader
            self.train_data, self.train_loader = self._get_data(flag='train') 
            self.max_iterations = self.train_data.get_max_iterations(self.args.batch_size)
            self.input_channels = self.train_data.data_x.shape[1]
        
        if self.args.cluster=='True': # Use FeSC algorithm to obtain a new sample set
            if args.pretrain_mode == 'one2many': # Cross-dataset mode
                self.cluster_pretrain_data, self.cluster_pretrain_loader = self._get_data(flag='train', cluster_data=True,pretrain_stage=True)
                self.cluster_max_iterations = self.cluster_pretrain_data.get_max_iterations(self.args.batch_size)
            else:
                self.cluster_train_data, self.cluster_train_loader = self._get_data(flag='train',cluster_data=True,pretrain_stage=True)  # _get_data calls data_provider to get training data  
                # The number of samples changes, so the maximum number of iterations in an epoch also changes
                self.cluster_max_iterations = self.cluster_train_data.get_max_iterations(self.args.batch_size)
        
        
        
        self.criterion = self._select_criterion()
        self.early_stopping = IterationEarlyStopping(patience_epochs=args.patience_epochs,patience_iterations=args.patience_iterations,min_iterations=args.min_iterations,verbose=True,delta=args.improved_delta)  # Set early stopping mechanism
        self.device = self._acquire_device()
        # Define and create the path to save the results of this task
        self.folder_path = folder_path
        self.plot_dir = plot_dir

        # Model initialization
        self._init_models()

    def _init_models(self):
        """Initialize models based on mode"""
        if self.args.pretrain_mode == 'one2many': # one2many mode: initialize pretraining and fine-tuning models separately
            self._init_one2many_models()
        else: # Single mode
            self._init_single_mode_models()

    def _init_single_mode_models(self):
        """Model initialization for Single mode """
        if self.args.model_used in ['ELFNet','ELFNet_depthwise', 'ELFNet_no_disentanglement', 'ELFNet_no_contrastive', 'ELFNet_dilution']:
            self.pretrain_model =None
            self.finetune_model = None
            if self.args.model_used =='ELFNet': # Complete version of ELFNet
                print('Standard VG-HCS ELFNet in single mode...')
                groups = ast.literal_eval(self.args.groups)
                self.model = ELFNet(
                    self.args, self.train_data.targetidx, self.input_channels, 
                    device=self.device, groups=groups
                ).to(self.device)
            elif self.args.model_used == 'ELFNet_dilution':
                print('ELFNet without VG-HCS in ablation experiment (CD)...')
                self.model = ELFNet(
                    self.args, self.train_data.targetidx, self.input_channels, 
                    device=self.device
                ).to(self.device)
            elif self.args.model_used == 'ELFNet_depthwise':
                print('ELFNet without VG-HCS in ablation experiment (CI)...')
                self.model = ELFNet_depthwise.ELFNet_depthwise(
                    self.args, self.train_data.targetidx, self.input_channels, 
                    device=self.device
                ).to(self.device)
            elif self.args.model_used == 'ELFNet_no_disentanglement':
                print('ELFNet without disentanglement in ablation experiment...')
                groups = ast.literal_eval(self.args.groups)
                self.model = ELFNet_no_disentanglement.ELFNet_no_disentanglement(
                    self.args, self.train_data.targetidx, self.input_channels, 
                    device=self.device, groups=groups
                ).to(self.device)
            elif self.args.model_used == 'ELFNet_no_contrastive':
                print('ELFNet without contrastive learning in ablation experiment...')
                groups = ast.literal_eval(self.args.groups)
                self.model = ELFNet_no_contrastive.ELFNet_no_contrastive(
                    self.args, self.train_data.targetidx, self.input_channels, 
                    device=self.device, groups=groups
                ).to(self.device)
            else:
                print('Please input the correct model name')
            
            self.optimizer = getattr(optim, self.args.optimizername)(self.model.parameters(), lr=self.args.lr)
            self.model.optimizer = self.optimizer
        else:
            # baseline model (keep original logic)
            self.CompareModel_dict = {
                'TimesNet': TimesNet,
                'ADDSTCN': ADDSTCN,
                'Informer': Informer,
                'SegRNN': SegRNN,
                'DLinear': DLinear
            }
            self.CompareModel = self._build_CompareModel().to(self.device)
            self.optimizer = getattr(optim, self.args.optimizername)(self.CompareModel.parameters(), lr=self.args.lr)



    def _init_one2many_models(self):
        """Initialize models for one2many mode"""
        print("Channel Independent ELFNet-Base in one2many mode ...")
        # 1. Pretraining stage: use ELFNet-Base (channel independent)
        self.pretrain_model = ELFNet_Base(self.args, self.device, hidden_dim=64).to(self.device)
        self.pretrain_optimizer = getattr(optim, self.args.optimizername)(self.pretrain_model.parameters(), lr=self.args.lr)
        # 2. Fine-tuning stage: construct later during training (requires group information)
        self.finetune_model = None
        self.finetune_optimizer = None
        print("✓ one2many mode model initialization completed")

    def _build_finetune_model(self):
        """Initialize fine-tuning model for target dataset and transfer weights in one2many mode"""
        if self.finetune_model is not None:
            return self.finetune_model

        print("Initializing fine-tuning model...")
        # Get input channel count and group information for target dataset
        input_channels = self.finetune_data.data_x.shape[1]
        target_idx = self.finetune_data.targetidx
        
        # Use groups parsed from args or variable adaptive grouping algorithm
        groups = ast.literal_eval(self.args.groups) if self.args.groups else self._get_default_groups(input_channels)
        
        print(f"Fine-tuning dataset info: Channels={input_channels}, Target Index={target_idx}, Groups={groups}")
        
        # Construct standard ELFNet
        if self.args.model_used =='ELFNet':
            self.finetune_model = ELFNet(
                self.args, target_idx, input_channels, 
                device=self.device, groups=groups, stage2=True
            ).to(self.device)
        else:
            # Handle ablation experiment models
            self.finetune_model = self._build_ablation_model(
                self.args.model_used, target_idx, input_channels, groups
            )
        
        # Transfer pretraining weights
        if self.args.model_path is not None: 
            print('Loading trained saved model to finetune')
            self.pretrain_model.load_state_dict(torch.load(os.path.join(self.args.model_path ,'model.pth')))
        
        self.finetune_model = load_pretrained_weights(self.pretrain_model, self.finetune_model, groups)

        
        # Set optimizer
        self.finetune_optimizer = getattr(optim, self.args.optimizername)(
            self.finetune_model.parameters(), lr=self.args.lr
        )
        
        print("✓ Fine-tuning model construction completed")
        return self.finetune_model

    def _build_ablation_model(self, model_type, target_idx, input_channels, groups):
        """
        Construct ablation experiment models in one2many mode (called in _build_finetune_model function).
        Note: ELFNet_no_contrastive ablation does not use contrastive learning, so it does not support two-stage training in one2many mode.
        """
        if model_type == 'ELFNet_dilution': # Channel dependent
            return ELFNet(
                self.args, target_idx, input_channels, 
                device=self.device, stage2=True
            ).to(self.device)
        elif model_type == 'ELFNet_depthwise': # Channel independent
            return ELFNet_depthwise(
                self.args, target_idx, input_channels, 
                device=self.device, stage2=True
            ).to(self.device)
        elif model_type == 'ELFNet_no_disentanglement': # Without disentanglement
            return ELFNet_no_disentanglement(
                self.args, target_idx, input_channels, 
                device=self.device, groups=groups, stage2=True
            ).to(self.device)
        else:
            raise ValueError(f"Unknown model type: {model_type}")

    def _build_CompareModel(self):
        """Initialize baseline model"""
        model = self.CompareModel_dict[self.args.model_used](self.args)
        model = model.float()  

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

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
        
        model_path = os.path.join(self.folder_path, 'baseline_model')
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        model_name = f"{self.args.model_used}_{os.path.splitext(self.args.data_path.split('/')[-1])[0]}"
        full_model_path = os.path.join(model_path,f"{model_name}.pth")
        
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
                    early_stopping(recent_avg_loss, self.CompareModel, model_path, model_name,is_iteration=True, current_iteration=global_iteration)
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

            early_stopping(vali_loss, self.CompareModel, model_path,model_name,is_iteration=False,current_iteration=global_iteration)
            
            if self.args.plot: # 可视化训练损失
                self._plot_losses(losses, 'compare_train')
            
            if early_stopping.early_stop :
                break

            adjust_learning_rate(optimizer, epoch + 1, self.args)
        
        total_training_time = time.time() - t
        print(f"Total Training Time: {total_training_time:.2f}s, 总迭代次数: {global_iteration}")

        return total_training_time,full_model_path
    

    def _train_ELFNet_family(self):  
        """ ELFNet 训练流程，支持one2many 和 single 两种模式"""
        
        if self.args.pretrain_mode == 'one2many':
            print("one2many mode:")
            if self.args.model_path is not None:
                training_time_stage1 = 0
                training_time_stage2,model_path = self._train_stage2_one2many()
            else:
                print("阶段1-预训练：使用通道独立的 ELFNet-Base ...")
                training_time_stage1 = self._train_stage1_one2many()
                print("阶段2-微调：构建并迁移权重到标准 ELFNet...")
                training_time_stage2,model_path = self._train_stage2_one2many()
            total_training_time = training_time_stage1 + training_time_stage2   
        else:  # Single 模式
            if self.args.model_used == 'ELFNet_no_contrastive' or self.args.model_path is not None:
                training_time_stage1 = 0
                training_time_stage2,model_path = self.train_stage2()  
            else:
                training_time_stage1 = self.train_stage1()
                training_time_stage2,model_path = self.train_stage2()
            total_training_time = training_time_stage1 + training_time_stage2   
        
        return training_time_stage1, training_time_stage2, total_training_time,model_path

    def _train_stage1_one2many(self):
        """one2many 模式的阶段1：自监督预训练"""
        epochs = self.args.train_epochs1
        optimizer = self.pretrain_optimizer
        early_stopping = self.early_stopping
        model = self.pretrain_model

        model_path = os.path.join(self.folder_path, 'pretrained_ELFNet_family')
        model_name = f"{self.args.model_used}_{os.path.splitext(self.args.pretrain_data_path.split('/')[-1])[0]}"
        
        t = time.time()
        model.train()
        losses = []

        # 选择数据加载器
        if self.args.cluster == 'True':
            train_loader = self.cluster_pretrain_loader
            max_iterations = self.cluster_max_iterations
        else:
            train_loader = self.pretrain_loader
            max_iterations = self.max_iterations

        global_iteration = 0

        for epoch in range(epochs):
            train_loss = []
            for iteration, (batch_x, _, batch_y, _) in enumerate(train_loader):
                optimizer.zero_grad()
                loss = model.compute_loss(batch_x.to(self.device), batch_y.to(self.device), self.plot_dir, epoch)
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
                    early_stopping(recent_avg_loss, model, model_path,model_name, is_iteration=True, current_iteration=global_iteration)
                    if early_stopping.is_loss_stable(threshold=0.001): # 额外检查损失是否稳定
                        print(f"损失已趋于稳定，考虑早停")
                        early_stopping.early_stop = True
                    if early_stopping.early_stop:
                        print("iteration级别训练早停")
                        break # 跳出当前iteration

                if (iteration + 1) % self.args.log_interval == 0:
                    print(f"预训练 Iter: {iteration+1}, Loss: {loss:.7f}")
            
            train_loss = np.average(train_loss)
            print(f"预训练 Epoch: {epoch+1}, Loss: {train_loss:.7f}")

            if self.args.plot:
                self._plot_losses(losses, 'pretrain_stage1')
            
            print(f"预训练时间: {time.time() - t:.2f}s")
            
            # ELFNet第一阶段预训练比较特殊，因为其早停只涉及训练集损失
            # 因此如果已经iteration级别早停，不用判断是否epoch早停，直接终止训练
            # 但是其它模型或者ELFNet微调训练阶段训练集损失iteration级别早停后还要进一步判断验证集上epoch级别早停
            if early_stopping.early_stop : 
                break

            # epoch级别早停机制
            early_stopping(train_loss, model, model_path,model_name,is_iteration=False,current_iteration=global_iteration)
            if early_stopping.early_stop:
                print("epoch级别训练早停")
                break

            adjust_learning_rate(optimizer, epoch + 1, self.args)
        
        training_time = time.time() - t
        print(f"预训练总时间: {training_time:.2f}s")
        return training_time

    def _train_stage2_one2many(self):
        """one2many 模式的阶段2训练"""
        # 构建微调模型并迁移权重
        self._build_finetune_model()
        
        epochs = self.args.train_epochs2
        optimizer = self.finetune_optimizer
        model = self.finetune_model
        
        model_path = os.path.join(self.folder_path, 'finetuned_ELFNet_family')
        model_name = f"{self.args.model_used}_{os.path.splitext(self.args.pretrain_data_path.split('/')[-1])[0]}"
        full_model_path = os.path.join(model_path,f"{model_name}.pth")
        
        # 重置早停器
        self.early_stopping = IterationEarlyStopping(patience_epochs=self.args.patience_epochs,patience_iterations=self.args.patience_iterations,min_iterations=self.args.min_iterations,verbose=True,delta=self.args.improved_delta)
        early_stopping = self.early_stopping
        t = time.time()
        model.train()
        losses = []
        global_iteration = 0
        for epoch in range(epochs):
            train_loss = []
            for iteration, (batch_x, _, batch_y, _) in enumerate(self.finetune_loader):
                optimizer.zero_grad()

                batch_x = batch_x.transpose(1, 2)
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                outputs = model(batch_x)
                
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
                    early_stopping(recent_avg_loss, model, model_path,model_name, is_iteration=True, current_iteration=global_iteration)
                    if early_stopping.is_loss_stable(threshold=0.001): # 额外检查损失是否稳定
                        print(f"损失已趋于稳定，考虑早停")
                        early_stopping.early_stop = True
                    if early_stopping.early_stop:
                        print("iteration级别训练早停")
                        break

                if (iteration + 1) % self.args.log_interval == 0:
                    print(f"微调 Iter: {iteration+1}, Loss: {loss:.7f}")
            
            if self.args.plot:
                self._plot_losses(losses, 'finetune_stage2')

            train_loss = np.average(train_loss)
            print(f"微调 Epoch: {epoch+1}, Loss: {train_loss:.7f}")

            # 验证
            vali_data, vali_loader = self._get_data(flag='val')
            vali_loss = self.vali(vali_loader)
            print(f"微调 Vali Loss: {vali_loss:.7f}")
            print(f"微调时间: {time.time() - t:.2f}s")
            
            early_stopping(vali_loss, model, model_path,model_name,is_iteration=False,current_iteration=global_iteration)
            if early_stopping.early_stop:
                break

            adjust_learning_rate(optimizer, epoch + 1, self.args)
        
        training_time = time.time() - t
        print(f"微调总时间: {training_time:.2f}s")
        return training_time,full_model_path

    def train_stage1(self):

        epochs = self.args.train_epochs1
        optimizer = self.optimizer
        early_stopping = self.early_stopping
        model_path = os.path.join(self.folder_path, 'pretrained_ELFNet_family')
        model_name = f"{self.args.model_used}_{os.path.splitext(self.args.data_path.split('/')[-1])[0]}"
        
        t = time.time()
        print('=============Starting to train ELFNet: stage1==============')
        self.model.train()
        losses = []

        # 是否聚类以减少样本
        if self.args.cluster=='True':
            if self.args.pretrain_mode != 'single':
                train_loader=self.cluster_pretrain_loader
            else:
                train_loader=self.cluster_train_loader
            max_iterations = self.cluster_max_iterations
        else:
            if self.args.pretrain_mode != 'single':
                train_loader=self.pretrain_loader
            else:
                train_loader=self.train_loader
            max_iterations = self.max_iterations

        global_iteration = 0
        for epoch in range(epochs):
            train_loss = []
            for iteration, (batch_x, _,batch_y,_) in enumerate(train_loader): # batch_x: tensor (b, seq_len, c)   batch_x: (b,pred_len,1)
                optimizer.zero_grad()
                loss = self.model.compute_loss(batch_x.to(self.device), batch_y.to(self.device),self.plot_dir,epoch)
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
                    early_stopping(recent_avg_loss, self.model, model_path,model_name, is_iteration=True, current_iteration=global_iteration)
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

            if self.args.plot:
                self._plot_losses(losses, 'train_stage1')
            
            print(f"Training time in stage1 until now: {time.time() - t:.2f}s")
           
            if early_stopping.early_stop : 
                break

            early_stopping(train_loss, self.model, model_path,model_name,is_iteration=False,current_iteration=global_iteration)
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
        model_path = os.path.join(self.folder_path, 'finetuned_ELFNet_family')
        model_name = f"{self.args.model_used}_{os.path.splitext(self.args.data_path.split('/')[-1])[0]}"
        full_model_path = os.path.join(model_path,f"{model_name}.pth")

        self.early_stopping = IterationEarlyStopping(patience_epochs=self.args.patience_epochs,patience_iterations=self.args.patience_iterations,min_iterations=self.args.min_iterations,verbose=True,delta=self.args.improved_delta)
        early_stopping = self.early_stopping
 
        t = time.time()
        if self.args.model_path is not None:
          print('Loading trained saved model to finetune')
          self.model.load_state_dict(torch.load(os.path.join(self.args.model_path ,'model.pth')))
        
        # 冻结 必要的层，冻结的层和预训练阶段二一致
        print('=============Starting to train ELFNet: stage2==============')
        self.model.train() 
        
        # 冻结 必要的层
        self.model.stage2 = True
        
        losses = []
        
        if self.args.pretrain_mode == 'one2many':
            train_loader=self.finetune_loader
        else:
            train_loader=self.train_loader

        global_iteration = 0
        for epoch in range(epochs):
            train_loss = []
            for iteration, (batch_x, _,batch_y,_) in enumerate(train_loader): 
                optimizer.zero_grad()

                batch_x = batch_x.transpose(1, 2)
                batch_x = batch_x.float().to(self.device)
                batch_y = (batch_y.float().to(self.device))
                outputs = self.model(batch_x)
                
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
                    early_stopping(recent_avg_loss, self.model,model_path,model_name, is_iteration=True, current_iteration=global_iteration)
                    if early_stopping.is_loss_stable(threshold=0.001): # 额外检查损失是否稳定
                        print(f"损失已趋于稳定，考虑早停")
                        early_stopping.early_stop = True
                    if early_stopping.early_stop:
                        print("iteration级别训练早停")
                        break

                if (iteration+1) % self.args.log_interval==0:
                    print(f"Iter: {iteration+1}, Train Loss in Stage2: {loss:.7f}")
            
            if self.args.plot:    
                self._plot_losses(losses,  'train_stage2')

            train_loss = np.average(train_loss)
            print(f"Epoch: {epoch+1}, Train Loss in Stage2: {train_loss:.7f}")

            vali_data, vali_loader = self._get_data(flag='val')
            vali_loss = self.vali(vali_loader)
            print(f"Vali Loss: {vali_loss:.7f}")
            print(f"Training Time in Stage2 until now: {time.time() - t:.2f}s")
            
            early_stopping(vali_loss, self.model, model_path,model_name,is_iteration=False,current_iteration=global_iteration)
            if early_stopping.early_stop :
                break

            adjust_learning_rate(optimizer, epoch + 1, self.args)
            
        training_time = time.time() - t
        print(f"Total training time in stage2: {training_time:.2f}s")
        
        return training_time,full_model_path
        
        
    def vali(self, vali_loader):
        total_loss = []
        with torch.no_grad():
            for batch_x, batch_x_mark,batch_y,batch_y_mark in vali_loader:
                batch_x = (batch_x.float().to(self.device)).transpose(1, 2) # (b,c,seq_len)
                batch_y = batch_y.float().to(self.device)
                batch_y = batch_y[:, -self.args.pred_len:, :].to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                if self.args.model_used in ['ELFNet','ELFNet_depthwise','ELFNet_no_disentanglement','ELFNet_no_contrastive','ELFNet_no_contrastive','ELFNet_dilution'] :
                    if self.finetune_model is None: # single模式或者'ELFNet_no_contrastive'
                        self.model.eval()
                        outputs = self.model(batch_x)
                        self.model.train()
                    else:
                        self.finetune_model.eval()
                        outputs = self.finetune_model(batch_x)
                        self.finetune_model.train()
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


    def test(self,full_model_path, setting, test):
        test_data, test_loader = self._get_data(flag='test')
        if test:
            print('Loading model')
            if self.args.model_used in ['ELFNet','ELFNet_depthwise','ELFNet_no_disentanglement','ELFNet_no_contrastive','ELFNet_no_contrastive','ELFNet_dilution'] :
                if self.finetune_model is None:
                    self.model.load_state_dict(torch.load(full_model_path))
                else:
                    self.finetune_model.load_state_dict(torch.load(full_model_path))
            else:
                self.CompareModel.load_state_dict(torch.load(full_model_path))
        preds, trues = [], []
        
        # derive folder_path from provided model_path (use model file name as prefix + "_test_results")
        model_dir = os.path.dirname(os.path.abspath(full_model_path))
        model_name = os.path.splitext(os.path.basename(full_model_path))[0]
        folder_path = os.path.join(model_dir, f"{model_name}_test_results")
        
        if not os.path.exists(folder_path):
            os.makedirs(folder_path, exist_ok=True)
        with torch.no_grad():
            for i,(batch_x, batch_x_mark,  batch_y,batch_y_mark )in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device).transpose(1, 2) 
                batch_y = batch_y.float().to(self.device) 
                batch_y = batch_y[:, -self.args.pred_len:, :].to(self.device)
                if self.args.model_used in ['ELFNet','ELFNet_depthwise','ELFNet_no_disentanglement','ELFNet_no_contrastive','ELFNet_no_contrastive','ELFNet_dilution'] :
                    if self.finetune_model is None:
                        self.model.eval()
                        outputs = self.model(batch_x)
                    else:
                        self.finetune_model.eval()
                        outputs = self.finetune_model(batch_x)   
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
        plt.savefig(os.path.join(self.plot_dir, f'losses.png'))
        plt.close()

    def _get_default_groups(self,input_channels):
        pass
