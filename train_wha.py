import torch
import torch.nn as nn
import os
import numpy as np
import loss_wha
import cv2
import func_utils
from tqdm import tqdm
import pdb
import json
from torch.utils.tensorboard import SummaryWriter

def collater(data):
    out_data_dict_cp = {}
    for name in data[0][0]:
        out_data_dict_cp[name] = []
    for sample in data:
        # print(sample)
        for name in sample[0]:
            out_data_dict_cp[name].append(torch.from_numpy(sample[0][name]))
    for name in out_data_dict_cp:
        out_data_dict_cp[name] = torch.stack(out_data_dict_cp[name], dim=0)
        
    out_data_dict_pc = {}
    for name in data[0][1]:
        out_data_dict_pc[name] = []
    for sample in data:
        for name in sample[1]:
            out_data_dict_pc[name].append(torch.from_numpy(sample[1][name]))
    for name in out_data_dict_pc:
        out_data_dict_pc[name] = torch.stack(out_data_dict_pc[name], dim=0)
    out_data_dict = [out_data_dict_cp , out_data_dict_pc]
    return out_data_dict_cp , out_data_dict_pc

class TrainModule(object):
    def __init__(self, dataset, num_classes, model, decoder, down_ratio):
        torch.manual_seed(317)
        self.dataset = dataset
        self.dataset_phase = {'radiate': ['train'],
                              'hrsc': ['train', 'test']}
        self.num_classes = num_classes
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = model
        self.decoder = decoder
        self.down_ratio = down_ratio

    def save_model(self, path, epoch, model, optimizer1, optimizer2):
        if isinstance(model, torch.nn.DataParallel):
            state_dict = model.module.state_dict()
        else:
            state_dict = model.state_dict()
        torch.save({
            'epoch': epoch,
            'model_state_dict': state_dict,
            'optimizer1_state_dict': optimizer1.state_dict(),
            'optimizer2_state_dict': optimizer2.state_dict(),
            # 'loss': loss
        }, path)

    def load_model(self, model, optimizer1, optimizer2, resume, strict=True):
        checkpoint = torch.load(resume, map_location=lambda storage, loc: storage)
        print('loaded weights from {}, epoch {}'.format(resume, checkpoint['epoch']))
        state_dict_ = checkpoint['model_state_dict']
        state_dict = {}
        for k in state_dict_:
            if k.startswith('module') and not k.startswith('module_list'):
                state_dict[k[7:]] = state_dict_[k]
            else:
                state_dict[k] = state_dict_[k]
        model_state_dict = model.state_dict()
        if not strict:
            for k in state_dict:
                if k in model_state_dict:
                    if state_dict[k].shape != model_state_dict[k].shape:
                        print('Skip loading parameter {}, required shape{}, ' \
                              'loaded shape{}.'.format(k, model_state_dict[k].shape, state_dict[k].shape))
                        state_dict[k] = model_state_dict[k]
                else:
                    print('Drop parameter {}.'.format(k))
            for k in model_state_dict:
                if not (k in state_dict):
                    print('No param {}.'.format(k))
                    state_dict[k] = model_state_dict[k]
        model.load_state_dict(state_dict, strict=False)
        optimizer1.load_state_dict(checkpoint['optimizer1_state_dict'])
        optimizer2.load_state_dict(checkpoint['optimizer2_state_dict'])
        for state in optimizer1.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.cuda()
        for state in optimizer2.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.cuda()
        epoch = checkpoint['epoch']
        # loss = checkpoint['loss']
        return model, optimizer1, optimizer2, epoch

    def train_network(self, args):

        self.optimizer = torch.optim.Adam(self.model.parameters(), args.init_lr, weight_decay=0)
        self.optimizer_weight_decay = torch.optim.Adam(self.model.parameters(), args.init_lr, weight_decay=1e-2)
        # self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=5, gamma=1e-2)
        origin_save_path = os.path.join('weights',args.backbone+'_'+args.train_mode)
        save_path = os.path.join(origin_save_path + '_0')
        start_epoch = 1
        
        if not args.resume_train:
            exp_number = 1
            while os.path.exists(save_path):
                save_path = os.path.join(origin_save_path+'_{}'.format(exp_number))
                exp_number += 1
            os.makedirs(save_path, exist_ok=True)
        # add resume part for continuing training when break previously, 10-16-2020
        if args.resume_train:
            check_point = os.path.join(save_path, args.resume_train)
            self.model, self.optimizer, self.optimizer_weight_decay, start_epoch = self.load_model(self.model, 
                                                                                                    self.optimizer,
                                                                                                    self.optimizer_weight_decay, 
                                                                                                    check_point, 
                                                                                                    strict=True)
        # end
        self.save_exp_condition(save_path, args)
        
        
        if args.ngpus>1:
            if torch.cuda.device_count() > 1:
                print("Let's use", torch.cuda.device_count(), "GPUs!")
                # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
                self.model = nn.DataParallel(self.model)
        self.model.to(self.device)

        criterion = loss_wha.LossAll()
        print('Setting up data...')

        dataset_module = self.dataset[args.dataset]

        dsets = {x: dataset_module(data_dir=args.data_dir,
                                   train_mode=args.train_mode,
                                   phase=x,
                                   max_objs = args.max_objs,
                                   only_objects = args.only_objects,
                                   input_h=args.input_h,
                                   input_w=args.input_w,
                                   down_ratio=self.down_ratio)
                 for x in self.dataset_phase[args.dataset]}

        dsets_loader = {}
        dsets_loader['train'] = torch.utils.data.DataLoader(dsets['train'],
                                                           batch_size=args.batch_size,
                                                           shuffle=True,
                                                           num_workers=args.num_workers,
                                                           pin_memory=True,
                                                           drop_last=True,
                                                           collate_fn=collater)
        print('Starting training...')
        log_path = os.path.join(save_path, 'log')
        os.makedirs(log_path, exist_ok=True)
        writer = SummaryWriter(log_path)
        train_loss = []
        ap_list = []
        for epoch in range(start_epoch, args.num_epoch+1):
            print('-'*10)
            print('Epoch: {}/{} '.format(epoch, args.num_epoch))
            epoch_loss, epoch_loss_cp, epoch_hm_loss_cp, \
            epoch_wh_loss_cp, epoch_off_loss_cp, epoch_cls_theta_loss_cp, \
            epoch_loss_pc, epoch_hm_loss_pc, epoch_wh_loss_pc, \
            epoch_off_loss_pc, epoch_cls_theta_loss_pc = self.run_epoch(phase='train',
                                                                        data_loader=dsets_loader['train'],
                                                                        criterion=criterion,
                                                                        gradient_accumulation = args.grad_step)
            # train_loss.append(epoch_loss)
            # self.scheduler.step()

            # np.savetxt(os.path.join(save_path, 'train_loss.txt'), train_loss, fmt='%.6f')            
            if epoch % 5 == 0:
                self.save_model(os.path.join(save_path, 'model_{}.pth'.format(epoch)),
                                epoch,
                                self.model,
                                self.optimizer,
                                self.optimizer_weight_decay)

            if 'test' in self.dataset_phase[args.dataset] and epoch%5==0:
                mAP = self.dec_eval(args, dsets['test'])
                ap_list.append(mAP)
                np.savetxt(os.path.join(save_path, 'ap_list.txt'), ap_list, fmt='%.6f')

            self.save_model(os.path.join(save_path, 'model_last.pth'),
                            epoch,
                            self.model,
                            self.optimizer,
                            self.optimizer_weight_decay)
            writer.add_scalars("Loss", {'total': epoch_loss, 'cp': epoch_loss_cp, 'pc': epoch_loss_pc}, epoch)
            writer.add_scalars("hm_Loss", {'cp': epoch_hm_loss_cp, 'pc': epoch_hm_loss_pc}, epoch)
            writer.add_scalars("wh_Loss", {'cp': epoch_wh_loss_cp, 'pc': epoch_wh_loss_pc}, epoch)
            writer.add_scalars("off_Loss", {'cp': epoch_off_loss_cp, 'pc': epoch_off_loss_pc}, epoch)
            writer.add_scalars("angle_Loss", {'cp': epoch_cls_theta_loss_cp, 'pc': epoch_cls_theta_loss_pc}, epoch)
            writer.add_scalar("lr/train", self.optimizer.param_groups[0]['lr'], epoch)
            
    def run_epoch(self, phase, data_loader, criterion, gradient_accumulation = None):
        if phase == 'train':
            self.model.train()
        # else:
        #     self.model.eval()
            
        running_loss = 0.
        running_hm_loss_cp = 0.
        running_wh_loss_cp = 0.
        running_off_loss_cp = 0.
        running_cls_theta_loss_cp = 0.
        running_loss_cp = 0.
        
        running_hm_loss_pc = 0.
        running_wh_loss_pc = 0.
        running_off_loss_pc = 0.
        running_cls_theta_loss_pc = 0.
        running_loss_pc = 0.
        
        self.optimizer.zero_grad()
        self.optimizer_weight_decay.zero_grad()
        # for i, (data_dict_cp, data_dict_pc) in enumerate(tqdm(data_loader)):
        for i, (data_dict_cp, data_dict_pc) in enumerate(tqdm(data_loader)):
            # pdb.set_trace()
            for name in data_dict_cp:
                data_dict_cp[name] = data_dict_cp[name].to(device=self.device, non_blocking=True)
            for name in data_dict_pc:
                data_dict_pc[name] = data_dict_pc[name].to(device=self.device, non_blocking=True)    
           
            # if phase == 'train':
            
            with torch.enable_grad():
                pr_decs_cp, pr_decs_pc = self.model(data_dict_cp['input'], data_dict_pc['input'])
                # pr_decs_pc = self.model(data_dict_pc['input'])
                # pdb.set_trace()
                # print(data_dict_cp['reg'][0][0])
                # print(data_dict_pc['reg'][0][0])
                hm_loss_cp, wh_loss_cp, off_loss_cp, cls_theta_loss_cp = criterion(pr_decs_cp, data_dict_cp)
                hm_loss_pc, wh_loss_pc, off_loss_pc, cls_theta_loss_pc = criterion(pr_decs_pc, data_dict_pc)
                # print(hm_loss_cp.data, wh_loss_cp, off_loss_cp, cls_theta_loss_cp)
                # print(hm_loss_pc, wh_loss_pc, off_loss_pc, cls_theta_loss_pc)
                loss_cp = hm_loss_cp + wh_loss_cp + off_loss_cp + cls_theta_loss_cp
                loss_pc = hm_loss_pc + wh_loss_pc + off_loss_pc + cls_theta_loss_pc
                loss = loss_cp + loss_pc
                 
                if gradient_accumulation is None:
                    self.optimizer.zero_grad()
                    self.optimizer_weight_decay.zero_grad()
                    loss.backward()
                    self.optimizer_weight_decay.step()
                    
                    # if (epoch+1) % 5 == 0:
                    #     self.optimizer_weight_decay.step()
                    # else:
                    #     self.optimizer.step()
                                
                else:
                    loss = loss / gradient_accumulation
                    loss.backward()
                    if (i+1) % gradient_accumulation == 0:
                        # if (epoch+1) % 5 == 0:
                        #     self.optimizer_weight_decay.step()
                        # else:
                        #     self.optimizer.step()
                        self.optimizer_weight_decay.step()
                        self.model.zero_grad()
                    
            # else:
            #     with torch.no_grad():
            #         pr_decs = self.model(data_dict['input'])
            #         hm_loss, wh_loss, off_loss, cls_theta_loss = criterion(pr_decs, data_dict)
            #         loss = hm_loss + wh_loss + off_loss + cls_theta_loss
            running_loss += loss.item()

            running_loss_cp += loss_cp.item()
            running_hm_loss_cp += hm_loss_cp.item()
            running_wh_loss_cp += wh_loss_cp.item()
            running_off_loss_cp += off_loss_cp.item()
            running_cls_theta_loss_cp += cls_theta_loss_cp.item()
            
            running_loss_pc += loss_pc.item()
            running_hm_loss_pc += hm_loss_pc.item()
            running_wh_loss_pc += wh_loss_pc.item()
            running_off_loss_pc += off_loss_pc.item()
            running_cls_theta_loss_pc += cls_theta_loss_pc.item()
            
        epoch_loss = running_loss / len(data_loader)    
        
        epoch_loss_cp = running_loss_cp / len(data_loader)  
        epoch_hm_loss_cp = running_hm_loss_cp / len(data_loader)
        epoch_wh_loss_cp = running_wh_loss_cp / len(data_loader)
        epoch_off_loss_cp = running_off_loss_cp / len(data_loader)
        epoch_cls_theta_loss_cp = running_cls_theta_loss_cp / len(data_loader)
        
        epoch_loss_pc = running_loss_pc / len(data_loader)  
        epoch_hm_loss_pc = running_hm_loss_pc / len(data_loader)
        epoch_wh_loss_pc = running_wh_loss_pc / len(data_loader)
        epoch_off_loss_pc = running_off_loss_pc / len(data_loader)
        epoch_cls_theta_loss_pc = running_cls_theta_loss_pc / len(data_loader)

        # print('{} loss: {}'.format(phase, epoch_loss))
        return epoch_loss ,epoch_loss_cp, epoch_hm_loss_cp, epoch_wh_loss_cp, epoch_off_loss_cp, epoch_cls_theta_loss_cp, \
               epoch_loss_pc, epoch_hm_loss_pc, epoch_wh_loss_pc, epoch_off_loss_pc, epoch_cls_theta_loss_pc


    def dec_eval(self, args, dsets):
        result_path = 'result_'+args.dataset
        if not os.path.exists(result_path):
            os.mkdir(result_path)

        self.model.eval()
        func_utils.write_results(args,
                                 self.model,dsets,
                                 self.down_ratio,
                                 self.device,
                                 self.decoder,
                                 result_path)
        ap = dsets.dec_evaluation(result_path)
        return ap
    
    def save_exp_condition(self, save_path, args):
        file_path = os.path.join(save_path, 'param.json')
        data = {}
        data['backbone'] = args.backbone
        data['train_mode'] = args.train_mode
        data['freeze'] = args.freeze
        data['n_layer'] = args.n_layer
        data['d_model'] = args.d_model
        data['h'] = args.h
        data['d_position'] = args.d_p
        data['topK'] = args.topK
        data['max_objs'] = args.max_objs
        data['lr'] = args.init_lr
        data['batch size'] = args.batch_size
        data['gard_step'] = args.grad_step
        
        with open(file_path, 'w') as of:
            json.dump(data, of, indent=2)