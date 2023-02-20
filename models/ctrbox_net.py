import torch.nn as nn
import numpy as np
import torch
from .model_parts import CombinationModule
from . import resnet
from .TRL import TemporalRelationalLayer
from .positional_encoding import *
import copy

import pdb
class CTRBOX(nn.Module):
    def __init__(self, heads, backbone, freeze, pretrained, down_ratio, final_kernel, head_conv, topK, n_layer, d_model, h, d_p):
        '''
        topK: number of interesting
        n_layer: number of TRL
        d_model: dimension of attention
        h: number of head
        d_p: dimension of position encoding
        '''
        super(CTRBOX, self).__init__()
        assert down_ratio in [2, 4, 8, 16]
        self.l1 = int(np.log2(down_ratio))
        
        self.freeze = freeze
        self.base_network = None
        self.dec_c2 = None
        self.dec_c3 = None
        self.dec_c4 = None
        self.channels = None
        self._get_backbone(backbone, pretrained)
        

        self.n_layer = n_layer
        self.topK = topK
        self.dp = d_p       
        self.position_encoding = PositionalEncoding(self.dp, self.channels[self.l1], self.channels[self.l1]).cuda()
         
        # self.TRLlayer = TemporalRelationalLayer(d_model=d_model, h=h,
        #                     qk_fc=nn.Linear(self.channels[self.l1]+d_p, d_model),
        #                     v_fc=nn.Linear(self.channels[self.l1], d_model), 
        #                     out_fc=nn.Linear(d_model, 64),
        #                     dr_rate=0)
        
        self.TRLlayers = nn.ModuleList([copy.deepcopy(TemporalRelationalLayer(d_model=d_model, h=h,
                            qk_fc=nn.Linear(self.channels[self.l1]+d_p, d_model),
                            v_fc=nn.Linear(self.channels[self.l1], d_model), 
                            out_fc=nn.Linear(d_model, self.channels[self.l1]),
                            topK = self.topK,
                            dr_rate=0)) for _ in range(self.n_layer)])
        
        self.heads = heads
        # pdb.set_trace()
        for head in self.heads:
            classes = self.heads[head]
            if head == 'wh':
                fc = nn.Sequential(nn.Conv2d(self.channels[self.l1], head_conv, kernel_size=3, padding=1, bias=True),
                                   nn.BatchNorm2d(head_conv),   # BN not used in the paper, but would help stable training
                                   nn.ReLU(inplace=True),
                                   nn.Conv2d(head_conv, classes, kernel_size=3, padding=1, bias=True))
            else:
                fc = nn.Sequential(nn.Conv2d(self.channels[self.l1], head_conv, kernel_size=3, padding=1, bias=True),
                                   nn.BatchNorm2d(head_conv),   # BN not used in the paper, but would help stable training
                                   nn.ReLU(inplace=True),
                                   nn.Conv2d(head_conv, classes, kernel_size=final_kernel, stride=1, padding=final_kernel // 2, bias=True))
            if 'hm' in head:
                fc[-1].bias.data.fill_(-2.19)
            else:
                self.fill_fc_weights(fc)

            self.__setattr__(head, fc)


    def fill_fc_weights(self, m):
        if isinstance(m, nn.Conv2d):
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x_cp, x_pc):
        
        x_cp = self.base_network(x_cp)
        c4_combine_cp = self.dec_c4(x_cp[-1], x_cp[-2])
        c3_combine_cp = self.dec_c3(c4_combine_cp, x_cp[-3])
        c2_combine_cp = self.dec_c2(c3_combine_cp, x_cp[-4])
        # pdb.set_trace()
            
        x_pc = self.base_network(x_pc)
        c4_combine_pc = self.dec_c4(x_pc[-1], x_pc[-2])
        c3_combine_pc = self.dec_c3(c4_combine_pc, x_pc[-3])
        c2_combine_pc = self.dec_c2(c3_combine_pc, x_pc[-4])
        # pdb.set_trace()
        
        # run TRL
        
        # todo
        H_cp, position_cp, indexs_feature_cp, indexs_feature_pc = self._get_TRL_input(c2_combine_cp, c2_combine_pc) # (B, 2k, C), (B, 2k, d_p)
        H_cp_pos = torch.cat((H_cp, position_cp), dim=2) # (B, 2k, C+d_p)
        for layer in self.TRLlayers:
            H_cp = layer(H_cp_pos, H_cp_pos, H_cp)            
            H_cp_pos = torch.cat((H_cp, position_cp), dim=2) # (B, 2k, C+d_p)
        
        c2_combine_cp, c2_combine_pc = self._get_feature(H_cp, c2_combine_cp, c2_combine_pc, indexs_feature_cp, indexs_feature_pc)
        
        dec_dict_cp = {}
        for head in self.heads:
            dec_dict_cp[head] = self.__getattr__(head)(c2_combine_cp)
            if 'hm' in head or 'cls' in head:
                dec_dict_cp[head] = torch.sigmoid(dec_dict_cp[head])
        
        dec_dict_pc = {}
        for head in self.heads:
            dec_dict_pc[head] = self.__getattr__(head)(c2_combine_pc)
            if 'hm' in head or 'cls' in head:
                dec_dict_pc[head] = torch.sigmoid(dec_dict_pc[head])
        
        
        return dec_dict_cp, dec_dict_pc 

    def _get_backbone(self, backbone, pretrained):
        if backbone == 'ResNet18':
            self.channels = [3, 64, 64, 128, 256, 512]
            self.base_network = resnet.resnet18(pretrained=pretrained)
            self.dec_c2 = CombinationModule(128, 64, batch_norm=True)
            self.dec_c3 = CombinationModule(256, 128, batch_norm=True)
            self.dec_c4 = CombinationModule(512, 256, batch_norm=True)
        elif backbone == 'ResNet34':
            self.channels = [3, 64, 64, 128, 256, 512]
            self.base_network = resnet.resnet34(pretrained=pretrained)
            self.dec_c2 = CombinationModule(128, 64, batch_norm=True)
            self.dec_c3 = CombinationModule(256, 128, batch_norm=True)
            self.dec_c4 = CombinationModule(512, 256, batch_norm=True)
        elif backbone == 'ResNet101':
            self.channels = [3, 64, 256, 512, 1024, 2048]
            self.base_network = resnet.resnet101(pretrained=pretrained)    
            self.dec_c2 = CombinationModule(512, 256, batch_norm=True)
            self.dec_c3 = CombinationModule(1024, 512, batch_norm=True)
            self.dec_c4 = CombinationModule(2048, 1024, batch_norm=True)

        first_layer = self.base_network.conv1
        assert isinstance(first_layer, nn.Conv2d), f'Change of input channels only supported with Conv2d, found {first_layer.__class__.__name__}'
        assert getattr(first_layer, 'in_channels') == 3, f'Unexpected number of input channels, found {getattr(first_layer, "in_channels")} while expecting 3'
        params = {attr:getattr(first_layer, attr) for attr in 'out_channels kernel_size stride padding dilation groups padding_mode'.split()}
        params['bias'] = getattr(first_layer, 'bias') is not None
        params['in_channels'] = 2
        new_layer = nn.Conv2d(**params)
        
        weight_mean = first_layer.weight.data.mean(dim=1, keepdim=True)
        new_layer.weight.data = weight_mean.expand_as(new_layer.weight.data).clone()
        self.base_network.conv1 = new_layer

        if self.freeze:
            for child in self.base_network.children():
                for param in child.parameters():
                    param.requires_grad = False

        # return channels
        
    def _get_TRL_input(self, featuremap_cp, featuremap_pc):

        B = featuremap_cp.size(0)
        position_encoding_flatten = self.position_encoding.repeat(B,1,1,1).flatten(start_dim=-2)
        
        pred_hm_cp = self.__getattr__('hm')(featuremap_cp) # (B, 1, H/s, W/s)
        pred_hm_cp = torch.sigmoid(pred_hm_cp)
        
        pred_hm_pc = self.__getattr__('hm')(featuremap_pc) # (B, 1, H/s, W/s)
        pred_hm_pc = torch.sigmoid(pred_hm_pc)
        
        topK_cp = pred_hm_cp.squeeze(1) # (B, H, W)
        topK_cp = topK_cp.flatten(start_dim=-2) # (B, H x W)
        _, indexs_cp = torch.topk(topK_cp, self.topK, dim=-1) # (B, k)


        topK_pc = pred_hm_pc.squeeze(1) # (B, H, W)
        topK_pc = topK_pc.flatten(start_dim=-2) # (B, H x W)
        _, indexs_pc = torch.topk(topK_pc, self.topK, dim=-1) # (B, k)

        featuremap_flat_cp = featuremap_cp.flatten(start_dim=-2) # (B, C, H x W)
        featuremap_flat_pc = featuremap_pc.flatten(start_dim=-2) # (B, C, H x W)

        indexs_feature_cp = indexs_cp[:, None, :].repeat(1, featuremap_flat_cp.size(1), 1) # (B, C, H x W)
        indexs_feature_pc = indexs_pc[:, None, :].repeat(1, featuremap_flat_pc.size(1), 1) # (B, C, H x W)
        indexs_position_cp = indexs_cp[:, None, :].repeat(1, self.dp, 1) # (B, C, H x W)
        indexs_position_pc = indexs_pc[:, None, :].repeat(1, self.dp, 1) # (B, C, H x W)

        H_c = featuremap_flat_cp.gather(2, indexs_feature_cp) # (B, C, k)
        H_p = featuremap_flat_pc.gather(2, indexs_feature_pc) # (B, C, k)
        H_cp = torch.cat((H_c, H_p), dim=2).transpose(-2,-1) # (B, 2k, C)

        position_c = position_encoding_flatten.gather(2, indexs_position_cp)
        position_p = position_encoding_flatten.gather(2, indexs_position_pc)
        position_cp = torch.cat((position_c, position_p), dim=2).transpose(-2,-1)

        return H_cp, position_cp, indexs_feature_cp, indexs_feature_pc
    
    def _get_feature(self, H_cp, c2_combine_cp, c2_combine_pc, indexs_feature_cp, indexs_feature_pc):
        
        B, C, H, W = c2_combine_cp.shape
        H_cp_T = H_cp.transpose(-2,-1)
        
        featuremap_flat_cp = c2_combine_cp.flatten(start_dim=-2) # (B, C, H x W)
        featuremap_flat_pc = c2_combine_pc.flatten(start_dim=-2) # (B, C, H x W)
        
        featuremap_flat_cp = featuremap_flat_cp.scatter(2, indexs_feature_cp, H_cp_T[:,:,:self.topK])
        featuremap_flat_pc = featuremap_flat_pc.scatter(2, indexs_feature_pc, H_cp_T[:,:,self.topK:])
                
        featuremap_cp = featuremap_flat_cp.view(B, C, H, W)
        featuremap_pc = featuremap_flat_pc.view(B, C, H, W)
        
        return featuremap_cp, featuremap_pc