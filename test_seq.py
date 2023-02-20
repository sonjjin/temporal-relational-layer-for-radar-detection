import argparse
import train_wha
import test
import eval
from datasets.dataset_radiate_temporal import RADIATE
from models import ctrbox_net
import decoder
import os
import numpy as np



def parse_args():
    parser = argparse.ArgumentParser(description='BBAVectors Implementation')
    parser.add_argument('--num_epoch', type=int, default=1, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=1, help='Number of batch size')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers')
    parser.add_argument('--init_lr', type=float, default=5e-4, help='Initial learning rate')
    parser.add_argument('--input_h', type=int, default=256, help='Resized image height')
    parser.add_argument('--input_w', type=int, default=256, help='Resized image width')
    parser.add_argument('--max_objs', type=int, default=500, help='Maximum of objects')
    parser.add_argument('--conf_thresh', type=float, default=0.1, help='Confidence threshold, 0.1 for general evaluation')
    parser.add_argument('--ngpus', type=int, default=1, help='Number of gpus, ngpus>1 for multigpu')
    parser.add_argument('--device', type=int, default=0, help='Device for training')
    parser.add_argument('--resume_train', type=str, default='', help='Weights resumed in training')
    parser.add_argument('--resume', type=str, default='model_5.pth', help='Weights resumed in testing and evaluation')
    parser.add_argument('--dataset', type=str, default='radiate', help='Name of dataset')
    parser.add_argument('--data_dir', type=str, default='/workspace/dataset/radiate_centercrop', help='Data directory')
    parser.add_argument('--phase', type=str, default='train', help='Phase choice= {train, test, eval}')
    parser.add_argument('--only_objects', type=bool, default=True, help='Whether to use only datasets with objects')
    parser.add_argument('--backbone', type=str, default='ResNet18', help='network backbone')
    parser.add_argument('--train_mode', type=str, default='train_good_and_bad_weather', help='train_mode')
    parser.add_argument('--grad_step', type=int, default=None, help='gradient accumulation step')
    parser.add_argument('--exp', type=str, default='0', help='experiment number')
    parser.add_argument('--topK', type=int, default=2, help='the number of K potential objects features')
    parser.add_argument('--n_layer', type=int, default=2, help='number of TRL')
    parser.add_argument('--d_model', type=int, default=512, help='dimension of attention')
    parser.add_argument('--h', type=int, default=8, help='number of attention head')
    parser.add_argument('-d_p', type=int, default=64, help='dimension of position encoding')
    parser.add_argument('--freeze', type=bool, default=False, help='weather freeze parameter of backbone')
    parser.add_argument('--wh_channels', type=int, default=8, help='Number of channels for the vectors (4x2)')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"]= str(args.device)
    # dataset = {'dota': DOTA, 'hrsc': HRSC}
    # num_classes = {'dota': 15, 'hrsc': 1}
    dataset = {'radiate': RADIATE}
    num_classes = {'radiate': 1}

    heads = {'hm': num_classes[args.dataset],
             'wh': 2,
             'reg': 2,
             'angle': 2
             }
    down_ratio = 4

    model = ctrbox_net.CTRBOX(heads=heads,
                              backbone=args.backbone,
                              pretrained=True,
                              freeze=args.freeze,
                              down_ratio=down_ratio,
                              final_kernel=1,
                              head_conv=256,
                              topK=args.topK,
                              n_layer = args.n_layer,
                              d_model = args.d_model,
                              h = args.h,
                              d_p = args.d_p)

    decoder = decoder.DecDecoder(max_objs=args.max_objs,
                                 conf_thresh=args.conf_thresh,
                                 num_classes=num_classes[args.dataset])
    

    ctrbox_obj = test.TestModule(dataset=dataset, num_classes=num_classes, model=model, decoder=decoder)
    ctrbox_obj.test(args, down_ratio=down_ratio)