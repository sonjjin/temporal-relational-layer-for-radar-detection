import torch
import os
import func_utils


class EvalModule(object):
    def __init__(self, dataset, num_classes, model, decoder):
        torch.manual_seed(317)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.dataset = dataset
        self.num_classes = num_classes
        self.model = model
        self.decoder = decoder


    def load_model(self, model, resume):
        checkpoint = torch.load(resume, map_location=lambda storage, loc: storage)
        print('loaded weights from {}, epoch {}'.format(resume, checkpoint['epoch']))
        state_dict_ = checkpoint['model_state_dict']
        model.load_state_dict(state_dict_, strict=False)
        return model

    def evaluation(self, args, down_ratio):
        weights_path = os.path.join('weights', args.backbone+'_'+args.train_mode+'_'+args.exp, args.resume)
        # self.model = self.load_model(self.model, os.path.join(save_path, args.resume))
        
        self.model = self.load_model(self.model, weights_path)
        
        self.model = self.model.to(self.device)
        self.model.eval()

        result_path = os.path.join('result', args.backbone+'_'+args.train_mode+'_'+args.exp, args.resume[:-4])
        if not os.path.exists(result_path):
            os.makedirs(result_path)

        dataset_module = self.dataset[args.dataset]
        dsets = dataset_module(data_dir=args.data_dir,
                               train_mode = args.train_mode,
                               phase='test',
                               max_objs = args.max_objs,
                               only_objects = args.only_objects,
                               input_h=args.input_h,
                               input_w=args.input_w,
                               down_ratio=down_ratio)       
        func_utils.write_results(args,
                                 self.model,
                                 dsets,
                                 down_ratio,
                                 self.device,
                                 self.decoder,
                                 result_path,
                                 print_ps=False)

        ap = dsets.dec_evaluation(result_path)
        return ap
        # if args.dataset == 'dota':
        #     merge_path = 'merge_'+args.dataset
        #     if not os.path.exists(merge_path):
        #         os.mkdir(merge_path)
        #     dsets.merge_crop_image_results(result_path, merge_path)
        #     return None
        # else:
        #     ap = dsets.dec_evaluation(result_path)
        #     return ap