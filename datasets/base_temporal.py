import torch.utils.data as data
import cv2
import torch
import numpy as np
import math
from .draw_gaussian import draw_umich_gaussian, gaussian_radius
from .transforms_radiate import random_flip, load_affine_matrix, random_crop_info, ex_box_jaccard
from . import data_augment_temporal
import pdb
import copy

class BaseDataset(data.Dataset):
    def __init__(self, data_dir, train_mode, phase, max_objs, only_objects=True, input_h=None, input_w=None, down_ratio=None):
        super(BaseDataset, self).__init__()
        self.data_dir = data_dir
        self.train_mode = train_mode
        self.phase = phase
        self.input_h = input_h
        self.input_w = input_w
        self.down_ratio = down_ratio
        self.img_ids = None
        self.num_classes = None
        self.img_mean = None
        self.img_std = None
        self.max_objs = max_objs
        self.only_objects = only_objects
        self.image_distort =  data_augment_temporal.PhotometricDistort()

    def load_img_ids(self):
        """
        Definition: generate self.img_ids
        Usage: index the image properties (e.g. image name) for training, testing and evaluation
        Format: self.img_ids = [list]
        Return: self.img_ids
        """
        return None

    def load_image(self, index):
        """
        Definition: read images online
        Input: index, the index of the image in self.img_ids
        Return: image with H x W x 3 format
        """
        return None

    def load_annoFolder(self, img_id):
        """
        Return: the path of annotation
        Note: You may not need this function
        """
        return None

    def load_annotation(self, index):
        """
        Return: dictionary of {'pts': float np array of [bl, tl, tr, br], 
                                'cat': int np array of class_index}
        Explaination:
                bl: bottom left point of the bounding box, format [x, y]
                tl: top left point of the bounding box, format [x, y]
                tr: top right point of the bounding box, format [x, y]
                br: bottom right point of the bounding box, format [x, y]
                class_index: the category index in self.category
                    example: self.category = ['ship]
                             class_index of ship = 0
        """
        return None

    def dec_evaluation(self, result_path):
        return None

    def data_transform(self, img_cp, img_pc, annotation_cp, annotation_pc, id):
        # only do random_flip augmentation to original images
        crop_size = None
        crop_center = None
        crop_size, crop_center = random_crop_info(h=img_cp.shape[0], w=img_cp.shape[1])
        img_cp, img_pc, annotation_cp['pts'], annotation_pc['pts'], crop_center = random_flip(img_cp, img_pc, annotation_cp['pts'], annotation_pc['pts'], crop_center)
        if crop_center is None:
            crop_center = np.asarray([float(img_cp.shape[1])/2, float(img_cp.shape[0])/2], dtype=np.float32)
        if crop_size is None:
            crop_size = [max(img_cp.shape[1], img_cp.shape[0]), max(img_cp.shape[1], img_cp.shape[0])]  # init
        M = load_affine_matrix(crop_center=crop_center,
                               crop_size=crop_size,
                               dst_size=(self.input_w, self.input_h),
                               inverse=False,
                               rotation=False)
        img_cp = cv2.warpAffine(src=img_cp, M=M, dsize=(self.input_w, self.input_h), flags=cv2.INTER_LINEAR)
        img_pc = cv2.warpAffine(src=img_pc, M=M, dsize=(self.input_w, self.input_h), flags=cv2.INTER_LINEAR)
        
        centerpoint_cp = None
        centerpoint_pc = None
        if len(annotation_cp['pts']):
            centerpoint_cp = annotation_cp['pts'][:,:2]
        if len(annotation_pc['pts']):
            centerpoint_pc = annotation_pc['pts'][:,:2]

        if centerpoint_cp is not None:
            centerpoint_cp = np.concatenate([centerpoint_cp, np.ones((centerpoint_cp.shape[0], 1))], axis=1)
            centerpoint_cp = np.matmul(centerpoint_cp, np.transpose(M))
            annotation_cp['pts'][:,:2] = np.asarray(centerpoint_cp, np.float32)

        if centerpoint_pc is not None:
            centerpoint_pc = np.concatenate([centerpoint_pc, np.ones((centerpoint_pc.shape[0], 1))], axis=1)
            centerpoint_pc = np.matmul(centerpoint_pc, np.transpose(M))
            annotation_pc['pts'][:,:2] = np.asarray(centerpoint_pc, np.float32)

        out_annotations_cp = {}
        size_thresh = 3
        out_rects_cp = []
        out_cat_cp = []
        for pt, cat in zip(annotation_cp['pts'] , annotation_cp['cat']):
            if pt[2]<size_thresh and pt[3]<size_thresh:
                continue
            if pt[0] < 0 or pt[1] < 0:
                continue
            if pt[0] > img_cp.shape[1] or pt[1] > img_cp.shape[0]:
                continue
            out_rects_cp.append([pt[0], pt[1], pt[2], pt[3], pt[4]])
            out_cat_cp.append(cat)
        out_annotations_cp['rect'] = np.asarray(out_rects_cp, np.float32)
        out_annotations_cp['cat'] = np.asarray(out_cat_cp, np.uint8)
        
        
        out_annotations_pc = {}
        size_thresh = 3
        out_rects_pc = []
        out_cat_pc = []
        for pt, cat in zip(annotation_pc['pts'] , annotation_pc['cat']):
            if pt[2]<size_thresh and pt[3]<size_thresh:
                continue
            if pt[0] < 0 or pt[1] < 0:
                continue
            if pt[0] > img_pc.shape[1] or pt[1] > img_pc.shape[0]:
                continue
            out_rects_pc.append([pt[0], pt[1], pt[2], pt[3], pt[4]])
            out_cat_pc.append(cat)
        out_annotations_pc['rect'] = np.asarray(out_rects_pc, np.float32)
        out_annotations_pc['cat'] = np.asarray(out_cat_pc, np.uint8)
        
        return img_cp, img_pc, out_annotations_cp, out_annotations_pc

    def __len__(self):
        return len(self.img_ids)

    def processing_test(self, image, input_h, input_w):
        image = cv2.resize(image, (input_w, input_h))
        if self.img_mean is None:
            out_image = image.astype(np.float32) / 255.
            out_image = out_image - 0.5
            out_image = out_image.transpose(2, 0, 1).reshape(1, 2, input_h, input_w)
            out_image = torch.from_numpy(out_image)

        else:
            out_image = image.astype(np.float32) - self.img_mean
            out_image = out_image / self.img_std
            out_image = out_image.transpose(2, 0, 1).reshape(1, 2, input_h, input_w)
            out_image = torch.from_numpy(out_image)
        
        return out_image

    def cal_bbox_wh(self, pts_4):
        x1 = np.min(pts_4[:,0])
        x2 = np.max(pts_4[:,0])
        y1 = np.min(pts_4[:,1])
        y2 = np.max(pts_4[:,1])
        return x2-x1, y2-y1


    def cal_bbox_pts(self, pts_4):
        x1 = np.min(pts_4[:,0])
        x2 = np.max(pts_4[:,0])
        y1 = np.min(pts_4[:,1])
        y2 = np.max(pts_4[:,1])
        bl = [x1, y2]
        tl = [x1, y1]
        tr = [x2, y1]
        br = [x2, y2]
        return np.asarray([bl, tl, tr, br], np.float32)

    def reorder_pts(self, tt, rr, bb, ll):
        pts = np.asarray([tt,rr,bb,ll],np.float32)
        l_ind = np.argmin(pts[:,0])
        r_ind = np.argmax(pts[:,0])
        t_ind = np.argmin(pts[:,1])
        b_ind = np.argmax(pts[:,1])
        tt_new = pts[t_ind,:]
        rr_new = pts[r_ind,:]
        bb_new = pts[b_ind,:]
        ll_new = pts[l_ind,:]
        return tt_new,rr_new,bb_new,ll_new


    def generate_ground_truth(self, image, annotation):
        image = np.asarray(np.clip(image, a_min=0., a_max=255.), np.float32)
        image = self.image_distort(np.asarray(image, np.float32))
        image = np.asarray(np.clip(image, a_min=0., a_max=255.), np.float32)
        if self.img_mean is None:
            image = np.transpose(image / 255. - 0.5, (2, 0, 1))
        else:
            image = np.transpose((image - self.img_mean)/self.img_std, (2, 0, 1))

        image_h = self.input_h // self.down_ratio
        image_w = self.input_w // self.down_ratio

        hm = np.zeros((self.num_classes, image_h, image_w), dtype=np.float32)
        wh = np.zeros((self.max_objs, 2), dtype=np.float32)
        ## add
        angle = np.zeros((self.max_objs, 2), dtype=np.float32)
        ## add end
        reg = np.zeros((self.max_objs, 2), dtype=np.float32)
        ind = np.zeros((self.max_objs), dtype=np.int64)
        reg_mask = np.zeros((self.max_objs), dtype=np.uint8)
        num_objs = min(annotation['rect'].shape[0], self.max_objs)
        # ###################################### view Images #######################################
        # copy_image1 = cv2.resize(image, (image_w, image_h))
        # copy_image2 = copy_image1.copy()
        # ##########################################################################################

        for k in range(num_objs):
            rect = annotation['rect'][k, :]
            cen_x, cen_y, bbox_w, bbox_h, theta = rect
            cen_x, cen_y, bbox_w, bbox_h = cen_x/self.down_ratio, cen_y/self.down_ratio, bbox_w/self.down_ratio, bbox_h/self.down_ratio
            # print(theta)
            radius = gaussian_radius((math.ceil(bbox_h), math.ceil(bbox_w)))
            radius = max(0, int(radius))
            ct = np.asarray([cen_x, cen_y], dtype=np.float32)
            ct_int = ct.astype(np.int32)
            draw_umich_gaussian(hm[annotation['cat'][k]], ct_int, radius)
            ind[k] = ct_int[1] * image_w + ct_int[0]
            reg[k] = ct - ct_int
            reg_mask[k] = 1
            angle[k, :] = [np.sin(theta), np.cos(theta)]
            wh[k, :] = [bbox_w, bbox_h]
            if(cen_x < 0) or (cen_y < 0):
                print(cen_x, cen_y)
            ind[ind > 4095] = 0
            # print(ind[k])
            # if ind[k].any() > 4096:
            #     print(ind[k])
            # generate wh ground_truth
            # pts_4 = cv2.boxPoints(((cen_x, cen_y), (bbox_w, bbox_h), theta))  # 4 x 2

            # bl = pts_4[0,:]
            # tl = pts_4[1,:]
            # tr = pts_4[2,:]
            # br = pts_4[3,:]

            # tt = (np.asarray(tl,np.float32)+np.asarray(tr,np.float32))/2
            # rr = (np.asarray(tr,np.float32)+np.asarray(br,np.float32))/2
            # bb = (np.asarray(bl,np.float32)+np.asarray(br,np.float32))/2
            # ll = (np.asarray(tl,np.float32)+np.asarray(bl,np.float32))/2

            # if theta in [-90.0, -0.0, 0.0]:  # (-90, 0]
            #     tt,rr,bb,ll = self.reorder_pts(tt,rr,bb,ll)
            # rotational channel
            # wh[k, 0:2] = tt - ct
            # wh[k, 2:4] = rr - ct
            # wh[k, 4:6] = bb - ct
            # wh[k, 6:8] = ll - ct
            #####################################################################################
            # # draw
            # cv2.line(copy_image1, (cen_x, cen_y), (int(tt[0]), int(tt[1])), (0, 0, 255), 1, 1)
            # cv2.line(copy_image1, (cen_x, cen_y), (int(rr[0]), int(rr[1])), (255, 0, 255), 1, 1)
            # cv2.line(copy_image1, (cen_x, cen_y), (int(bb[0]), int(bb[1])), (0, 255, 255), 1, 1)
            # cv2.line(copy_image1, (cen_x, cen_y), (int(ll[0]), int(ll[1])), (255, 0, 0), 1, 1)
            #####################################################################################
            # horizontal channel
            # w_hbbox, h_hbbox = self.cal_bbox_wh(pts_4)
            # wh[k, 8:10] = 1. * w_hbbox, 1. * h_hbbox
            #####################################################################################
            # # draw
            # cv2.line(copy_image2, (cen_x, cen_y), (int(cen_x), int(cen_y-wh[k, 9]/2)), (0, 0, 255), 1, 1)
            # cv2.line(copy_image2, (cen_x, cen_y), (int(cen_x+wh[k, 8]/2), int(cen_y)), (255, 0, 255), 1, 1)
            # cv2.line(copy_image2, (cen_x, cen_y), (int(cen_x), int(cen_y+wh[k, 9]/2)), (0, 255, 255), 1, 1)
            # cv2.line(copy_image2, (cen_x, cen_y), (int(cen_x-wh[k, 8]/2), int(cen_y)), (255, 0, 0), 1, 1)
            #####################################################################################
            # v0
            # if abs(theta)>3 and abs(theta)<90-3:
            #     cls_theta[k, 0] = 1
            # v1
            # jaccard_score = ex_box_jaccard(pts_4.copy(), self.cal_bbox_pts(pts_4).copy())
            # if jaccard_score<0.95:
            #     cls_theta[k, 0] = 1
        # ###################################### view Images #####################################
        # # hm_show = np.uint8(cv2.applyColorMap(np.uint8(hm[0, :, :] * 255), cv2.COLORMAP_JET))
        # # copy_image = cv2.addWeighted(np.uint8(copy_image), 0.4, hm_show, 0.8, 0)
        #     if jaccard_score>0.95:
        #         print(theta, jaccard_score, cls_theta[k, 0])
        #         cv2.imshow('img1', cv2.resize(np.uint8(copy_image1), (image_w*4, image_h*4)))
        #         cv2.imshow('img2', cv2.resize(np.uint8(copy_image2), (image_w*4, image_h*4)))
        #         key = cv2.waitKey(0)&0xFF
        #         if key==ord('q'):
        #             cv2.destroyAllWindows()
        #             exit()
        # #########################################################################################

        ret = {'input': image,
               'hm': hm,
               'reg_mask': reg_mask,
               'ind': ind,
               'wh': wh,
               'angle': angle,
               'reg': reg,
               }
        return ret

    def __getitem__(self, index):
        img_cp, img_pc = self.load_image(index)
        image_h, image_w, c = img_cp.shape
        if self.phase == 'test':
            img_id = self.img_ids[index]
            image_cp = self.processing_test(img_cp, self.input_h, self.input_w)
            image_pc = self.processing_test(img_pc, self.input_h, self.input_w)
            return {'image_cp': image_cp,
                    'image_pc': image_pc,
                    'img_id': img_id,
                    'image_w': image_w,
                    'image_h': image_h}

        elif self.phase == 'train':
            annotation_cp, annotation_pc = self.load_annotation(index)
            # check
            # annotation_cp_ori = copy.deepcopy(annotation_cp)
            # annotation_pc_ori = copy.deepcopy(annotation_pc)
            # img_cp_ori = copy.deepcopy(img_cp)
            # img_pc_ori = copy.deepcopy(img_pc)
            
            
            img_cp, img_pc, annotation_cp, annotation_pc = self.data_transform(img_cp, img_pc, annotation_cp, annotation_pc, index)
            data_dict_cp = self.generate_ground_truth(img_cp, annotation_cp)
            data_dict_pc = self.generate_ground_truth(img_pc, annotation_pc)
            
            
            # data_dict_cp['annotation_ori'] = annotation_cp_ori
            # data_dict_pc['annotation_ori'] = annotation_pc_ori
            # data_dict_cp['annotation'] = annotation_cp
            # data_dict_pc['annotation'] = annotation_pc
            # data_dict_cp['input_ori'] = img_cp_ori
            # data_dict_pc['input_ori'] = img_pc_ori
            
            return data_dict_cp, data_dict_pc


