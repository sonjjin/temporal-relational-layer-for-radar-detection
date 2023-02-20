from .base_temporal import BaseDataset
import os
import cv2
import numpy as np
from .DOTA_devkit.ResultMerge_multi_process import mergebypoly
from .radiate_evaluation_task1 import voc_eval
import pdb
class RADIATE(BaseDataset):
    def __init__(self, data_dir, train_mode, phase, max_objs, only_objects=True, input_h=None, input_w=None, down_ratio=None):
        super(RADIATE, self).__init__(data_dir, train_mode, phase, max_objs, only_objects, input_h, input_w, down_ratio)
        self.data_dir = data_dir
        self.traindata_dir = os.path.join(data_dir, train_mode)
        self.category = ['vehicle']
        self.color_pans = [(204,78,210)]
        self.num_classes = len(self.category)
        self.cat_ids = {cat:i for i,cat in enumerate(self.category)}
        self.image_path = os.path.join(self.traindata_dir, 'images')
        self.label_path = os.path.join(self.traindata_dir, 'labels_angle')
        self.img_ids = self.load_img_ids()
        self.img_mean = 31.484315963620194
        self.img_std = 3.050721641184311

    def load_img_ids(self):
        if self.phase == 'train':
            if self.only_objects:
                image_set_index_file = os.path.join(self.traindata_dir, 'labels_temporal_with_object.txt')
            else:
                image_set_index_file = os.path.join(self.traindata_dir, 'labels_temporal.txt')
        else:
            if self.only_objects:
                image_set_index_file = os.path.join(self.data_dir, 'test/labels_temporal_with_object.txt')
            else:              
                image_set_index_file = os.path.join(self.data_dir, 'test/labels_temporal.txt')
        assert os.path.exists(image_set_index_file), 'Path does not exist: {}'.format(image_set_index_file)
        with open(image_set_index_file, 'r') as f:
            lines = f.readlines()
        image_lists = [line.strip() for line in lines]
        return image_lists

    def load_image(self, index):
        if self.phase == 'train':
            img_id = self.img_ids[index]
            img_id = int(img_id)
            imgFile_cur = os.path.join(self.image_path, str(img_id).zfill(5)+'.png')
            imgFile_pre = os.path.join(self.image_path, str(img_id-3).zfill(5)+'.png')
            assert os.path.exists(imgFile_cur), 'image {} not existed'.format(imgFile_cur)
            assert os.path.exists(imgFile_cur), 'image {} not existed'.format(imgFile_pre)

        else:
            img_path = os.path.join(self.data_dir, 'test/images_add')
            img_id = self.img_ids[index]
            img_id = int(img_id)
            imgFile_cur = os.path.join(img_path, str(img_id).zfill(5)+'.png')
            imgFile_pre = os.path.join(img_path, str(img_id-3).zfill(5)+'.png')
            assert os.path.exists(imgFile_cur), 'image {} not existed'.format(imgFile_cur)
            assert os.path.exists(imgFile_cur), 'image {} not existed'.format(imgFile_pre)

        
        img_cur = cv2.imread(imgFile_cur, cv2.IMREAD_GRAYSCALE)
        img_pre = cv2.imread(imgFile_pre, cv2.IMREAD_GRAYSCALE)
        img_cp = np.stack((img_cur, img_pre), axis=2)
        img_pc = np.stack((img_pre, img_cur), axis=2)    
        return img_cp, img_pc
        

    def load_annoFolder(self, img_id):
        return os.path.join(self.label_path, img_id+'.txt')

    def load_annotation(self, index):
        img_cp, _ = self.load_image(index)
        img_h,img_w,c = img_cp.shape
        valid_pts_cp = []
        valid_cat_cp = []
        valid_dif_cp = []
        valid_pts_pc = []
        valid_cat_pc = []
        valid_dif_pc = []
        
        img_id = self.img_ids[index]
        img_id = int(img_id)
        
        # load cur+pre
        with open(self.load_annoFolder(str(img_id).zfill(5)), 'r') as f:
            for i, line in enumerate(f.readlines()):
                obj = line.split(' ')  # list object
                if len(obj)>5:
                    cx = min(max(float(obj[0]), 0), img_w - 1)
                    cy = min(max(float(obj[1]), 0), img_h - 1)
                    w = min(max(float(obj[2]), 0), img_w - 1)
                    h = min(max(float(obj[3]), 0), img_h - 1)
                    angle = float(obj[4])
                    # TODO: filter small instances
                    if (w > 5) and (h > 5):
                        valid_pts_cp.append([cx, cy, w, h, angle])
                        valid_cat_cp.append(self.cat_ids[obj[5]])
                        valid_dif_cp.append(int(obj[6]))
        
        annotation_cp = {}
        annotation_cp['pts'] = np.asarray(valid_pts_cp, np.float32)
        annotation_cp['cat'] = np.asarray(valid_cat_cp, np.int32)
        annotation_cp['dif'] = np.asarray(valid_dif_cp, np.int32)
        
        # load pre+cur
        with open(self.load_annoFolder(str(img_id-3).zfill(5)), 'r') as f:
            for i, line in enumerate(f.readlines()):
                obj = line.split(' ')  # list object
                if len(obj)>5:
                    cx = min(max(float(obj[0]), 0), img_w - 1)
                    cy = min(max(float(obj[1]), 0), img_h - 1)
                    w = min(max(float(obj[2]), 0), img_w - 1)
                    h = min(max(float(obj[3]), 0), img_h - 1)
                    angle = float(obj[4])
                    # TODO: filter small instances
                    if (w > 5) and (h > 5):
                        valid_pts_pc.append([cx, cy, w, h, angle])
                        valid_cat_pc.append(self.cat_ids[obj[5]])
                        valid_dif_pc.append(int(obj[6]))

        annotation_pc = {}
        annotation_pc['pts'] = np.asarray(valid_pts_pc, np.float32)
        annotation_pc['cat'] = np.asarray(valid_cat_pc, np.int32)
        annotation_pc['dif'] = np.asarray(valid_dif_pc, np.int32)
        # pts0 = np.asarray(valid_pts, np.float32)
        # img = self.load_image(index)
        # for i in range(pts0.shape[0]):
        #     pt = pts0[i, :, :]
        #     tl = pt[0, :]
        #     tr = pt[1, :]
        #     br = pt[2, :]
        #     bl = pt[3, :]
        #     cv2.line(img, (int(tl[0]), int(tl[1])), (int(tr[0]), int(tr[1])), (0, 0, 255), 1, 1)
        #     cv2.line(img, (int(tr[0]), int(tr[1])), (int(br[0]), int(br[1])), (255, 0, 255), 1, 1)
        #     cv2.line(img, (int(br[0]), int(br[1])), (int(bl[0]), int(bl[1])), (0, 255, 255), 1, 1)
        #     cv2.line(img, (int(bl[0]), int(bl[1])), (int(tl[0]), int(tl[1])), (255, 0, 0), 1, 1)
        #     cv2.putText(img, '{}:{}'.format(valid_dif[i], self.category[valid_cat[i]]), (int(tl[0]), int(tl[1])), cv2.FONT_HERSHEY_TRIPLEX, 0.6,
        #                 (0, 0, 255), 1, 1)
        # cv2.imshow('img', np.uint8(img))
        # k = cv2.waitKey(0) & 0xFF
        # if k == ord('q'):
        #     cv2.destroyAllWindows()
        #     exit()
        # pdb.set_trace()
        return annotation_cp, annotation_pc
        
    def dec_evaluation(self, result_path):
        detpath = os.path.join(result_path, 'Task1_{}.txt')
        # annopath = os.path.join(self.label_path, '{}.xml')  # change the directory to the path of val/labelTxt, if you want to do evaluation on the valset
        annopath = os.path.join(self.data_dir, 'test/labelTxt', '{}.txt')
        # pdb.set_trace()
        imagesetfile = os.path.join(self.data_dir, 'test/labels_temporal.txt')
        classaps = []
        map = 0
        for classname in self.category:
            if classname == 'background':
                continue
            print('classname:', classname)
            rec, prec, ap = voc_eval(detpath,
                                     annopath,
                                     imagesetfile,
                                     classname,
                                     ovthresh=0.5,
                                     use_07_metric=True)
            map = map + ap
            # print('rec: ', rec, 'prec: ', prec, 'ap: ', ap)
            print('{}:{} '.format(classname, ap*100))
            classaps.append(ap)
            # umcomment to show p-r curve of each category
            # plt.figure(figsize=(8,4))
            # plt.xlabel('recall')
            # plt.ylabel('precision')
            # plt.plot(rec, prec)
        # plt.show()
        map = map / len(self.category)
        print('map:', map*100)
        # classaps = 100 * np.array(classaps)
        # print('classaps: ', classaps)
        return map