import torch.nn.functional as F
import torch
import pdb

class DecDecoder(object):
    def __init__(self, max_objs, conf_thresh, num_classes):
        self.max_objs = max_objs
        self.conf_thresh = conf_thresh
        self.num_classes = num_classes

    def _topk(self, scores):
        batch, cat, height, width = scores.size()

        topk_scores, topk_inds = torch.topk(scores.view(batch, cat, -1), self.max_objs)

        topk_inds = topk_inds % (height * width)
        # topk_ys = (topk_inds // width).int().float()
        topk_ys = torch.div(topk_inds, width, rounding_mode='trunc')
        topk_xs = (topk_inds % width).int().float()

        topk_score, topk_ind = torch.topk(topk_scores.view(batch, -1), self.max_objs)
        # topk_clses = (topk_ind // self.K).int()
        topk_clses = torch.div(topk_ind, self.max_objs, rounding_mode='trunc')
        topk_inds = self._gather_feat( topk_inds.view(batch, -1, 1), topk_ind).view(batch, self.max_objs)
        topk_ys = self._gather_feat(topk_ys.view(batch, -1, 1), topk_ind).view(batch, self.max_objs)
        topk_xs = self._gather_feat(topk_xs.view(batch, -1, 1), topk_ind).view(batch, self.max_objs)

        return topk_score, topk_inds, topk_clses, topk_ys, topk_xs


    def _nms(self, heat, kernel=3):
        hmax = F.max_pool2d(heat, (kernel, kernel), stride=1, padding=(kernel - 1) // 2)
        keep = (hmax == heat).float()
        return heat * keep

    def _gather_feat(self, feat, ind, mask=None):
        dim = feat.size(2)
        ind = ind.unsqueeze(2).expand(ind.size(0), ind.size(1), dim)
        feat = feat.gather(1, ind)
        if mask is not None:
            mask = mask.unsqueeze(2).expand_as(feat)
            feat = feat[mask]
            feat = feat.view(-1, dim)
        return feat

    def _tranpose_and_gather_feat(self, feat, ind):
        feat = feat.permute(0, 2, 3, 1).contiguous()
        feat = feat.view(feat.size(0), -1, feat.size(3))
        feat = self._gather_feat(feat, ind)
        return feat

    def _gen_boundingbox(self, bbox, angle):
        # bbox: B x K x 4(cx, cy, w, h)
        # angle: B x K x 1(rad)
        theta = -angle / 180 * torch.pi
        R1 = torch.cat([torch.cos(theta), -torch.sin(theta)], dim=2)
        R2 = torch.cat([torch.sin(theta), torch.cos(theta)], dim=2)
        R = torch.stack([R1, R2], dim=2)
        
        x1 = bbox[:,:,0] - bbox[:,:,2] / 2 # B x K x 1
        y1 = bbox[:,:,1] - bbox[:,:,3] / 2
        x2 = bbox[:,:,0] + bbox[:,:,2] / 2
        y2 = bbox[:,:,1] + bbox[:,:,3] / 2
        
        tr = torch.stack([x2, y1], dim=2) # B x K x 2
        br = torch.stack([x2, y2], dim=2)
        bl = torch.stack([x1, y2], dim=2)
        tl = torch.stack([x1, y1], dim=2)
        points = torch.stack([tr, br, bl, tl], dim=3)

        cx = bbox[:,:,0]
        cy = bbox[:,:,1]

        T = torch.stack([cx, cy], dim=2)
        T = T.unsqueeze(3)
        T = T.expand(-1,self.max_objs,2,4)

        points = points - T # B x K x 2 x 4
        points = torch.matmul(R, points) + T
        return points # B x K x 2 x 4

    def ctdet_decode(self, pr_decs):
        heat = pr_decs['hm']
        wh = pr_decs['wh']
        reg = pr_decs['reg']
        sin_cos = pr_decs['angle']
        

        batch, c, height, width = heat.size()
        heat = self._nms(heat)

        scores, inds, clses, ys, xs = self._topk(heat)
        # pdb.set_trace()
        reg = self._tranpose_and_gather_feat(reg, inds)
        reg = reg.view(batch, self.max_objs, 2)
        xs = xs.view(batch, self.max_objs, 1) + reg[:, :, 0:1]
        ys = ys.view(batch, self.max_objs, 1) + reg[:, :, 1:2]
        clses = clses.view(batch, self.max_objs, 1).float()
        scores = scores.view(batch, self.max_objs, 1)
        wh = self._tranpose_and_gather_feat(wh, inds)
        wh = wh.view(batch, self.max_objs, 2)
        # add
        bbox = torch.cat([xs, ys], dim = 2)
        bbox = torch.cat([bbox, wh], dim = 2)
        sin_cos = self._tranpose_and_gather_feat(sin_cos, inds)
        sin_cos = sin_cos.view(batch, self.max_objs, 2)
        angle = torch.atan2(sin_cos[:, :, 0:1], sin_cos[:, :, 1:2])
        points = self._gen_boundingbox(bbox, angle) # B x K x 2 x 4 (tr, br, bl, tl)
        tr_x = points[:,:,0,0].unsqueeze(2)
        tr_y = points[:,:,1,0].unsqueeze(2)
        br_x = points[:,:,0,1].unsqueeze(2)
        br_y = points[:,:,1,1].unsqueeze(2)
        bl_x = points[:,:,0,2].unsqueeze(2)
        bl_y = points[:,:,1,2].unsqueeze(2)
        tl_x = points[:,:,0,3].unsqueeze(2)
        tl_y = points[:,:,1,3].unsqueeze(2)

        #
        detections = torch.cat([tr_x,
                                tr_y,
                                br_x,
                                br_y,
                                bl_x,                                        
                                bl_y,
                                tl_x,
                                tl_y,
                                scores,
                                clses],
                               dim=2)

        index = (scores>self.conf_thresh).squeeze(0).squeeze(1)
        detections = detections[:,index,:]
        # pdb.set_trace()
        return detections.data.cpu().numpy()