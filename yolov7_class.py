import matplotlib
import matplotlib.pyplot as plt
import torch
import cv2
import yaml
from torchvision import transforms
import numpy as np

import sys
import os

sys.path.append('yolov7')
print(sys.path)

from yolov7.utils.datasets import letterbox
from yolov7.utils.general import non_max_suppression_mask_conf

from detectron2.modeling.poolers import ROIPooler
from detectron2.structures import Boxes
from detectron2.utils.memory import retry_if_cuda_oom
from detectron2.layers import paste_masks_in_image


class Segmentation: 
    def __init__(self):
        # self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.device = torch.device("cpu")
        print(self.device)
        with open('yolov7/data/hyp.scratch.mask.yaml') as f:
            self.hyp = yaml.load(f, Loader=yaml.FullLoader)
        weigths = torch.load('yolov7/yolov7-mask.pt')
        model = weigths['model']

        if self.device == torch.device("cuda:0"):
            self.model = model.half().to(self.device)
        else:
            self.model = model.to(self.device)
            self.model = self.model.float()
        # model = model.to(device)
        _ = self.model.eval()

    def apply_yolo_model(self, image):
        image = letterbox(image, 640, stride=64, auto=True)[0]
        image_ = image.copy()
        image = transforms.ToTensor()(image)
        image = torch.tensor(np.array([image.numpy()]))
        if self.device == torch.device("cuda:0"):
            self.image = image.half().to(self.device)
        else:
            self.image = image.to(self.device)
        # self.image = self.image.half()

        self.output = self.model(self.image)
        
    def post_process(self):
        inf_out, train_out, attn, mask_iou, bases, sem_output = self.output['test'], self.output['bbox_and_cls'], self.output['attn'], self.output['mask_iou'], self.output['bases'], self.output['sem']
        bases = torch.cat([bases, sem_output], dim=1)
        nb, _, height, width = self.image.shape
        names = self.model.names
        pooler_scale = self.model.pooler_scale
        pooler = ROIPooler(output_size=self.hyp['mask_resolution'], scales=(pooler_scale,), sampling_ratio=1, pooler_type='ROIAlignV2', canonical_level=2)

        output, output_mask, output_mask_score, output_ac, output_ab = non_max_suppression_mask_conf(inf_out, attn, bases, pooler, self.hyp, conf_thres=0.25, iou_thres=0.65, merge=False, mask_iou=None)

        pred, pred_masks = output[0], output_mask[0]
        base = bases[0]
        bboxes = Boxes(pred[:, :4])
        original_pred_masks = pred_masks.view(-1, self.hyp['mask_resolution'], self.hyp['mask_resolution'])
        pred_masks = retry_if_cuda_oom(paste_masks_in_image)( original_pred_masks, bboxes, (height, width), threshold=0.5)
        pred_masks_np = pred_masks.detach().cpu().numpy()
        pred_cls = pred[:, 5].detach().cpu().numpy()
        pred_conf = pred[:, 4].detach().cpu().numpy()
        nimg = self.image[0].permute(1, 2, 0) * 255
        nimg = nimg.cpu().numpy().astype(np.uint8)
        nimg = cv2.cvtColor(nimg, cv2.COLOR_RGB2BGR)
        nbboxes = bboxes.tensor.detach().cpu().numpy().astype(int) # replaced np.int with int. Can specify np.int64/32 for precision
        pnimg = nimg.copy()

        for one_mask, bbox, cls, conf in zip(pred_masks_np, nbboxes, pred_cls, pred_conf):
            if conf < 0.25:
                continue
            color = [np.random.randint(255), np.random.randint(255), np.random.randint(255)]
                                
                                
            pnimg[one_mask] = pnimg[one_mask] * 0.5 + np.array(color, dtype=np.uint8) * 0.5
            pnimg = cv2.rectangle(pnimg, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)

        return pnimg

segmentation = Segmentation()

image = cv2.imread('/home/reuben/Project/YOLOV7-mask_branch/yolov7/inference/images/image3.jpg')  # 504x378 image

segmentation.apply_yolo_model(image)

output = segmentation.post_process()

cv2.imshow('image', output)
cv2.waitKey(0)
cv2.destroyAllWindows()


