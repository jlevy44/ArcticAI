"""Nuclei Detection
==========
Functions to support cell localization. 
"""
import os
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from torchvision.ops import nms
# import some detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg

from .utils import get_groups_dict

class BasePredictor:
    @property
    def type(self):
        raise NotImplementedError
    @property
    def classes(self):
        raise NotImplementedError
    @property
    def n(self):
        raise NotImplementedError
    @property
    def base_model(self):
        raise NotImplementedError
        
    def __init__(self, output_dir, model_path, threshold=0.2):
        self.cfg = self.build_cfg(output_dir, model_path)
        self.build_predictor(threshold)
    
    def build_cfg(self, output_dir, model_path):
        cfg = get_cfg()
        cfg.merge_from_file(model_zoo.get_config_file(self.base_model))
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = self.n
        cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES = self.n
        cfg.OUTPUT_DIR=output_dir
        cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, model_path)  # path to the model we just trained
        return cfg
    
    def build_predictor(self, threshold):
        self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = threshold
        self.predictor = DefaultPredictor(self.cfg)
        
    def predict(self, im, threshold=0.0):
        # Image should have BGR channels
        output = self.predictor(im)
        labels = output['instances']._fields['pred_classes'].cpu().detach().numpy().astype(int)
        # TODO: may want to check this.
        masks = output['instances']._fields['pred_masks'].cpu().detach().numpy().astype(np.uint8)
        boxes = output['instances']._fields['pred_boxes'].tensor.cpu().detach().numpy().astype(int)
        scores = output['instances']._fields['scores'].cpu().detach().numpy().astype(int)
        if threshold > 0:
            keep_idx=nms(boxes = output['instances']._fields['pred_boxes'].tensor.cpu().detach(), scores = output['instances']._fields['scores'].cpu().detach(), iou_threshold=threshold).numpy()
            masks, boxes, labels = masks[keep_idx], boxes[keep_idx], labels[keep_idx]
        return masks, boxes, labels

class InstanceNucleiPredictor(BasePredictor): 
    type = 'all'
    classes = get_groups_dict()[type]
    n = len(classes['map'])
    base_model = "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"

class InstanceSingleClassNucleiPredictor(BasePredictor):
    type = 'single'
    classes = get_groups_dict()[type]
    n = len(classes['map'])
    base_model = "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"

class PanopticNucleiPredictor(BasePredictor):
    type = 'all'
    classes = get_groups_dict()[type]
    n = len(classes['map'])
    base_model = "COCO-PanopticSegmentation/panoptic_fpn_R_101_3x.yaml"

class PanopticTwoClassNucleiPredictor(BasePredictor):
    type = 'two'
    classes = get_groups_dict()[type]
    n = len(classes['map'])
    base_model = "COCO-PanopticSegmentation/panoptic_fpn_R_101_3x.yaml"
        
class PanopticSingleClassNucleiPredictor(BasePredictor):
    type = 'single'
    classes = get_groups_dict()[type]
    n = len(classes['map'])
    base_model = "COCO-PanopticSegmentation/panoptic_fpn_R_101_3x.yaml"

def load_predictor(classifier_type, dir, file, threshold=0.05):
    return classifier_type(dir, file, threshold)