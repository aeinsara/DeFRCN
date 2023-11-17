import torch
import logging
from torch import nn
from detectron2.structures import ImageList
from detectron2.utils.logger import log_first_n
from detectron2.modeling.backbone import build_backbone
from detectron2.modeling.postprocessing import detector_postprocess
from detectron2.modeling.proposal_generator import build_proposal_generator
from detectron2.modeling.backbone.fpn import build_resnet_fpn_backbone
from .build import META_ARCH_REGISTRY
from .gdl import decouple_layer, AffineLayer
from defrcn.modeling.roi_heads import build_roi_heads
from detectron2.modeling import FPN, RPN_HEAD_REGISTRY, build_rpn_head
from PIL import Image
import numpy as np
import cv2
__all__ = ["GeneralizedRCNN"]


@META_ARCH_REGISTRY.register()
class GeneralizedRCNN(nn.Module):

    def __init__(self, cfg):
        super().__init__()

        self.cfg = cfg
        self.device = torch.device(cfg.MODEL.DEVICE)
        self.backbone = build_backbone(cfg)
#         self.neck = self.build_neck(cfg) # ............. sara added .............
        self._SHAPE_ = self.backbone.output_shape()
        self.proposal_generator = build_proposal_generator(cfg, self._SHAPE_)

        self.roi_heads = build_roi_heads(cfg, self._SHAPE_)
        self.normalizer = self.normalize_fn()
        self.affine_rpn = AffineLayer(num_channels=self._SHAPE_['res4'].channels, bias=True)
        self.affine_rcnn = AffineLayer(num_channels=self._SHAPE_['res4'].channels, bias=True)
#         print("output_shape() == ", self.neck.output_shape())
#         self.affine_rpn = AffineLayer(num_channels=256, bias=True)
#         self.affine_rcnn = AffineLayer(num_channels=256, bias=True)
        self.to(self.device)
        print("iiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiii cfg.MODEL.BACKBONE.FREEZE ", cfg.MODEL.BACKBONE.FREEZE)
        if cfg.MODEL.BACKBONE.FREEZE: # False
            for p in self.backbone.parameters():
                p.requires_grad = False
#             for p in self.neck.parameters():
#                 p.requires_grad = False
            print("froze backbone parameters")
        print("iiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiii cfg.MODEL.RPN.FREEZE ", cfg.MODEL.RPN.FREEZE)
        if cfg.MODEL.RPN.FREEZE: # False
            for p in self.proposal_generator.parameters():
                p.requires_grad = False
            print("froze proposal generator parameters")
#         print("self.roi_heads ============= ", self.roi_heads)
        print("iiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiii cfg.MODEL.ROI_HEADS.FREEZE_FEAT ", cfg.MODEL.ROI_HEADS.FREEZE_FEAT)
        if cfg.MODEL.ROI_HEADS.FREEZE_FEAT:# True
            for p in self.roi_heads.res5.parameters():
                p.requires_grad = False
            print("froze roi_box_head parameters")
    def build_neck(self, cfg):
        in_features0 = ["res4"]
        backbone_shape = self.backbone.output_shape()
        return FPN(bottom_up=self.backbone,in_features=in_features0, out_channels=1024)
    
    def forward(self, batched_inputs):
        if not self.training:
            return self.inference(batched_inputs)
        assert "instances" in batched_inputs[0]
        gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
        proposal_losses, detector_losses, _, _ = self._forward_once_(batched_inputs, gt_instances)
        losses = {}
        losses.update(detector_losses)
        losses.update(proposal_losses)
        return losses

    def inference(self, batched_inputs):
        assert not self.training
        _, _, results, image_sizes = self._forward_once_(batched_inputs, None)
        processed_results = []
        for r, input, image_size in zip(results, batched_inputs, image_sizes):
            height = input.get("height", image_size[0])
            width = input.get("width", image_size[1])
            r = detector_postprocess(r, height, width)
            processed_results.append({"instances": r})
        return processed_results

    def _forward_once_(self, batched_inputs, gt_instances=None):
#         x1,y1,x2,y2 = gt_instances[0].gt_boxes.tensor.data[0].tolist()
#         image = np.transpose(batched_inputs[0]['image'].cpu().detach().numpy(), (1, 2, 0))
#         image = self.crop(image, int(x1), int(y1), int(x2-x1), int(y2-y1), 1)
#         cv2.rectangle(image, (x1, y1), (x2, y2), (0,0,255), 3)

#         image = Image.fromarray(image.astype('uint8'))
#         image.save('/home/aeen/fewshot/DeFRCN/results/pcb/'+str(batched_inputs[0]['file_name'].split('/')[-1])+'.jpg')
#         cv2.imwrite('/home/aeen/fewshot/DeFRCN/results/pcb/'+str(int(batched_inputs[0]['instances'].gt_classes[0]))+'.jpg', image)#batched_inputs['image']
        images = self.preprocess_image(batched_inputs,gt_instances)
#         print('images.shepe() === ',len(gt_instances), len(images), len(images[0].tolist()))
        features = self.backbone(images.tensor)
#         features = self.neck(features) # ...................................
        features_de_rpn = features
        if self.cfg.MODEL.RPN.ENABLE_DECOUPLE:
            scale = self.cfg.MODEL.RPN.BACKWARD_SCALE
            features_de_rpn = {k: self.affine_rpn(decouple_layer(features[k], scale)) for k in features}
        proposals, proposal_losses = self.proposal_generator(images, features_de_rpn, gt_instances)

        features_de_rcnn = features
        if self.cfg.MODEL.ROI_HEADS.ENABLE_DECOUPLE:
            scale = self.cfg.MODEL.ROI_HEADS.BACKWARD_SCALE
            features_de_rcnn = {k: self.affine_rcnn(decouple_layer(features[k], scale)) for k in features}
        results, detector_losses = self.roi_heads(images, features_de_rcnn, proposals, gt_instances)

        return proposal_losses, detector_losses, results, images.image_sizes

    def preprocess_image(self, batched_inputs, gt_instances):
#         for i,img in enumerate(batched_inputs):
#             image = img["image"]
#             if gt_instances is None:
#                 continue
#             x1,y1,x2,y2 = gt_instances[i].gt_boxes.tensor.data[0].tolist()
#             image = np.transpose(batched_inputs[i]['image'].cpu().detach().numpy(), (1, 2, 0))
#             image = self.crop(image, int(x1), int(y1), int(x2-x1), int(y2-y1), 1)
# #             images.append(torch.from_numpy(image.transpose((2, 0, 1))).to(self.device))
#             batched_inputs[i]['image'] = torch.from_numpy(image.transpose((2, 0, 1)))
        images = [x["image"].to(self.device) for x in batched_inputs]
        images = [self.normalizer(x) for x in images]
        images = ImageList.from_tensors(images, self.backbone.size_divisibility)
        return images

    def normalize_fn(self):
        assert len(self.cfg.MODEL.PIXEL_MEAN) == len(self.cfg.MODEL.PIXEL_STD)
        num_channels = len(self.cfg.MODEL.PIXEL_MEAN)
        pixel_mean = (torch.Tensor(
            self.cfg.MODEL.PIXEL_MEAN).to(self.device).view(num_channels, 1, 1))
        pixel_std = (torch.Tensor(
            self.cfg.MODEL.PIXEL_STD).to(self.device).view(num_channels, 1, 1))
        return lambda x: (x - pixel_mean) / pixel_std

    def crop(self, image, x1, y1, w, h, scale):
        # create a mask
        mask = np.zeros(image.shape[:2], np.uint8)
        if scale == 1:
            new_x1 = x1
            new_y1 = y1
            new_x2 = x1+w
            new_y2 = y1+h
        else:
            new_x1 = int(max(x1 - ((scale / 2) * w), 0))
            new_y1 = int(max(y1 - ((scale / 2) * h), 0))
            new_x2 = int(min((x1 + (scale / 2) * w + w), image.shape[1]))
            new_y2 = int(min((y1 + (scale / 2) * h + h), image.shape[0]))
        
        # ............ For useing mask image ..............
        mask[new_y1:new_y2, new_x1:new_x2] = 255

        # compute the bitwise AND using the mask
        masked_img = cv2.bitwise_and(image,image,mask = mask)
        return masked_img