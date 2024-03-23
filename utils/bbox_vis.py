# import matplotlib.pyplot as plt
import pickle
from typing import List, Optional, Union
import cv2
import numpy as np
import math
import supervision as sv
from supervision.detection.core import Detections
from supervision.draw.color import Color, ColorPalette
import torch
from ipdb import set_trace
from torchvision.ops import box_convert
from torchvision.utils import make_grid

class BoxAnnotate(sv.BoxAnnotator):
    def Rbbox_annotate(
        self,
        scene: np.ndarray,
        xywh: np.ndarray,
        rad: np.ndarray,
        detections: Detections,
        labels: Optional[List[str]] = None,
        skip_label: bool = False,
    ) -> np.ndarray:
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        for i in range(len(detections)):
            x1, y1, x2, y2 = detections.xyxy[i].astype(int)
            class_id = (
                detections.class_id[i] if detections.class_id is not None else None
            )
            idx = class_id if class_id is not None else i
            color = (
                self.color.by_idx(idx)
                if isinstance(self.color, ColorPalette)
                else self.color
            )
            # cv2.rectangle(
            #     img=scene,
            #     pt1=(x1, y1),
            #     pt2=(x2, y2),
            #     color=color.as_bgr(),
            #     thickness=self.thickness,
            # )

            """draw rbboxes"""
            tmp = rad[i]
            theta = math.degrees(math.atan2(tmp[0], tmp[1]))
            if theta > 90:
                theta = 180 - theta
            x, y, w, h = xywh[i]
            rect = ((x, y), (w, h), theta)
            
            box_points = cv2.boxPoints(rect)
            box_points = np.intp(box_points)
            # cv2.drawContours(img, [box_points], 0, (0, 0, 255), 2)
            cv2.polylines(scene, [box_points], isClosed=True, color=color.as_bgr(), thickness=self.thickness)
            # cv2.imwrite(f"./test/outputs/mask/rbbox_{i}.png", img)

            if skip_label:
                continue

            text = (
                f"{class_id}"
                if (labels is None or len(detections) != len(labels))
                else labels[i]
            )

            text_width, text_height = cv2.getTextSize(
                text=text,
                fontFace=font,
                fontScale=self.text_scale,
                thickness=self.text_thickness,
            )[0]
            
            # x1, y1 = box_points[0][0], box_points[0][1] ### 可以注释掉
            text_x = x1 + self.text_padding
            text_y = y1 - self.text_padding

            text_background_x1 = x1
            text_background_y1 = y1 - 2 * self.text_padding - text_height

            text_background_x2 = x1 + 2 * self.text_padding + text_width
            text_background_y2 = y1

            cv2.rectangle(
                img=scene,
                pt1=(text_background_x1, text_background_y1),
                pt2=(text_background_x2, text_background_y2),
                color=color.as_bgr(),
                thickness=cv2.FILLED,
            )
            cv2.putText(
                img=scene,
                text=text,
                org=(text_x, text_y),
                fontFace=font,
                fontScale=self.text_scale,
                color=self.text_color.as_rgb(),
                thickness=self.text_thickness,
                lineType=cv2.LINE_AA,
            )
        return scene
    

class MaskAnnotate(sv.MaskAnnotator):
    def annotate(
        self, scene: np.ndarray, detections: Detections, opacity: float = 0.5
    ) -> np.ndarray:
        """
        Overlays the masks on the given image based on the provided detections, with a specified opacity.

        Args:
            scene (np.ndarray): The image on which the masks will be overlaid
            detections (Detections): The detections for which the masks will be overlaid
            opacity (float): The opacity of the masks, between 0 and 1, default is 0.5

        Returns:
            np.ndarray: The image with the masks overlaid
        """
        if detections.mask is None:
            return scene

        masks, masks_sort, mask_label, sorted_index = [], [], [], []
        bg_img = np.zeros_like(scene, dtype=np.uint8)
        for i in np.flip(np.argsort(detections.area)):
            sorted_index.append(i)
            class_id = (
                detections.class_id[i] if detections.class_id is not None else None
            )
            idx = class_id if class_id is not None else i
            mask_label.append(idx)
            color = (
                self.color.by_idx(idx)
                if isinstance(self.color, ColorPalette)
                else self.color
            )

            mask = detections.mask[i]
            colored_mask = np.zeros_like(scene, dtype=np.uint8)
            colored_mask[:] = color.as_bgr()

            # annotate each mask to background
            single_mask = np.where(
                np.expand_dims(mask, axis=-1),
                np.uint8(opacity * colored_mask + (1 - opacity) * bg_img),
                bg_img,
            )
            masks_sort.append(single_mask)

            scene = np.where(
                np.expand_dims(mask, axis=-1),
                np.uint8(opacity * colored_mask + (1 - opacity) * scene),
                scene,
            )

        ### masks order match bbox ###
        

        # return scene, masks, masks_sort, mask_label
        return scene, masks_sort, mask_label, sorted_index

    def rgb_annotate(
        self, scene: np.ndarray, detections: Detections, opacity: float = 0.5
    ) -> np.ndarray:
        """
        Overlays the masks on the given image based on the provided detections, with a specified opacity.

        Args:
            scene (np.ndarray): The image on which the masks will be overlaid
            detections (Detections): The detections for which the masks will be overlaid
            opacity (float): The opacity of the masks, between 0 and 1, default is 0.5

        Returns:
            np.ndarray: The image with the masks overlaid
        """
        if detections.mask is None:
            return scene

        # masks, masks_sort, mask_label = [], [], []
        area_sort = []
        bg_img = np.zeros_like(scene, dtype=np.uint8)
        for i in np.flip(np.argsort(detections.area)):
            area_sort.append(i)
            mask = detections.mask[i]

            # annotate each mask to background
            scene = cv2.add(scene, mask)

        ### masks order match bbox ###
        

        # return scene, masks, masks_sort, mask_label
        return scene, area_sort


def bbox_vis(res_batch, category_dict): # show_box(box.numpy(), plt.gca(), label)
    w, h = 512, 512
    ptr = res_batch.ptr
    imgs = []
    for idx in range(ptr.shape[0]-1):
        pos = res_batch.x[ptr[idx]:ptr[idx+1]]
        geo = res_batch.geo[ptr[idx]:ptr[idx+1]]
        category = res_batch.category[ptr[idx]:ptr[idx+1]]

        label = [category_dict[int(key)] for key in category]
        box = torch.cat((pos, geo), dim=1)
        #######################################################
        ### if sampler clamp [-1, 1], do some change to box ###
        #######################################################
        box = (box + 1) / 2.0
        boxes = box * torch.Tensor([w, h, w, h]).to(device='cuda')
        xyxy = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy").cpu().numpy()
        detections = sv.Detections(xyxy=xyxy)

        box_annotator = BoxAnnotate()
        bbox_image = np.zeros((h, w, 3))
        bbox_img = box_annotator.annotate(scene=bbox_image, detections=detections, labels=label)
        imgs.append(bbox_img)
    
    batch_imgs = np.stack(imgs, axis=0)
    bbox_imgs = torch.tensor(batch_imgs).permute(0, 3, 1, 2)  
    bbox_imgs = make_grid(bbox_imgs.float(), padding=2, nrow=2, normalize=True) 
    return bbox_imgs
    


def Rotated_bbox_vis(res_batch, category_dict): # show_box(box.numpy(), plt.gca(), label)
    w, h = 512, 512
    ptr = res_batch.ptr
    imgs = []
    for idx in range(ptr.shape[0]-1):
        pos = res_batch.x[ptr[idx]:ptr[idx+1]]
        geo = res_batch.geo[ptr[idx]:ptr[idx+1]]
        category = res_batch.category[ptr[idx]:ptr[idx+1]]

        label = [category_dict[int(key)] for key in category]
        # print(box.shape)
        # print(geo.shape)
        center = pos[:, :2]
        rad = pos[:, 2:] # [sin, cos] -> rad
        box = torch.cat((center, geo), dim=1)
        #######################################################
        ### if sampler clamp [-1, 1], do some change to box ###
        #######################################################
        box[:, :2] = (box[:, :2] + 1.0) / 2.0
        boxes = box * torch.Tensor([w, h, w, h]).to(device='cuda')
        boxes_xywh = boxes.cpu().numpy()
        xyxy = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy").cpu().numpy()
        detections = sv.Detections(xyxy=xyxy)

        box_annotator = BoxAnnotate()
        bbox_image = np.zeros((h, w, 3))
        bbox_img = box_annotator.Rbbox_annotate(scene=bbox_image, xywh=boxes_xywh, rad=rad, detections=detections, labels=label)
        imgs.append(bbox_img)
    
    batch_imgs = np.stack(imgs, axis=0)
    bbox_imgs = torch.tensor(batch_imgs).permute(0, 3, 1, 2)  
    bbox_imgs = make_grid(bbox_imgs.float(), padding=2, nrow=2, normalize=True) 

    return bbox_imgs


def mask_translate(translation, rgb, masks):

    result_masks = []
    for i in range(len(translation)):
        mask = masks[i]
        rgb_mask = np.zeros_like(rgb)
        rgb_mask[mask] = rgb[mask]
        tx, ty = translation[i]

        # build translation matrix
        translation_matrix = np.float32([[1, 0, tx], [0, 1, ty]])
        translated_mask = cv2.warpAffine(rgb_mask, translation_matrix, (rgb_mask.shape[1], rgb_mask.shape[0]), borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        result_masks.append(translated_mask)

    return np.array(result_masks)


def rgb_vis(res_batch, category_dict, bbox_data, rgb, masks): # show_box(box.numpy(), plt.gca(), label)

    w, h = 640, 640
    sample_color = rgb[10, 10, :]
        
    init_pos = np.zeros([len(bbox_data), 2])
    for i in range(len(bbox_data)):
        init_pos[i, :] = bbox_data[i][:2]

    ptr = res_batch.ptr
    # imgs = []
    for idx in range(ptr.shape[0]-1):
        pos = res_batch.x[ptr[idx]:ptr[idx+1]]
        geo = res_batch.geo[ptr[idx]:ptr[idx+1]]
        category = res_batch.category[ptr[idx]:ptr[idx+1]]

        label = [category_dict[int(key)] for key in category]
        ### calculate bbox translation ###
        pos_trans = ((pos+ 1.0) / 2.0).cpu().numpy() - init_pos
        pos_trans *= [w, h]

        box = torch.cat((pos, geo), dim=1)
        box[:, :2] = (box[:, :2] + 1.0) / 2.0

        boxes = box * torch.Tensor([w, h, w, h]).to(device='cuda')
        xyxy = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy").cpu().numpy()
        detections = sv.Detections(xyxy=xyxy)
        detections.mask = mask_translate(pos_trans, rgb, masks)
        
        box_annotator = BoxAnnotate()
        mask_annotator = MaskAnnotate()
        bbox_image = np.zeros((h, w, 3), dtype=np.uint8)
        # set_trace()
        annotated_image, area_sort = mask_annotator.rgb_annotate(scene=bbox_image.copy(), detections=detections)
        bbox_img = box_annotator.annotate(scene=annotated_image, detections=detections, labels=label)

        ### change background ###
        mask = cv2.inRange(bbox_img, np.array([0, 0, 0]), np.array([0, 0, 0]))
        bbox_img[mask > 0] = sample_color
        
    return bbox_img, area_sort


def Rotated_rgb_vis(res_batch, category_dict): # show_box(box.numpy(), plt.gca(), label)
    w, h = 512, 512
    
    ptr = res_batch.ptr
    for idx in range(ptr.shape[0]-1):
        pos = res_batch.x[ptr[idx]:ptr[idx+1]]
        geo = res_batch.geo[ptr[idx]:ptr[idx+1]]
        category = res_batch.category[ptr[idx]:ptr[idx+1]]

        label = [category_dict[int(key)] for key in category]
        center = pos[:, :2]
        rad = pos[:, 2:]
        box = torch.cat((center, geo), dim=1)
        box[:, :2] = (box[:, :2] + 1.0) / 2.0 # [-1, 1] -> [0, 1]
        boxes = box * torch.Tensor([w*2, h*2, w, h]).to(device='cuda')
        boxes_xywh = boxes.cpu().numpy()
        xyxy = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy").cpu().numpy()
        detections = sv.Detections(xyxy=xyxy)
 
        box_annotator = BoxAnnotate()
        bbox_image = np.zeros((h*2, w*2, 3))
        bbox_img = box_annotator.Rbbox_annotate(scene=bbox_image, xywh=boxes_xywh, rad=rad, detections=detections, labels=label)

    return bbox_img
    