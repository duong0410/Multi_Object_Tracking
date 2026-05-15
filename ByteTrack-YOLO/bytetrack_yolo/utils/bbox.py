"""
Bounding box utilities
"""

import numpy as np


def tlbr_to_tlwh(tlbr):
    """
    Convert bounding box from tlbr to tlwh format
    
    Args:
        tlbr: Array [top, left, bottom, right]
    
    Returns:
        tlwh: Array [top, left, width, height]
    """
    ret = np.asarray(tlbr).copy()
    ret[2:] -= ret[:2]
    return ret


def tlwh_to_tlbr(tlwh):
    """
    Convert bounding box from tlwh to tlbr format
    
    Args:
        tlwh: Array [top, left, width, height]
    
    Returns:
        tlbr: Array [top, left, bottom, right]
    """
    ret = np.asarray(tlwh).copy()
    ret[2:] += ret[:2]
    return ret


def tlwh_to_xyah(tlwh):
    """
    Convert tlwh to xyah format (center x, center y, aspect ratio, height)
    
    Args:
        tlwh: Array [top, left, width, height]
    
    Returns:
        xyah: Array [center_x, center_y, aspect_ratio, height]
    """
    ret = np.asarray(tlwh).copy()
    ret[:2] += ret[2:] / 2
    ret[2] /= ret[3]
    return ret


def bbox_iou(box1, box2):
    """
    Calculate IoU (Intersection over Union) between two boxes
    
    Args:
        box1: Array [x1, y1, x2, y2]
        box2: Array [x1, y1, x2, y2]
    
    Returns:
        iou: IoU value [0, 1]
    """
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2
    
    # Calculate intersection
    inter_x_min = max(x1_min, x2_min)
    inter_y_min = max(y1_min, y2_min)
    inter_x_max = min(x1_max, x2_max)
    inter_y_max = min(y1_max, y2_max)
    
    if inter_x_max < inter_x_min or inter_y_max < inter_y_min:
        return 0.0
    
    inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)
    
    # Calculate union
    box1_area = (x1_max - x1_min) * (y1_max - y1_min)
    box2_area = (x2_max - x2_min) * (y2_max - y2_min)
    union_area = box1_area + box2_area - inter_area
    
    return inter_area / union_area if union_area > 0 else 0.0


def bbox_ious(boxes1, boxes2):
    """
    Calculate IoU matrix between two sets of boxes
    
    Args:
        boxes1: Array [N, 4] in format [x1, y1, x2, y2]
        boxes2: Array [M, 4] in format [x1, y1, x2, y2]
    
    Returns:
        ious: Array [N, M] with IoU values
    """
    ious = np.zeros((len(boxes1), len(boxes2)), dtype=np.float32)
    if ious.size == 0:
        return ious
    
    for i, box1 in enumerate(boxes1):
        for j, box2 in enumerate(boxes2):
            ious[i, j] = bbox_iou(box1, box2)
    
    return ious
