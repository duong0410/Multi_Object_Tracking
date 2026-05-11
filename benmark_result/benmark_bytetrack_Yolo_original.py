#!/usr/bin/env python3
"""
ByteTrack + YOLOx implementation for MOT17 benchmark
This script runs YOLOx detector + ByteTrack tracking algorithm on MOT17 dataset and evaluates performance
"""

import os
import numpy as np
import cv2
from collections import defaultdict
import argparse
from pathlib import Path
import motmetrics as mm
from typing import List, Tuple, Optional
import scipy.linalg
import lap
import torch
import torch.nn as nn
from tqdm import tqdm


# ============================================================================
# YOLOx Detector Class
# ============================================================================

class YOLOxDetector:
    """YOLOx detector loaded from PyTorch Lightning checkpoint"""
    
    def __init__(self, checkpoint_path, device='cuda', conf_thresh=0.01, nms_thresh=0.7):
        """
        Args:
            checkpoint_path: Path to PyTorch Lightning checkpoint folder (archive/)
            device: Device to run inference on
            conf_thresh: Confidence threshold for detections
            nms_thresh: NMS threshold
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.conf_thresh = conf_thresh
        self.nms_thresh = nms_thresh
        
        print(f"Loading YOLOx model from checkpoint: {checkpoint_path}")
        
        try:
            checkpoint_path = Path(checkpoint_path)
            
            # Try to load distributed checkpoint (Lightning format with data/ folder)
            if checkpoint_path.is_dir() and (checkpoint_path / 'data.pkl').exists():
                print("Detected distributed checkpoint format (Lightning)")
                
                # Load using Lightning's distributed checkpoint loader
                try:
                    from lightning.fabric.utilities.cloud_io import _load as pl_load
                    state_dict = pl_load(checkpoint_path, map_location=self.device)
                except ImportError:
                    # Fallback: Try PyTorch Lightning 1.x
                    try:
                        from pytorch_lightning.utilities.cloud_io import load as pl_load
                        state_dict = pl_load(checkpoint_path, map_location=self.device)
                    except ImportError:
                        # Manual load for distributed checkpoint
                        print("Loading distributed checkpoint manually...")
                        import pickle
                        import io
                        
                        # Load the metadata
                        with open(checkpoint_path / 'data.pkl', 'rb') as f:
                            # Create custom unpickler for persistent storage
                            class PersistentUnpickler(pickle.Unpickler):
                                def __init__(self, file, data_path):
                                    super().__init__(file)
                                    self.data_path = data_path
                                    
                                def persistent_load(self, pid):
                                    # pid is the persistent ID
                                    # It should be a tuple like ('storage', storage_type, key, ...)
                                    if isinstance(pid, tuple) and len(pid) > 0 and pid[0] == 'storage':
                                        storage_type, key, location, size = pid[1:]
                                        # Load the actual data from the data/ folder
                                        storage_path = self.data_path / str(key)
                                        
                                        # Read the binary data into memory first
                                        with open(storage_path, 'rb') as f:
                                            data_bytes = f.read()
                                        
                                        # Create UntypedStorage from bytes
                                        storage = torch.UntypedStorage.from_buffer(data_bytes, dtype=torch.uint8)
                                        
                                        # Wrap in TypedStorage
                                        return storage_type(wrap_storage=storage)
                                    else:
                                        raise pickle.UnpicklingError(f"Unsupported persistent id: {pid}")
                            
                            unpickler = PersistentUnpickler(f, checkpoint_path / 'data')
                            state_dict = unpickler.load()
                
                # Extract model state dict from checkpoint
                if isinstance(state_dict, dict):
                    # PyTorch Lightning format: {'model': {...}, 'optimizer': {...}, 'epoch': ...}
                    if 'model' in state_dict and isinstance(state_dict['model'], dict):
                        print(f"  Extracting 'model' key from checkpoint (PyTorch Lightning format)")
                        state_dict = state_dict['model']
                    elif 'state_dict' in state_dict:
                        state_dict = state_dict['state_dict']
                    
                    # Remove 'model.' prefix if present in keys
                    if any(k.startswith('model.') for k in state_dict.keys()):
                        print(f"  Removing 'model.' prefix from keys")
                        state_dict = {k.replace('model.', ''): v for k, v in state_dict.items()}
            
            # Try to load regular checkpoint file
            elif checkpoint_path.is_dir():
                # Find the checkpoint file
                ckpt_files = list(checkpoint_path.glob('**/*.ckpt')) + list(checkpoint_path.glob('**/*.pth')) + list(checkpoint_path.glob('**/*.pt'))
                if len(ckpt_files) == 0:
                    raise FileNotFoundError(f"No checkpoint file found in {checkpoint_path}")
                checkpoint_path = ckpt_files[0]
                print(f"Loading checkpoint file: {checkpoint_path}")
                
                # Load the checkpoint
                checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
                
                # Extract model state dict
                if 'state_dict' in checkpoint:
                    state_dict = checkpoint['state_dict']
                    # Remove 'model.' prefix if present
                    state_dict = {k.replace('model.', ''): v for k, v in state_dict.items()}
                else:
                    state_dict = checkpoint
            
            else:
                # Single file checkpoint
                checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
                
                # Extract model state dict
                if 'state_dict' in checkpoint:
                    state_dict = checkpoint['state_dict']
                    # Remove 'model.' prefix if present
                    state_dict = {k.replace('model.', ''): v for k, v in state_dict.items()}
                else:
                    state_dict = checkpoint
            
            # Auto-detect model size from checkpoint architecture
            # Check stem conv layer width to determine model variant
            model_variant = 'yolox-s'  # default
            if 'backbone.backbone.stem.conv.conv.weight' in state_dict:
                stem_width = state_dict['backbone.backbone.stem.conv.conv.weight'].shape[0]
                # YOLOx variants by stem width (first conv layer output channels):
                # s: 32, m: 48, l: 64, x: 80
                if stem_width == 32:
                    model_variant = 'yolox-s'
                elif stem_width == 48:
                    model_variant = 'yolox-m'
                elif stem_width == 64:
                    model_variant = 'yolox-l'
                elif stem_width == 80:
                    model_variant = 'yolox-x'
                print(f"  Detected model variant: {model_variant.upper()} (stem width={stem_width})")
            
            # Try to create model from YOLOx library
            try:
                from yolox.exp import get_exp
                exp = get_exp(None, model_variant)
                exp.num_classes = 1  # CRITICAL: Model trained for single class (person detection)
                self.model = exp.get_model()
                print(f"Using YOLOx model from yolox.exp: {model_variant.upper()}")
                print(f"  Model configured for {exp.num_classes} class (person detection)")
            except Exception as exp_err:
                # Fallback: create a basic YOLOx model structure with correct variant
                print(f"Could not use get_exp: {exp_err}")
                print(f"Creating basic YOLOx model structure for {model_variant.upper()}...")
                try:
                    from yolox.models import YOLOX, YOLOPAFPN, YOLOXHead
                    import torch.nn as nn
                    
                    def init_yolo(M):
                        for m in M.modules():
                            if isinstance(m, nn.BatchNorm2d):
                                m.eps = 1e-3
                                m.momentum = 0.03
                    
                    # Model variant parameters (depth, width)
                    variant_params = {
                        'yolox-s': (0.33, 0.5),
                        'yolox-m': (0.67, 0.75),
                        'yolox-l': (1.0, 1.0),
                        'yolox-x': (1.33, 1.25)
                    }
                    depth, width = variant_params.get(model_variant, (0.33, 0.5))
                    
                    in_channels = [256, 512, 1024]
                    backbone = YOLOPAFPN(depth=depth, width=width, in_channels=in_channels)
                    head = YOLOXHead(num_classes=1, width=width, in_channels=in_channels)  # Single class
                    self.model = YOLOX(backbone, head)
                    self.model.apply(init_yolo)
                    print(f"✓ Created basic {model_variant.upper()} model (1 class, depth={depth}, width={width})")
                except ImportError:
                    raise ImportError(
                        "YOLOx library is required. Please install it:\n"
                        "  pip install yolox\n"
                        "  or clone from: https://github.com/Megvii-BaseDetection/YOLOX"
                    )
            
            # Load state dict
            print(f"Loading weights from checkpoint...")
            
            # Debug: Check key matching
            model_keys = set(self.model.state_dict().keys())
            ckpt_keys = set(state_dict.keys())
            
            print(f"  Model expects {len(model_keys)} keys")
            print(f"  Checkpoint has {len(ckpt_keys)} keys")
            
            missing_keys = model_keys - ckpt_keys
            unexpected_keys = ckpt_keys - model_keys
            
            if missing_keys:
                print(f"  ⚠️  Missing {len(missing_keys)} keys in checkpoint")
                if len(missing_keys) <= 10:
                    for k in list(missing_keys)[:10]:
                        print(f"      - {k}")
            
            if unexpected_keys:
                print(f"  ⚠️  Unexpected {len(unexpected_keys)} keys in checkpoint")
                if len(unexpected_keys) <= 10:
                    for k in list(unexpected_keys)[:10]:
                        print(f"      + {k}")
            
            result = self.model.load_state_dict(state_dict, strict=False)
            print(f"  Missing keys: {len(result.missing_keys)}")
            print(f"  Unexpected keys: {len(result.unexpected_keys)}")

            self.model.to(self.device)
            if self.device.type == 'cuda':
                self.model.half()
            self.model.eval()
        except Exception as e:
            print(f"❌ Error loading YOLOx checkpoint: {e}")
            import traceback
            traceback.print_exc()
            raise

        # Test shape
        self.test_size = (800, 1440)  # MOT17 default size
        
    def preprocess(self, img):
        """Preprocess image for YOLOx"""
        if len(img.shape) == 3:
            padded_img = np.ones((self.test_size[0], self.test_size[1], 3), dtype=np.uint8) * 114
        else:
            padded_img = np.ones(self.test_size, dtype=np.uint8) * 114
        
        r = min(self.test_size[0] / img.shape[0], self.test_size[1] / img.shape[1])
        resized_img = cv2.resize(
            img,
            (int(img.shape[1] * r), int(img.shape[0] * r)),
            interpolation=cv2.INTER_LINEAR,
        ).astype(np.uint8)
        
        padded_img[: int(img.shape[0] * r), : int(img.shape[1] * r)] = resized_img
        
        # Convert to CHW format and normalize to [0, 1]
        padded_img = padded_img.transpose((2, 0, 1))
        padded_img = np.ascontiguousarray(padded_img, dtype=np.float32) / 255.0  # Normalize!
        
        return padded_img, r
    
    def postprocess(self, outputs, img_size, ratio, debug_first_frame=False):
        """Postprocess YOLOx raw outputs - decode and filter detections"""
        # Raw outputs shape: (1, n_anchors, 5+num_classes) where 5 = [x, y, w, h, obj]
        # For 1 class: (1, n_anchors, 6) = [x, y, w, h, obj, cls0]
        if outputs is None or len(outputs) == 0:
            return np.empty((0, 5))
        
        predictions = outputs[0]  # Remove batch dimension -> (n_anchors, 6)
        
        if debug_first_frame:
            print(f"    [DEBUG] Raw predictions shape: {predictions.shape}")
            print(f"    [DEBUG] Raw predictions stats - min: {predictions.min():.4f}, max: {predictions.max():.4f}, mean: {predictions.mean():.4f}")
        
        # YOLOx raw format: [x_center, y_center, w, h, obj_conf, cls0, cls1, ...]
        # Coordinates are in stride/grid space, NOT pixel space yet!
        boxes = predictions[:, :4]  # [x, y, w, h] in grid space
        
        # Model output is already post-sigmoid (probabilities), NOT logits
        # col4 range: [0, 1], col5 range: [0, 1]
        # DO NOT apply sigmoid again (was causing double sigmoid bug)
        obj_conf = predictions[:, 4]  # Already probabilities from model
        class_conf = predictions[:, 5:]  # Already probabilities from model
        
        if debug_first_frame:
            print(f"    [DEBUG] col4 (obj_conf) range: [{predictions[:, 4].min():.4f}, {predictions[:, 4].max():.4f}]")
            print(f"    [DEBUG] col5 (class_conf) range: [{predictions[:, 5].min():.4f}, {predictions[:, 5].max():.4f}]")
        
        # Get best class
        class_scores = np.max(class_conf, axis=1)
        class_ids = np.argmax(class_conf, axis=1)
        
        # Final score = objectness * class_confidence (direct multiplication)
        scores = obj_conf * class_scores
        
        if debug_first_frame:
            print(f"    [DEBUG] obj_conf max: {obj_conf.max():.4f}, class_conf max: {class_scores.max():.4f}")
            print(f"    [DEBUG] Score range: [{scores.min():.4f}, {scores.max():.4f}]")
        
        # Filter by confidence threshold
        valid_mask = scores > self.conf_thresh
        boxes = boxes[valid_mask]
        scores = scores[valid_mask]
        class_ids = class_ids[valid_mask]
        
        if debug_first_frame:
            print(f"    [DEBUG] After conf filter (>{self.conf_thresh}): {len(boxes)} detections")
        
        # For single-class model, all are person class
        # For multi-class, filter to person (class_id = 0)
        if len(boxes) > 0 and class_conf.shape[1] > 1:  # Multi-class
            person_mask = class_ids == 0
            boxes = boxes[person_mask]
            scores = scores[person_mask]
            if debug_first_frame:
                print(f"    [DEBUG] After person filter: {len(boxes)} detections")
        
        if len(boxes) == 0:
            return np.empty((0, 5))
        
        # Limit to top-k before NMS for speed
        max_dets = 1000
        if len(scores) > max_dets:
            top_k = np.argsort(scores)[-max_dets:]
            boxes = boxes[top_k]
            scores = scores[top_k]
            if debug_first_frame:
                print(f"    [DEBUG] After top-k: {len(boxes)} detections")
        
        if debug_first_frame:
            print(f"    [DEBUG] Boxes in grid space: x=[{boxes[:, 0].min():.1f}, {boxes[:, 0].max():.1f}], y=[{boxes[:, 1].min():.1f}, {boxes[:, 1].max():.1f}]")
            print(f"    [DEBUG] Box sizes: w=[{boxes[:, 2].min():.1f}, {boxes[:, 2].max():.1f}], h=[{boxes[:, 3].min():.1f}, {boxes[:, 3].max():.1f}]")
        
        # Convert from [x_center, y_center, w, h] to [x1, y1, x2, y2]
        # NOTE: Boxes are already in pixel coordinates relative to padded image!
        # YOLOx outputs are already decoded by the model
        x_center = boxes[:, 0]
        y_center = boxes[:, 1]
        w = boxes[:, 2]
        h = boxes[:, 3]
        
        x1 = x_center - w / 2.0
        y1 = y_center - h / 2.0
        x2 = x_center + w / 2.0
        y2 = y_center + h / 2.0
        boxes = np.stack([x1, y1, x2, y2], axis=1)
        
        if debug_first_frame:
            print(f"    [DEBUG] After center->corner: x=[{boxes[:, 0].min():.1f}, {boxes[:, 2].max():.1f}], y=[{boxes[:, 1].min():.1f}, {boxes[:, 3].max():.1f}]")
        
        # Scale from padded image size to original image size
        boxes = boxes / ratio
        
        if debug_first_frame:
            print(f"    [DEBUG] After scaling by ratio={ratio:.4f}: x=[{boxes[:, 0].min():.1f}, {boxes[:, 2].max():.1f}], y=[{boxes[:, 1].min():.1f}, {boxes[:, 3].max():.1f}]")
            print(f"    [DEBUG] Original image: {img_size['height']}x{img_size['width']}")
        
        # Clip boxes to image boundaries
        boxes[:, 0] = np.clip(boxes[:, 0], 0, img_size['width'])
        boxes[:, 1] = np.clip(boxes[:, 1], 0, img_size['height'])
        boxes[:, 2] = np.clip(boxes[:, 2], 0, img_size['width'])
        boxes[:, 3] = np.clip(boxes[:, 3], 0, img_size['height'])
        
        # Filter out invalid boxes (x2 <= x1 or y2 <= y1)
        valid_boxes = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
        boxes = boxes[valid_boxes]
        scores = scores[valid_boxes]
        
        if debug_first_frame:
            print(f"    [DEBUG] After clipping and validation: {len(boxes)} detections")
        
        if len(boxes) == 0:
            return np.empty((0, 5))
        
        # Apply NMS
        from torchvision.ops import nms
        import torch
        keep = nms(
            torch.from_numpy(boxes).float().cuda(),
            torch.from_numpy(scores).float().cuda(),
            self.nms_thresh
        )
        boxes = boxes[keep.cpu().numpy()]
        scores = scores[keep.cpu().numpy()]
        
        if debug_first_frame:
            print(f"    [DEBUG] After NMS: {len(boxes)} detections")
            if len(boxes) > 0:
                print(f"    [DEBUG] Final score range: [{scores.min():.4f}, {scores.max():.4f}]")
        
        # Concatenate to [x1, y1, x2, y2, score]
        detections = np.concatenate([boxes, scores[:, None]], axis=1)
        
        return detections
    
    def detect(self, img, debug=False):
        """
        Run detection on an image
        
        Args:
            img: BGR image from cv2.imread
            debug: if True, print debug info
            
        Returns:
            detections: numpy array of shape (N, 5) with [x1, y1, x2, y2, score]
        """
        img_info = {"height": img.shape[0], "width": img.shape[1]}
        
        # Preprocess
        img_preprocessed, ratio = self.preprocess(img)
        
        if debug:
            print(f"    [DEBUG] Input image shape: {img.shape}")
            print(f"    [DEBUG] Preprocessed shape: {img_preprocessed.shape}")
            print(f"    [DEBUG] Preprocessed stats - min: {img_preprocessed.min():.4f}, max: {img_preprocessed.max():.4f}, mean: {img_preprocessed.mean():.4f}")
            print(f"    [DEBUG] Ratio: {ratio}")
        
        # Convert to tensor
        img_tensor = torch.from_numpy(img_preprocessed).unsqueeze(0)
        if self.device.type == 'cuda':
            img_tensor = img_tensor.half()
        else:
            img_tensor = img_tensor.float()
        img_tensor = img_tensor.to(self.device)
        # Inference - YOLOx model outputs raw predictions
        with torch.no_grad():
            outputs = self.model(img_tensor)
            
            # YOLOx outputs raw predictions in format (batch, n_anchors, 5+num_classes)
            # For 1 class: (1, 8400, 6) = [x, y, w, h, obj, cls0]
            # These are NOT yet decoded to pixel coordinates!
            
        if debug:
            if outputs is not None:
                print(f"    [DEBUG] Raw model outputs shape: {outputs.shape}")
                print(f"    [DEBUG] Raw outputs range: min={outputs.min():.2f}, max={outputs.max():.2f}")
        
        # Postprocess
        if outputs is None:
            return np.empty((0, 5))
        
        # Convert to numpy
        outputs = outputs.cpu().numpy()
        
        detections = self.postprocess(outputs, img_info, ratio, debug_first_frame=debug)
        
        return detections
# ============================================================================
# ByteTrack Core Classes
# ============================================================================

class STrack:
    """Single target track with Kalman filter state"""
    
    shared_kalman = None
    track_id_count = 0
    
    def __init__(self, tlwh, score):
        # tlwh format: top-left width height
        self._tlwh = np.asarray(tlwh, dtype=np.float32)
        self.kalman_filter = None
        self.mean, self.covariance = None, None
        self.is_activated = False
        
        self.score = score
        self.tracklet_len = 0
        
        # These will be set in activate()
        self.track_id = 0
        self.frame_id = 0
        self.start_frame = 0
        
        self.state = TrackState.New
        
    def predict(self):
        """Predict next state using Kalman filter"""
        mean_state = self.mean.copy()
        if self.state != TrackState.Tracked:
            mean_state[7] = 0
        self.mean, self.covariance = self.kalman_filter.predict(mean_state, self.covariance)
        
    @staticmethod
    def multi_predict(stracks):
        """Predict multiple tracks"""
        if len(stracks) > 0:
            multi_mean = np.asarray([st.mean.copy() for st in stracks])
            multi_covariance = np.asarray([st.covariance for st in stracks])
            for i, st in enumerate(stracks):
                if st.state != TrackState.Tracked:
                    multi_mean[i][7] = 0
            multi_mean, multi_covariance = STrack.shared_kalman.multi_predict(multi_mean, multi_covariance)
            for i, (mean, cov) in enumerate(zip(multi_mean, multi_covariance)):
                stracks[i].mean = mean
                stracks[i].covariance = cov
                
    def activate(self, kalman_filter, frame_id):
        """Start a new tracklet"""
        self.kalman_filter = kalman_filter
        self.track_id = self.next_id()
        self.mean, self.covariance = self.kalman_filter.initiate(self.tlwh_to_xyah(self._tlwh))
        
        self.tracklet_len = 0
        self.state = TrackState.Tracked
        # ByteTrack official: activate on frame 1, otherwise need confirmation
        if frame_id == 1:
            self.is_activated = True
        # For other frames, will be activated in update() after min_hits
        self.frame_id = frame_id
        self.start_frame = frame_id
        
    def re_activate(self, new_track, frame_id, new_id=False):
        """Reactivate a lost track"""
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, self.tlwh_to_xyah(new_track.tlwh)
        )
        # ByteTrack official: reset tracklet_len to 0
        self.tracklet_len = 0
        self.state = TrackState.Tracked
        self.is_activated = True
        self.frame_id = frame_id
        if new_id:
            self.track_id = self.next_id()
        self.score = new_track.score
        
    def update(self, new_track, frame_id, min_hits=1):
        """
        Update a matched track
        
        Args:
            new_track: New detection to update with
            frame_id: Current frame ID
            min_hits: Minimum hits before track is activated (default: 1, matching BoxMOT)
        """
        self.frame_id = frame_id
        self.tracklet_len += 1
        
        new_tlwh = new_track.tlwh
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, self.tlwh_to_xyah(new_tlwh)
        )
        
        # BoxMOT-style: Activate immediately (min_hits=1 by default)
        if self.tracklet_len >= min_hits:
            self.is_activated = True
            self.state = TrackState.Tracked
        
        self.score = new_track.score
    
    def mark_lost(self):
        """Mark track as lost"""
        self.state = TrackState.Lost
    
    def mark_removed(self):
        """Mark track as removed"""
        self.state = TrackState.Removed
        
    @property
    def tlwh(self):
        """Get current position in tlwh format"""
        if self.mean is None:
            return self._tlwh.copy()
        ret = self.mean[:4].copy()
        ret[2] *= ret[3]
        ret[:2] -= ret[2:] / 2
        return ret
    
    @property
    def tlbr(self):
        """Convert tlwh to tlbr (top-left, bottom-right)"""
        ret = self.tlwh.copy()
        ret[2:] += ret[:2]
        return ret
    
    @staticmethod
    def tlwh_to_xyah(tlwh):
        """Convert tlwh to xyah (center x, center y, aspect ratio, height)"""
        ret = np.asarray(tlwh).copy()
        ret[:2] += ret[2:] / 2
        ret[2] /= ret[3]
        return ret
    
    @property
    def end_frame(self):
        """Get the last frame id when the track was updated"""
        return self.frame_id
    
    @staticmethod
    def next_id():
        STrack.track_id_count += 1
        return STrack.track_id_count
    
    def __repr__(self):
        return f'Track_{self.track_id}_({self.start_frame}-{self.frame_id})'


class TrackState:
    """Enumeration type for track state"""
    New = 0
    Tracked = 1
    Lost = 2
    Removed = 3


class KalmanFilter:
    """Kalman filter for track state estimation"""
    
    def __init__(self):
        ndim, dt = 4, 1.
        
        # Create Kalman filter model matrices
        self._motion_mat = np.eye(2 * ndim, 2 * ndim)
        for i in range(ndim):
            self._motion_mat[i, ndim + i] = dt
        self._update_mat = np.eye(ndim, 2 * ndim)
        
        # Motion and observation uncertainty
        self._std_weight_position = 1. / 20
        self._std_weight_velocity = 1. / 160
        
    def initiate(self, measurement):
        """Create track from unassociated measurement"""
        mean_pos = measurement
        mean_vel = np.zeros_like(mean_pos)
        mean = np.r_[mean_pos, mean_vel]
        
        std = [
            2 * self._std_weight_position * measurement[3],
            2 * self._std_weight_position * measurement[3],
            1e-2,
            2 * self._std_weight_position * measurement[3],
            10 * self._std_weight_velocity * measurement[3],
            10 * self._std_weight_velocity * measurement[3],
            1e-5,
            10 * self._std_weight_velocity * measurement[3]
        ]
        covariance = np.diag(np.square(std))
        return mean, covariance
    
    def predict(self, mean, covariance):
        """Run Kalman filter prediction step"""
        std_pos = [
            self._std_weight_position * mean[3],
            self._std_weight_position * mean[3],
            1e-2,
            self._std_weight_position * mean[3]
        ]
        std_vel = [
            self._std_weight_velocity * mean[3],
            self._std_weight_velocity * mean[3],
            1e-5,
            self._std_weight_velocity * mean[3]
        ]
        motion_cov = np.diag(np.square(np.r_[std_pos, std_vel]))
        
        mean = np.dot(self._motion_mat, mean)
        covariance = np.linalg.multi_dot((
            self._motion_mat, covariance, self._motion_mat.T)) + motion_cov
        
        return mean, covariance
    
    def project(self, mean, covariance):
        """Project state distribution to measurement space"""
        std = [
            self._std_weight_position * mean[3],
            self._std_weight_position * mean[3],
            1e-1,
            self._std_weight_position * mean[3]
        ]
        innovation_cov = np.diag(np.square(std))
        
        mean = np.dot(self._update_mat, mean)
        covariance = np.linalg.multi_dot((
            self._update_mat, covariance, self._update_mat.T))
        return mean, covariance + innovation_cov
    
    def update(self, mean, covariance, measurement):
        """Run Kalman filter correction step"""
        projected_mean, projected_cov = self.project(mean, covariance)
        
        chol_factor, lower = scipy.linalg.cho_factor(
            projected_cov, lower=True, check_finite=False)
        kalman_gain = scipy.linalg.cho_solve(
            (chol_factor, lower), np.dot(covariance, self._update_mat.T).T,
            check_finite=False).T
        innovation = measurement - projected_mean
        
        new_mean = mean + np.dot(innovation, kalman_gain.T)
        new_covariance = covariance - np.linalg.multi_dot((
            kalman_gain, projected_cov, kalman_gain.T))
        return new_mean, new_covariance
    
    def multi_predict(self, mean, covariance):
        """Run prediction step for multiple tracks"""
        std_pos = [
            self._std_weight_position * mean[:, 3],
            self._std_weight_position * mean[:, 3],
            1e-2 * np.ones_like(mean[:, 3]),
            self._std_weight_position * mean[:, 3]
        ]
        std_vel = [
            self._std_weight_velocity * mean[:, 3],
            self._std_weight_velocity * mean[:, 3],
            1e-5 * np.ones_like(mean[:, 3]),
            self._std_weight_velocity * mean[:, 3]
        ]
        sqr = np.square(np.r_[std_pos, std_vel]).T
        
        motion_cov = []
        for i in range(len(mean)):
            motion_cov.append(np.diag(sqr[i]))
        motion_cov = np.asarray(motion_cov)
        
        mean = np.dot(mean, self._motion_mat.T)
        left = np.dot(self._motion_mat, covariance).transpose((1, 0, 2))
        covariance = np.dot(left, self._motion_mat.T) + motion_cov
        
        return mean, covariance


class BYTETracker:
    """
    ByteTrack multi-object tracker
    
    ByteTrack uses 2-phase matching strategy:
    - Phase 1: Match tracks with high-score detections (det_conf_high)
    - Phase 2: Match remaining tracks with low-score detections (det_conf_low)
    
    This approach reduces miss/fragmentation during occlusion and motion blur.
    """
    
    def __init__(self, det_conf_high=0.6, det_conf_low=0.1, new_track_thresh=0.6,
                 match_thresh_high=0.9, match_thresh_low=0.5, track_buffer=30, min_hits=1):
        """
        Args:
            det_conf_high: Confidence threshold for high-score detections (default: 0.6, ByteTrack official track_thresh)
            det_conf_low: Confidence threshold for low-score detections (default: 0.1, ByteTrack official)
            new_track_thresh: Threshold for creating new tracks (default: 0.6, same as track_thresh)
            match_thresh_high: Cost threshold for Phase 1 association (default: 0.9, ByteTrack official)
            match_thresh_low: Cost threshold for Phase 2 association (default: 0.5, ByteTrack official, hardcoded)
            track_buffer: Number of frames to keep lost tracks (default: 30, ByteTrack official)
            min_hits: Always 1 - ByteTrack activates tracks immediately at frame 1
        """
        self.tracked_stracks = []  # type: list[STrack]
        self.lost_stracks = []  # type: list[STrack]
        self.removed_stracks = []  # type: list[STrack]
        
        self.frame_id = 0
        
        # ByteTrack thresholds
        self.det_conf_high = det_conf_high
        self.det_conf_low = det_conf_low
        self.new_track_thresh = new_track_thresh  # For creating new tracks
        self.match_thresh_high = match_thresh_high
        self.match_thresh_low = match_thresh_low
        self.max_time_lost = track_buffer
        self.min_hits = min_hits  # BoxMOT-style: minimum hits before confirmed
        
        self.kalman_filter = KalmanFilter()
        STrack.shared_kalman = self.kalman_filter
        
    def update(self, output_results, img_shape):
        """
        Update tracker with new detections using ByteTrack algorithm (Algorithm 1)
        
        Following the pseudo-code:
        - Line 6-13: Split detections D into D_high and D_low by threshold τ
        - Line 14-16: Predict T with Kalman Filter
        - Line 17: First association - Match T and D_high (Similarity#1)
        - Line 20: Second association - Match T_remain and D_low (Similarity#2)
        - Line 22: Delete unmatched tracks T_re-remain from T
        - Line 23-25: Initialize new tracks from D_remain
        
        Args:
            output_results: numpy array of detections [x1, y1, x2, y2, score, class]
            img_shape: tuple of (height, width)
            
        Returns:
            list of active tracks T
        """
        self.frame_id += 1  # Frame j_t in algorithm
        activated_stracks = []
        refind_stracks = []
        lost_stracks = []
        removed_stracks = []
        
        # Debug first frame
        if self.frame_id == 1:
            print(f"    [DEBUG TRACKER] Frame {self.frame_id}: Received {len(output_results)} detections")
            if len(output_results) > 0:
                print(f"    [DEBUG TRACKER] Detection shape: {output_results.shape}")
                print(f"    [DEBUG TRACKER] First detection: {output_results[0]}")
        
        # Parse detection results from detector Det(f_t)
        if output_results.shape[1] == 5:
            scores = output_results[:, 4]
            bboxes = output_results[:, :4]
        else:
            scores = output_results[:, 4] * output_results[:, 5]
            bboxes = output_results[:, :4]
        
        if self.frame_id == 1 and len(output_results) > 0:
            print(f"    [DEBUG TRACKER] Parsed {len(bboxes)} boxes, score range: [{scores.min():.4f}, {scores.max():.4f}]")
        
        # ======================================================================
        # Algorithm 1, Line 6-13: Split detections by score threshold
        # for d in D_t do
        #     if d.score > τ then D_high ← D_high ∪ {d}
        #     else D_low ← D_low ∪ {d}
        # ======================================================================
        # D_high: High-score detections (threshold τ = det_conf_high ≈ 0.5)
        remain_inds_high = scores > self.det_conf_high
        dets_high = bboxes[remain_inds_high]
        scores_high = scores[remain_inds_high]
        
        # D_low: Low-score detections (det_conf_low ≈ 0.1 < score ≤ det_conf_high)
        # These help recover tracks during occlusion/motion blur
        remain_inds_low = np.logical_and(scores > self.det_conf_low, scores <= self.det_conf_high)
        dets_low = bboxes[remain_inds_low]
        scores_low = scores[remain_inds_low]
        
        if self.frame_id == 1:
            print(f"    [DEBUG TRACKER] High-score dets (>{self.det_conf_high}): {len(dets_high)}")
            print(f"    [DEBUG TRACKER] Low-score dets ({self.det_conf_low}<score<={self.det_conf_high}): {len(dets_low)}")
        
        # Convert to STrack objects
        if len(dets_high) > 0:
            detections_high = [STrack(tlbr_to_tlwh(tlbr), s) for tlbr, s in zip(dets_high, scores_high)]
        else:
            detections_high = []
            
        if len(dets_low) > 0:
            detections_low = [STrack(tlbr_to_tlwh(tlbr), s) for tlbr, s in zip(dets_low, scores_low)]
        else:
            detections_low = []
        
        if self.frame_id == 1:
            print(f"    [DEBUG TRACKER] Created {len(detections_high)} high STracks, {len(detections_low)} low STracks")
            
        # Separate confirmed tracks (T) and unconfirmed tracks
        # T: tracked stracks that are already activated
        unconfirmed = []
        tracked_stracks = []  # This is T in the algorithm
        for track in self.tracked_stracks:
            if not track.is_activated:
                unconfirmed.append(track)
            else:
                tracked_stracks.append(track)
        
        # ======================================================================
        # CRITICAL FIX: First association with BOTH tracked AND lost tracks!
        # ======================================================================
        # Combine tracked and lost tracks into one pool
        strack_pool = joint_stracks(tracked_stracks, self.lost_stracks)
        
        # Predict ALL tracks (both tracked and lost) with Kalman Filter
        STrack.multi_predict(strack_pool)
        
        # ======================================================================
        # Algorithm 1, Line 17: First Association
        # Associate (T + Lost) with D_high using high IoU threshold
        # ======================================================================
        dists = iou_distance(strack_pool, detections_high)
        matches, u_track, u_detection_high = linear_assignment(dists, thresh=self.match_thresh_high)
        
        # Update matched tracks
        for itracked, idet in matches:
            track = strack_pool[itracked]
            det = detections_high[idet]
            if track.state == TrackState.Tracked:
                track.update(det, self.frame_id, self.min_hits)
                activated_stracks.append(track)
            else:  # Lost track found again
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)
        
        # ======================================================================
        # Algorithm 1, Line 18-19: Get remaining detections and tracks
        # D_remain ← remaining detection boxes from D_high
        # T_remain ← remaining TRACKED tracks only (not lost!)
        # ======================================================================
        dets_remain_high = [detections_high[i] for i in u_detection_high]
        
        # CRITICAL: Only take TRACKED tracks from unmatched pool
        # ByteTrack official: lost tracks don't go to second association
        r_tracked_stracks = [strack_pool[i] for i in u_track if strack_pool[i].state == TrackState.Tracked]
        
        # ======================================================================
        # Algorithm 1, Line 20: Second Association
        # Associate remaining TRACKED tracks with D_low
        # ======================================================================
        dists_low = iou_distance(r_tracked_stracks, detections_low)
        matches_low, u_track_remain, u_detection_low = linear_assignment(dists_low, thresh=self.match_thresh_low)
        
        # Update tracks matched with low-score detections
        for itracked, idet in matches_low:
            track = r_tracked_stracks[itracked]
            det = detections_low[idet]
            if track.state == TrackState.Tracked:
                track.update(det, self.frame_id, self.min_hits)
                activated_stracks.append(track)
            else:  # Should not happen, but handle it
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)
        
        # ======================================================================
        # Mark unmatched TRACKED tracks as lost
        # ======================================================================
        for it in u_track_remain:
            track = r_tracked_stracks[it]
            if not track.state == TrackState.Lost:
                track.mark_lost()
                lost_stracks.append(track)
        
        # ======================================================================
        # Handle unconfirmed tracks (newly created but not yet stable)
        # Match with REMAINING high detections
        # ======================================================================
        detections = [detections_high[i] for i in u_detection_high]
        dists = iou_distance(unconfirmed, detections)
        # Use same threshold as second association
        matches, u_unconfirmed, u_detection = linear_assignment(dists, thresh=0.5)
        
        for itracked, idet in matches:
            unconfirmed[itracked].update(detections[idet], self.frame_id, self.min_hits)
            # Add to activated regardless - they need to stay in tracked_stracks to accumulate hits
            activated_stracks.append(unconfirmed[itracked])
        
        for it in u_unconfirmed:
            track = unconfirmed[it]
            track.mark_removed()
            removed_stracks.append(track)
        
        # ======================================================================
        # Initialize new tracks
        # ByteTrack official: Use new_track_thresh (higher than track_high_thresh)
        # This prevents creating tracks from false positive detections
        # ======================================================================
        for inew in u_detection:
            track = detections[inew]
            if track.score < self.new_track_thresh:
                continue
            track.activate(self.kalman_filter, self.frame_id)
            # Add to activated - track needs to be in tracked_stracks to accumulate hits
            # Output filtering will handle only returning confirmed tracks
            activated_stracks.append(track)
            
        # Step 5: Remove lost tracks that exceeded max_time_lost
        for track in self.lost_stracks:
            if self.frame_id - track.end_frame > self.max_time_lost:
                track.mark_removed()
                removed_stracks.append(track)
                
        # Update state
        self.tracked_stracks = [t for t in self.tracked_stracks if t.state == TrackState.Tracked]
        self.tracked_stracks = joint_stracks(self.tracked_stracks, activated_stracks)
        self.tracked_stracks = joint_stracks(self.tracked_stracks, refind_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.tracked_stracks)
        self.lost_stracks.extend(lost_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.removed_stracks)
        self.removed_stracks.extend(removed_stracks)
        self.tracked_stracks, self.lost_stracks = remove_duplicate_stracks(self.tracked_stracks, self.lost_stracks)
        
        # Get current active tracks
        # BoxMOT-style: Output all activated tracks (min_hits=1 means immediate activation)
        output_stracks = [track for track in self.tracked_stracks 
                         if track.is_activated]
        
        return output_stracks


# ============================================================================
# Utility Functions
# ============================================================================

def tlbr_to_tlwh(tlbr):
    """Convert tlbr to tlwh format"""
    ret = np.asarray(tlbr).copy()
    ret[2:] -= ret[:2]
    return ret


def iou_distance(atracks, btracks):
    """
    Compute cost based on IoU between tracks
    """
    if len(atracks) > 0 and isinstance(atracks[0], np.ndarray) or len(btracks) > 0 and isinstance(btracks[0], np.ndarray):
        atlbrs = atracks
        btlbrs = btracks
    else:
        atlbrs = [track.tlbr for track in atracks]
        btlbrs = [track.tlbr for track in btracks]
        
    ious = np.zeros((len(atlbrs), len(btlbrs)), dtype=np.float32)
    if ious.size == 0:
        return ious
    
    ious = bbox_ious(np.ascontiguousarray(atlbrs, dtype=np.float32),
                     np.ascontiguousarray(btlbrs, dtype=np.float32))
    
    cost_matrix = 1 - ious
    return cost_matrix


def bbox_ious(atlbrs, btlbrs):
    """Compute IoU between two sets of boxes"""
    if len(atlbrs) == 0 or len(btlbrs) == 0:
        return np.zeros((len(atlbrs), len(btlbrs)), dtype=np.float32)

    atlbrs = np.asarray(atlbrs, dtype=np.float32)
    btlbrs = np.asarray(btlbrs, dtype=np.float32)

    tl = np.maximum(atlbrs[:, None, :2], btlbrs[None, :, :2])
    br = np.minimum(atlbrs[:, None, 2:], btlbrs[None, :, 2:])
    wh = np.clip(br - tl, a_min=0, a_max=None)
    inter = wh[:, :, 0] * wh[:, :, 1]

    area_a = (atlbrs[:, 2] - atlbrs[:, 0]) * (atlbrs[:, 3] - atlbrs[:, 1])
    area_b = (btlbrs[:, 2] - btlbrs[:, 0]) * (btlbrs[:, 3] - btlbrs[:, 1])
    union = area_a[:, None] + area_b[None, :] - inter
    union = np.clip(union, a_min=1e-6, a_max=None)
    return inter / union


def bbox_iou(box1, box2):
    """Compute IoU between two boxes"""
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2
    
    inter_x_min = max(x1_min, x2_min)
    inter_y_min = max(y1_min, y2_min)
    inter_x_max = min(x1_max, x2_max)
    inter_y_max = min(y1_max, y2_max)
    
    if inter_x_max < inter_x_min or inter_y_max < inter_y_min:
        return 0.0
    
    inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)
    box1_area = (x1_max - x1_min) * (y1_max - y1_min)
    box2_area = (x2_max - x2_min) * (y2_max - y2_min)
    union_area = box1_area + box2_area - inter_area
    
    return inter_area / union_area if union_area > 0 else 0.0


def linear_assignment(cost_matrix, thresh):
    """
    Perform linear assignment using Hungarian algorithm
    
    Args:
        cost_matrix: Cost matrix where cost = 1 - IoU
        thresh: Cost threshold - matches accepted if cost < thresh
                This means IoU > (1 - thresh)
                Example: thresh=0.8 means IoU > 0.2
                         thresh=0.5 means IoU > 0.5
    """
    if cost_matrix.size == 0:
        return np.empty((0, 2), dtype=int), tuple(range(cost_matrix.shape[0])), tuple(range(cost_matrix.shape[1]))
    
    matches, unmatched_a, unmatched_b = [], [], []
    cost, x, y = lap.lapjv(cost_matrix, extend_cost=True, cost_limit=thresh)
    
    for ix, mx in enumerate(x):
        if mx >= 0:
            matches.append([ix, mx])
    unmatched_a = np.where(x < 0)[0]
    unmatched_b = np.where(y < 0)[0]
    matches = np.asarray(matches)
    
    return matches, unmatched_a, unmatched_b


def joint_stracks(tlista, tlistb):
    """Join two lists of tracks"""
    exists = {}
    res = []
    for t in tlista:
        exists[t.track_id] = 1
        res.append(t)
    for t in tlistb:
        tid = t.track_id
        if not exists.get(tid, 0):
            exists[tid] = 1
            res.append(t)
    return res


def sub_stracks(tlista, tlistb):
    """Remove tracks in tlistb from tlista"""
    stracks = {}
    for t in tlista:
        stracks[t.track_id] = t
    for t in tlistb:
        tid = t.track_id
        if stracks.get(tid, 0):
            del stracks[tid]
    return list(stracks.values())


def remove_duplicate_stracks(stracksa, stracksb):
    """Remove duplicate tracks"""
    pdist = iou_distance(stracksa, stracksb)
    pairs = np.where(pdist < 0.15)
    dupa, dupb = [], []
    for p, q in zip(*pairs):
        timep = stracksa[p].frame_id - stracksa[p].start_frame
        timeq = stracksb[q].frame_id - stracksb[q].start_frame
        if timep > timeq:
            dupb.append(q)
        else:
            dupa.append(p)
    resa = [t for i, t in enumerate(stracksa) if not i in dupa]
    resb = [t for i, t in enumerate(stracksb) if not i in dupb]
    return resa, resb


# ============================================================================
# MOT17 Dataset Handler
# ============================================================================

class MOT17Dataset:
    """Handle MOT17 dataset loading and processing"""
    
    def __init__(self, data_root, split='train', use_detector=False):
        self.data_root = Path(data_root)
        self.split = split
        self.use_detector = use_detector
        self.split_dir = None  # Will be set by _get_sequences
        self.sequences = self._get_sequences()
        
    def _get_sequences(self):
        """Get all sequence names for the split"""
        split_dir = self.data_root / self.split
        
        # Try different directory structures
        if not split_dir.exists():
            # Check if structure is MOT17/MOT17/train or MOT20/MOT20/train
            dataset_name = self.data_root.name  # Get the last part of path
            alt_split_dir = self.data_root / dataset_name / self.split
            if alt_split_dir.exists():
                split_dir = alt_split_dir
            else:
                raise ValueError(f"Split directory not found: {split_dir} or {alt_split_dir}")
        
        self.split_dir = split_dir  # Store for later use
        
        sequences = []
        if self.use_detector:
            # For YOLOx mode: get only one variant per base sequence (prefer DPM)
            # E.g., MOT17-02-DPM, MOT17-04-DPM, ... (not all 3 variants)
            seen_bases = set()
            for seq_dir in sorted(split_dir.iterdir()):
                if seq_dir.is_dir() and not seq_dir.name.startswith('.'):
                    # Extract base name: MOT17-02-DPM -> MOT17-02
                    base_name = seq_dir.name.replace('-DPM', '').replace('-FRCNN', '').replace('-SDP', '')
                    
                    # Only take first occurrence (DPM variant if available)
                    if base_name not in seen_bases and '-DPM' in seq_dir.name:
                        sequences.append(seq_dir.name)
                        seen_bases.add(base_name)
        else:
            # For precomputed mode: get all sequences including all detector variants
            for seq_dir in sorted(split_dir.iterdir()):
                if seq_dir.is_dir() and not seq_dir.name.startswith('.'):
                    sequences.append(seq_dir.name)
        
        return sequences
    
    def get_sequence_info(self, seq_name):
        """Get sequence information"""
        seq_dir = self.split_dir / seq_name
        seqinfo_path = seq_dir / 'seqinfo.ini'
        
        info = {}
        if seqinfo_path.exists():
            with open(seqinfo_path, 'r') as f:
                for line in f:
                    if '=' in line and not line.startswith('['):
                        key, value = line.strip().split('=')
                        info[key] = value
        return info
    
    def get_image_path(self, seq_name, frame_id):
        """Get path to image for a specific frame"""
        seq_dir = self.split_dir / seq_name
        img_dir = seq_dir / 'img1'
        
        # MOT17 uses 6-digit frame numbers
        img_path = img_dir / f"{frame_id:06d}.jpg"
        
        if not img_path.exists():
            # Try alternative format
            img_path = img_dir / f"{frame_id:06d}.png"
        
        return img_path
    
    def load_detections(self, seq_name, detector='DPM'):
        """Load detection results for a sequence (for baseline mode)"""
        det_file = self.split_dir / seq_name / 'det' / 'det.txt'
        
        if not det_file.exists():
            print(f"Warning: Detection file not found: {det_file}")
            return {}
        
        detections = defaultdict(list)
        with open(det_file, 'r') as f:
            for line in f:
                parts = line.strip().split(',')
                if len(parts) < 7:
                    continue
                    
                frame_id = int(parts[0])
                x, y, w, h = map(float, parts[2:6])
                score = float(parts[6]) if len(parts) > 6 else 1.0
                
                # Convert to [x1, y1, x2, y2, score]
                detection = [x, y, x + w, y + h, score]
                detections[frame_id].append(detection)
        
        return detections


# ============================================================================
# MOT20 Dataset Handler
# ============================================================================

class MOT20Dataset:
    """Handle MOT20 dataset loading and processing"""
    
    def __init__(self, data_root, split='train', use_detector=False):
        self.data_root = Path(data_root)
        self.split = split
        self.use_detector = use_detector
        self.split_dir = None  # Will be set by _get_sequences
        self.sequences = self._get_sequences()
        
    def _get_sequences(self):
        """Get all sequence names for the split"""
        split_dir = self.data_root / self.split
        
        # Try different directory structures
        if not split_dir.exists():
            # Check if structure is MOT20/MOT20/train or MOT17/MOT17/train
            dataset_name = self.data_root.name  # Get the last part of path
            alt_split_dir = self.data_root / dataset_name / self.split
            if alt_split_dir.exists():
                split_dir = alt_split_dir
            else:
                raise ValueError(f"Split directory not found: {split_dir} or {alt_split_dir}")
        
        self.split_dir = split_dir  # Store for later use
        
        sequences = []
        # MOT20 has a simpler structure than MOT17 - no detector variants
        for seq_dir in sorted(split_dir.iterdir()):
            if seq_dir.is_dir() and not seq_dir.name.startswith('.'):
                sequences.append(seq_dir.name)
        
        return sequences
    
    def get_sequence_info(self, seq_name):
        """Get sequence information"""
        seq_dir = self.split_dir / seq_name
        seqinfo_path = seq_dir / 'seqinfo.ini'
        
        info = {}
        if seqinfo_path.exists():
            with open(seqinfo_path, 'r') as f:
                for line in f:
                    if '=' in line and not line.startswith('['):
                        key, value = line.strip().split('=')
                        info[key] = value
        return info
    
    def get_image_path(self, seq_name, frame_id):
        """Get path to image for a specific frame"""
        seq_dir = self.split_dir / seq_name
        img_dir = seq_dir / 'img1'
        
        # MOT20 uses 6-digit frame numbers, same as MOT17
        img_path = img_dir / f"{frame_id:06d}.jpg"
        
        if not img_path.exists():
            # Try alternative format
            img_path = img_dir / f"{frame_id:06d}.png"
        
        return img_path
    
    def load_detections(self, seq_name, detector=None):
        """Load detection results for a sequence (for baseline mode)"""
        det_file = self.split_dir / seq_name / 'det' / 'det.txt'
        
        if not det_file.exists():
            print(f"Warning: Detection file not found: {det_file}")
            return {}
        
        detections = defaultdict(list)
        with open(det_file, 'r') as f:
            for line in f:
                parts = line.strip().split(',')
                if len(parts) < 7:
                    continue
                    
                frame_id = int(parts[0])
                x, y, w, h = map(float, parts[2:6])
                score = float(parts[6]) if len(parts) > 6 else 1.0
                
                # Convert to [x1, y1, x2, y2, score]
                detection = [x, y, x + w, y + h, score]
                detections[frame_id].append(detection)
        
        return detections


# ============================================================================
# Evaluation Functions
# ============================================================================

def evaluate_mot17(tracker, dataset, output_dir, args, detector=None):
    """
    Run tracker on MOT17 dataset and evaluate
    
    Args:
        tracker: BYTETracker instance
        dataset: MOT17Dataset instance
        output_dir: directory to save results
        args: command line arguments
        detector: YOLOxDetector instance (optional, if None uses precomputed detections)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results = {}
    
    use_yolox = detector is not None
    
    for seq_name in dataset.sequences:
        print(f"\nProcessing sequence: {seq_name}")
        
        # ============================================================
        # For YOLOx mode: use consistent parameters
        # For precomputed detections: adjust per detector type
        # ============================================================
        if use_yolox:
            # YOLOx mode: use args parameters
            tracker.det_conf_high = args.track_thresh
            tracker.det_conf_low = args.det_conf_low
            tracker.new_track_thresh = args.track_thresh
            tracker.match_thresh_high = args.match_thresh_high
            tracker.match_thresh_low = args.match_thresh_low
        
        # Reset tracker state for new sequence
        tracker.frame_id = 0
        tracker.tracked_stracks = []
        tracker.lost_stracks = []
        tracker.removed_stracks = []
        STrack.track_id_count = 0
        
        # Load detections or prepare for detection
        if use_yolox:
            detections = None  # Will detect on-the-fly
            print(f"  Using YOLOx detector for real-time detection")
        else:
            detections = dataset.load_detections(seq_name)
            print(f"  Using precomputed detections")
        
        seq_info = dataset.get_sequence_info(seq_name)
        
        img_height = int(seq_info.get('imHeight', 1080))
        img_width = int(seq_info.get('imWidth', 1920))
        img_shape = (img_height, img_width)
        
        # Process each frame
        seq_results = []
        num_frames = int(seq_info.get('seqLength', len(detections) if detections else 0))
        
        print(f"  Processing {num_frames} frames...")
        
        import time
        start_time = time.time()
        frame_times = []
        
        for frame_id in range(1, num_frames + 1):
            frame_start = time.time()
            
            if use_yolox:
                # Load image and run detector
                img_path = dataset.get_image_path(seq_name, frame_id)
                
                if not img_path.exists():
                    dets = np.empty((0, 5))
                else:
                    # Read image
                    img = cv2.imread(str(img_path))
                    if img is None:
                        dets = np.empty((0, 5))
                    else:
                        # Run detection with debug on first frame
                        debug_this_frame = False  # (frame_id == 1)  # Disabled for clean output
                        frame_dets = detector.detect(img, debug=debug_this_frame)
                        
                        if debug_this_frame and len(frame_dets) > 0:
                            print(f"    [DEBUG] Detector returned {len(frame_dets)} detections")
                            print(f"    [DEBUG] Detection format: {frame_dets.dtype}, shape: {frame_dets.shape}")
                            print(f"    [DEBUG] First detection: {frame_dets[0]}")
                        
                        if len(frame_dets) == 0:
                            dets = np.empty((0, 5))
                        else:
                            # frame_dets is already numpy array from detector
                            dets = frame_dets
                            
                        if debug_this_frame:
                            print(f"    [DEBUG] Passing {len(dets)} detections to tracker")
                            if len(dets) > 0:
                                print(f"    [DEBUG] Score range in dets: [{dets[:, 4].min():.4f}, {dets[:, 4].max():.4f}]")
            else:
                # Use precomputed detections
                frame_dets = detections.get(frame_id, [])
                
                if len(frame_dets) == 0:
                    dets = np.empty((0, 5))
                else:
                    # Convert to numpy array
                    dets = np.array(frame_dets)
            
            # Update tracker (even with empty detections to handle lost tracks)
            online_targets = tracker.update(dets, img_shape)
            
            # Debug first frame output
            if debug_this_frame:
                print(f"    [DEBUG] Tracker returned {len(online_targets)} confirmed tracks")
                if len(online_targets) > 0:
                    for i, t in enumerate(online_targets[:3]):
                        print(f"      Track {i}: ID={t.track_id}, score={t.score:.4f}, state={t.state}")
            
            # Save results
            for track in online_targets:
                tlwh = track.tlwh
                tid = track.track_id
                seq_results.append([frame_id, tid, tlwh[0], tlwh[1], tlwh[2], tlwh[3]])
            
            frame_times.append(time.time() - frame_start)
            
            # Print progress every 100 frames
            if frame_id % 100 == 0 or frame_id == num_frames:
                avg_time = sum(frame_times[-100:]) / len(frame_times[-100:])
                fps = 1.0 / avg_time if avg_time > 0 else 0
                print(f"    Frame {frame_id}/{num_frames} - {len(online_targets)} tracks, {len(seq_results)} total detections - {fps:.2f} FPS")
        
        # Save sequence results
        # For YOLOx: save with base name (MOT17-02) for comparison with GT
        if use_yolox:
            # Remove detector suffix: MOT17-02-DPM -> MOT17-02
            base_seq_name = seq_name.replace('-DPM', '').replace('-FRCNN', '').replace('-SDP', '')
            output_file = output_dir / f"{base_seq_name}.txt"
        else:
            output_file = output_dir / f"{seq_name}.txt"
        
        with open(output_file, 'w') as f:
            for row in seq_results:
                f.write(f"{row[0]},{row[1]},{row[2]:.2f},{row[3]:.2f},{row[4]:.2f},{row[5]:.2f},1,-1,-1,-1\n")
        
        print(f"  Saved {len(seq_results)} detections to {output_file.name}\n")
        results[seq_name] = seq_results
    
    return results


def evaluate_mot20(tracker, dataset, output_dir, args, detector=None):
    """
    Run tracker on MOT20 dataset and evaluate
    
    Args:
        tracker: BYTETracker instance
        dataset: MOT20Dataset instance
        output_dir: directory to save results
        args: command line arguments
        detector: YOLOxDetector instance (optional, if None uses precomputed detections)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results = {}
    
    use_yolox = detector is not None
    
    for seq_name in dataset.sequences:
        print(f"\nProcessing sequence: {seq_name}")
        
        # Reset tracker state for new sequence
        tracker.frame_id = 0
        tracker.tracked_stracks = []
        tracker.lost_stracks = []
        tracker.removed_stracks = []
        STrack.track_id_count = 0
        
        # Load detections or prepare for detection
        if use_yolox:
            detections = None  # Will detect on-the-fly
            print(f"  Using YOLOx detector for real-time detection")
        else:
            detections = dataset.load_detections(seq_name)
            print(f"  Using precomputed detections")
        
        seq_info = dataset.get_sequence_info(seq_name)
        
        img_height = int(seq_info.get('imHeight', 1080))
        img_width = int(seq_info.get('imWidth', 1920))
        img_shape = (img_height, img_width)
        
        # Process each frame
        seq_results = []
        num_frames = int(seq_info.get('seqLength', len(detections) if detections else 0))
        
        print(f"  Processing {num_frames} frames...")
        
        import time
        start_time = time.time()
        frame_times = []
        
        for frame_id in range(1, num_frames + 1):
            frame_start = time.time()
            
            if use_yolox:
                # Load image and run detector
                img_path = dataset.get_image_path(seq_name, frame_id)
                
                if not img_path.exists():
                    dets = np.empty((0, 5))
                else:
                    # Read image
                    img = cv2.imread(str(img_path))
                    if img is None:
                        dets = np.empty((0, 5))
                    else:
                        # Run detection
                        frame_dets = detector.detect(img, debug=False)
                        
                        if len(frame_dets) == 0:
                            dets = np.empty((0, 5))
                        else:
                            # frame_dets is already numpy array from detector
                            dets = frame_dets
            else:
                # Use precomputed detections
                frame_dets = detections.get(frame_id, [])
                
                if len(frame_dets) == 0:
                    dets = np.empty((0, 5))
                else:
                    # Convert to numpy array
                    dets = np.array(frame_dets)
            
            # Update tracker (even with empty detections to handle lost tracks)
            online_targets = tracker.update(dets, img_shape)
            
            # Save results
            for track in online_targets:
                tlwh = track.tlwh
                tid = track.track_id
                seq_results.append([frame_id, tid, tlwh[0], tlwh[1], tlwh[2], tlwh[3]])
            
            frame_times.append(time.time() - frame_start)
            
            # Print progress every 100 frames
            if frame_id % 100 == 0 or frame_id == num_frames:
                avg_time = sum(frame_times[-100:]) / len(frame_times[-100:])
                fps = 1.0 / avg_time if avg_time > 0 else 0
                print(f"    Frame {frame_id}/{num_frames} - {len(online_targets)} tracks, {len(seq_results)} total detections - {fps:.2f} FPS")
        
        # Save sequence results
        output_file = output_dir / f"{seq_name}.txt"
        
        with open(output_file, 'w') as f:
            for row in seq_results:
                f.write(f"{row[0]},{row[1]},{row[2]:.2f},{row[3]:.2f},{row[4]:.2f},{row[5]:.2f},1,-1,-1,-1\n")
        
        print(f"  Saved {len(seq_results)} detections to {output_file.name}\n")
        results[seq_name] = seq_results
    
    return results


def compute_mot_metrics(data_root, pred_dir, use_yolox=False):
    """
    Compute MOT metrics using py-motmetrics
    For YOLOx mode: compute overall metrics
    For precomputed detections: separate by detector type (DPM, FRCNN, SDP)
    
    Args:
        data_root: root directory of MOT17 dataset (contains train folder with sequences)
        pred_dir: directory containing prediction files
        use_yolox: whether YOLOx detector was used (affects grouping)
    """
    data_root = Path(data_root)
    pred_dir = Path(pred_dir)
    
    # Get all prediction files (exclude summary.txt)
    pred_files = sorted([f for f in pred_dir.glob('*.txt') if f.name != 'summary.txt' and f.name != 'evaluation_summary.txt'])
    
    if len(pred_files) == 0:
        print(f"No prediction files found in {pred_dir}")
        return None
    
    print(f"\nFound {len(pred_files)} prediction files")
    
    # Group by detector type - store accumulators per sequence
    if use_yolox:
        # YOLOx mode: single group
        detector_metrics = {'YOLOx': []}
    else:
        # Precomputed mode: group by detector
        detector_metrics = defaultdict(list)
    
    for pred_file in tqdm(pred_files, desc="Processing sequences", unit="seq"):
        seq_name = pred_file.stem
        
        # Determine detector type and ground truth path
        if use_yolox:
            det_type = 'YOLOx'
            # For YOLOx: predictions saved with base name (MOT17-02.txt)
            # but GT is in DPM variant folder (MOT17-02-DPM/gt/gt.txt)
            gt_file = data_root / f"{seq_name}-DPM" / 'gt' / 'gt.txt'
        else:
            # For precomputed: determine detector type
            det_type = None
            if 'DPM' in seq_name:
                det_type = 'DPM'
            elif 'FRCNN' in seq_name:
                det_type = 'FRCNN'
            elif 'SDP' in seq_name:
                det_type = 'SDP'
            else:
                continue
            
            gt_file = data_root / seq_name / 'gt' / 'gt.txt'
        
        if not gt_file.exists():
            print(f"Warning: Ground truth file not found for {seq_name}")
            continue
        
        # Create separate accumulator for this sequence
        acc = mm.MOTAccumulator(auto_id=True)
        
        # Load ground truth
        gt_data = defaultdict(list)
        with open(gt_file, 'r') as f:
            for line in f:
                parts = line.strip().split(',')
                if len(parts) < 8:
                    continue
                frame_id = int(parts[0])
                track_id = int(parts[1])
                x, y, w, h = map(float, parts[2:6])
                conf = float(parts[6])
                cls = int(parts[7]) if len(parts) > 7 else 1
                
                # Filter: only consider pedestrian class (cls==1) and confident annotations (conf==1)
                if cls == 1 and conf == 1:
                    gt_data[frame_id].append({
                        'id': track_id,
                        'bbox': [x, y, x+w, y+h]
                    })
        
        # Load predictions
        pred_data = defaultdict(list)
        with open(pred_file, 'r') as f:
            for line in f:
                parts = line.strip().split(',')
                if len(parts) < 6:
                    continue
                frame_id = int(parts[0])
                track_id = int(parts[1])
                x, y, w, h = map(float, parts[2:6])
                
                pred_data[frame_id].append({
                    'id': track_id,
                    'bbox': [x, y, x+w, y+h]
                })
        
        # Compute metrics per frame
        all_frames = sorted(set(list(gt_data.keys()) + list(pred_data.keys())))
        
        for frame_id in tqdm(all_frames, desc=f"  {seq_name} frames", unit="f", leave=False):
            gt_ids = [obj['id'] for obj in gt_data.get(frame_id, [])]
            gt_bboxes = [obj['bbox'] for obj in gt_data.get(frame_id, [])]
            
            pred_ids = [obj['id'] for obj in pred_data.get(frame_id, [])]
            pred_bboxes = [obj['bbox'] for obj in pred_data.get(frame_id, [])]
            
            # Compute IoU distance matrix
            if len(gt_bboxes) > 0 and len(pred_bboxes) > 0:
                # Compute IoU manually to match BoxMOT
                ious = np.zeros((len(gt_bboxes), len(pred_bboxes)))
                for i, gt_box in enumerate(gt_bboxes):
                    for j, pred_box in enumerate(pred_bboxes):
                        xx1 = max(gt_box[0], pred_box[0])
                        yy1 = max(gt_box[1], pred_box[1])
                        xx2 = min(gt_box[2], pred_box[2])
                        yy2 = min(gt_box[3], pred_box[3])
                        
                        w = max(0, xx2 - xx1)
                        h = max(0, yy2 - yy1)
                        inter = w * h
                        
                        area_gt = (gt_box[2] - gt_box[0]) * (gt_box[3] - gt_box[1])
                        area_pred = (pred_box[2] - pred_box[0]) * (pred_box[3] - pred_box[1])
                        union = area_gt + area_pred - inter
                        
                        ious[i, j] = inter / union if union > 0 else 0
                
                dists = 1 - ious
            else:
                dists = np.empty((len(gt_ids), len(pred_ids)))
            
            acc.update(gt_ids, pred_ids, dists)
        
        # Store accumulator with sequence name
        detector_metrics[det_type].append((seq_name, acc))
    
    # Store results for each detector
    all_results = {}
    
    # Compute metrics for each detector using compute_many
    detector_types = ['YOLOx'] if use_yolox else ['DPM', 'FRCNN', 'SDP']
    
    for det_type in tqdm(detector_types, desc="Computing metrics by detector", unit="det"):
        if det_type not in detector_metrics or len(detector_metrics[det_type]) == 0:
            continue
        
        print(f"\n{'='*80}")
        print(f"Evaluating {det_type} sequences")
        print(f"{'='*80}")
        
        # Get accumulators and names
        accs = [acc for _, acc in detector_metrics[det_type]]
        names = [name for name, _ in detector_metrics[det_type]]
        
        # Compute metrics using compute_many
        mh = mm.metrics.create()
        summary = mh.compute_many(
            accs,
            metrics=['mota', 'motp', 'idf1', 'precision', 'recall',
                    'num_switches', 'num_fragmentations', 'num_false_positives', 'num_misses'],
            names=names
        )
        
        all_results[det_type] = summary
    
    return all_results


def compute_mot20_metrics(data_root, pred_dir, use_yolox=False):
    """
    Compute MOT metrics for MOT20 dataset using py-motmetrics
    
    Args:
        data_root: root directory path (may include /train or /test suffix)
        pred_dir: directory containing prediction files
        use_yolox: whether YOLOx detector was used
    """
    data_root = Path(data_root)
    pred_dir = Path(pred_dir)
    
    # Auto-detect data root structure
    # Handle case where data_root ends with 'train' or 'test'
    split_dir = None
    if data_root.name in ['train', 'test']:
        split_name = data_root.name
        parent_path = data_root.parent
        
        # Check if parent has MOT20-01 (flat structure: MOT20/{train,test}/MOT20-01)
        if (parent_path / 'MOT20-01').exists():
            split_dir = parent_path
        # Check if grandparent has nested structure (MOT20/MOT20/{train,test}/MOT20-01)
        elif (parent_path.parent / 'MOT20' / split_name / 'MOT20-01').exists():
            split_dir = parent_path.parent / 'MOT20' / split_name
        else:
            split_dir = data_root
    else:
        # data_root points to dataset root (MOT20)
        # Check if it's flat structure  
        if (data_root / 'MOT20-01').exists():
            split_dir = data_root
        # Check for nested structure (MOT20/MOT20/{train,test})
        elif (data_root / 'MOT20' / 'train' / 'MOT20-01').exists():
            split_dir = data_root / 'MOT20' / 'train'
        elif (data_root / 'MOT20' / 'test' / 'MOT20-01').exists():
            split_dir = data_root / 'MOT20' / 'test'
        else:
            split_dir = data_root
    
    # Get all prediction files (exclude summary.txt)
    pred_files = sorted([f for f in pred_dir.glob('*.txt') if f.name != 'summary.txt' and f.name != 'evaluation_summary.txt'])
    
    if len(pred_files) == 0:
        print(f"No prediction files found in {pred_dir}")
        return None
    
    print(f"\nFound {len(pred_files)} prediction files")
    print(f"Using split directory: {split_dir}")
    
    # MOT20 doesn't have detector variants like MOT17, so single accumulator
    accs = []
    names = []
    
    for pred_file in tqdm(pred_files, desc="Processing sequences", unit="seq"):
        seq_name = pred_file.stem
        
        # Ground truth file
        gt_file = split_dir / seq_name / 'gt' / 'gt.txt'
        
        if not gt_file.exists():
            print(f"Warning: Ground truth file not found for {seq_name}: {gt_file}")
            continue
        
        # Load predictions
        pred_data = defaultdict(lambda: [[], []])
        with open(pred_file, 'r') as f:
            for line in f:
                parts = line.strip().split(',')
                frame_id = int(parts[0])
                track_id = int(parts[1])
                x, y, w, h = map(float, parts[2:6])
                
                pred_data[frame_id][0].append([x, y, x + w, y + h])
                pred_data[frame_id][1].append(track_id)
        
        # Load ground truth
        gt_data = defaultdict(lambda: [[], []])
        with open(gt_file, 'r') as f:
            for line in f:
                parts = line.strip().split(',')
                frame_id = int(parts[0])
                track_id = int(parts[1])
                x, y, w, h = map(float, parts[2:6])
                label = int(parts[6])  # 1 = pedestrian, -1 = ignored
                conf = int(parts[7])    # Detection confidence
                visibility = float(parts[8]) # Visibility (0-1)
                
                if label == 1 and conf == 1:  # Only use annotated pedestrians
                    gt_data[frame_id][0].append([x, y, x + w, y + h])
                    gt_data[frame_id][1].append(track_id)
        
        # Create accumulator
        mh = mm.metrics.create()
        acc = mm.MOTAccumulator(auto_id=True)
        
        # Process each frame
        all_frames = sorted(set(list(pred_data.keys()) + list(gt_data.keys())))
        
        for frame_id in tqdm(all_frames, desc=f"  {seq_name} frames", unit="f", leave=False):
            gt_boxes = gt_data[frame_id][0]
            gt_ids = gt_data[frame_id][1]
            pred_boxes = pred_data[frame_id][0]
            pred_ids = pred_data[frame_id][1]
            
            # Compute distance matrix
            if len(gt_boxes) == 0 and len(pred_boxes) == 0:
                continue
            
            if len(gt_boxes) == 0:
                distances = np.full((0, len(pred_boxes)), np.nan)
            elif len(pred_boxes) == 0:
                distances = np.full((len(gt_boxes), 0), np.nan)
            else:
                distances = np.zeros((len(gt_boxes), len(pred_boxes)))
                for i, gt_box in enumerate(gt_boxes):
                    for j, pred_box in enumerate(pred_boxes):
                        distances[i, j] = 1 - bbox_iou(gt_box, pred_box)  # Convert IoU to distance
            
            acc.update(gt_ids, pred_ids, distances)
        
        accs.append(acc)
        names.append(seq_name)
    
    if len(accs) == 0:
        print("No valid sequences found for evaluation")
        return None
    
    # Compute metrics using compute_many
    print("\nComputing MOT20 metrics...")
    mh = mm.metrics.create()
    summary = mh.compute_many(
        accs,
        metrics=['mota', 'motp', 'idf1', 'precision', 'recall',
                'num_switches', 'num_fragmentations', 'num_false_positives', 'num_misses'],
        names=names
    )
    print("✓ Metrics computed successfully")
    
    return {'MOT20': summary}


def run_dataset_precheck(dataset, args):
    """Validate dataset files before running debug/tracking."""
    print("\n" + "=" * 80)
    print("PRECHECK: Verify dataset and GT/detection files")
    print("=" * 80)

    errors = []
    warnings = []

    if not hasattr(dataset, 'split_dir') or dataset.split_dir is None:
        errors.append("Dataset split_dir is not initialized")
    elif not dataset.split_dir.exists():
        errors.append(f"Split directory does not exist: {dataset.split_dir}")

    if not getattr(dataset, 'sequences', None):
        errors.append("No sequences found in selected split")

    print(f"Split directory: {getattr(dataset, 'split_dir', None)}")
    print(f"Total sequences: {len(getattr(dataset, 'sequences', []))}")

    for seq_name in dataset.sequences:
        seq_dir = dataset.split_dir / seq_name
        seq_errors = []

        seqinfo_path = seq_dir / 'seqinfo.ini'
        if not seqinfo_path.exists():
            seq_errors.append(f"Missing seqinfo.ini ({seqinfo_path})")

        img_dir = seq_dir / 'img1'
        if not img_dir.exists():
            seq_errors.append(f"Missing image directory ({img_dir})")

        gt_path = seq_dir / 'gt' / 'gt.txt'
        if args.split == 'train':
            if not gt_path.exists():
                seq_errors.append(f"Missing GT file ({gt_path})")
            else:
                # Parse one valid GT row to ensure file can be read.
                gt_parsed = False
                with open(gt_path, 'r') as f:
                    for line in f:
                        parts = line.strip().split(',')
                        if len(parts) >= 6:
                            int(parts[0])
                            int(parts[1])
                            float(parts[2])
                            float(parts[3])
                            float(parts[4])
                            float(parts[5])
                            gt_parsed = True
                            break
                if not gt_parsed:
                    warnings.append(f"GT exists but no valid row found: {gt_path}")

        if not args.use_yolox:
            det_path = seq_dir / 'det' / 'det.txt'
            if not det_path.exists():
                seq_errors.append(f"Missing detection file ({det_path})")
            else:
                # Parse one valid detection row to ensure file can be read.
                det_parsed = False
                with open(det_path, 'r') as f:
                    for line in f:
                        parts = line.strip().split(',')
                        if len(parts) >= 7:
                            int(parts[0])
                            float(parts[2])
                            float(parts[3])
                            float(parts[4])
                            float(parts[5])
                            float(parts[6])
                            det_parsed = True
                            break
                if not det_parsed:
                    warnings.append(f"Det exists but no valid row found: {det_path}")

        if seq_errors:
            errors.extend([f"{seq_name}: {msg}" for msg in seq_errors])
            print(f"[ERR] {seq_name}")
        else:
            print(f"[OK ] {seq_name}")

    if warnings:
        print("\nPrecheck warnings:")
        for msg in warnings:
            print(f"  - {msg}")

    if errors:
        print("\nPrecheck errors:")
        for msg in errors:
            print(f"  - {msg}")
        raise RuntimeError("Precheck failed. Fix dataset/GT/detection files before running debug/tracking.")

    print("Precheck passed. Dataset/GT files are loadable.")


# ============================================================================
# Main Script
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='ByteTrack + YOLOx on MOT17/MOT20')
    
    # Dataset selection
    parser.add_argument('--dataset', type=str, default='mot17',
                       choices=['mot17', 'mot20'],
                       help='Dataset to benchmark: mot17 or mot20')
    parser.add_argument('--data_root', type=str, 
                       default=r'd:\Learn\Year4\KLTN\Dataset\MOT17',
                       help='Path to dataset root directory')
    parser.add_argument('--split', type=str, default='train',
                       choices=['train', 'test'],
                       help='Dataset split to use')
    parser.add_argument('--output_dir', type=str, default='./results_yolox',
                       help='Directory to save tracking results')
    
    # YOLOx detector options
    parser.add_argument('--use_yolox', action='store_true',
                       help='Use YOLOx detector instead of precomputed detections')
    parser.add_argument('--yolox_checkpoint', type=str, 
                       default=r'd:\Learn\Year4\KLTN\benmark_result\archive_s',
                       help='Path to YOLOx checkpoint (PyTorch Lightning format)')
    parser.add_argument('--yolox_checkpoint_mot20', type=str, 
                       default=r'd:\Learn\Year4\KLTN\benmark_result\archive_x_mot20',
                       help='Path to YOLOx checkpoint for MOT20 (PyTorch Lightning format)')
    parser.add_argument('--yolox_conf_thresh', type=float, default=0.01,
                       help='YOLOx confidence threshold for detections (default: 0.01, matching ByteTrack official)')
    parser.add_argument('--yolox_nms_thresh', type=float, default=0.65,
                       help='YOLOx NMS threshold (lower = fewer overlapping boxes)')
    parser.add_argument('--device', type=str, default='cuda',
                       choices=['cuda', 'cpu'],
                       help='Device to run YOLOx on')
    
    # ByteTrack parameters (matching benmark_bytetrack.py)
    parser.add_argument('--track_thresh', type=float, default=0.6,
                       help='High-score detection threshold (default: 0.6, ByteTrack official)')
    parser.add_argument('--det_conf_low', type=float, default=0.1,
                       help='Low-score detection threshold (default: 0.1, ByteTrack official)')
    parser.add_argument('--match_thresh_high', type=float, default=0.9,
                       help='Cost threshold for Phase 1 association (default: 0.9, ByteTrack official)')
    parser.add_argument('--match_thresh_low', type=float, default=0.5,
                       help='Cost threshold for second association (default: 0.5, IoU > 0.5)')
    parser.add_argument('--track_buffer', type=int, default=30,
                       help='Frames to keep lost tracks (default: 30, ByteTrack official)')
    # Note: ByteTrack official activates tracks immediately (frame 1)
    # No min_hits parameter exposed - always min_hits=1 for ByteTrack
    
    args = parser.parse_args()
    
    # Update default data root based on dataset
    if args.data_root == r'd:\Learn\Year4\KLTN\Dataset\MOT17' and args.dataset == 'mot20':
        args.data_root = r'd:\Learn\Year4\KLTN\Dataset\MOT20'
    
    # Update default output dir based on dataset
    if args.output_dir == './results_yolox':
        if args.dataset == 'mot20':
            args.output_dir = './results_mot20_yolox' if args.use_yolox else './results_mot20'
        else:
            args.output_dir = './results_mot17_yolox' if args.use_yolox else './results_mot17'
    
    # Select appropriate YOLOx checkpoint for MOT20
    if args.dataset == 'mot20' and args.use_yolox:
        if args.yolox_checkpoint == r'd:\Learn\Year4\KLTN\benmark_result\archive_s':
            # Use MOT20 checkpoint if user didn't specify custom one
            args.yolox_checkpoint = args.yolox_checkpoint_mot20
    
    dataset_name = 'MOT17' if args.dataset == 'mot17' else 'MOT20'
    
    print("="*80)
    if args.use_yolox:
        print(f"ByteTrack + YOLOx on {dataset_name}")
    else:
        print(f"ByteTrack on {dataset_name} (with precomputed detections)")
    print("="*80)
    print(f"Dataset: {dataset_name}")
    print(f"Data root: {args.data_root}")
    print(f"Split: {args.split}")
    print(f"Output dir: {args.output_dir}")
    
    if args.use_yolox:
        print(f"\nYOLOx Detector:")
        print(f"  Checkpoint: {args.yolox_checkpoint}")
        print(f"  Conf thresh: {args.yolox_conf_thresh}")
        print(f"  NMS thresh: {args.yolox_nms_thresh}")
        print(f"  Device: {args.device}")
    
    print("\nByteTrack Parameters (matching official ByteTrack):") 
    print(f"  track_thresh:      {args.track_thresh} (Phase 1 high-conf detections)")
    print(f"  det_conf_low:      {args.det_conf_low} (Phase 2 low-conf detections)")
    print(f"  match_thresh_high: {args.match_thresh_high} (Phase 1: IoU > {1-args.match_thresh_high:.2f})")
    print(f"  match_thresh_low:  {args.match_thresh_low} (Phase 2: IoU > {1-args.match_thresh_low:.2f})")
    print(f"  track_buffer:      {args.track_buffer} frames")
    print(f"  Note: min_hits=1, activate at frame 1 (ByteTrack official)")
    print("="*80)
    
    # Initialize YOLOx detector if requested
    detector = None
    if args.use_yolox:
        print("\nInitializing YOLOx detector...")
        try:
            detector = YOLOxDetector(
                checkpoint_path=args.yolox_checkpoint,
                device=args.device,
                conf_thresh=args.yolox_conf_thresh,
                nms_thresh=args.yolox_nms_thresh
            )
        except Exception as e:
            print(f"Error initializing YOLOx detector: {e}")
            print("Falling back to precomputed detections...")
            args.use_yolox = False
    
    # Initialize dataset
    if args.dataset == 'mot20':
        dataset = MOT20Dataset(args.data_root, args.split, use_detector=args.use_yolox)
    else:
        dataset = MOT17Dataset(args.data_root, args.split, use_detector=args.use_yolox)

    # Check GT/detection files first before entering any debug/tracking flow.
    run_dataset_precheck(dataset, args)
    
    print(f"\nFound {len(dataset.sequences)} sequences")
    
    tracker = BYTETracker(
        det_conf_high=args.track_thresh,
        det_conf_low=args.det_conf_low,
        new_track_thresh=args.track_thresh,  # ByteTrack: same as track_thresh
        match_thresh_high=args.match_thresh_high,
        match_thresh_low=args.match_thresh_low,
        track_buffer=args.track_buffer,
        min_hits=1  # ByteTrack always: activate frame 1
    )
    
    # Run tracking
    if args.use_yolox:
        print(f"\nRunning ByteTrack with YOLOx detector on {dataset_name}...")
    else:
        print(f"\nRunning ByteTrack with precomputed detections on {dataset_name}...")
    
    if args.dataset == 'mot20':
        results = evaluate_mot20(tracker, dataset, args.output_dir, args, detector=detector)
    else:
        results = evaluate_mot17(tracker, dataset, args.output_dir, args, detector=detector)
    
    print("\n" + "="*80)
    print("Tracking completed!")
    print(f"Results saved to: {args.output_dir}")
    print("="*80)
    
    # Compute metrics if ground truth available
    if args.split == 'train':
        print("\nComputing MOT metrics...")
        # Use dataset-resolved split_dir so metrics path matches precheck path.
        gt_dir = Path(dataset.split_dir)
        
        try:
            if args.dataset == 'mot20':
                all_results = compute_mot20_metrics(gt_dir, Path(args.output_dir), use_yolox=args.use_yolox)
                metric_title = f"MOT20 EVALUATION RESULTS - ByteTrack"
            else:
                all_results = compute_mot_metrics(gt_dir, Path(args.output_dir), use_yolox=args.use_yolox)
                metric_title = f"MOT17 EVALUATION RESULTS - ByteTrack"
            
            if all_results is not None and len(all_results) > 0:
                print("\n" + "="*100)
                print(f"{metric_title:^100}")
                print("="*100)
                
                # Prepare summary data
                summary_data = []
                
                # Determine which detector types to process
                if args.dataset == 'mot20':
                    det_types = ['MOT20']
                elif args.use_yolox:
                    det_types = ['YOLOx']
                else:
                    det_types = ['DPM', 'FRCNN', 'SDP']
                
                for det_type in det_types:
                    if det_type not in all_results:
                        continue
                    
                    summary = all_results[det_type]
                    
                    # Process each sequence and calculate average
                    if len(summary) > 0:
                        # First, print detailed per-sequence results
                        print(f"\n{det_type} - Per-Sequence Results:")
                        print(f"{'Sequence':<20} {'MOTA':<10} {'IDF1':<10} {'MOTP':<10} {'Precision':<12} {'Recall':<10} {'ID_Sw':<10} {'Frag':<10} {'FP':<10} {'FN':<10}")
                        print("-" * 120)
                        
                        seq_metrics = []
                        for seq_name in summary.index:
                            row = summary.loc[seq_name]
                            seq_data = {
                                'Sequence': seq_name,
                                'MOTA': row['mota'] * 100,
                                'IDF1': row['idf1'] * 100,
                                'MOTP': row['motp'],
                                'Precision': row['precision'] * 100,
                                'Recall': row['recall'] * 100,
                                'ID_Sw': int(row['num_switches']),
                                'Frag': int(row['num_fragmentations']),
                                'FP': int(row['num_false_positives']),
                                'FN': int(row['num_misses'])
                            }
                            seq_metrics.append(seq_data)
                            
                            # Print sequence result
                            print(f"{seq_data['Sequence']:<20} "
                                  f"{seq_data['MOTA']:>6.2f}%   "
                                  f"{seq_data['IDF1']:>6.2f}%   "
                                  f"{seq_data['MOTP']:>6.3f}    "
                                  f"{seq_data['Precision']:>8.2f}%   "
                                  f"{seq_data['Recall']:>6.2f}%   "
                                  f"{seq_data['ID_Sw']:>6}    "
                                  f"{seq_data['Frag']:>6}   "
                                  f"{seq_data['FP']:>6}   "
                                  f"{seq_data['FN']:>8}")
                        
                        # Calculate average across all sequences
                        avg_data = {
                            'Detector': f"{det_type} (AVERAGE)",
                            'MOTA': np.mean([d['MOTA'] for d in seq_metrics]),
                            'IDF1': np.mean([d['IDF1'] for d in seq_metrics]),
                            'MOTP': np.mean([d['MOTP'] for d in seq_metrics]),
                            'Precision': np.mean([d['Precision'] for d in seq_metrics]),
                            'Recall': np.mean([d['Recall'] for d in seq_metrics]),
                            'ID_Sw': int(np.sum([d['ID_Sw'] for d in seq_metrics])),  # Total, not average
                            'Frag': int(np.sum([d['Frag'] for d in seq_metrics])),    # Total, not average
                            'FP': int(np.sum([d['FP'] for d in seq_metrics])),        # Total, not average
                            'FN': int(np.sum([d['FN'] for d in seq_metrics]))         # Total, not average
                        }
                        summary_data.append(avg_data)
                
                # Print formatted table
                print(f"\n{'='*120}")
                print(f"OVERALL SUMMARY")
                print(f"{'='*120}")
                print(f"\n{'Detector':<20} {'MOTA':<10} {'IDF1':<10} {'MOTP':<10} {'Precision':<12} {'Recall':<10} {'ID_Sw':<10} {'Frag':<10} {'FP':<10} {'FN':<10}")
                print("-" * 120)
                
                for data in summary_data:
                    print(f"{data['Detector']:<20} "
                          f"{data['MOTA']:>6.2f}%   "
                          f"{data['IDF1']:>6.2f}%   "
                          f"{data['MOTP']:>6.3f}    "
                          f"{data['Precision']:>8.2f}%   "
                          f"{data['Recall']:>6.2f}%   "
                          f"{data['ID_Sw']:>6}    "
                          f"{data['Frag']:>6}   "
                          f"{data['FP']:>6}   "
                          f"{data['FN']:>8}")
                
                print(f"{'='*120}\n")
                
                # Save detailed summary to file
                summary_file = Path(args.output_dir) / 'evaluation_summary.txt'
                with open(summary_file, 'w') as f:
                    f.write("="*120 + "\n")
                    f.write(f"{metric_title:^120}\n")
                    f.write("="*120 + "\n\n")
                    
                    f.write("Configuration:\n")
                    if args.use_yolox:
                        f.write(f"  Detector: YOLOx\n")
                        f.write(f"  Checkpoint: {args.yolox_checkpoint}\n")
                    else:
                        f.write(f"  Detector: Precomputed detections\n")
                    f.write(f"  track_thresh:      {args.track_thresh}\n")
                    f.write(f"  det_conf_low:      {args.det_conf_low}\n")
                    f.write(f"  match_thresh_high: {args.match_thresh_high}\n")
                    f.write(f"  match_thresh_low:  {args.match_thresh_low}\n")
                    f.write(f"  track_buffer:      {args.track_buffer}\n\n")
                    
                    # Write per-sequence results for each detector
                    for det_type in det_types:
                        if det_type not in all_results:
                            continue
                        
                        summary = all_results[det_type]
                        if len(summary) == 0:
                            continue
                        
                        f.write("="*120 + "\n")
                        f.write(f"{det_type} - Per-Sequence Results\n")
                        f.write("="*120 + "\n")
                        f.write(f"{'Sequence':<20} {'MOTA':>8} {'IDF1':>8} {'MOTP':>8} {'Precision':>10} {'Recall':>8} {'ID_Sw':>8} {'Frag':>8} {'FP':>8} {'FN':>8}\n")
                        f.write("-"*120 + "\n")
                        
                        for seq_name in summary.index:
                            row = summary.loc[seq_name]
                            f.write(f"{seq_name:<20} "
                                  f"{row['mota']*100:>7.2f}% "
                                  f"{row['idf1']*100:>7.2f}% "
                                  f"{row['motp']:>8.3f} "
                                  f"{row['precision']*100:>9.2f}% "
                                  f"{row['recall']*100:>7.2f}% "
                                  f"{int(row['num_switches']):>8d} "
                                  f"{int(row['num_fragmentations']):>8d} "
                                  f"{int(row['num_false_positives']):>8d} "
                                  f"{int(row['num_misses']):>8d}\n")
                        f.write("\n")
                    
                    # Write overall summary
                    f.write("="*120 + "\n")
                    f.write("OVERALL SUMMARY\n")
                    f.write("="*120 + "\n")
                    f.write(f"{'Detector':<20} {'MOTA':>8} {'IDF1':>8} {'MOTP':>8} {'Precision':>10} {'Recall':>8} {'ID_Sw':>8} {'Frag':>8} {'FP':>8} {'FN':>8}\n")
                    f.write("-"*120 + "\n")
                    
                    for data in summary_data:
                        f.write(f"{data['Detector']:<20} "
                              f"{data['MOTA']:>7.2f}% "
                              f"{data['IDF1']:>7.2f}% "
                              f"{data['MOTP']:>8.3f} "
                              f"{data['Precision']:>9.2f}% "
                              f"{data['Recall']:>7.2f}% "
                              f"{data['ID_Sw']:>8d} "
                              f"{data['Frag']:>8d} "
                              f"{data['FP']:>8d} "
                              f"{data['FN']:>8d}\n")
                    
                    f.write("="*120 + "\n\n")
                
                print(f"\nDetailed summary saved to: {summary_file}")
            
        except Exception as e:
            print(f"Warning: Could not compute metrics: {e}")
            import traceback
            traceback.print_exc()
            print("Make sure py-motmetrics is installed: pip install motmetrics")


if __name__ == '__main__':
    # Install required packages if not available
    print("Checking required packages...")
    try:
        import scipy.linalg
        import lap
        import torch
        print("All packages available!")
    except ImportError as e:
        print(f"\nMissing package: {e}")
        print("Installing required packages...")
        import subprocess
        subprocess.check_call(['pip', 'install', 
                             'scipy', 'lap', 'motmetrics', 'opencv-python', 
                             'torch', 'torchvision', 'pytorch-lightning'])
        print("\nNote: You may also need to install YOLOX:")
        print("  pip install yolox")
        print("  or clone from: https://github.com/Megvii-BaseDetection/YOLOX")
        print("\nPackages installed successfully! Please run the script again.")
        exit(0)
    
    main()
