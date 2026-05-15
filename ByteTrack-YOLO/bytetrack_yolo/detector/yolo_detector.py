"""
YOLO Detector wrapper for object detection
"""

import numpy as np
from pathlib import Path


class YOLODetector:
    """YOLO detector wrapper for ByteTrack"""
    
    def __init__(self, model_path='yolo11n.pt', conf_threshold=0.1, device='cuda',
                 min_box_area=400, edge_margin=10):
        """
        Initialize YOLO detector
        
        Args:
            model_path: Path to YOLO model weights (.pt file)
            conf_threshold: Minimum confidence threshold for detections
            device: Device to run inference on ('cuda' or 'cpu')
            min_box_area: Minimum bounding box area (pixels) to keep detection (default: 400)
            edge_margin: Margin from frame edge (pixels) to filter out objects leaving scene (default: 10)
        """
        try:
            from ultralytics import YOLO
            
            if not Path(model_path).exists():
                raise FileNotFoundError(f"Model not found: {model_path}")
            
            self.model = YOLO(model_path)
            self.conf_threshold = conf_threshold
            self.device = device
            self.min_box_area = min_box_area
            self.edge_margin = edge_margin
            
            # Load class names from model
            self.class_names = self.model.names
            
            print(f" YOLO model loaded: {Path(model_path).name}")
            print(f"  Device: {device}")
            print(f"  Confidence threshold: {conf_threshold}")
            print(f"  Min box area: {min_box_area} pixels")
            print(f"  Edge margin: {edge_margin} pixels")
            print(f"  Classes: {list(self.class_names.values())}")
            
        except ImportError:
            raise ImportError(
                "ultralytics package not found! "
                "Install with: pip install ultralytics"
            )
    
    def get_class_name(self, class_id):
        """
        Get class name from class ID
        
        Args:
            class_id: Integer class ID
        
        Returns:
            Class name string
        """
        return self.class_names.get(int(class_id), f"class_{int(class_id)}")
    
    def filter_detections(self, detections, frame_width, frame_height):
        """
        Filter detections by size and edge position
        
        Args:
            detections: numpy array [N, 6] with format [x1, y1, x2, y2, conf, class]
            frame_width: Width of the frame
            frame_height: Height of the frame
            
        Returns:
            filtered_detections: Detections passing the filters
        """
        if len(detections) == 0:
            return detections
        
        filtered = []
        for det in detections:
            x1, y1, x2, y2 = det[:4]
            
            # Calculate box area
            box_width = x2 - x1
            box_height = y2 - y1
            box_area = box_width * box_height
            
            # Filter 1: Remove very small boxes (likely false positives or too far away)
            if box_area < self.min_box_area:
                continue
            
            # Filter 2: Remove objects too close to frame edges (leaving scene)
            # Check if box center is near edge
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            
            # Allow some margin but filter out objects clearly leaving the frame
            if (center_x < self.edge_margin or 
                center_x > frame_width - self.edge_margin or
                center_y < self.edge_margin or 
                center_y > frame_height - self.edge_margin):
                # Additional check: only filter if box is also partially outside
                if (x1 < self.edge_margin or x2 > frame_width - self.edge_margin or
                    y1 < self.edge_margin or y2 > frame_height - self.edge_margin):
                    continue
            
            filtered.append(det)
        
        return np.array(filtered) if len(filtered) > 0 else np.empty((0, 6))
    
    def detect(self, frame):
        """
        Run detection on a frame
        
        Args:
            frame: Input frame (BGR format)
            
        Returns:
            detections: numpy array [N, 6] with format [x1, y1, x2, y2, conf, class]
        """
        # Run inference
        results = self.model(
            frame, 
            conf=self.conf_threshold, 
            device=self.device, 
            verbose=False
        )
        
        # Extract detections - USE LIST TO ACCUMULATE
        all_detections = []
        for result in results:
            boxes = result.boxes
            if boxes is not None and len(boxes) > 0:
                # Get boxes in xyxy format
                xyxy = boxes.xyxy.cpu().numpy()
                conf = boxes.conf.cpu().numpy()
                cls = boxes.cls.cpu().numpy()
                
                # Combine into detections array
                if len(xyxy) > 0:
                    batch_detections = np.concatenate([
                        xyxy,
                        conf.reshape(-1, 1),
                        cls.reshape(-1, 1)
                    ], axis=1)
                    all_detections.append(batch_detections)
        
        # Combine all detections from all results
        if len(all_detections) == 0:
            return np.empty((0, 6))
        
        # Concatenate all batches
        detections = np.vstack(all_detections) if len(all_detections) > 1 else all_detections[0]
        
        # Apply size and edge filters
        frame_height, frame_width = frame.shape[:2]
        detections = self.filter_detections(detections, frame_width, frame_height)
        
        return detections
    
    def detect_batch(self, frames):
        """
        Run detection on multiple frames (batch processing)
        
        Args:
            frames: List of frames
            
        Returns:
            List of detection arrays
        """
        results = self.model(frames, conf=self.conf_threshold, 
                           device=self.device, verbose=False)
        
        all_detections = []
        for result in results:
            boxes = result.boxes
            if boxes is not None and len(boxes) > 0:
                xyxy = boxes.xyxy.cpu().numpy()
                conf = boxes.conf.cpu().numpy()
                cls = boxes.cls.cpu().numpy()
                
                if len(xyxy) > 0:
                    detections = np.concatenate([
                        xyxy,
                        conf.reshape(-1, 1),
                        cls.reshape(-1, 1)
                    ], axis=1)
                else:
                    detections = np.empty((0, 6))
            else:
                detections = np.empty((0, 6))
            
            all_detections.append(detections)
        
        return all_detections
