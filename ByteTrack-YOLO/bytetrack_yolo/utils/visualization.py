"""
Visualization utilities for tracking results
"""

import cv2
import numpy as np


class TrackVisualizer:
    """Visualize tracking results on video frames"""
    
    def __init__(self, class_names=None, seed=42):
        """
        Initialize visualizer
        
        Args:
            class_names: Dictionary mapping class IDs to names
            seed: Random seed for color generation
        """
        self.class_names = class_names or {}
        
        # Generate random colors for track IDs
        np.random.seed(seed)
        self.colors = np.random.randint(0, 255, size=(1000, 3), dtype=np.uint8)
        
    def draw_tracks(self, frame, tracks, draw_info=True):
        """
        Draw tracks on frame
        
        Args:
            frame: Input frame (BGR)
            tracks: List of STrack objects
            draw_info: Whether to draw track info (ID, class)
        
        Returns:
            frame: Frame with drawn tracks
        """
        for track in tracks:
            if not track.is_activated:
                continue
            
            # Get bounding box
            tlbr = track.tlbr
            x1, y1, x2, y2 = map(int, tlbr)
            
            # Get track color
            track_id = track.track_id
            color = self.colors[track_id % len(self.colors)].tolist()
            
            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            if draw_info:
                # Get class name
                class_name = "unknown"
                if track.class_id is not None:
                    class_name = self.class_names.get(
                        int(track.class_id), 
                        f"class_{int(track.class_id)}"
                    )
                
                # Create label
                label = f"ID:{track_id} | {class_name}"
                
                # Get text size
                (label_w, label_h), _ = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
                )
                
                # Draw background for text
                cv2.rectangle(
                    frame, 
                    (x1, y1 - label_h - 10), 
                    (x1 + label_w + 10, y1), 
                    color, 
                    -1
                )
                
                # Draw text
                cv2.putText(
                    frame, label, (x1 + 5, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2
                )
        
        return frame
    
    def draw_info_panel(self, frame, frame_id, total_frames, num_detections, 
                       num_tracks, fps):
        """
        Draw information panel on frame
        
        Args:
            frame: Input frame
            frame_id: Current frame number
            total_frames: Total number of frames
            num_detections: Number of detections
            num_tracks: Number of tracks
            fps: Current FPS
        
        Returns:
            frame: Frame with info panel
        """
        info_text = [
            f"Frame: {frame_id}/{total_frames}",
            f"Detections: {num_detections}",
            f"Tracks: {num_tracks}",
            f"FPS: {fps:.1f}"
        ]
        
        y_offset = 30
        for text in info_text:
            cv2.putText(
                frame, text, (10, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2
            )
            y_offset += 25
        
        return frame
