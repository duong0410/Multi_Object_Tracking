"""
ROI (Region of Interest) utilities for video processing
"""

import numpy as np
import cv2
from typing import List, Tuple, Optional


def create_roi_mask(frame_shape, roi_polygon):
    """
    Create binary mask from ROI polygon
    
    Args:
        frame_shape: (height, width) of frame
        roi_polygon: List of (x, y) points
        
    Returns:
        mask: Binary mask (255 inside ROI, 0 outside)
    """
    if roi_polygon is None or len(roi_polygon) < 3:
        return None
    
    mask = np.zeros(frame_shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [np.array(roi_polygon, dtype=np.int32)], 255)
    return mask


def filter_detections_by_roi(detections, roi_mask):
    """
    Filter detections to keep only those inside ROI
    
    Args:
        detections: Array [N, 6] of [x1, y1, x2, y2, conf, class]
        roi_mask: Binary mask
        
    Returns:
        filtered_detections: Detections inside ROI
    """
    if roi_mask is None or len(detections) == 0:
        return detections
    
    filtered = []
    for det in detections:
        x1, y1, x2, y2 = map(int, det[:4])
        # Check if center point is inside ROI
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        if roi_mask[cy, cx] > 0:
            filtered.append(det)
    
    return np.array(filtered) if len(filtered) > 0 else np.empty((0, 6))


class ROISelector:
    """Interactive ROI polygon selector"""
    
    def __init__(self, frame, canvas, callback):
        """
        Initialize ROI selector
        
        Args:
            frame: Video frame to draw on
            canvas: tkinter Canvas to draw on
            callback: Callback function when finished - receives polygon points
        """
        self.frame = frame.copy()
        self.frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.canvas = canvas
        self.callback = callback
        self.points = []
        self.scale = 1.0
        self.offset_x = 0
        self.offset_y = 0
        self.canvas_width = canvas.winfo_width()
        self.canvas_height = canvas.winfo_height()
        self.is_active = True
        
        # Bind canvas click
        self.canvas.bind("<Button-1>", self._on_canvas_click)
        
        # Initial draw
        self._update_display()
        
        print("\n" + "="*60)
        print("Drawing Polygon on Canvas:")
        print("  - Click on canvas to add polygon points")
        print("  - Need at least 3 points to finish")
        print("  - Press ENTER to finish")
        print("  - Press 'C' to clear all points")
        print("="*60 + "\n")
    
    def _on_canvas_click(self, event):
        """Handle canvas click"""
        if not self.is_active:
            return
        
        # Convert canvas coordinates to frame coordinates
        x_canvas = event.x - self.offset_x
        y_canvas = event.y - self.offset_y
        
        x_frame = int(x_canvas / self.scale)
        y_frame = int(y_canvas / self.scale)
        
        # Check if click is within frame bounds
        if 0 <= x_frame < self.frame_rgb.shape[1] and 0 <= y_frame < self.frame_rgb.shape[0]:
            self.points.append((x_frame, y_frame))
            print(f"Point added: ({x_frame}, {y_frame}) - Total: {len(self.points)}")
            self._update_display()
    
    def _update_display(self):
        """Update canvas display with frame and points"""
        try:
            from PIL import Image, ImageTk
        except ImportError:
            print("Error: PIL not installed")
            return
        
        # Resize frame to fit canvas
        h, w = self.frame_rgb.shape[:2]
        self.canvas_width = self.canvas.winfo_width()
        self.canvas_height = self.canvas.winfo_height()
        
        if self.canvas_width > 1 and self.canvas_height > 1:
            self.scale = min(self.canvas_width / w, self.canvas_height / h)
            new_w = int(w * self.scale)
            new_h = int(h * self.scale)
            
            frame_scaled = cv2.resize(self.frame_rgb, (new_w, new_h))
            
            # Draw points on frame
            frame_draw = frame_scaled.copy()
            for i, pt in enumerate(self.points):
                pt_scaled = (int(pt[0] * self.scale), int(pt[1] * self.scale))
                cv2.circle(frame_draw, pt_scaled, 5, (0, 255, 0), -1)
                cv2.circle(frame_draw, pt_scaled, 8, (0, 255, 0), 2)
                cv2.putText(frame_draw, str(i+1), (pt_scaled[0]+10, pt_scaled[1]-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                if i > 0:
                    prev_pt = (int(self.points[i-1][0] * self.scale), int(self.points[i-1][1] * self.scale))
                    cv2.line(frame_draw, prev_pt, pt_scaled, (0, 255, 0), 2)
            
            # Draw closing line if we have points
            if len(self.points) > 2:
                pt0 = (int(self.points[0][0] * self.scale), int(self.points[0][1] * self.scale))
                pt_last = (int(self.points[-1][0] * self.scale), int(self.points[-1][1] * self.scale))
                cv2.line(frame_draw, pt_last, pt0, (0, 255, 0), 2)
                # Fill polygon with transparency
                points_scaled = np.array([(int(p[0] * self.scale), int(p[1] * self.scale)) for p in self.points], dtype=np.int32)
                overlay = frame_draw.copy()
                cv2.fillPoly(overlay, [points_scaled], (0, 255, 0))
                frame_draw = cv2.addWeighted(frame_draw, 0.7, overlay, 0.3, 0)
            
            # Convert to PIL and display
            frame_pil = Image.fromarray(frame_draw)
            photo = ImageTk.PhotoImage(frame_pil)
            self.canvas.create_image(0, 0, image=photo, anchor='nw')
            self.canvas.image = photo
    
    def clear_points(self):
        """Clear all points"""
        self.points = []
        self._update_display()
    
    def finish(self):
        """Finish selection and return points"""
        if len(self.points) >= 3:
            self.is_active = False
            self.callback(self.points)
            print(f"ROI polygon selected with {len(self.points)} points")
            return self.points
        else:
            print(f"Error: Need at least 3 points, got {len(self.points)}")
            return None
