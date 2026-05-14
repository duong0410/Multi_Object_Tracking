#!/usr/bin/env python3
"""
ByteTrack GUI - Vehicle Tracking with YOLO11 and Traffic Violation Detection
GUI Interface matching bytetrack_test.py but using modular src/ packages

Usage:
    python app_gui.py

Features:
    - YOLO11 object detection
    - ByteTrack multi-object tracking
    - Traffic violation detection
    - Real-time video visualization
    - Interactive lane configuration (ROI drawing)
    - Video saving support
"""

import sys
import os
import time
import queue
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from pathlib import Path
from typing import List, Tuple, Optional, Dict
from collections import Counter
from datetime import timedelta
import csv

import cv2
import numpy as np

try:
    import pandas as pd
except ImportError:
    pd = None

try:
    from PIL import Image, ImageTk
except ImportError:
    print("Error: PIL not installed. Run: pip install pillow")
    sys.exit(1)

# Add src to path for modular imports
sys.path.insert(0, str(Path(__file__).parent))

from src.detector import YOLODetector
from src.tracker import (
    BYTETracker,
    STrack,
    TrackState,
    ViolationDetector,
    TrafficLane,
    ViolationType,
    NoParkingZone,
)
from src.utils.bbox import bbox_iou


class TrackLogger:
    """Collects per-track lifecycle data and exports Excel/CSV logs."""

    def __init__(self):
        self.entries = {}
        self.fps = 30

    @staticmethod
    def format_timestamp(seconds: float) -> str:
        td = timedelta(seconds=float(seconds))
        total_seconds = int(td.total_seconds())
        millis = int((td.total_seconds() - total_seconds) * 1000)
        hours, remainder = divmod(total_seconds, 3600)
        minutes, secs = divmod(remainder, 60)
        return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millis:03d}"

    def reset(self):
        self.entries = {}
        self.fps = 30

    def update_frame(self, tracks: List, frame_id: int, fps: int, detector):
        self.fps = fps
        for track in tracks:
            if not track.is_activated:
                continue

            track_id = track.track_id
            class_name = detector.get_class_name(track.class_id) if track.class_id is not None else "unknown"
            violations = set(getattr(track, "violation_types", []))

            entry = self.entries.get(track_id)
            if entry is None:
                entry = {
                    "track_id": track_id,
                    "class_id": track.class_id,
                    "class_name": class_name,
                    "first_frame": track.start_frame,
                    "first_time_s": track.start_frame / fps,
                    "last_frame": frame_id,
                    "last_time_s": frame_id / fps,
                    "violations": violations,
                }
                self.entries[track_id] = entry
            else:
                entry["class_name"] = class_name or entry["class_name"]
                entry["class_id"] = track.class_id if entry["class_id"] is None else entry["class_id"]
                entry["first_frame"] = min(entry["first_frame"], track.start_frame)
                entry["first_time_s"] = entry["first_frame"] / fps
                entry["last_frame"] = max(entry["last_frame"], frame_id)
                entry["last_time_s"] = entry["last_frame"] / fps
                entry["violations"].update(violations)

    def build_report(self):
        rows = []
        for entry in sorted(self.entries.values(), key=lambda e: e["track_id"]):
            duration_frames = max(0, entry["last_frame"] - entry["first_frame"] + 1)
            duration_seconds = max(0.0, entry["last_time_s"] - entry["first_time_s"])
            violation_names = [v.name for v in sorted(entry["violations"], key=lambda x: x.value)]
            rows.append({
                "Track ID": entry["track_id"],
                "Class": entry["class_name"],
                "Class ID": entry["class_id"],
                "First Frame": entry["first_frame"],
                "First Timestamp": self.format_timestamp(entry["first_time_s"]),
                "Last Frame": entry["last_frame"],
                "Last Timestamp": self.format_timestamp(entry["last_time_s"]),
                "Duration (frames)": duration_frames,
                "Duration (seconds)": round(duration_seconds, 3),
                "Violations": ", ".join(violation_names) if violation_names else "None",
            })

        counts_by_class = Counter(entry["class_name"] for entry in self.entries.values())
        counts_by_violation = Counter(
            v.name for entry in self.entries.values() for v in entry["violations"]
        )

        summary_rows = [
            {"Metric": "Total distinct tracks", "Value": len(self.entries)},
        ]
        summary_rows.extend(
            {"Metric": f"Total {class_name}", "Value": count}
            for class_name, count in sorted(counts_by_class.items())
        )
        if counts_by_violation:
            summary_rows.append({"Metric": "" , "Value": ""})
            summary_rows.append({"Metric": "Violation counts", "Value": ""})
            summary_rows.extend(
                {"Metric": violation_name.replace("_", " "), "Value": count}
                for violation_name, count in sorted(counts_by_violation.items())
            )

        return rows, summary_rows

    def save(self, base_path: str):
        if not self.entries:
            return None

        base_path = Path(base_path)
        output_dir = base_path.parent if base_path.exists() else base_path.parent
        log_stem = f"{base_path.stem}_tracking_log"
        excel_path = output_dir / f"{log_stem}.xlsx"
        csv_path = output_dir / f"{log_stem}.csv"

        rows, summary_rows = self.build_report()

        if pd is not None:
            try:
                df_tracks = pd.DataFrame(rows)
                df_summary = pd.DataFrame(summary_rows)
                with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
                    df_tracks.to_excel(writer, index=False, sheet_name="tracks")
                    df_summary.to_excel(writer, index=False, sheet_name="summary")
                return excel_path
            except Exception:
                pass

        # Fallback to CSV if Excel writer is unavailable
        with open(csv_path, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

        return csv_path


# ============================================================================
# ROI Selector - Interactive polygon drawing for lane definition
# ============================================================================

class ROISelector:
    """Interactive ROI polygon selector - exactly 4 points for lane"""
    
    def __init__(self, frame, canvas, callback):
        """
        Initialize ROI selector to draw exactly 4 points
        
        Args:
            frame: Video frame to draw on
            canvas: tkinter Canvas to draw on
            callback: Callback function when 4 points finished - receives polygon points
        """
        self.frame = frame.copy()
        self.frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.canvas = canvas
        self.callback = callback
        self.points = []
        self.scale = 1.0
        self.offset_x = 0
        self.offset_y = 0
        self.photo = None
        self.is_active = True
        self.canvas_id = None
        
        # Bind canvas click
        self.canvas_id = self.canvas.bind("<Button-1>", self._on_canvas_click)
        
        # Initial draw
        self._update_display()
        
        print("\n" + "="*60)
        print("Draw Lane ROI (4 POINTS):")
        print("  - Click 4 points on canvas to define ROI area")
        print("  - Points: 0/4")
        print("="*60 + "\n")
    
    def _on_canvas_click(self, event):
        """Handle canvas click"""
        if not self.is_active or len(self.points) >= 4:
            return
        
        # Convert canvas coordinates to frame coordinates
        x_canvas = event.x - self.offset_x
        y_canvas = event.y - self.offset_y
        
        x_frame = int(x_canvas / self.scale)
        y_frame = int(y_canvas / self.scale)
        
        # Check if click is within frame bounds
        if 0 <= x_frame < self.frame_rgb.shape[1] and 0 <= y_frame < self.frame_rgb.shape[0]:
            self.points.append((x_frame, y_frame))
            print(f"Point {len(self.points)}/4: ({x_frame}, {y_frame})")
            self._update_display()
            
            # If 4 points are selected, finish automatically
            if len(self.points) == 4:
                print("4 points selected! Processing...")
                self._finish()
    
    def _update_display(self):
        """Update canvas display with frame and points"""
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
                cv2.circle(frame_draw, pt_scaled, 8, (0, 255, 0), -1)
                cv2.circle(frame_draw, pt_scaled, 12, (0, 255, 0), 2)
                cv2.putText(frame_draw, str(i+1), (pt_scaled[0]+20, pt_scaled[1]-20),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 0), 2)
                if i > 0:
                    prev_pt = (int(self.points[i-1][0] * self.scale), int(self.points[i-1][1] * self.scale))
                    cv2.line(frame_draw, prev_pt, pt_scaled, (0, 255, 0), 2)
            
            # Draw closing line if we have 4 points
            if len(self.points) == 4:
                pt0 = (int(self.points[0][0] * self.scale), int(self.points[0][1] * self.scale))
                pt_last = (int(self.points[-1][0] * self.scale), int(self.points[-1][1] * self.scale))
                cv2.line(frame_draw, pt_last, pt0, (0, 255, 0), 2)
                # Fill polygon with transparency
                points_scaled = np.array([(int(p[0] * self.scale), int(p[1] * self.scale)) for p in self.points], dtype=np.int32)
                overlay = frame_draw.copy()
                cv2.fillPoly(overlay, [points_scaled], (0, 255, 0))
                cv2.addWeighted(overlay, 0.2, frame_draw, 0.8, 0, frame_draw)
            
            # Add text instructions on frame
            cv2.putText(frame_draw, f"Points: {len(self.points)}/4", (20, 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)
            
            if len(self.points) < 4:
                cv2.putText(frame_draw, "Click to add points", (20, 100),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            else:
                cv2.putText(frame_draw, "READY! Configuring...", (20, 100),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 3)
            
            # Convert to PIL and display
            try:
                pil_img = Image.fromarray(frame_draw)
                self.photo = ImageTk.PhotoImage(pil_img)
                
                # Center image on canvas
                self.offset_x = (self.canvas_width - new_w) // 2
                self.offset_y = (self.canvas_height - new_h) // 2
                
                # Clear and redraw
                self.canvas.delete("all")
                self.canvas.create_image(self.offset_x, self.offset_y, image=self.photo, anchor='nw')
            except Exception as e:
                print(f"Error updating display: {e}")
    
    def _finish(self):
        """Finish polygon selection after 4 points"""
        if len(self.points) != 4:
            return
        
        self.is_active = False
        self.canvas.unbind("<Button-1>", self.canvas_id)
        self.callback(self.points)
        print(f"✓ ROI polygon created with 4 points")


# ============================================================================
# GUI Application
# ============================================================================

class ByteTrackGUI:
    """GUI Application for ByteTrack + YOLO video tracking"""
    
    def __init__(self, root):
        self.root = root
        self.root.title("ByteTrack Tracker")
        self.root.geometry("1800x900")
        
        # State variables
        self.video_path = None
        self.output_path = None
        self.is_processing = False
        self.should_stop = False
        self.cap = None
        self.tracker = None
        self.detector = None
        
        # Traffic lane & violation management
        self.violation_detector = ViolationDetector()
        self.traffic_lanes = {}  # {lane_id: TrafficLane}
        self.no_parking_zones = {}  # {zone_id: NoParkingZone}
        self.track_logger = TrackLogger()
        
        # video_fps: FPS gốc của video (giới hạn dưới cho display)
        # actual_fps: FPS hệ thống xử lý được realtime (giới hạn trên cho display)
        # display_fps = min(video_fps, actual_fps)
        self.video_fps = 30
        self.actual_fps = 30
        self.person_vehicle_iou_threshold = 0.2

        # Queue size=1: chỉ giữ frame mới nhất, không bao giờ block
        self.frame_queue = queue.Queue(maxsize=1)
        
        # Create GUI
        self.create_widgets()
        
    def filter_person_riding_vehicle(self, detections: np.ndarray, iou_threshold: float = 0.2) -> np.ndarray:
        """Keep actual pedestrians and remove riders sitting on vehicles."""
        if detections.size == 0:
            return detections

        vehicle_indices = []
        person_indices = []
        for idx, det in enumerate(detections):
            class_id = int(det[5])
            class_name = self.detector.get_class_name(class_id).lower()
            if class_name == "person":
                person_indices.append(idx)
            elif any(vehicle in class_name for vehicle in ("car", "bus", "truck", "bicycle", "motor")):
                vehicle_indices.append(idx)

        if not person_indices or not vehicle_indices:
            return detections

        keep_indices = [i for i in range(len(detections)) if i not in person_indices]
        for p_idx in person_indices:
            person_box = detections[p_idx, :4]
            is_rider = False
            for v_idx in vehicle_indices:
                vehicle_box = detections[v_idx, :4]
                if bbox_iou(person_box, vehicle_box) >= iou_threshold:
                    is_rider = True
                    break
            if not is_rider:
                keep_indices.append(p_idx)

        keep_indices.sort()
        return detections[keep_indices]

    def create_widgets(self):
        """Create all GUI widgets"""
        
        # Configure main grid - settings on left, video on right
        self.root.columnconfigure(0, weight=0)  # Settings - fixed width
        self.root.columnconfigure(1, weight=1)  # Video - takes remaining space
        self.root.rowconfigure(0, weight=1)
        
        # LEFT PANEL: Settings
        settings_frame = ttk.LabelFrame(self.root, text="⚙ Settings", padding="8")
        settings_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), 
                           padx=3, pady=3, ipadx=5, ipady=5)
        settings_frame.columnconfigure(0, weight=1)
        
        # Video Selection
        ttk.Label(settings_frame, text="Video:", font=("Arial", 8, "bold")).grid(
            row=0, column=0, sticky=tk.W, pady=2)
        
        self.video_label = ttk.Label(settings_frame, text="None", 
                                     foreground="gray", font=("Arial", 7), wraplength=140)
        self.video_label.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=(0, 3))
        
        ttk.Button(settings_frame, text="Browse", command=self.browse_video, width=15).grid(
            row=2, column=0, sticky=(tk.W, tk.E), pady=(0, 5))
        
        # Model Path
        ttk.Label(settings_frame, text="Model:", font=("Arial", 8, "bold")).grid(
            row=3, column=0, sticky=tk.W, pady=2)
        
        self.model_path = r"D:\Learn\Year4\KLTN\ByteTrack-YOLO\models\traffic_yolo_v11m\best.pt"
        self.model_label = ttk.Label(settings_frame, text=Path(self.model_path).name, 
                               foreground="blue", font=("Arial", 7))
        self.model_label.grid(row=4, column=0, sticky=tk.W, pady=(0, 2))
        
        ttk.Button(settings_frame, text="Browse Model", 
                   command=self.browse_model, width=15).grid(
                   row=5, column=0, sticky=(tk.W, tk.E), pady=(0, 5))
        
        # Device Selection
        ttk.Label(settings_frame, text="Device:", font=("Arial", 8, "bold")).grid(
            row=6, column=0, sticky=tk.W, pady=2)
        
        self.device_var = tk.StringVar(value="cuda")
        device_frame = ttk.Frame(settings_frame)
        device_frame.grid(row=7, column=0, sticky=(tk.W, tk.E), pady=(0, 5))
        
        ttk.Radiobutton(device_frame, text="GPU", variable=self.device_var, 
                       value="cuda").pack(side=tk.LEFT, padx=(0, 10))
        ttk.Radiobutton(device_frame, text="CPU", variable=self.device_var, 
                       value="cpu").pack(side=tk.LEFT)
        
        # Separator
        ttk.Separator(settings_frame, orient=tk.HORIZONTAL).grid(
            row=8, column=0, sticky=(tk.W, tk.E), pady=3)
        
        # Detection Confidence
        ttk.Label(settings_frame, text="Detection:", font=("Arial", 8, "bold")).grid(
            row=9, column=0, sticky=tk.W, pady=2)
        self.det_conf_var = tk.DoubleVar(value=0.01)
        det_conf_scale = ttk.Scale(settings_frame, from_=0.0, to=1.0, 
                                   variable=self.det_conf_var, orient=tk.HORIZONTAL)
        det_conf_scale.grid(row=10, column=0, sticky=(tk.W, tk.E), pady=(0, 1))
        self.det_conf_label = ttk.Label(settings_frame, text="0.01", font=("Arial", 7))
        self.det_conf_label.grid(row=11, column=0, sticky=tk.W)
        det_conf_scale.config(command=lambda v: self.det_conf_label.config(
            text=f"{float(v):.2f}"))
        
        # Separator
        ttk.Separator(settings_frame, orient=tk.HORIZONTAL).grid(
            row=12, column=0, sticky=(tk.W, tk.E), pady=3)
        
        # Track Settings
        ttk.Label(settings_frame, text="Track Buf:", font=("Arial", 7)).grid(
            row=13, column=0, sticky=tk.W, pady=2)
        self.track_buffer_var = tk.IntVar(value=30)
        buffer_spinbox = ttk.Spinbox(settings_frame, from_=1, to=100, 
                                     textvariable=self.track_buffer_var, width=12)
        buffer_spinbox.grid(row=14, column=0, sticky=(tk.W, tk.E), pady=1)
        
        ttk.Label(settings_frame, text="Min Area:", font=("Arial", 7)).grid(
            row=15, column=0, sticky=tk.W, pady=2)
        self.min_box_area_var = tk.IntVar(value=400)
        area_spinbox = ttk.Spinbox(settings_frame, from_=100, to=5000, increment=100,
                                   textvariable=self.min_box_area_var, width=12)
        area_spinbox.grid(row=16, column=0, sticky=(tk.W, tk.E), pady=1)
        
        ttk.Label(settings_frame, text="Edge:", font=("Arial", 7)).grid(
            row=17, column=0, sticky=tk.W, pady=2)
        self.edge_margin_var = tk.IntVar(value=10)
        margin_spinbox = ttk.Spinbox(settings_frame, from_=0, to=100, increment=5,
                                     textvariable=self.edge_margin_var, width=12)
        margin_spinbox.grid(row=18, column=0, sticky=(tk.W, tk.E), pady=1)
        
        # Output Settings
        self.save_output_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(settings_frame, text="Save output", 
                       variable=self.save_output_var,
                       command=self.toggle_output).grid(row=19, column=0, 
                                                        sticky=tk.W, pady=5)
        
        # Separator
        ttk.Separator(settings_frame, orient=tk.HORIZONTAL).grid(
            row=20, column=0, sticky=(tk.W, tk.E), pady=3)
        
        # ===== VIOLATION DETECTION SECTION =====
        ttk.Label(settings_frame, text="Violation Detection", 
                 font=("Arial", 9, "bold")).grid(row=22, column=0, sticky=tk.W, pady=(8, 3))
        
        self.enable_violation_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(settings_frame, text="Enable violation check", 
                       variable=self.enable_violation_var).grid(
                       row=23, column=0, sticky=tk.W, pady=(1, 5))
        
        # ===== TRAFFIC LANES SECTION =====
        ttk.Label(settings_frame, text="Traffic Lanes", 
                 font=("Arial", 8, "bold")).grid(row=24, column=0, sticky=tk.W, pady=(3, 2))
        
        lane_btn_frame = ttk.Frame(settings_frame)
        lane_btn_frame.grid(row=25, column=0, sticky=(tk.W, tk.E), pady=(0, 2))
        
        ttk.Button(lane_btn_frame, text="New Lane", 
                  command=self.setup_traffic_lanes, width=12).pack(side=tk.LEFT, padx=(0, 2), fill=tk.X, expand=True)
        ttk.Button(lane_btn_frame, text="Modify", 
                  command=self.modify_traffic_lane, width=8).pack(side=tk.LEFT, padx=1, fill=tk.X, expand=True)
        ttk.Button(lane_btn_frame, text="Delete", 
                  command=self.delete_traffic_lane, width=8).pack(side=tk.LEFT, padx=(1, 0), fill=tk.X, expand=True)

        # Lane Info Display
        self.lane_list_label = ttk.Label(settings_frame, 
                                        text="0 lanes", 
                                        foreground="orange", 
                                        font=("Arial", 6, "bold"))
        self.lane_list_label.grid(row=26, column=0, sticky=tk.W, pady=(2, 0))
        
        self.lane_detail_label = ttk.Label(settings_frame, 
                                          text="", 
                                          foreground="blue", 
                                          font=("Arial", 5), 
                                          wraplength=140, 
                                          justify=tk.LEFT)
        self.lane_detail_label.grid(row=27, column=0, sticky=(tk.W, tk.E), pady=(0, 3))
        
        # ===== NO-PARKING ZONES SECTION =====
        ttk.Label(settings_frame, text="No-Parking Zones", 
                 font=("Arial", 8, "bold")).grid(row=28, column=0, sticky=tk.W, pady=(4, 1))
        
        ttk.Label(settings_frame, text="Timeout (frames):", 
                 font=("Arial", 7)).grid(row=29, column=0, sticky=tk.W, pady=(1, 1))
        self.no_park_frames_var = tk.IntVar(value=45)
        no_park_spinbox = ttk.Spinbox(settings_frame, from_=10, to=300,
                                      textvariable=self.no_park_frames_var, width=12)
        no_park_spinbox.grid(row=30, column=0, sticky=(tk.W, tk.E), pady=(0, 2))

        no_parking_btn_frame = ttk.Frame(settings_frame)
        no_parking_btn_frame.grid(row=31, column=0, sticky=(tk.W, tk.E), pady=(0, 2))
        ttk.Button(no_parking_btn_frame, text="New Zone", 
              command=self.setup_no_parking_zone, width=13).pack(side=tk.LEFT, padx=(0, 2), fill=tk.X, expand=True)
        ttk.Button(no_parking_btn_frame, text="Clear All", 
              command=self.clear_no_parking_zones, width=9).pack(side=tk.LEFT, padx=(2, 0), fill=tk.X, expand=True)
        
        # No-Parking Info Display
        self.zone_list_label = ttk.Label(settings_frame, 
                                        text="0 zones", 
                                        foreground="orange", 
                                        font=("Arial", 6, "bold"))
        self.zone_list_label.grid(row=32, column=0, sticky=tk.W, pady=(2, 0))
        
        self.zone_detail_label = ttk.Label(settings_frame, 
                                          text="", 
                                          foreground="blue", 
                                          font=("Arial", 5), 
                                          wraplength=140, 
                                          justify=tk.LEFT)
        self.zone_detail_label.grid(row=33, column=0, sticky=(tk.W, tk.E), pady=(0, 3))
        
        # Combined info label (kept for backwards compatibility)
        self.lane_info_label = ttk.Label(settings_frame, 
                                        text="", 
                                        foreground="green", 
                                        font=("Arial", 6), 
                                        wraplength=140)
        self.lane_info_label.grid(row=34, column=0, sticky=(tk.W, tk.E), pady=(0, 3))
        
        # Separator
        ttk.Separator(settings_frame, orient=tk.HORIZONTAL).grid(
            row=35, column=0, sticky=(tk.W, tk.E), pady=2)
        
        # Control Buttons
        button_frame = ttk.Frame(settings_frame)
        button_frame.grid(row=36, column=0, sticky=(tk.W, tk.E), pady=(3, 0))
        
        self.start_button = ttk.Button(button_frame, text="Start", 
                                       command=self.start_processing)
        self.start_button.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 2))
        
        self.stop_button = ttk.Button(button_frame, text="Stop", 
                                      command=self.stop_processing, 
                                      state=tk.DISABLED)
        self.stop_button.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(2, 0))
        
        # Add spacer
        ttk.Label(settings_frame, text="").grid(row=37, column=0, sticky=(tk.W, tk.E), pady=5)
        settings_frame.rowconfigure(37, weight=1)
        
        # RIGHT PANEL: Video display
        right_frame = ttk.Frame(self.root, padding="5")
        right_frame.grid(row=0, column=1, sticky=(tk.W, tk.E, tk.N, tk.S), padx=3, pady=3)
        right_frame.columnconfigure(0, weight=1)
        right_frame.rowconfigure(0, weight=1)
        right_frame.rowconfigure(1, weight=0)
        
        # Video Display Canvas
        self.canvas = tk.Canvas(right_frame, width=1300, height=750, bg="black")
        self.canvas.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 5))
        
        # Status & Progress
        status_frame = ttk.Frame(right_frame)
        status_frame.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=(0, 0))
        
        self.status_label = ttk.Label(status_frame, text="Ready", font=("Arial", 9))
        self.status_label.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(status_frame, variable=self.progress_var, 
                                           maximum=100, mode='determinate', length=300)
        self.progress_bar.pack(side=tk.RIGHT, padx=(10, 0))
        
    def browse_video(self):
        """Open file dialog to select video"""
        filename = filedialog.askopenfilename(
            title="Select Video File",
            filetypes=[
                ("Video files", "*.mp4 *.avi *.mov *.mkv"),
                ("All files", "*.*")
            ]
        )
        
        if filename:
            self.video_path = filename
            self.video_label.config(text=Path(filename).name, foreground="black")
            print(f"✓ Video selected: {filename}")
            
            # Clear lanes when video changes
            self.traffic_lanes.clear()
            self.no_parking_zones.clear()
            self.violation_detector.lanes.clear()
            self.violation_detector.no_parking_zones.clear()
            self.violation_detector.no_parking_state.clear()
            self.violation_detector.persistent_violations.clear()
            self.update_lane_info_display()
            print("  Violation polygons cleared for new video")
            
            # Suggest output path
            if self.save_output_var.get():
                output_name = Path(filename).stem + "_tracked.mp4"
                self.output_path = str(Path(filename).parent / output_name)
                
    def browse_model(self):
        """Open file dialog to select a YOLO checkpoint"""
        filename = filedialog.askopenfilename(
            title="Select YOLO checkpoint",
            filetypes=[
                ("PyTorch checkpoint", "*.pt *.pth"),
                ("All files", "*.*")
            ]
        )
        if filename:
            self.model_path = filename
            self.model_label.config(text=Path(filename).name)
            print(f"✓ Model checkpoint selected: {filename}")

    def toggle_output(self):
        """Toggle output video saving"""
        if self.save_output_var.get() and self.video_path:
            output_name = Path(self.video_path).stem + "_tracked.mp4"
            self.output_path = str(Path(self.video_path).parent / output_name)
        else:
            self.output_path = None
    
    def setup_traffic_lanes(self):
        """Actually setup traffic lanes - draw 4 points then configure vehicle types"""
        # Open video to get first frame
        cap = cv2.VideoCapture(self.video_path)
        ret, frame = cap.read()
        cap.release()
        
        if not ret:
            messagebox.showerror("Error", "Cannot read video")
            return
        
        # Load detector to get class names
        try:
            detector_temp = YOLODetector(
                model_path=self.model_path,
                conf_threshold=self.det_conf_var.get(),
                device=self.device_var.get()
            )
            class_names = detector_temp.class_names
            del detector_temp
        except Exception as e:
            messagebox.showerror("Error", f"Cannot load model: {str(e)}")
            return
        
        # Show canvas
        self.root.deiconify()
        self.canvas.focus_set()
        
        polygon_result = [None]
        
        def on_polygon_finished(polygon):
            """Called when 4 points are drawn - show vehicle type dialog"""
            polygon_result[0] = polygon
            
            # Clear canvas
            self.canvas.delete("all")
            
            # Create dialog for vehicle type selection
            config_window = tk.Toplevel(self.root)
            config_window.title("Select Allowed Vehicle Types")
            config_window.geometry("450x450")
            config_window.resizable(False, False)
            
            # Title
            ttk.Label(config_window, text=f"Lane {len(self.traffic_lanes) + 1}: Select Vehicle Types", 
                     font=("Arial", 11, "bold")).pack(pady=10)
            ttk.Label(config_window, text="Which vehicles are ALLOWED to use this lane?", 
                     font=("Arial", 9)).pack(pady=(0, 10))
            
            # Help text
            ttk.Label(config_window, text=" Check = allowed in this lane\n✗ Uncheck = violation if detected", 
                     font=("Arial", 8, "italic"), foreground="gray").pack(pady=(0, 10))
            
            # Checkboxes for vehicle types
            class_vars = {}
            ttk.Label(config_window, text="Vehicle Types:", 
                     font=("Arial", 9, "bold")).pack(pady=(5, 5), anchor=tk.W, padx=15)
            
            cb_frame = ttk.Frame(config_window)
            cb_frame.pack(fill=tk.BOTH, expand=True, padx=15, pady=5)
            
            # Create scrollable list
            canvas_scroll = tk.Canvas(cb_frame, height=200)
            scrollbar = ttk.Scrollbar(cb_frame, orient=tk.VERTICAL, command=canvas_scroll.yview)
            scrollable_frame = ttk.Frame(canvas_scroll)
            
            scrollable_frame.bind(
                "<Configure>",
                lambda e: canvas_scroll.configure(scrollregion=canvas_scroll.bbox("all"))
            )
            
            canvas_scroll.create_window((0, 0), window=scrollable_frame, anchor="nw")
            canvas_scroll.configure(yscrollcommand=scrollbar.set)
            
            for class_id, class_name in sorted(class_names.items()):
                var = tk.BooleanVar(value=False)
                if 'car' in class_name.lower() or 'motorcycle' in class_name.lower():
                    var.set(True)
                
                cb = ttk.Checkbutton(scrollable_frame, text=f"{class_name} (ID:{class_id})", variable=var)
                cb.pack(anchor=tk.W, pady=2)
                class_vars[class_id] = var
            
            canvas_scroll.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
            scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
            
            # Button frame
            btn_frame = ttk.Frame(config_window)
            btn_frame.pack(fill=tk.X, padx=15, pady=15)
            
            def on_ok():
                """Save the lane with selected vehicle types"""
                allowed_classes = [cid for cid, var in class_vars.items() if var.get()]
                
                if not allowed_classes:
                    messagebox.showerror("Error", "Please select at least one vehicle type!")
                    return
                
                try:
                    lane_id = len(self.traffic_lanes) + 1
                    lane_name = f"Lane {lane_id}"
                    
                    lane = TrafficLane(
                        lane_id=lane_id,
                        name=lane_name,
                        polygon=polygon_result[0],
                        allowed_classes=allowed_classes,
                        direction_vector=(1.0, 0.0)
                    )
                    
                    self.traffic_lanes[lane_id] = lane
                    self.violation_detector.add_lane(lane)
                    
                    self._display_frame_on_canvas(frame)
                    
                    vehicle_types = ', '.join([class_names.get(cid, f'ID{cid}') for cid in allowed_classes])
                    messagebox.showinfo(" Success!", f"Lane {lane_id} created successfully!\n\nAllowed vehicles: {vehicle_types}")
                    
                    self.update_lane_info_display()
                    config_window.destroy()
                    
                except Exception as e:
                    messagebox.showerror("Error", f"Failed to create lane: {str(e)}")
            
            ttk.Button(btn_frame, text=" OK - Save Lane", command=on_ok).pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
            ttk.Button(btn_frame, text=" Cancel", command=config_window.destroy).pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        
        # Create ROI selector
        roi_selector = ROISelector(frame, self.canvas, on_polygon_finished)

    def setup_no_parking_zone(self):
        """Actually setup a no-parking polygon zone"""
        cap = cv2.VideoCapture(self.video_path)
        ret, frame = cap.read()
        cap.release()

        if not ret:
            messagebox.showerror("Error", "Cannot read video")
            return

        self.root.deiconify()
        self.canvas.focus_set()

        def on_polygon_finished(polygon):
            self.canvas.delete("all")

            try:
                zone_id = len(self.no_parking_zones) + 1
                timeout_frames = self.no_park_frames_var.get()
                zone = NoParkingZone(
                    zone_id=zone_id,
                    name=f"NoParking {zone_id}",
                    polygon=polygon,
                    parking_frame_threshold=timeout_frames,
                    movement_threshold=6.0,
                )
                self.no_parking_zones[zone_id] = zone
                self.violation_detector.add_no_parking_zone(zone)
                self._display_frame_on_canvas(frame)
                self.update_lane_info_display()
                messagebox.showinfo(
                    "Success!",
                    f"No-parking zone {zone_id} created successfully!\n\n"
                    f"Parking timeout: {timeout_frames} frames\n"
                    f"(vehicle must stay > {timeout_frames} frames = violation)"
                )
            except Exception as e:
                messagebox.showerror("Error", f"Failed to create no-parking zone: {str(e)}")

        roi_selector = ROISelector(frame, self.canvas, on_polygon_finished)

    def clear_no_parking_zones(self):
        """Delete all no-parking zones"""
        if len(self.no_parking_zones) == 0:
            messagebox.showerror("Error", "No no-parking zones to clear!")
            return

        if messagebox.askyesno("Confirm", "Delete all no-parking zones?"):
            self.no_parking_zones.clear()
            self.violation_detector.no_parking_zones.clear()
            self.violation_detector.no_parking_state.clear()
            self.update_lane_info_display()
            messagebox.showinfo("Success", "All no-parking zones deleted!")
    
    def modify_traffic_lane(self):
        """Modify an existing lane"""
        if len(self.traffic_lanes) == 0:
            messagebox.showerror("Error", "No lanes to modify!")
            return
        
        if not self.video_path:
            messagebox.showerror("Error", "Please select a video first!")
            return
        
        select_window = tk.Toplevel(self.root)
        select_window.title("Select Lane to Modify")
        select_window.geometry("300x250")
        select_window.resizable(False, False)
        
        ttk.Label(select_window, text="Select a lane to modify:", 
                 font=("Arial", 10, "bold")).pack(pady=10)
        
        listbox = tk.Listbox(select_window, height=8, font=("Arial", 9))
        listbox.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        for lane_id, lane in self.traffic_lanes.items():
            class_names_str = ", ".join([self.detector.class_names.get(cid, f"ID{cid}") 
                                        for cid in lane.allowed_classes])
            listbox.insert(tk.END, f"Lane {lane_id}: {class_names_str}")
        
        def on_select():
            try:
                idx = listbox.curselection()[0]
                selected_lane_id = list(self.traffic_lanes.keys())[idx]
                select_window.destroy()
                self._edit_lane(selected_lane_id)
            except IndexError:
                messagebox.showerror("Error", "Please select a lane!")
        
        ttk.Button(select_window, text="Modify Selected", command=on_select).pack(pady=10)
    
    def _edit_lane(self, lane_id):
        """Edit vehicle types for a specific lane"""
        if not self.video_path:
            messagebox.showerror("Error", "Please select a video first!")
            return
        
        lane = self.traffic_lanes[lane_id]
        
        edit_window = tk.Toplevel(self.root)
        edit_window.title(f"Modify Lane {lane_id}")
        edit_window.geometry("400x400")
        edit_window.resizable(False, False)
        
        ttk.Label(edit_window, text=f"Lane {lane_id}: Edit Allowed Vehicles", 
                 font=("Arial", 10, "bold")).pack(pady=10)
        
        class_vars = {}
        cb_frame = ttk.Frame(edit_window)
        cb_frame.pack(fill=tk.BOTH, expand=True, padx=15, pady=10)
        
        for class_id, class_name in sorted(self.detector.class_names.items()):
            var = tk.BooleanVar(value=class_id in lane.allowed_classes)
            cb = ttk.Checkbutton(cb_frame, text=f"{class_name} (ID:{class_id})", variable=var)
            cb.pack(anchor=tk.W, pady=3)
            class_vars[class_id] = var
        
        def on_save():
            allowed_classes = [cid for cid, var in class_vars.items() if var.get()]
            
            if not allowed_classes:
                messagebox.showerror("Error", "Please select at least one vehicle type!")
                return
            
            try:
                lane.allowed_classes = allowed_classes
                self.violation_detector.add_lane(lane)
                edit_window.destroy()
                messagebox.showinfo("Success", f"Lane {lane_id} updated successfully!")
                self.update_lane_info_display()
            except Exception as e:
                messagebox.showerror("Error", f"Failed to update lane: {str(e)}")
        
        btn_frame = ttk.Frame(edit_window)
        btn_frame.pack(fill=tk.X, padx=15, pady=15)
        ttk.Button(btn_frame, text="✓ Save Changes", command=on_save).pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        ttk.Button(btn_frame, text="✗ Cancel", command=edit_window.destroy).pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
    
    def delete_traffic_lane(self):
        """Delete a traffic lane"""
        if len(self.traffic_lanes) == 0:
            messagebox.showerror("Error", "No lanes to delete!")
            return
        
        select_window = tk.Toplevel(self.root)
        select_window.title("Select Lane to Delete")
        select_window.geometry("300x250")
        select_window.resizable(False, False)
        
        ttk.Label(select_window, text="Select a lane to delete:", 
                 font=("Arial", 10, "bold")).pack(pady=10)
        
        listbox = tk.Listbox(select_window, height=8, font=("Arial", 9))
        listbox.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        for lane_id, lane in self.traffic_lanes.items():
            class_names_str = ", ".join([self.detector.class_names.get(cid, f"ID{cid}") 
                                        for cid in lane.allowed_classes])
            listbox.insert(tk.END, f"Lane {lane_id}: {class_names_str}")
        
        def on_delete():
            try:
                idx = listbox.curselection()[0]
                selected_lane_id = list(self.traffic_lanes.keys())[idx]
                
                if messagebox.askyesno("Confirm", f"Delete Lane {selected_lane_id}?"):
                    del self.traffic_lanes[selected_lane_id]
                    if selected_lane_id in self.violation_detector.lanes:
                        del self.violation_detector.lanes[selected_lane_id]
                    if len(self.violation_detector.lanes) == 0:
                        self.violation_detector.persistent_violations.clear()
                    select_window.destroy()
                    messagebox.showinfo("Success", f"Lane {selected_lane_id} deleted!")
                    self.update_lane_info_display()
            except IndexError:
                messagebox.showerror("Error", "Please select a lane!")
        
        ttk.Button(select_window, text="Delete Selected", command=on_delete).pack(pady=10)
    
    def _display_frame_on_canvas(self, frame):
        """Helper method to display BGR frame on canvas"""
        try:
            canvas_width = self.canvas.winfo_width()
            canvas_height = self.canvas.winfo_height()
            
            if canvas_width > 1 and canvas_height > 1:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                h, w = frame_rgb.shape[:2]
                scale = min(canvas_width / w, canvas_height / h)
                new_w, new_h = int(w * scale), int(h * scale)
                frame_resized = cv2.resize(frame_rgb, (new_w, new_h))
                
                img = Image.fromarray(frame_resized)
                photo = ImageTk.PhotoImage(image=img)
                
                self.canvas.delete("all")
                self.canvas.create_image(canvas_width // 2, canvas_height // 2,
                                       image=photo, anchor=tk.CENTER)
                self.canvas.image = photo
        except Exception as e:
            print(f"Error displaying frame: {e}")
    
    def update_lane_info_display(self):
        """Update lane info display in settings - show lanes and zones separately"""
        lane_count = len(self.traffic_lanes)
        no_parking_count = len(self.no_parking_zones)
        
        # Update lane count label
        if lane_count == 0:
            self.lane_list_label.config(text="0 lanes configured", foreground="orange")
            self.lane_detail_label.config(text="")
        else:
            self.lane_list_label.config(text=f"{lane_count} lane(s) configured", foreground="green")
            lanes_detail = ""
            for lane_id, lane in sorted(self.traffic_lanes.items()):
                vehicle_types = ", ".join([self.detector.class_names.get(cid, f"ID{cid}") 
                                          if self.detector else f"ID{cid}"
                                          for cid in lane.allowed_classes])
                lanes_detail += f"  Lane {lane_id}: {vehicle_types}\n"
            self.lane_detail_label.config(text=lanes_detail.rstrip())
        
        # Update no-parking zone count label
        if no_parking_count == 0:
            self.zone_list_label.config(text="0 zones configured", foreground="orange")
            self.zone_detail_label.config(text="")
        else:
            self.zone_list_label.config(text=f"✓ {no_parking_count} zone(s) configured", foreground="green")
            zones_detail = ""
            for zone_id, zone in sorted(self.no_parking_zones.items()):
                zones_detail += f"  Zone {zone_id}: {zone.parking_frame_threshold} frames\n"
            self.zone_detail_label.config(text=zones_detail.rstrip())
    
    def start_processing(self):
        """Start video processing in separate thread"""
        if not self.video_path:
            messagebox.showerror("Error", "Please select a video file first!")
            return
        
        if not Path(self.video_path).exists():
            messagebox.showerror("Error", f"Video file not found: {self.video_path}")
            return
        
        if not Path(self.model_path).exists():
            messagebox.showerror("Error", 
                f"YOLO model not found!\n\nExpected path:\n{self.model_path}")
            return
        
        self.start_button.config(state=tk.DISABLED)
        self.stop_button.config(state=tk.NORMAL)
        self.is_processing = True
        self.should_stop = False
        
        STrack.track_id_count = 0
        
        thread = threading.Thread(target=self.process_video, daemon=True)
        thread.start()
        
        self.update_display()
    
    def stop_processing(self):
        """Stop video processing"""
        self.should_stop = True
    
    def process_video(self):
            """Process video with ByteTrack (runs in separate thread)"""
            try:
                print("="*60)
                print("Initializing YOLO11 Traffic detector...")
                self.detector = YOLODetector(
                    model_path=self.model_path,
                    conf_threshold=self.det_conf_var.get(),
                    device=self.device_var.get(),
                    min_box_area=self.min_box_area_var.get(),
                    edge_margin=self.edge_margin_var.get()
                )
                
                print("Initializing ByteTrack...")
                self.tracker = BYTETracker(
                    track_buffer=self.track_buffer_var.get(),
                    min_hits=3
                )
                print("ByteTrack initialized")
                
                print(f"Opening video: {Path(self.video_path).name}")
                self.cap = cv2.VideoCapture(self.video_path)
                
                if not self.cap.isOpened():
                    print("ERROR: Cannot open video file!")
                    return
                
                fps = int(self.cap.get(cv2.CAP_PROP_FPS))
                width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
                
                self.video_fps = fps if fps > 0 else 30
                self.actual_fps = self.video_fps
                print(f"Video info: {width}x{height} @ {fps} FPS, {total_frames} frames")

                target_width = 1280
                target_height = 720
                if width > target_width or height > target_height:
                    scale = min(target_width / width, target_height / height)
                    proc_width = int(width * scale)
                    proc_height = int(height * scale)
                    print(f"Resizing frames for detector to: {proc_width}x{proc_height}")
                else:
                    proc_width = width
                    proc_height = height
                    print("No downscale needed; using original frame size for detection")
                
                writer = None
                if self.save_output_var.get() and self.output_path:
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    writer = cv2.VideoWriter(self.output_path, fourcc, fps, (width, height))
                    print(f"Output will be saved to: {Path(self.output_path).name}")
                
                self.violation_detector.persistent_violations.clear()
                self.violation_detector.no_parking_state.clear()
                self.track_logger.reset()
                
                print("="*60)
                print("Processing started...")
                print(f"{'Frame':<8} {'detect':>8} {'track':>8} {'violation':>10} {'draw':>8} {'puttext':>9} {'queue':>7} {'total':>8} {'FPS':>7}")
                print("-"*80)
                
                frame_id = 0
                fps_list = []
                
                while not self.should_stop:
                    ret, frame = self.cap.read()
                    if not ret:
                        break
                    
                    frame_id += 1
                    t0 = time.time()
                    
                    # ── 1. Detect ──────────────────────────────────────────────
                    frame_proc = frame
                    if proc_width != width or proc_height != height:
                        frame_proc = cv2.resize(frame, (proc_width, proc_height))

                    detections = self.detector.detect(frame_proc)
                    if detections.size > 0 and (proc_width != width or proc_height != height):
                        scale_x = width / proc_width
                        scale_y = height / proc_height
                        detections[:, [0, 2]] *= scale_x
                        detections[:, [1, 3]] *= scale_y

                    detections = self.filter_person_riding_vehicle(
                        detections,
                        iou_threshold=self.person_vehicle_iou_threshold
                    )
                    t1 = time.time()

                    # ── 2. Track ───────────────────────────────────────────────
                    img_shape = (height, width)
                    online_tracks = self.tracker.update(detections, img_shape)
                    
                    for track in online_tracks:
                        if not hasattr(track, 'violation_type'):
                            track.violation_type = ViolationType.NONE
                    t2 = time.time()

                    # ── 3. Violation ───────────────────────────────────────────
                    violations_enabled = self.enable_violation_var.get()
                    has_zones_lanes = (len(self.traffic_lanes) > 0 or len(self.no_parking_zones) > 0)
                    
                    if violations_enabled and has_zones_lanes:
                        violations = self.violation_detector.detect_violations(online_tracks, frame_id)
                        for track in online_tracks:
                            if track.track_id in violations:
                                violation_types_list = violations[track.track_id]
                                class_name = self.detector.get_class_name(track.class_id) if track.class_id is not None else "unknown"
                                if class_name.lower() == "person":
                                    violation_types_list = [v for v in violation_types_list if v != ViolationType.NO_PARKING]
                                track.violation_types = violation_types_list
                                track.violation_type = violation_types_list[0] if violation_types_list else ViolationType.NONE
                            else:
                                track.violation_types = []
                                track.violation_type = ViolationType.NONE
                    else:
                        for track in online_tracks:
                            track.violation_types = []
                            track.violation_type = ViolationType.NONE
                    t3 = time.time()

                    self.track_logger.update_frame(
                        online_tracks,
                        frame_id,
                        self.video_fps,
                        self.detector,
                    )

                    # ── 4. Draw tracks ─────────────────────────────────────────
                    frame_vis = self.draw_tracks(frame, online_tracks, detections)
                    t4 = time.time()

                    # ── 5. Put text info ───────────────────────────────────────
                    elapsed_so_far = t4 - t0
                    fps_val_so_far = 1.0 / elapsed_so_far if elapsed_so_far > 0 else 0
                    info_text = [
                        f"Frame: {frame_id}/{total_frames}",
                        f"Detections: {len(detections)}",
                        f"Tracks: {len(online_tracks)}",
                        f"FPS: {fps_val_so_far:.1f}"
                    ]
                    y_offset = 30
                    for text in info_text:
                        cv2.putText(frame_vis, text, (10, y_offset),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                        y_offset += 30
                    t5 = time.time()

                    # ── 6. Queue + UI update ───────────────────────────────────
                    try:
                        self.frame_queue.get_nowait()
                    except queue.Empty:
                        pass
                    self.frame_queue.put_nowait(frame_vis)

                    if frame_id % 5 == 0:
                        progress = (frame_id / total_frames) * 100
                        self.progress_var.set(progress)
                        self.status_label.config(
                            text=f"Processing: Frame {frame_id}/{total_frames} ({progress:.1f}%) - FPS: {fps_val_so_far:.1f}")
                    t6 = time.time()

                    # ── FPS tracking ───────────────────────────────────────────
                    elapsed = t6 - t0
                    fps_val = 1.0 / elapsed if elapsed > 0 else 0
                    fps_list.append(fps_val)
                    n = min(len(fps_list), 10)
                    self.actual_fps = sum(fps_list[-n:]) / n

                    # ── Log mỗi 30 frame ───────────────────────────────────────
                    if frame_id % 30 == 0 or frame_id == 1:
                        detect_ms    = (t1 - t0) * 1000
                        track_ms     = (t2 - t1) * 1000
                        violation_ms = (t3 - t2) * 1000
                        draw_ms      = (t4 - t3) * 1000
                        puttext_ms   = (t5 - t4) * 1000
                        queue_ms     = (t6 - t5) * 1000
                        total_ms     = elapsed * 1000
                        print(f"{frame_id:<8} {detect_ms:>7.1f}ms {track_ms:>7.1f}ms {violation_ms:>9.1f}ms {draw_ms:>7.1f}ms {puttext_ms:>8.1f}ms {queue_ms:>6.1f}ms {total_ms:>7.1f}ms {fps_val:>6.1f}")

                    if writer:
                        writer.write(frame_vis)
                
                self.cap.release()
                if writer:
                    writer.release()

                log_path = self.track_logger.save(self.output_path if self.output_path else self.video_path)
                if log_path is not None:
                    print(f"Track log saved to: {log_path}")
                
                print("\n" + "="*60)
                if self.should_stop:
                    print(" Processing stopped by user")
                else:
                    print(" Processing completed!")
                print(f"  Total frames processed: {frame_id}")
                if len(fps_list) > 0:
                    print(f"  Average FPS: {np.mean(fps_list):.2f}")
                print(f"  Total tracks created: {STrack.track_id_count}")
                print("="*60)
                
                self.status_label.config(text=" Processing completed!")
                self.progress_var.set(100)
                
                if not self.should_stop:
                    messagebox.showinfo("Success", 
                                    f"Video processing completed!\n\n"
                                    f"Frames processed: {frame_id}\n"
                                    f"Average FPS: {np.mean(fps_list):.1f}\n"
                                    f"Total tracks: {STrack.track_id_count}")
                
            except Exception as e:
                print(f" ERROR: {str(e)}")
                import traceback
                print(traceback.format_exc())
                messagebox.showerror("Error", f"An error occurred:\n{str(e)}")
                
            finally:
                self.is_processing = False
                self.start_button.config(state=tk.NORMAL)
                self.stop_button.config(state=tk.DISABLED)
    
    def draw_tracks(self, frame, tracks, detections):
        """Draw tracking results on frame"""
        for track in tracks:
            if not track.is_activated:
                continue
            
            tlbr = track.tlbr
            x1, y1, x2, y2 = map(int, tlbr)
            track_id = track.track_id
            
            # Check if track has any violations
            violation_types_list = getattr(track, 'violation_types', [])
            has_violation = len(violation_types_list) > 0
            
            if has_violation:
                color = (0, 0, 255)  # Red for violations
                line_thickness = 3
            else:
                color = (0, 255, 0)  # Green for normal
                line_thickness = 2
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, line_thickness)
            
            class_name = "unknown"
            if track.class_id is not None:
                class_name = self.detector.get_class_name(track.class_id)
            
            label = f"ID:{track_id} {class_name}"
            
            # Append all violations to label
            if has_violation:
                violation_names = {
                    ViolationType.WRONG_VEHICLE_TYPE: "WRONG TYPE",
                    ViolationType.NO_PARKING: "NO PARKING",
                }
                violation_strs = [violation_names.get(v, 'VIOLATION') for v in violation_types_list]
                label += f" | {' + '.join(violation_strs)}"
            
            (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(frame, (x1, y1 - label_h - 10), (x1 + label_w + 10, y1), color, -1)
            cv2.putText(frame, label, (x1 + 5, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        if len(self.traffic_lanes) > 0:
            for lane_id, lane in self.traffic_lanes.items():
                polygon = np.array(lane.polygon, dtype=np.int32)
                overlay = frame.copy()
                cv2.fillPoly(overlay, [polygon], (0, 255, 0))
                cv2.addWeighted(overlay, 0.1, frame, 0.9, 0, frame)
                cv2.polylines(frame, [polygon], True, (0, 255, 0), 2)
                cv2.putText(frame, f"Lane {lane_id}", (polygon[0][0], polygon[0][1] - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        if len(self.no_parking_zones) > 0:
            for zone_id, zone in self.no_parking_zones.items():
                polygon = np.array(zone.polygon, dtype=np.int32)
                overlay = frame.copy()
                cv2.fillPoly(overlay, [polygon], (0, 165, 255))
                cv2.addWeighted(overlay, 0.12, frame, 0.88, 0, frame)
                cv2.polylines(frame, [polygon], True, (0, 165, 255), 2)
                cv2.putText(frame, f"NoPark {zone_id}", (polygon[0][0], polygon[0][1] - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
        
        return frame
    
    def update_display(self):
        """Update canvas with latest frame"""
        if self.is_processing:
            try:
                frame = self.frame_queue.get_nowait()
                
                canvas_width = self.canvas.winfo_width()
                canvas_height = self.canvas.winfo_height()
                
                if canvas_width > 1 and canvas_height > 1:
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    
                    h, w = frame_rgb.shape[:2]
                    scale = min(canvas_width / w, canvas_height / h)
                    new_w, new_h = int(w * scale), int(h * scale)
                    frame_resized = cv2.resize(frame_rgb, (new_w, new_h))
                    
                    img = Image.fromarray(frame_resized)
                    photo = ImageTk.PhotoImage(image=img)
                    
                    self.canvas.delete("all")
                    self.canvas.create_image(canvas_width // 2, canvas_height // 2,
                                           image=photo, anchor=tk.CENTER)
                    self.canvas.image = photo
                    
            except queue.Empty:
                pass
        
        # display_fps = min(video_fps, actual_fps)
        # - video_fps: không hiển thị nhanh hơn video gốc
        # - actual_fps: không hiển thị nhanh hơn hệ thống xử lý được
        display_fps = min(self.video_fps, self.actual_fps)
        interval_ms = max(10, int(1000 / display_fps))
        self.root.after(interval_ms, self.update_display)


def main():
    """Launch GUI application"""
    root = tk.Tk()
    
    style = ttk.Style()
    try:
        style.theme_use('clam')
    except:
        pass
    
    app = ByteTrackGUI(root)
    
    print("=" * 60)
    print("ByteTrack GUI Started")
    print("=" * 60)
    
    root.mainloop()


if __name__ == '__main__':
    main()