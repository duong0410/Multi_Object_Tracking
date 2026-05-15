#!/usr/bin/env python3
"""
ByteTrack-YOLO: Vehicle Tracking System with Traffic Violation Detection
Main entry point - launches GUI

Usage:
    python main.py

Features:
    - YOLO11 object detection
    - ByteTrack multi-object tracking
    - Traffic violation detection
    - Real-time video visualization
    - Interactive lane configuration (ROI drawing)
    - Video saving support
"""

if __name__ == '__main__':
    from app_gui import main
    main()

