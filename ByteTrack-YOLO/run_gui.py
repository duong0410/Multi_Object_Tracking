"""
ByteTrack-YOLO GUI Launcher
Launch the graphical user interface for video tracking
"""

import sys
from pathlib import Path

from app_gui import main as app_main


def main():
    """Main entry point for GUI"""
    print("=" * 60)
    print("ByteTrack-YOLO: Multi-Object Tracking System")
    print("GUI Mode")
    print("=" * 60)
    print()
    print("Launching GUI using app_gui.py...")
    app_main()


if __name__ == '__main__':
    main()
