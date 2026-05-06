#  Multi-Object Tracking System

Hệ thống theo dõi đa đối tượng (Multi-Object Tracking) sử dụng YOLO detector và ByteTrack tracker cho bài toán theo dõi phương tiện giao thông và người đi bộ.

##  Mục lục

- [Tổng quan](#tổng-quan)
- [Tính năng](#tính-năng)
- [Cài đặt](#cài-đặt)
- [Sử dụng](#sử-dụng)
- [Cấu trúc dự án](#cấu-trúc-dự-án)
- [Kết quả](#kết-quả)
- [Công nghệ](#công-nghệ)

##  Tổng quan

Dự án này triển khai hệ thống tracking đa đối tượng cho hai bài toán chính:

1. **Traffic Tracking**: Theo dõi các phương tiện giao thông

### Trackers được triển khai:

- **ByteTrack** (custom implementation)

##  Tính năng

### GUI Application
- **GUI tracking video** với YOLO11 + ByteTrack
- ROI selection (chọn vùng quan tâm)
- Real-time visualization
- Track filtering và statistics

###  Video Tracking

- **Real-time tracking** với ByteTrack
- Metrics: MOTA, IDF1, MOTP, Precision, Recall, ID Switches, Fragmentations


###  Detection Models

- **YOLO11** (Ultralytics YOLO11)
- Custom trained on traffic datasets

##  Cài đặt

### Yêu cầu hệ thống

- Python 3.10+
- CUDA 11.8+ (nếu dùng GPU)
- 8GB+ RAM
- 4GB+ VRAM (GPU)


##  Sử dụng

```bash
python ByteTrack-YOLO/main.py
```

**Features:**
- Chọn video input
- Chọn vùng ROI (Region of Interest)
- Điều chỉnh tracking parameters
- Xem kết quả real-time
- Export tracking results

##  Cấu trúc dự án

```
KLTN/
├── 📄 bytetrack_test.py           # GUI tracking application
├── 📄 yolo11_bytetrack.py        # YOLO + ByteTrack integration
├── 📄 test_yolo11s.py            # YOLO detection testing
├── 📄 Train_Yolo.ipynb           # Training notebook
│
├── 📂 ByteTrack-YOLO/        # ByteTrack custom implementation
│     ├── 📂 Detector_train/             
│       ├── yolov11m.ipynb                   # Final code training
│       └── load-data-coco-ua-detrac.ipynb   # not important 
│     ├── main.py
│     ├── models
│        ├── traffic_yolo_v11m
│             ├── best.pt                # checkpoint yolov11m train on custom data
│     ├── run_gui.py
│     ├── model.py
│     ├── setup.py
│     ├── track_and_detect.py
│        └── src/
├── 📂 Dataset/                   # ⚠️ NOT INCLUDED IN GIT
│
├── 📄 .gitignore                 # Git ignore rules
├── 📄 README.md                  # This file

```

##  Kết quả

### ByteTrack MOT17 Tracking Results

 Evaluation results on MOT17 dataset train using detection checkpoint from origin paper

| Model | MOTA | IDF1 | MOTP | Precision | Recall | ID_Sw | FP | FN |
|-------|------|------|------|-----------|--------|-------|----|----|
| bytetrack_s_mot17 | 72.6% | 77.7% | 0.439 | 100.0% | 73.1% | 390 | 23 | 23791 |
| bytetrack_m_mot17 | 83.5% | 84.1% | 0.432 | 99.7% | 84.3% | 431 | 188 | 13006 |
| bytetrack_l_mot17 | 86.4% | 85.3% | 0.436 | 99.8% | 87.1% | 413 | 127 | 10633 |
| bytetrack_x_mot17 | 87.3% | 85.4% | 0.398 | 99.8% | 88.1% | 412 | 163 | 9887 |


##  Công nghệ

### Deep Learning Frameworks
- **PyTorch** 2.0+
- **Ultralytics YOLO** 11

### Tracking Algorithms
- **ByteTrack**: 2-phase matching strategy
- **Kalman Filter**: Motion prediction

### Evaluation Metrics
- **MOT Metrics** (via motmetrics)
  - MOTA (Multi-Object Tracking Accuracy)
  - IDF1 (ID F1 Score)
  - MOTP (Multi-Object Tracking Precision)
  - Precision, Recall
  - ID Switches, Fragmentations

##  Dataset Setup (Không bao gồm trong repo)

### MOT17 Dataset

MOT17 dataset được sử dụng cho đánh giá performance của ByteTrack tracker.

**Download**: [MOT Challenge Website](https://motchallenge.net/data/MOT17/)

**Dataset Structure**:
```
Dataset/MOT17/
├── train/
│   ├── MOT17-02-DPM/
│   ├── MOT17-02-FRCNN/
│   ├── MOT17-02-SDP/
│   └── ...
└── test/
    └── ...
```

**Download**: [MOT Challenge](https://motchallenge.net/data/MOT17/)

##  Acknowledgments

- **ByteTrack**: [arXiv:2110.06864](https://arxiv.org/abs/2110.06864)
- **YOLO**: [Ultralytics](https://github.com/ultralytics/ultralytics)
- **MOT Challenge**: [motchallenge.net](https://motchallenge.net/)

##  Contact

- Email: tranthaidaiduong0@gmail.com
- GitHub: [@duong0410](https://github.com/duong0410)

---

**⭐ If you find this project helpful, please give it a star!**

---
