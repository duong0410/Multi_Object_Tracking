"""
Traffic violation detection module
Detects traffic violations based on lane rules and vehicle behavior
"""

import numpy as np
import cv2
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Tuple, Optional, Dict


class ViolationType(Enum):
    NONE = 0
    WRONG_LANE = 1
    WRONG_DIRECTION = 2
    WRONG_VEHICLE_TYPE = 3
    NO_PARKING = 4


@dataclass
class TrafficLane:
    lane_id: int
    name: str
    polygon: List[Tuple[int, int]]
    allowed_classes: List[int]
    direction_vector: Tuple[float, float] = (1.0, 0.0)

    def point_in_lane(self, x: int, y: int) -> bool:
        if len(self.polygon) < 3:
            return False
        point = np.array([x, y], dtype=np.float32)
        polygon = np.array(self.polygon, dtype=np.int32)
        return cv2.pointPolygonTest(polygon, tuple(point), False) >= 0

    def bbox_in_lane(self, tlbr: Tuple[int, int, int, int]) -> bool:
        x1, y1, x2, y2 = tlbr
        return self.point_in_lane((x1 + x2) // 2, (y1 + y2) // 2)


@dataclass
class NoParkingZone:
    zone_id: int
    name: str
    polygon: List[Tuple[int, int]]

    # --- Trigger: phát hiện đỗ ---
    parking_frame_threshold: int = 60
    # Số frame tối đa trong sliding window được phép "chập chờn" (miss/jitter)
    # mà vẫn tính là đứng yên. Nên đặt ~10-15% của parking_frame_threshold.
    allowed_jitter_frames: int = 8
    # pixel/frame: dưới ngưỡng này → frame đó tính là "đứng yên"
    movement_threshold: float = 4.0

    # --- Clear: xe di chuyển thật ---
    # Ngưỡng tối thiểu movement/frame để được tính vào accumulator khi đang flagged.
    # Mục đích: lọc bbox jitter (~1-2px) không bị nhầm là xe di chuyển.
    # Đặt thấp hơn movement_threshold để xe đi chậm vẫn tích lũy được,
    # nhưng đủ cao để loại nhiễu detector (thường 1-2px).
    min_clear_movement_per_frame: float = 2.5
    # Tổng displacement tích lũy (px) để clear vi phạm.
    clear_distance_threshold: float = 60.0

    # Chỉ áp dụng cho các class này. None = tất cả.
    # Mặc định bỏ person (class 5): [0=car, 1=truck, 2=bus, 3=motor, 4=bicycle]
    applicable_classes: Optional[List[int]] = field(default_factory=lambda: [0, 1, 2, 3, 4])

    def point_in_zone(self, x: int, y: int) -> bool:
        if len(self.polygon) < 3:
            return False
        point = np.array([x, y], dtype=np.float32)
        polygon = np.array(self.polygon, dtype=np.int32)
        return cv2.pointPolygonTest(polygon, tuple(point), False) >= 0

    def bbox_in_zone(self, tlbr: Tuple[int, int, int, int]) -> bool:
        x1, y1, x2, y2 = tlbr
        return self.point_in_zone((x1 + x2) // 2, (y1 + y2) // 2)


@dataclass
class TrackViolation:
    track_id: int
    violation_type: ViolationType
    lane_id: Optional[int] = None
    confidence: float = 1.0
    frame_detected: int = 0


class ViolationDetector:

    def __init__(self):
        self.lanes: Dict[int, TrafficLane] = {}
        self.no_parking_zones: Dict[int, NoParkingZone] = {}
        self.persistent_violations: Dict[int, ViolationType] = {}
        # State no-parking: (track_id, zone_id) -> dict
        self.no_parking_state: Dict[Tuple[int, int], Dict] = {}

    def add_lane(self, lane: TrafficLane):
        self.lanes[lane.lane_id] = lane

    def add_no_parking_zone(self, zone: NoParkingZone):
        self.no_parking_zones[zone.zone_id] = zone

    def _get_center(self, x1: int, y1: int, x2: int, y2: int) -> Tuple[int, int]:
        """
        Dùng bbox center thực tế để tính movement.
        KHÔNG dùng Kalman (track.mean) vì Kalman smooth quá →
        movement luôn nhỏ dù xe đang chạy → không clear được.
        """
        return (x1 + x2) // 2, (y1 + y2) // 2

    def _check_wrong_vehicle_type(
        self,
        tlbr: Tuple[int, int, int, int],
        class_id: Optional[int],
        track_id: int,
        violations_list: List[ViolationType],
    ):
        """Check xem vehicle có đi đúng làn không."""
        for lane in self.lanes.values():
            if lane.bbox_in_lane(tlbr):
                if class_id is not None and class_id not in lane.allowed_classes:
                    if ViolationType.WRONG_VEHICLE_TYPE not in violations_list:
                        violations_list.append(ViolationType.WRONG_VEHICLE_TYPE)
                        self.persistent_violations[track_id] = ViolationType.WRONG_VEHICLE_TYPE
                break  # mỗi xe chỉ thuộc 1 làn

    def _check_no_parking(
        self,
        tlbr: Tuple[int, int, int, int],
        cx: int,
        cy: int,
        track_id: int,
        class_id: Optional[int],
        violations_list: List[ViolationType],
    ):
        """
        Logic no-parking với sliding window chịu được bbox chập chờn:

        TRIGGER (phát hiện đỗ):
        ─────────────────────────────────────────────────────────────────
        - Dùng sliding window kích thước `parking_frame_threshold`.
        - Mỗi frame, ghi nhận movement so với frame liền trước có dữ liệu.
        - Trong window, đếm số frame "stationary" (movement <= movement_threshold).
        - Số frame "jitter" = window_size - stationary_count.
        - Nếu jitter <= allowed_jitter_frames → coi là đứng yên đủ lâu → flagged.
        - Lợi ích: bbox detector miss 1-2 frame không làm streak bị reset về 0.

        CLEAR (xe di chuyển đi):
        ─────────────────────────────────────────────────────────────────
        - Chỉ tích lũy displacement khi movement >= min_clear_movement_per_frame
          (mặc định 2.5px) → lọc jitter bbox (1-2px) không bị clear nhầm.
        - Ngưỡng này thấp hơn movement_threshold (4px) nên xe đi chậm thật
          (3-4px/frame) vẫn tích lũy được.
        - Tích lũy đủ `clear_distance_threshold` px → clear vi phạm.
        - Khi clear: reset toàn bộ state (window + accumulator) để có thể
          detect lại nếu xe đỗ trở lại.

        EXIT ZONE:
        ─────────────────────────────────────────────────────────────────
        - Xe rời zone → reset hoàn toàn, không còn vi phạm.
        """
        for zone_id, zone in self.no_parking_zones.items():
            key = (track_id, zone_id)

            # Bỏ qua nếu class không thuộc diện áp dụng (vd: person)
            if zone.applicable_classes is not None and class_id not in zone.applicable_classes:
                self.no_parking_state.pop(key, None)
                continue

            if not zone.bbox_in_zone(tlbr):
                # Xe rời zone → reset state hoàn toàn
                self.no_parking_state.pop(key, None)
                continue

            # Xe đang trong zone
            state = self.no_parking_state.get(key)

            if state is None:
                # Frame đầu tiên vào zone
                state = {
                    # Sliding window: mỗi phần tử là movement (px) của 1 frame
                    'movement_window': deque(maxlen=zone.parking_frame_threshold),
                    'last_center': (cx, cy),
                    'is_flagged': False,
                    'moving_accumulator': 0.0,
                }
                # Frame đầu không có prev → movement = 0 (coi là đứng yên)
                state['movement_window'].append(0.0)
                self.no_parking_state[key] = state
            else:
                prev_cx, prev_cy = state['last_center']

                # Displacement tuyệt đối so với frame trước có dữ liệu.
                # Bắt được tất cả hướng: tiến, lùi, trái, phải, chéo.
                movement = float(np.hypot(cx - prev_cx, cy - prev_cy))

                if state['is_flagged']:
                    # ── Đang vi phạm: tích lũy displacement để clear ──────────
                    # Chỉ cộng khi movement >= min_clear_movement_per_frame
                    # để lọc bbox jitter (1-2px) không bị clear nhầm.
                    # Ngưỡng này thấp hơn movement_threshold nên xe đi chậm
                    # vẫn tích lũy được, chỉ loại nhiễu thuần túy của detector.
                    if movement >= zone.min_clear_movement_per_frame:
                        state['moving_accumulator'] += movement

                    if state['moving_accumulator'] >= zone.clear_distance_threshold:
                        # Clear vi phạm, reset hoàn toàn để detect lại nếu xe đỗ tiếp
                        state['is_flagged'] = False
                        state['moving_accumulator'] = 0.0
                        state['movement_window'].clear()
                        state['movement_window'].append(0.0)

                else:
                    # ── Chưa vi phạm: cập nhật sliding window ────────────────
                    state['movement_window'].append(movement)

                    # Chỉ evaluate khi window đã đủ parking_frame_threshold frame
                    if len(state['movement_window']) >= zone.parking_frame_threshold:
                        stationary_count = sum(
                            1 for m in state['movement_window']
                            if m <= zone.movement_threshold
                        )
                        jitter_count = len(state['movement_window']) - stationary_count

                        if jitter_count <= zone.allowed_jitter_frames:
                            state['is_flagged'] = True
                            state['moving_accumulator'] = 0.0

                state['last_center'] = (cx, cy)

            # Ghi nhận vi phạm nếu đang flagged
            if state['is_flagged']:
                if ViolationType.NO_PARKING not in violations_list:
                    violations_list.append(ViolationType.NO_PARKING)

    def detect_violations(self, tracks: List, frame_id: int) -> Dict[int, List]:
        """
        Detect violations cho tất cả tracks.

        Returns:
            {track_id: [ViolationType, ...]}
            Một track có thể bị nhiều loại violation cùng lúc.
        """
        violations: Dict[int, List[ViolationType]] = {}
        active_track_ids: set = set()

        for track in tracks:
            if not track.is_activated:
                continue

            track_id = track.track_id
            active_track_ids.add(track_id)

            x1, y1, x2, y2 = map(int, track.tlbr)
            tlbr = (x1, y1, x2, y2)
            class_id = getattr(track, 'class_id', None)

            # Bbox center thực tế, không dùng Kalman
            cx, cy = self._get_center(x1, y1, x2, y2)

            violations_list: List[ViolationType] = []

            # Lấy WRONG_VEHICLE_TYPE đã persist từ frame trước
            if self.persistent_violations.get(track_id) == ViolationType.WRONG_VEHICLE_TYPE:
                violations_list.append(ViolationType.WRONG_VEHICLE_TYPE)

            # Check vi phạm làn đường
            self._check_wrong_vehicle_type(tlbr, class_id, track_id, violations_list)

            # Check đỗ sai chỗ
            self._check_no_parking(tlbr, cx, cy, track_id, class_id, violations_list)

            # Cập nhật violation lên track object
            if violations_list:
                violations[track_id] = violations_list
                track.violation_type = violations_list[0]  # primary violation
            else:
                track.violation_type = ViolationType.NONE

        # Cleanup: xóa state của track không còn active
        stale_tracks = set(self.persistent_violations) - active_track_ids
        for tid in stale_tracks:
            self.persistent_violations.pop(tid, None)

        stale_keys = [k for k in self.no_parking_state if k[0] not in active_track_ids]
        for key in stale_keys:
            del self.no_parking_state[key]

        return violations