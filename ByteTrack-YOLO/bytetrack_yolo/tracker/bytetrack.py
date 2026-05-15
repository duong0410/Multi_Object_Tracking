"""
ByteTrack: Multi-Object Tracking Algorithm
Implementation of ByteTrack with class-persistent tracking and violation detection
"""

import numpy as np

from ..utils.kalman_filter import KalmanFilter
from ..utils.bbox import tlbr_to_tlwh, tlwh_to_xyah
from ..utils.matching import (
    iou_distance, linear_assignment, joint_stracks, 
    sub_stracks, remove_duplicate_stracks
)
from .violation_detection import ViolationType


class TrackState:
    """Enumeration for track states"""
    New = 0
    Tracked = 1
    Lost = 2
    Removed = 3


class STrack:
    """
    Single target track with Kalman filter state and class persistence
    """
    
    shared_kalman = None
    track_id_count = 0
    
    def __init__(self, tlwh, score, class_id=None):
        """
        Initialize track
        
        Args:
            tlwh: Bounding box [top, left, width, height]
            score: Detection confidence score
            class_id: Object class ID (persisted throughout tracking)
        """
        self._tlwh = np.asarray(tlwh, dtype=np.float32)
        self.kalman_filter = None
        self.mean, self.covariance = None, None
        self.is_activated = False
        
        self.score = score
        self.tracklet_len = 0
        self.class_id = class_id  # Class ID is set once and never changed
        
        # Will be set in activate()
        self.track_id = 0
        self.frame_id = 0
        self.start_frame = 0
        
        # Traffic lane and violation tracking
        self.lane_id = None
        self.violation_type = ViolationType.NONE
        
        self.state = TrackState.New
        
    def predict(self):
        """Predict next state using Kalman filter"""
        mean_state = self.mean.copy()
        if self.state != TrackState.Tracked:
            mean_state[7] = 0
        self.mean, self.covariance = self.kalman_filter.predict(
            mean_state, self.covariance
        )
        
    @staticmethod
    def multi_predict(stracks):
        """Predict multiple tracks (vectorized)"""
        if len(stracks) > 0:
            multi_mean = np.asarray([st.mean.copy() for st in stracks])
            multi_covariance = np.asarray([st.covariance for st in stracks])
            
            for i, st in enumerate(stracks):
                if st.state != TrackState.Tracked:
                    multi_mean[i][7] = 0
                    
            multi_mean, multi_covariance = STrack.shared_kalman.multi_predict(
                multi_mean, multi_covariance
            )
            
            for i, (mean, cov) in enumerate(zip(multi_mean, multi_covariance)):
                stracks[i].mean = mean
                stracks[i].covariance = cov
                
    def activate(self, kalman_filter, frame_id):
        """
        Start a new tracklet
        
        Args:
            kalman_filter: Kalman filter instance
            frame_id: Current frame ID
        """
        self.kalman_filter = kalman_filter
        self.track_id = self.next_id()
        self.mean, self.covariance = self.kalman_filter.initiate(
            self.tlwh_to_xyah(self._tlwh)
        )
        
        self.tracklet_len = 0
        self.state = TrackState.Tracked
        
        # ByteTrack: activate on frame 1, otherwise need confirmation
        if frame_id == 1:
            self.is_activated = True
            
        self.frame_id = frame_id
        self.start_frame = frame_id
        
    def re_activate(self, new_track, frame_id, new_id=False):
        """
        Reactivate a lost track
        
        Args:
            new_track: New detection to reactivate with
            frame_id: Current frame ID
            new_id: Whether to assign a new track ID
        """
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, self.tlwh_to_xyah(new_track.tlwh)
        )
        
        self.tracklet_len = 0
        self.state = TrackState.Tracked
        self.is_activated = True
        self.frame_id = frame_id
        
        if new_id:
            self.track_id = self.next_id()
            
        self.score = new_track.score
        # IMPORTANT: Keep original class_id, don't change it
        
    def update(self, new_track, frame_id, min_hits=1):
        """
        Update a matched track
        
        Args:
            new_track: New detection to update with
            frame_id: Current frame ID
            min_hits: Minimum hits before track is activated
        """
        self.frame_id = frame_id
        self.tracklet_len += 1
        
        new_tlwh = new_track.tlwh
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, self.tlwh_to_xyah(new_tlwh)
        )
        
        # Activate track after minimum hits
        if self.tracklet_len >= min_hits:
            self.is_activated = True
            self.state = TrackState.Tracked
        
        self.score = new_track.score
        
        # IMPORTANT: Only set class_id on first update, then keep it forever
        if self.class_id is None:
            self.class_id = new_track.class_id
    
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
        """Convert tlwh to xyah format"""
        ret = np.asarray(tlwh).copy()
        ret[:2] += ret[2:] / 2
        ret[2] /= ret[3]
        return ret
    
    @property
    def end_frame(self):
        """Get the last frame when track was updated"""
        return self.frame_id
    
    @staticmethod
    def next_id():
        """Generate next track ID"""
        STrack.track_id_count += 1
        return STrack.track_id_count
    
    def __repr__(self):
        return f'Track_{self.track_id}_({self.start_frame}-{self.frame_id})'


class BYTETracker:
    """
    ByteTrack: Multi-Object Tracking with Two-Phase Association
    
    Key features:
    - Two-phase matching: high-confidence and low-confidence detections
    - Class persistence: Once assigned, class ID never changes
    - Kalman filter for motion prediction
    """
    
    def __init__(self, det_conf_high=0.5, det_conf_low=0.1, new_track_thresh=0.6,
                 match_thresh_high=0.8, match_thresh_low=0.5, track_buffer=30, 
                 min_hits=1):
        """
        Initialize ByteTrack tracker
        
        Args:
            det_conf_high: Confidence threshold for high-score detections
            det_conf_low: Confidence threshold for low-score detections
            new_track_thresh: Threshold for creating new tracks
            match_thresh_high: Cost threshold for first association
            match_thresh_low: Cost threshold for second association
            track_buffer: Number of frames to keep lost tracks
            min_hits: Minimum hits before track is confirmed
        """
        self.tracked_stracks = []
        self.lost_stracks = []
        self.removed_stracks = []
        
        self.frame_id = 0
        
        # ByteTrack parameters
        self.det_conf_high = det_conf_high
        self.det_conf_low = det_conf_low
        self.new_track_thresh = new_track_thresh
        self.match_thresh_high = match_thresh_high
        self.match_thresh_low = match_thresh_low
        self.max_time_lost = track_buffer
        self.min_hits = min_hits
        
        # Initialize Kalman filter
        self.kalman_filter = KalmanFilter()
        STrack.shared_kalman = self.kalman_filter
        
    def update(self, output_results, img_shape):
        """
        Update tracker with new detections
        
        ByteTrack Algorithm:
        1. Split detections into high and low confidence
        2. Predict all tracks with Kalman filter
        3. First association: Match tracks with high-confidence detections
        4. Second association: Match remaining tracks with low-confidence detections
        5. Create new tracks from unmatched high-confidence detections
        6. Remove lost tracks that exceeded timeout
        
        Args:
            output_results: Detection array [N, 6] with format [x1, y1, x2, y2, conf, class]
            img_shape: Tuple (height, width)
            
        Returns:
            List of active STrack objects
        """
        self.frame_id += 1
        activated_stracks = []
        refind_stracks = []
        lost_stracks = []
        removed_stracks = []
        
        # Parse detections
        if output_results.shape[1] == 5:
            # No class info
            scores = output_results[:, 4]
            bboxes = output_results[:, :4]
            classes = np.zeros(len(scores), dtype=np.int32)
        else:
            # With class info
            scores = output_results[:, 4]
            bboxes = output_results[:, :4]
            classes = output_results[:, 5].astype(np.int32)
        
        # Split detections by confidence
        remain_inds_high = scores > self.det_conf_high
        dets_high = bboxes[remain_inds_high]
        scores_high = scores[remain_inds_high]
        classes_high = classes[remain_inds_high]
        
        remain_inds_low = np.logical_and(
            scores > self.det_conf_low, 
            scores <= self.det_conf_high
        )
        dets_low = bboxes[remain_inds_low]
        scores_low = scores[remain_inds_low]
        classes_low = classes[remain_inds_low]
        
        # Create STrack objects
        if len(dets_high) > 0:
            detections_high = [
                STrack(tlbr_to_tlwh(tlbr), s, c) 
                for tlbr, s, c in zip(dets_high, scores_high, classes_high)
            ]
        else:
            detections_high = []
            
        if len(dets_low) > 0:
            detections_low = [
                STrack(tlbr_to_tlwh(tlbr), s, c) 
                for tlbr, s, c in zip(dets_low, scores_low, classes_low)
            ]
        else:
            detections_low = []
            
        # Separate confirmed and unconfirmed tracks
        unconfirmed = []
        tracked_stracks = []
        for track in self.tracked_stracks:
            if not track.is_activated:
                unconfirmed.append(track)
            else:
                tracked_stracks.append(track)
        
        # Combine tracked and lost tracks for matching
        strack_pool = joint_stracks(tracked_stracks, self.lost_stracks)
        
        # Predict all tracks
        STrack.multi_predict(strack_pool)
        
        # ===== First Association: Track-Detection (High Confidence) =====
        dists = iou_distance(strack_pool, detections_high)
        matches, u_track, u_detection_high = linear_assignment(
            dists, thresh=self.match_thresh_high
        )
        
        for itracked, idet in matches:
            track = strack_pool[itracked]
            det = detections_high[idet]
            if track.state == TrackState.Tracked:
                track.update(det, self.frame_id, self.min_hits)
                activated_stracks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)
        
        # ===== Second Association: Remaining Tracks - Low Confidence =====
        r_tracked_stracks = [
            strack_pool[i] for i in u_track 
            if strack_pool[i].state == TrackState.Tracked
        ]
        
        dists_low = iou_distance(r_tracked_stracks, detections_low)
        matches_low, u_track_remain, u_detection_low = linear_assignment(
            dists_low, thresh=self.match_thresh_low
        )
        
        for itracked, idet in matches_low:
            track = r_tracked_stracks[itracked]
            det = detections_low[idet]
            if track.state == TrackState.Tracked:
                track.update(det, self.frame_id, self.min_hits)
                activated_stracks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)
        
        # Mark unmatched tracked tracks as lost
        for it in u_track_remain:
            track = r_tracked_stracks[it]
            if track.state != TrackState.Lost:
                track.mark_lost()
                lost_stracks.append(track)
        
        # ===== Handle unconfirmed tracks =====
        detections = [detections_high[i] for i in u_detection_high]
        dists = iou_distance(unconfirmed, detections)
        matches, u_unconfirmed, u_detection = linear_assignment(dists, thresh=0.5)
        
        for itracked, idet in matches:
            unconfirmed[itracked].update(
                detections[idet], self.frame_id, self.min_hits
            )
            activated_stracks.append(unconfirmed[itracked])
        
        for it in u_unconfirmed:
            track = unconfirmed[it]
            track.mark_removed()
            removed_stracks.append(track)
        
        # ===== Initialize new tracks =====
        for inew in u_detection:
            track = detections[inew]
            if track.score < self.new_track_thresh:
                continue
            track.activate(self.kalman_filter, self.frame_id)
            activated_stracks.append(track)
            
        # ===== Remove lost tracks =====
        for track in self.lost_stracks:
            if self.frame_id - track.end_frame > self.max_time_lost:
                track.mark_removed()
                removed_stracks.append(track)
                
        # Update state
        self.tracked_stracks = [
            t for t in self.tracked_stracks if t.state == TrackState.Tracked
        ]
        self.tracked_stracks = joint_stracks(self.tracked_stracks, activated_stracks)
        self.tracked_stracks = joint_stracks(self.tracked_stracks, refind_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.tracked_stracks)
        self.lost_stracks.extend(lost_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.removed_stracks)
        self.removed_stracks.extend(removed_stracks)
        self.tracked_stracks, self.lost_stracks = remove_duplicate_stracks(
            self.tracked_stracks, self.lost_stracks
        )
        
        # Return active tracks
        output_stracks = [
            track for track in self.tracked_stracks if track.is_activated
        ]
        
        return output_stracks
    
    def reset(self):
        """Reset tracker state"""
        self.tracked_stracks = []
        self.lost_stracks = []
        self.removed_stracks = []
        self.frame_id = 0
        STrack.track_id_count = 0
