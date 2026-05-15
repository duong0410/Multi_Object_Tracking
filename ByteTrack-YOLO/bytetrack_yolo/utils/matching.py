"""
Matching utilities for track-detection association
"""

import numpy as np
import lap

from .bbox import bbox_ious


def iou_distance(atracks, btracks):
    """
    Compute cost matrix based on IoU distance
    
    Args:
        atracks: List of tracks or numpy arrays
        btracks: List of tracks or numpy arrays
    
    Returns:
        cost_matrix: IoU distance matrix (1 - IoU)
    """
    if len(atracks) > 0 and isinstance(atracks[0], np.ndarray) or \
       len(btracks) > 0 and isinstance(btracks[0], np.ndarray):
        atlbrs = atracks
        btlbrs = btracks
    else:
        atlbrs = [track.tlbr for track in atracks]
        btlbrs = [track.tlbr for track in btracks]
        
    ious = np.zeros((len(atlbrs), len(btlbrs)), dtype=np.float32)
    if ious.size == 0:
        return ious
    
    ious = bbox_ious(
        np.ascontiguousarray(atlbrs, dtype=np.float32),
        np.ascontiguousarray(btlbrs, dtype=np.float32)
    )
    
    cost_matrix = 1 - ious
    return cost_matrix


def linear_assignment(cost_matrix, thresh):
    """
    Perform linear assignment using Hungarian algorithm
    
    Args:
        cost_matrix: Cost matrix where cost = 1 - IoU
        thresh: Cost threshold - matches accepted if cost < thresh
                (IoU > (1 - thresh))
    
    Returns:
        matches: Array [K, 2] of matched indices
        unmatched_a: Array of unmatched indices in A
        unmatched_b: Array of unmatched indices in B
    """
    if cost_matrix.size == 0:
        return (
            np.empty((0, 2), dtype=int), 
            tuple(range(cost_matrix.shape[0])), 
            tuple(range(cost_matrix.shape[1]))
        )
    
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
    """
    Join two lists of tracks, removing duplicates
    
    Args:
        tlista: First list of tracks
        tlistb: Second list of tracks
    
    Returns:
        Combined list without duplicates
    """
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
    """
    Remove tracks in tlistb from tlista
    
    Args:
        tlista: List of tracks to filter
        tlistb: List of tracks to remove
    
    Returns:
        Filtered list
    """
    stracks = {}
    for t in tlista:
        stracks[t.track_id] = t
    for t in tlistb:
        tid = t.track_id
        if stracks.get(tid, 0):
            del stracks[tid]
    return list(stracks.values())


def remove_duplicate_stracks(stracksa, stracksb):
    """
    Remove duplicate tracks based on IoU overlap
    
    Args:
        stracksa: First list of tracks
        stracksb: Second list of tracks
    
    Returns:
        resa: Filtered first list
        resb: Filtered second list
    """
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
    
    resa = [t for i, t in enumerate(stracksa) if i not in dupa]
    resb = [t for i, t in enumerate(stracksb) if i not in dupb]
    return resa, resb
