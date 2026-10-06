import cv2
from map import Map
from frame import Frame
from utils import feature_matching
import numpy as np


class MonocularSLAM:
    def __init__(self, K, min_parallax=15.0, min_inliers=30):
        self.K = K
        self.slam_map = Map()
        self.frame_count = 0
        # median pixel flow needed before a frame becomes a keyframe; with too little
        # baseline the essential matrix is mostly noise
        self.min_parallax = min_parallax
        self.min_inliers = min_inliers

    def process_image(self, image, timestamp=None, scale=None):
        """
        Returns the new keyframe's triangulated points, or None if the frame was skipped.
        Monocular translation has unit length per keyframe step; `scale` (e.g. the
        ground-truth distance since the last keyframe) rescales it.
        """
        f_next = Frame(self.frame_count, image, self.K, timestamp)
        self.frame_count += 1
        if self.slam_map.get_length_frames() == 0:
            self.slam_map.add_frame(f_next)
            return

        # f_cur is the last keyframe, not the previous image
        f_cur = self.slam_map.get_current_frame()
        id_cur, id_next = feature_matching(f_cur, f_next)
        if len(id_cur) < self.min_inliers:
            return

        pts_cur, pts_next = f_cur.pts[id_cur], f_next.pts[id_next]
        if np.median(np.linalg.norm(pts_cur - pts_next, axis=1)) < self.min_parallax:
            return

        E, mask = cv2.findEssentialMat(pts_cur, pts_next, self.K, method=cv2.RANSAC, prob=0.999, threshold=1.0)
        if E is None:
            return
        # recoverPose keeps only RANSAC inliers that also lie in front of both cameras
        _, R, t, mask = cv2.recoverPose(E[:3], pts_cur, pts_next, self.K, mask=mask)
        inliers = mask.ravel() > 0
        if inliers.sum() < self.min_inliers:
            return

        # T_21 maps points from f_cur's camera into f_next's; poses are camera-to-world
        T_21 = np.eye(4)
        T_21[:3, :3] = R
        T_21[:3, 3] = t.ravel() * (scale if scale is not None else 1.0)
        f_next.pose = f_cur.pose @ np.linalg.inv(T_21)

        # P = K [R | t] with world-to-camera extrinsics; for the first pair this is K [I | 0] and K [R | t]
        P1 = self.K @ np.linalg.inv(f_cur.pose)[:3]
        P2 = self.K @ np.linalg.inv(f_next.pose)[:3]
        pts4d = cv2.triangulatePoints(P1, P2, pts_cur[inliers].T, pts_next[inliers].T)
        pts3d = (pts4d[:3] / pts4d[3]).T

        # keep points in front of both cameras
        pts3d_h = np.vstack([pts3d.T, np.ones(len(pts3d))])
        depth_cur = (np.linalg.inv(f_cur.pose) @ pts3d_h)[2]
        depth_next = (np.linalg.inv(f_next.pose) @ pts3d_h)[2]
        good = (depth_cur > 0) & (depth_next > 0)

        self.slam_map.add_frame(f_next)
        for p in pts3d[good]:
            self.slam_map.add_point(p)
        return pts3d[good]
        
        
        
                
        
