import cv2
import numpy as np


class Frame:
    def __init__(self, frame_id, image, K, timestamp=None):
        self.K = K
        self.K_inv = np.linalg.inv(K)
        self.id = frame_id
        self.timestamp = timestamp

        self.pose = np.eye(4)
        self.h, self.w = image.shape[:2]

        self.kps, self.pts, self.descriptors = self._feature_extraction(image)


    def _feature_extraction(self, image):
        orb = cv2.ORB_create()
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        corners = cv2.goodFeaturesToTrack(gray, 1000, qualityLevel=0.01, minDistance=7)
        if corners is None:
            return [], np.empty((0, 2)), None

        kps = [cv2.KeyPoint(x=float(c[0][0]), y=float(c[0][1]), size=20) for c in corners]
        # compute may drop keypoints near the border, so build pts from what it returns
        kps, descriptors = orb.compute(gray, kps)
        pts = np.array([kp.pt for kp in kps], dtype=np.float64)
        return kps, pts, descriptors






