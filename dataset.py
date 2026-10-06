import os

import cv2
import numpy as np
from scipy.spatial.transform import Rotation


def read_tum_list(path):
    rows = []
    with open(path) as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            rows.append(line.split())
    return rows


class TUMDataset:
    """Yields (timestamp, bgr image, ground-truth camera-to-world 4x4 or None) for a TUM RGB-D sequence."""

    def __init__(self, root, max_dt=0.02):
        self.root = root
        self.rgb = [(float(ts), path) for ts, path in read_tum_list(os.path.join(root, "rgb.txt"))]

        gt = np.array(read_tum_list(os.path.join(root, "groundtruth.txt")), dtype=np.float64)
        self.gt_times = gt[:, 0]
        self.gt_data = gt[:, 1:]  # tx ty tz qx qy qz qw
        self.max_dt = max_dt

    def __len__(self):
        return len(self.rgb)

    def gt_pose(self, timestamp):
        # rgb and ground truth aren't synchronised, so take the nearest gt sample
        i = np.argmin(np.abs(self.gt_times - timestamp))
        if abs(self.gt_times[i] - timestamp) > self.max_dt:
            return None
        T = np.eye(4)
        T[:3, :3] = Rotation.from_quat(self.gt_data[i, 3:]).as_matrix()  # scipy uses x, y, z, w like TUM
        T[:3, 3] = self.gt_data[i, :3]
        return T

    def __iter__(self):
        for ts, path in self.rgb:
            yield ts, cv2.imread(os.path.join(self.root, path)), self.gt_pose(ts)
