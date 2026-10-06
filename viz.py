import numpy as np
import rerun as rr


class Viewer:
    """Live 3D view in rerun: current image, keyframe cameras, trajectory vs ground truth, map points."""

    def __init__(self, K, width, height, spawn=True):
        rr.init("monocular_slam", spawn=spawn)
        # world = first keyframe's camera frame: x right, y down, z forward
        rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Y_DOWN, static=True)
        rr.log("world/camera/image", rr.Pinhole(image_from_camera=K, resolution=[width, height],
                                                image_plane_distance=0.1), static=True)
        self.K = K
        self.est_xyz = []
        self.gt_xyz = []

    def log_image(self, frame_idx, image):
        rr.set_time("frame", sequence=frame_idx)
        rr.log("world/camera/image", rr.Image(image[..., ::-1]).compress(jpeg_quality=75))

    def log_keyframe(self, frame, image, pts3d, gt_xyz=None):
        """Call after log_image for the same frame, so it lands on the same timestamp."""
        rr.log("world/camera", rr.Transform3D(translation=frame.pose[:3, 3], mat3x3=frame.pose[:3, :3]))

        self.est_xyz.append(frame.pose[:3, 3])
        rr.log("world/trajectory/estimate", rr.LineStrips3D([self.est_xyz], colors=[255, 140, 0]))
        if gt_xyz is not None:
            self.gt_xyz.append(gt_xyz)
            rr.log("world/trajectory/ground_truth", rr.LineStrips3D([self.gt_xyz], colors=[30, 120, 255]))

        if pts3d is not None and len(pts3d):
            self._log_points(frame, image, pts3d)

    def _log_points(self, frame, image, pts3d):
        # colour each point by projecting it back into the keyframe that triangulated it
        T_cw = np.linalg.inv(frame.pose)
        pts_cam = (T_cw[:3, :3] @ pts3d.T).T + T_cw[:3, 3]
        depth = pts_cam[:, 2]
        # near-parallel rays triangulate far away; hide them so the cloud stays readable
        keep = depth < 3 * np.median(depth)
        pts3d, pts_cam = pts3d[keep], pts_cam[keep]

        uv = (self.K @ pts_cam.T).T
        uv = np.round(uv[:, :2] / uv[:, 2:]).astype(int)
        uv[:, 0] = np.clip(uv[:, 0], 0, image.shape[1] - 1)
        uv[:, 1] = np.clip(uv[:, 1], 0, image.shape[0] - 1)
        colors = image[uv[:, 1], uv[:, 0], ::-1]

        # one entity per keyframe, so earlier points stay visible
        rr.log(f"world/points/kf_{frame.id}", rr.Points3D(pts3d, colors=colors, radii=0.004))
