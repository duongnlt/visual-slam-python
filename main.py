import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation

from dataset import TUMDataset
from slam import MonocularSLAM

# calibrated intrinsics of the freiburg1 RGB camera
# (525 / 319.5 / 239.5 are the ROS defaults, fine for depth but too rough for monocular)
fx = 517.3  # focal length x
fy = 516.5  # focal length y
cx = 318.6  # optical center x
cy = 255.3  # optical center y
K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
# TODO: undistort points, fr1 distortion = [0.2624, -0.9531, -0.0054, 0.0026, 1.1633]


def save_tum(path, frames):
    """timestamp tx ty tz qx qy qz qw, same format as groundtruth.txt, so evo can read it."""
    with open(path, "w") as f:
        for fr in frames:
            t = fr.pose[:3, 3]
            q = Rotation.from_matrix(fr.pose[:3, :3]).as_quat()
            f.write(f"{fr.timestamp:.6f} {t[0]:.6f} {t[1]:.6f} {t[2]:.6f} {q[0]:.6f} {q[1]:.6f} {q[2]:.6f} {q[3]:.6f}\n")


def plot_trajectory(path, est_xyz, gt_xyz):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    # camera axes: x right, y down, z forward
    ax1.plot(gt_xyz[:, 0], gt_xyz[:, 2], label="ground truth")
    ax1.plot(est_xyz[:, 0], est_xyz[:, 2], label="estimate")
    ax1.set_title("top-down (x-z)")
    ax1.set_xlabel("x [m]")
    ax1.set_ylabel("z [m]")
    ax1.axis("equal")
    ax1.legend()

    ax2.plot(gt_xyz[:, 0], gt_xyz[:, 1], label="ground truth")
    ax2.plot(est_xyz[:, 0], est_xyz[:, 1], label="estimate")
    ax2.set_title("front (x-y)")
    ax2.set_xlabel("x [m]")
    ax2.set_ylabel("y [m]")
    ax2.axis("equal")
    ax2.invert_yaxis()

    fig.tight_layout()
    fig.savefig(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="./rgbd_dataset_freiburg1_xyz")
    parser.add_argument("--no-gt-scale", action="store_true",
                        help="don't borrow translation length from ground truth (every keyframe step = 1)")
    parser.add_argument("--plot", default="trajectory.png")
    parser.add_argument("--out", default="est_trajectory.txt")
    parser.add_argument("--viz", action="store_true", help="live 3D view in rerun")
    args = parser.parse_args()

    dataset = TUMDataset(args.data)
    slam = MonocularSLAM(K)
    viewer = None
    if args.viz:
        from viz import Viewer
        viewer = Viewer(K, 640, 480)

    gt_kf = []  # ground-truth pose for each keyframe
    for i, (ts, image, gt_pose) in enumerate(dataset):
        if gt_pose is None:
            continue

        scale = None
        if not args.no_gt_scale and gt_kf:
            scale = np.linalg.norm(gt_pose[:3, 3] - gt_kf[-1][:3, 3])

        if viewer:
            viewer.log_image(i, image)

        n_before = slam.slam_map.get_length_frames()
        pts3d = slam.process_image(image, ts, scale)
        if slam.slam_map.get_length_frames() > n_before:
            gt_kf.append(gt_pose)
            if viewer:
                gt_xyz = (np.linalg.inv(gt_kf[0]) @ gt_pose)[:3, 3]
                viewer.log_keyframe(slam.slam_map.get_current_frame(), image, pts3d, gt_xyz)

    frames = slam.slam_map.frames
    # express ground truth relative to the first keyframe so both start at the origin with the same axes
    gt_xyz = np.array([(np.linalg.inv(gt_kf[0]) @ T)[:3, 3] for T in gt_kf])
    est_xyz = np.array([f.pose[:3, 3] for f in frames])

    print(f"keyframes: {len(frames)} / {len(dataset)} frames, map points: {len(slam.slam_map.points)}")
    if not args.no_gt_scale:
        err = np.linalg.norm(est_xyz - gt_xyz, axis=1)
        print(f"position error (first-frame aligned): rmse {np.sqrt(np.mean(err ** 2)):.3f} m, final {err[-1]:.3f} m")

    plot_trajectory(args.plot, est_xyz, gt_xyz)
    save_tum(args.out, frames)
    print(f"saved {args.plot}, {args.out}")
    print(f"evaluate: evo_ape tum {args.data}/groundtruth.txt {args.out} -as -p")


if __name__ == "__main__":
    main()
