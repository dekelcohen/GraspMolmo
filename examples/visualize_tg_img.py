"""
Example script to visualize the TaskGrasp-Image dataset.
"""

import argparse

import numpy as np
from PIL import Image
import trimesh
from acronym_tools import create_gripper_marker

from graspmolmo.eval.utils import TaskGraspScanLibrary, img_to_pc
from graspmolmo.inference.utils import draw_grasp


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("tg_dir", help="Path to the TaskGrasp-Image dataset")
    parser.add_argument("object_id", help="The object ID, i.e. 001_squeezer")
    parser.add_argument("scan_id", type=int, help="The scan ID. Not every scan is included in the dataset.")

    subparsers = parser.add_subparsers(dest="mode", required=True)
    viz_3d = subparsers.add_parser("3d", help="Visualize using a 3D point cloud")
    viz_3d.add_argument("--far-clip", type=float, default=1.5, help="Far clip distance for rendering point cloud")

    subparsers.add_parser("2d", help="Visualize by drawing grasps on a 2D image")
    return parser.parse_args()


def viz_3d(args, library: TaskGraspScanLibrary):
    data = library.get(args.object_id, args.scan_id)
    cam_K: np.ndarray = data["cam_params"]
    rgb: Image.Image = data["rgb"]
    rgb_arr: np.ndarray = np.array(rgb)
    depth: np.ndarray = data["depth"]
    depth_mask = (depth < args.far_clip)

    points = img_to_pc(rgb_arr, depth, cam_K, depth_mask)
    rgba = np.full((len(points), 4), 255, dtype=np.uint8)
    rgba[:, :3] = points[:, 3:]
    pc = trimesh.PointCloud(points[:, :3], rgba)

    scene = trimesh.Scene([pc])
    for grasp in data["registered_grasps"]:
        marker: trimesh.Trimesh = create_gripper_marker()
        marker.apply_transform(grasp)
        scene.add_geometry(marker)

    scene.show()


def viz_2d(args, library: TaskGraspScanLibrary):
    data = library.get(args.object_id, args.scan_id)
    cam_K: np.ndarray = data["cam_params"]
    rgb: Image.Image = data["rgb"]

    for grasp in data["registered_grasps"]:
        draw_grasp(rgb, cam_K, grasp)

    rgb.show()


def main():
    args = get_args()
    library = TaskGraspScanLibrary(args.tg_dir)

    if args.mode == "3d":
        viz_3d(args, library)
    elif args.mode == "2d":
        viz_2d(args, library)
    else:
        raise ValueError(f"Invalid mode: {args.mode}")


if __name__ == "__main__":
    main()
