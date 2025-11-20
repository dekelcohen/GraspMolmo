'''
Demo script that loads M2T2 outputs and uses GraspMolmo to select a grasp.
'''
import numpy as np
from PIL import Image
import os

from graspmolmo.inference.grasp_predictor import GraspMolmo
from m2t2.meshcat_utils import (
    create_visualizer,
    make_frame,
    visualize_grasp,
    visualize_pointcloud
)

def main():
    input_dir = "M2T2_grasp_outputs"
    
    # Load data from m2t2_predictor.py
    try:
        rgb_image = Image.open(os.path.join(input_dir, "rgb_image.png"))
        point_cloud = np.load(os.path.join(input_dir, "point_cloud.npy"))
        grasps = np.load(os.path.join(input_dir, "grasps.npy"))
        cam_K = np.load(os.path.join(input_dir, "camera_intrinsics.npy"))
        cam_pose = np.load(os.path.join(input_dir, "camera_pose.npy"))
    except FileNotFoundError as e:
        print(f"Error loading data: {e}")
        print(f"Please run m2t2_predictor.py first to generate the necessary files in the '{input_dir}' directory.")
        return

    # GraspMolmo prediction
    task = "Carry the guitar horizontally from the table to another location."
    gm = GraspMolmo()

    print("Running GraspMolmo to select the best grasp...")
    selected_grasp_idx = gm.pred_grasp(rgb_image, point_cloud, task, grasps, cam_K, verbosity=1)

    if selected_grasp_idx is not None:
        print(f"GraspMolmo selected grasp index: {selected_grasp_idx}")
        selected_grasp = grasps[selected_grasp_idx]
        
        # Visualization
        print("Visualizing the scene and the selected grasp...")
        vis = create_visualizer()
        
        # Convert PIL image to numpy array for visualization
        rgb_np = np.array(rgb_image)
        
        # The point cloud from M2T2 is already in the camera frame.
        # For visualization, we transform it to the world frame.
        xyz_world = point_cloud @ cam_pose[:3, :3].T + cam_pose[:3, 3]
        
        make_frame(vis, 'camera', T=cam_pose)
        visualize_pointcloud(vis, 'scene', xyz_world, rgb_np, size=0.005)

        # The grasps are in the camera frame. For visualization, transform to world frame.
        selected_grasp_world = cam_pose @ selected_grasp
        visualize_grasp(vis, "selected_grasp", selected_grasp_world, [0, 255, 0], linewidth=0.5)
        print("\nVisualization is ready. Check your meshcat viewer.")
        print(f"Predicted grasp:\n{selected_grasp}")

    else:
        print("GraspMolmo did not select a grasp.")


if __name__ == '__main__':
    main()
