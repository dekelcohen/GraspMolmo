'''
Demo script that loads M2T2 outputs and uses GraspMolmo to select a grasp.
'''
import numpy as np
from PIL import Image
import os
import argparse

from graspmolmo.inference.grasp_predictor import GraspMolmo
from graspmolmo.inference.utils import draw_grasp

def main():
    parser = argparse.ArgumentParser(description="Select a grasp using GraspMolmo from M2T2 outputs.")
    parser.add_argument('--input_dir', type=str, default="../M2T2/examples/M2T2_grasp_outputs",
                        help='Directory containing the M2T2 grasp outputs.')
    parser.add_argument('--task_prompt', type=str,
                        help='The task prompt for GraspMolmo.')
    args = parser.parse_args()

    input_dir = args.input_dir
    
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
    gm = GraspMolmo()

    print("Running GraspMolmo to select the best grasp...")
    selected_grasp_idx = gm.pred_grasp(rgb_image, point_cloud, args.task_prompt, grasps, cam_K, verbosity=1)

    if selected_grasp_idx is not None:
        print(f"GraspMolmo selected grasp index: {selected_grasp_idx}")
        selected_grasp = grasps[selected_grasp_idx]
        
        # Visualization
        print("Visualizing the selected grasp on the 2D image...")
        
        # Draw the selected grasp on the image
        # The grasp is in camera frame, which is what draw_grasp expects
        draw_grasp(rgb_image, cam_K, selected_grasp, color="blue")
        
        # Save the image with the grasp visualization
        output_image_path = os.path.join(input_dir, "rgb_image_with_grasp.png")
        rgb_image.save(output_image_path)
        
        print(f"\nSaved visualization to {os.path.abspath(output_image_path)}")
        print(f"Predicted grasp:\n{selected_grasp}")

    else:
        print("GraspMolmo did not select a grasp.")


if __name__ == '__main__':
    main()
