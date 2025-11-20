'''
Demo script that loads M2T2 outputs and uses GraspMolmo to select a grasp.
'''
import numpy as np
from PIL import Image, ImageDraw
import os
import argparse

from graspmolmo.inference.grasp_predictor import GraspMolmo
from graspmolmo.inference.utils import draw_grasp, draw_grasp_points

def main():
    parser = argparse.ArgumentParser(description="Select a grasp using GraspMolmo from M2T2 outputs.")
    parser.add_argument('--input_dir', type=str, default="../M2T2/examples/M2T2_grasp_outputs",
                        help='Directory containing the M2T2 grasp outputs.')
    parser.add_argument('--task_prompt', type=str, default="Carry the guitar horizontally from the table to another location.",
                        help='The task prompt for GraspMolmo.')
    parser.add_argument('--verbosity', type=int, default=3,
                        help='Verbosity level for GraspMolmo output. 1 shows only the final grasp, 3 shows Molmo point and all candidate grasps.')
    args = parser.parse_args()

    input_dir = args.input_dir
    task = args.task_prompt
    verbosity = args.verbosity
    
    # Load data from m2t2_predictor.py
    try:
        rgb_image = Image.open(os.path.join(input_dir, "rgb.png"))
        point_cloud = np.load(os.path.join(input_dir, "point_cloud.npy"))
        grasps = np.load(os.path.join(input_dir, "grasps.npy"))
        print(f'len(grasps) = {grasps.shape[0]}')
        cam_K = np.load(os.path.join(input_dir, "camera_intrinsics.npy"))
        cam_pose = np.load(os.path.join(input_dir, "camera_pose.npy"))        
    except FileNotFoundError as e:
        print(f"Error loading data: {e}")
        print(f"Please run m2t2_predictor.py first to generate the necessary files in the '{input_dir}' directory.")
        return

    # GraspMolmo prediction
    gm = GraspMolmo()

    print("Running GraspMolmo to select the best grasp...")
    selected_grasp_idx = gm.pred_grasp(rgb_image, point_cloud, task, grasps, cam_K, verbosity=verbosity)

    if selected_grasp_idx is not None:
        print(f"GraspMolmo selected grasp index: {selected_grasp_idx}")
        selected_grasp = grasps[selected_grasp_idx]
        
        # Visualization
        print("Visualizing the selected grasp on the 2D image...")
        
        # If verbosity is 3 or higher, draw all candidate grasps in red
        if verbosity >= 3:
            for grasp in grasps:
                draw_grasp(rgb_image, cam_K, grasp, color="red")
            #draw_grasp_points(rgb_image, cam_K, point_cloud, grasps, r=3, color="red")

        # Draw the selected grasp on the image in blue
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
