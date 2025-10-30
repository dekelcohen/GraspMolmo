from prism_dataset import build_pointing_dataset
from graspmolmo.inference.grasp_predictor import GraspMolmo

if __name__ == "__main__":
	ds = build_pointing_dataset("test")
    
    task = "Carry the guitar horizontally from the table to another location."
    rgb, depth = get_image()
    camera_intrinsics = np.array(...)

    point_cloud = backproject(rgb, depth, camera_intrinsics)
    # grasps are in the camera reference frame
    grasps = predict_grasps(point_cloud)  # Using your favorite grasp predictor (e.g. M2T2)

    gm = GraspMolmo()
    idx = gm.pred_grasp(rgb, point_cloud, task, grasps)

    print(f"Predicted grasp: {grasps[idx]}")
