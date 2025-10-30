import os

import datasets
import huggingface_hub as hf_hub
import h5py
from PIL import Image
import numpy as np

def point_to_xml(grasp_pt: np.ndarray):
    if grasp_pt.ndim == 2:
        assert grasp_pt.shape == (1, 2)
        grasp_pt = grasp_pt[0]
    assert grasp_pt.shape == (2,)
    point_desc = "Where to grasp the object"
    return f"<point x=\"{grasp_pt[0]*100:.1f}\" y=\"{grasp_pt[1]*100:.1f}\" alt=\"{point_desc}\">{point_desc}</point>"

def map_sample(file_loc_map: dict[str, str], ex: dict):
    h5_path = file_loc_map[ex["scene_path"]]
    with h5py.File(h5_path, "r") as f:
        img = Image.fromarray(f[ex["view_id"]]["rgb"][:])
        grasp_pt_px = f[ex["view_id"]][ex["obs_id"]]["grasp_point_px"][:]
        grasp_pt_px = grasp_pt_px / np.array([img.width, img.height])
        xyz = f[ex["view_id"]]['xyz'][:]
    task = ex["task"]
    prompt = f"Point to the grasp that would accomplish the following task: {task}"
    point_xml = point_to_xml(grasp_pt_px)
    response = f"In order to accomplish the task \"{task}\", the optimal grasp is described as follows: \"{ex['matching_grasp_desc']}\".\n\n{point_xml}"

    return dict(
        image=img,
        prompt=prompt,
        text=response,
        style="pointing",
        h5_path=h5_path,
        xyz=xyz # point cloud for stable grasp predictor --> to generate candidates 
    )

def build_pointing_dataset(split: str, num_proc: int = 10, max_rows: int = None) -> datasets.Dataset:
    """
    split: str - test, train 
    num_proc: int = 10 - parallel processing on 10 cores by default 
    max_rows: int = None - limit the dataset to 5 rows for examples inference --> loading dataset (map) is instant
    """
    hf_fs = hf_hub.HfFileSystem()
    chunks = hf_fs.ls(f"datasets/allenai/PRISM/PRISM-{split}", detail=False)
    urls = []
    for chunk in chunks:
        path = chunk[len("datasets/allenai/PRISM/"):]
        urls.append(hf_hub.hf_hub_url(repo_id="allenai/PRISM", filename=path, repo_type="dataset"))

    dl_manager = datasets.DownloadManager(dataset_name="allenai/PRISM", record_checksums=False)
    paths = dl_manager.download_and_extract(urls)
    print('prism dataset split was downloaded to these local paths', paths)

    file_loc_map = {}
    for path in paths:
        path = str(path)
        for file in os.listdir(path):
            file_loc_map[file] = os.path.join(path, file)

    metadata_ds = datasets.load_dataset("allenai/PRISM", split=split)
    if not max_rows is None:
        metadata_ds = metadata_ds.select(range(max_rows))
    dataset = metadata_ds.map(lambda ex: map_sample(file_loc_map, ex), num_proc=num_proc)
    return dataset

if __name__ == "__main__":
    #build_pointing_dataset("train")
    ds = build_pointing_dataset("test")
