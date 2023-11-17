import copy
import json
import os
from typing import Dict, List, Optional, Tuple

import cv2
import mmcv
import numpy as np
import pyquaternion
from matplotlib import pyplot as plt

from ..bbox import LiDARInstance3DBoxes

__all__ = ["visualize_camera", "visualize_lidar", "visualize_map","export_results_aimotive","export_results_dict"]


OBJECT_PALETTE = {
    "car": (255, 158, 0),
    "truck": (255, 99, 71),
    "construction_vehicle": (233, 150, 70),
    "bus": (255, 69, 0),
    "trailer": (255, 140, 0),
    "barrier": (112, 128, 144),
    "motorcycle": (255, 61, 99),
    "bicycle": (220, 20, 60),
    "pedestrian": (0, 0, 230),
    "traffic_cone": (47, 79, 79),
}

MAP_PALETTE = {
    "drivable_area": (166, 206, 227),
    "road_segment": (31, 120, 180),
    "road_block": (178, 223, 138),
    "lane": (51, 160, 44),
    "ped_crossing": (251, 154, 153),
    "walkway": (227, 26, 28),
    "stop_line": (253, 191, 111),
    "carpark_area": (255, 127, 0),
    "road_divider": (202, 178, 214),
    "lane_divider": (106, 61, 154),
    "divider": (106, 61, 154),
}


def visualize_camera(
    fpath: str,
    image: np.ndarray,
    *,
    bboxes: Optional[LiDARInstance3DBoxes] = None,
    labels: Optional[np.ndarray] = None,
    bboxes_r: Optional[LiDARInstance3DBoxes] = None,
    labels_r: Optional[np.ndarray] = None,
    transform: Optional[np.ndarray] = None,
    classes: Optional[List[str]] = None,
    color: Optional[Tuple[int, int, int]] = None,
    thickness: float = 4,
) -> None:
    canvas = image.copy()
    canvas = cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR)


    if bboxes_r is not None and len(bboxes_r) > 0:
        corners = bboxes_r.corners
        num_bboxes = corners.shape[0]

        coords = np.concatenate(
            [corners.reshape(-1, 3), np.ones((num_bboxes * 8, 1))], axis=-1
        )
       
        transform = copy.deepcopy(transform).reshape(4, 4)
        coords = coords @ transform.T
        coords = coords.reshape(-1, 8, 4)

        indices = np.all(coords[..., 2] > 0, axis=1)
        coords = coords[indices]
        labels_r = labels_r[indices]

        indices = np.argsort(-np.min(coords[..., 2], axis=1))
        coords = coords[indices]
        labels_r = labels_r[indices]

        coords = coords.reshape(-1, 4)
        coords[:, 2] = np.clip(coords[:, 2], a_min=1e-5, a_max=1e5)
        coords[:, 0] /= coords[:, 2]
        coords[:, 1] /= coords[:, 2]

        coords = coords[..., :2].reshape(-1, 8, 2)
      
        for index in range(coords.shape[0]):
            name = classes[labels_r[index]]
           
            for start, end in [
                (0, 1),
                (0, 3),
                (0, 4),
                (1, 2),
                (1, 5),
                (3, 2),
                (3, 7),
                (4, 5),
                (4, 7),
                (2, 6),
                (5, 6),
                (6, 7),
            ]:
               
                cv2.line(
                    canvas,
                    coords[index, start].astype(np.int32),
                    coords[index, end].astype(np.int32),
                    (255,255,255),
                    thickness,
                    cv2.LINE_AA,
                )

    if bboxes is not None and len(bboxes) > 0:
        corners = bboxes.corners
        num_bboxes = corners.shape[0]

        coords = np.concatenate(
            [corners.reshape(-1, 3), np.ones((num_bboxes * 8, 1))], axis=-1
        )
       
        transform = copy.deepcopy(transform).reshape(4, 4)
        coords = coords @ transform.T
        coords = coords.reshape(-1, 8, 4)

        indices = np.all(coords[..., 2] > 0, axis=1)
        coords = coords[indices]
        labels = labels[indices]

        indices = np.argsort(-np.min(coords[..., 2], axis=1))
        coords = coords[indices]
        labels = labels[indices]

        coords = coords.reshape(-1, 4)
        coords[:, 2] = np.clip(coords[:, 2], a_min=1e-5, a_max=1e5)
        coords[:, 0] /= coords[:, 2]
        coords[:, 1] /= coords[:, 2]

        coords = coords[..., :2].reshape(-1, 8, 2)
      
        for index in range(coords.shape[0]):
            name = classes[labels[index]]
           
            for start, end in [
                (0, 1),
                (0, 3),
                (0, 4),
                (1, 2),
                (1, 5),
                (3, 2),
                (3, 7),
                (4, 5),
                (4, 7),
                (2, 6),
                (5, 6),
                (6, 7),
            ]:
               
                cv2.line(
                    canvas,
                    coords[index, start].astype(np.int32),
                    coords[index, end].astype(np.int32),
                    color or OBJECT_PALETTE[name],
                    thickness,
                    cv2.LINE_AA,
                )
        canvas = canvas.astype(np.uint8)
    canvas = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
    
    mmcv.mkdir_or_exist(os.path.dirname(fpath))
    mmcv.imwrite(canvas, fpath)


def visualize_lidar(
    fpath: str,
    lidar: Optional[np.ndarray] = None,
    *,
    bboxes: Optional[LiDARInstance3DBoxes] = None,
    labels: Optional[np.ndarray] = None,
    classes: Optional[List[str]] = None,
    xlim: Tuple[float, float] = (-50, 50),
    ylim: Tuple[float, float] = (-50, 50),
    color: Optional[Tuple[int, int, int]] = None,
    radius: float = 15,
    thickness: float = 25,
) -> None:
    fig = plt.figure(figsize=(xlim[1] - xlim[0], ylim[1] - ylim[0]))

    ax = plt.gca()
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect(1)
    ax.set_axis_off()

    if lidar is not None:
        plt.scatter(
            lidar[:, 0],
            lidar[:, 1],
            s=radius,
            c="white",
        )

    if bboxes is not None and len(bboxes) > 0:
        coords = bboxes.corners[:, [0, 3, 7, 4, 0], :2]
        for index in range(coords.shape[0]):
            name = classes[labels[index]]
            plt.plot(
                coords[index, :, 0],
                coords[index, :, 1],
                linewidth=thickness,
                color=np.array(color or OBJECT_PALETTE[name]) / 255,
            )

    mmcv.mkdir_or_exist(os.path.dirname(fpath))
    fig.savefig(
        fpath,
        dpi=10,
        facecolor="black",
        format="png",
        bbox_inches="tight",
        pad_inches=0,
    )
    plt.close()


def visualize_map(
    fpath: str,
    masks: np.ndarray,
    *,
    classes: List[str],
    background: Tuple[int, int, int] = (240, 240, 240),
) -> None:
    assert masks.dtype == np.bool, masks.dtype

    canvas = np.zeros((*masks.shape[-2:], 3), dtype=np.uint8)
    canvas[:] = background

    for k, name in enumerate(classes):
        if name in MAP_PALETTE:
            canvas[masks[k], :] = MAP_PALETTE[name]
    canvas = cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR)

    mmcv.mkdir_or_exist(os.path.dirname(fpath))
    mmcv.imwrite(canvas, fpath)


map_name={
    'car':"CAR",
    'pedestrian':"PEDESTRIAN",
    'truck':"TRUCK",
    'bicycle':"BICYCLE",
    'motorcycle':"MOTORCYCLE",
    'bus':"BUS",
}

def export_results_dict(
    gt_boxes: Dict,
    bboxes: LiDARInstance3DBoxes,
    labels: np.ndarray,
    scores: np.ndarray,
    classes: List[str],
    sequence: str,
    frameid: str,
    timestamp: float,
    pose: np.ndarray
) -> Dict:

  

    frame_dict={}
    frame_dict['frame_id'] = frameid
    out_boxes =  []
    out_scores = []
    classes_names = []
  
    frame_dict['pose']=pose
    frame_dict['sequence']=sequence
    if bboxes is not None and len(bboxes) > 0:
        centers = bboxes.gravity_center.numpy().astype(np.float64)
        dims = bboxes.dims.numpy().astype(np.float64)
        yaws=bboxes.yaw.numpy().astype(np.float64)

        for i in range(centers.shape[0]): 
          
            bbox=[centers[i,0],centers[i,1],centers[i,2],dims[i,1],dims[i,0],dims[i,2],np.pi/2-yaws[i]]
            out_boxes.append(bbox)
            classes_names.append(classes[labels[i]])
            out_scores.append(float(scores[i]))

    frame_dict['name'] = np.asarray(classes_names)
    frame_dict['score'] = np.asarray(out_scores)
    frame_dict['boxes_lidar'] = np.asarray(out_boxes)
    frame_dict['label'] = labels

  
    gt_boxes.append(frame_dict)
  
    return gt_boxes

def export_results_aimotive(
    fpath: str,
    bboxes: LiDARInstance3DBoxes,
    labels: np.ndarray,
    scores: np.ndarray,
    classes: List[str],
    frameid: str,
    timestamp: float,
) -> None:
    
    

    mmcv.mkdir_or_exist(os.path.dirname(fpath))

    if bboxes is not None and len(bboxes) > 0:
        centers = bboxes.gravity_center.numpy().astype(np.float64)
        dims = bboxes.dims.numpy().astype(np.float64)
        yaws=bboxes.yaw.numpy().astype(np.float64)
        outdict={}
        outboxes=[]
        
        for i in range(centers.shape[0]): 
            
            boxclass=map_name[classes[labels[i]]]
            
            box={}
            box["BoundingBox3D Origin X"]=centers[i,0]
            box["BoundingBox3D Origin Y"]=centers[i,1]
            box["BoundingBox3D Origin Z"]=centers[i,2]
            box["BoundingBox3D Extent X"]=dims[i,0]
            box["BoundingBox3D Extent Y"]=dims[i,1]
            box["BoundingBox3D Extent Z"]=dims[i,2]
            

            box['Absolute Acceleration X']=0.0
            box['Absolute Acceleration Y']=0.0
            box['Absolute Acceleration Z']=0.0

            box['Absolute Velocity X']=0.0
            box['Absolute Velocity Y']=0.0
            box['Absolute Velocity Z']=0.0

            box['ActorName']='{} {}'.format(boxclass,i)


            yaw=np.pi/2 - yaws[i]
            quat = pyquaternion.Quaternion(axis=[0, 0, 1], radians=yaw)
            
            box['BoundingBox3D Orientation Quat W']=quat.w
            box['BoundingBox3D Orientation Quat X']=quat.x
            box['BoundingBox3D Orientation Quat Y']=quat.y
            box['BoundingBox3D Orientation Quat Z']=quat.z
            
            box['ObjectId']=i
            box['ObjectType']=boxclass

            box['Relative Acceleration X']=0.0
            box['Relative Acceleration Y']=0.0
            box['Relative Acceleration Z']=0.0

            box['Relative Velocity X']=0.0
            box['Relative Velocity Y']=0.0
            box['Relative Velocity Z']=0.0
            box['Occluded']=0
            box['Truncated']=0
            box['Score']=float(scores[i])
            outboxes.append(box)

        outdict["CapturedObjects"]=outboxes
        outdict["FrameId"]=int(frameid)
        outdict["Timestamp"]=int(timestamp*1e3)
        outdict["TimestampMiddle"]=int(timestamp*1e3)
      
        with open(fpath, 'w') as json_file:
            json.dump(outdict, json_file,indent=4)