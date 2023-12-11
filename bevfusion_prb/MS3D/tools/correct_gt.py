import sys

import torch  # not used but prevents a bug

sys.path.append("../")
import argparse
import json
import math
import os
import pickle
from pathlib import Path

import cv2
import numpy as np
import yaml
from datatools.boxutils import fitBox, projectPoints, projectPoints_all
from datatools.create_segmentationmasks_m2f import ID2LABEL, create_infer_model
from matplotlib import pyplot as plt
from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets.augmentor.augmentor_utils import get_points_in_box
from pcdet.utils import ms3d_utils
from sklearn.cluster import DBSCAN
from tqdm import tqdm
from visual_utils import common_vis

OBJECT_PALETTE = {
    "car": (0, 200, 0),
    "truck": (255, 100, 71),
    "bus": (255, 40, 0),
    "motorcycle": (255, 255, 255),
    "bicycle": (0, 255, 230),
    "pedestrian": (0, 0, 230),
}

def plot3DBoxes(image, img_pts_box3d, z, color, id):
    """
    Draws a 3D bounding box on an image given its 8 corner points and depth.

    Args:
        image (numpy.ndarray): The image to draw the bounding box on.
        img_pts_box3d (numpy.ndarray): An array of shape (8, 2) containing the image coordinates of the 8 corner points of the 3D bounding box.
        z (numpy.ndarray): An array of shape (8,) containing the depth of the 8 corner points of the 3D bounding box.
        color (tuple): A tuple of 3 integers representing the RGB color of the bounding box.
        id (int): The ID of the bounding box.

    Returns:
        None
    """


    if z[0] > 0 and z[1] > 0:
        cv2.line(
            image,
            (img_pts_box3d[0, 0], img_pts_box3d[0, 1]),
            (img_pts_box3d[1, 0], img_pts_box3d[1, 1]),
            color,
            2,
        )
    if z[1] > 0 and z[2] > 0:
        cv2.line(
            image,
            (img_pts_box3d[1, 0], img_pts_box3d[1, 1]),
            (img_pts_box3d[2, 0], img_pts_box3d[2, 1]),
            color,
            2,
        )
    if z[2] > 0 and z[3] > 0:
        cv2.line(
            image,
            (img_pts_box3d[2, 0], img_pts_box3d[2, 1]),
            (img_pts_box3d[3, 0], img_pts_box3d[3, 1]),
            color,
            2,
        )
    if z[0] > 0 and z[3] > 0:
        cv2.line(
            image,
            (img_pts_box3d[0, 0], img_pts_box3d[0, 1]),
            (img_pts_box3d[3, 0], img_pts_box3d[3, 1]),
            color,
            2,
        )

    if z[4] > 0 and z[5] > 0:
        cv2.line(
            image,
            (img_pts_box3d[4, 0], img_pts_box3d[4, 1]),
            (img_pts_box3d[5, 0], img_pts_box3d[5, 1]),
            color,
            2,
        )
    if z[5] > 0 and z[6] > 0:
        cv2.line(
            image,
            (img_pts_box3d[5, 0], img_pts_box3d[5, 1]),
            (img_pts_box3d[6, 0], img_pts_box3d[6, 1]),
            color,
            2,
        )
    if z[6] > 0 and z[7] > 0:
        cv2.line(
            image,
            (img_pts_box3d[6, 0], img_pts_box3d[6, 1]),
            (img_pts_box3d[7, 0], img_pts_box3d[7, 1]),
            color,
            2,
        )
    if z[4] > 0 and z[7] > 0:
        cv2.line(
            image,
            (img_pts_box3d[4, 0], img_pts_box3d[4, 1]),
            (img_pts_box3d[7, 0], img_pts_box3d[7, 1]),
            color,
            2,
        )

    if z[0] > 0 and z[4] > 0:
        cv2.line(
            image,
            (img_pts_box3d[0, 0], img_pts_box3d[0, 1]),
            (img_pts_box3d[4, 0], img_pts_box3d[4, 1]),
            color,
            2,
        )
    if z[1] > 0 and z[5] > 0:
        cv2.line(
            image,
            (img_pts_box3d[1, 0], img_pts_box3d[1, 1]),
            (img_pts_box3d[5, 0], img_pts_box3d[5, 1]),
            color,
            2,
        )
    if z[2] > 0 and z[6] > 0:
        cv2.line(
            image,
            (img_pts_box3d[2, 0], img_pts_box3d[2, 1]),
            (img_pts_box3d[6, 0], img_pts_box3d[6, 1]),
            color,
            2,
        )
    if z[3] > 0 and z[7] > 0:
        cv2.line(
            image,
            (img_pts_box3d[3, 0], img_pts_box3d[3, 1]),
            (img_pts_box3d[7, 0], img_pts_box3d[7, 1]),
            color,
            2,
        )

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    thickness = 2
    color = (0, 0, 255)  # BGR format

    # Draw the text on the image
    cv2.putText(
        image,
        str(id),
        (img_pts_box3d[0, 0], img_pts_box3d[0, 1]),
        font,
        font_scale,
        color,
        thickness,
    )


class InstanceObjects:
    """
    A class that represents a set of instance objects in an image, along with their corresponding  masks.
    """

    def __init__(self, gt_dict, mask):
        """
        Initializes an instance of InstanceObjects with the given dictionary and mask.

        Args:
        - gt_dict (dict): A dictionary containing ground truth information, including classes names, classes index, and scores.
        - mask (numpy.ndarray): A numpy array representing the mask.

        Returns:
        - None
        """
        self.gt_dict = gt_dict
        self.mask = mask
        self.classes = gt_dict["classes_names"]
        self.classes_index = gt_dict["classes_index"]
        self.box2d = []
        self.matched = []
        self.scores = np.asarray(gt_dict["scores"])
        self.valid=[]
        for i in range(len(self.classes_index)):
            mask_i = np.where(mask == i + 1)
            class_idx = self.classes_index[i]
            class_name = self.classes[class_idx-1]
            valid=True
            self.matched.append(False)
            if mask_i[0].shape[0] == 0:
                self.box2d.append([0.0, 0.0, 0.0, 0.0])
                valid=False
            else:
                box = [min(mask_i[1]), min(mask_i[0]), max(mask_i[1]), max(mask_i[0])]
                self.box2d.append(box)

            if class_name=='bicycle' or class_name=='smallVehicle':
                if self.heigth(i)<60:
                    valid=False
            self.valid.append(valid)

    def size_instance(self, instance_idx):
        mask_idx = self.mask == instance_idx + 1
        return np.sum(mask_idx)

    def size(self, box):
        return (box[2] - box[0]) * (box[3] - box[1])

    def heigth(self, instance_idx):
        box = self.box2d[instance_idx]
        return box[3] - box[1]

    def computeMeanPointForInstance(self, instance_idx, pts3d, pts2d, img):
        """
        Computes the mean 3D point for a given instance in the image.

        Args:
            instance_idx (int): The index of the instance to compute the mean point for.
            pts3d (numpy.ndarray): An array of shape (N, 3) containing the 3D points of the current frame.
            pts2d (numpy.ndarray): An array of shape (N, 2) containing the 2D points of the current frame..
            img (numpy.ndarray): The input image.

        Returns:
            numpy.ndarray: An array of shape (3,) containing the mean 3D point of the instance.
        """
    
        instance_id = instance_idx + 1
        uv_mask_val = self.mask[pts2d[:, 1], pts2d[:, 0]]

        if np.size(uv_mask_val) == 0:
            return np.asarray([])

        pts2d_inst = pts2d[uv_mask_val == instance_id]
        pts3d_inst = pts3d[uv_mask_val == instance_id]

        if np.size(pts3d_inst) == 0:
            return np.asarray([])
        box_inst = self.box2d[instance_idx]
        instance_center = np.asarray(
            [(box_inst[0] + box_inst[2]) / 2.0, (box_inst[1] + box_inst[3]) / 2.0]
        )

        distance = np.linalg.norm(pts2d_inst - instance_center, axis=1)
        index = np.argmin(distance)
        pts3d_inst = pts3d_inst[index]
        cv2.circle(
            img,
            (int(pts2d_inst[index, 0]), int(pts2d_inst[index, 1])),
            3,
            (0, 0, 255),
            -1,
        )
        return np.asarray(pts3d_inst)

    def correctBox3dWidthandLength(
            self, box, instance_idx, pts3d, pts2d, is_person, img
        ):
            """
            Corrects the width and length of a 3D bounding box based on the 3D and 2D points of the object instance.

            Args:
                box (np.ndarray): The 3D bounding box to be corrected.
                instance_idx (int): The index of the object instance.
                pts3d (np.ndarray): The 3D points of the object instance.
                pts2d (np.ndarray): The 2D points of the object instance.
                is_person (bool): Whether the object instance is a person or not.
                img (np.ndarray): The image containing the object instance.

            Returns:
                np.ndarray: The corrected 3D bounding box.
            """
            
            if is_person:
                return box
            
            instance_id = instance_idx + 1
            center = box[0:3]
            
            direction = np.asarray(
                [math.cos(box[6] + np.pi / 2), math.sin(box[6] + np.pi / 2)]
            )

            uv_mask_val = self.mask[pts2d[:, 1], pts2d[:, 0]]

            if np.size(uv_mask_val) == 0:
                return box

            pts2d_inst = pts2d[uv_mask_val == instance_id]
            pts3d_inst = pts3d[uv_mask_val == instance_id]

            if np.size(pts3d_inst) == 0:
                return box

            # Set DBSCAN parameters
            epsilon = 0.8
            min_samples = 5

            # Apply DBSCAN clustering
            dbscan = DBSCAN(eps=epsilon, min_samples=min_samples)
            labels = dbscan.fit(pts3d_inst[:, 0:2]).labels_

            # Count the number of points in each cluster
            unique_labels = np.unique(labels)

            counts = []
            for label in unique_labels:
                if label == -1:
                    counts.append(0)
                else:
                    count = np.sum(labels == label)
                    counts.append(count)
            # Find the cluster with the maximum number of points
            max_count = np.max(counts)
            max_count_label = unique_labels[np.argmax(counts)]
            # Return the cluster with the maximum number of points
            pts3d_inst = pts3d_inst[labels == max_count_label]
            pts2d_inst = pts2d_inst[labels == max_count_label]

            if np.size(pts3d_inst) < 10:
                return box

            projections = (pts3d_inst[:, 0:2].copy() - center[0:2]).dot(direction)
            min_projection = np.min(projections)
            max_projection = np.max(projections)

            if min_projection >= 0 or max_projection <= 0:
                return box

            length = box[4]
            new_length = max_projection - min_projection

            if new_length <= 0 or new_length < length:
                return box

            instance_class = self.classes[self.classes_index[instance_idx] - 1]

            if instance_class == "truck":
                if new_length > 3 * length:
                    return box
            else:
                if new_length > 1.5 * length:
                    return box

            box[4] = new_length
            box[0:2] = center[0:2] + (min_projection + max_projection) / 2.0 * direction

            return box

    def correctBox3dHeight(self, box, corners, instance_idx, K, R, T, is_person, img):
        """
        Corrects the height of a 3D bounding box based on the projected 2D corners of the box onto an image.

        Args:
            box (np.ndarray):  the 3D bounding box. 
            corners (np.ndarray): A 2D array of shape (8, 3) representing the 3D corners of the box.
            instance_idx (int): The index of the instance in the mask.
            K (np.ndarray): the camera intrinsic matrix.
            R (np.ndarray): the camera rotation matrix.
            T (np.ndarray): camera translation vector.
            is_person (bool): Whether the box represents a person or not.
            img (np.ndarray): A 3D array representing the image.

        Returns:
            np.ndarray: the corrected 3D bounding box.
        """
       
        box = box.copy()
        mask_idx = self.mask == instance_idx + 1

        pts_img, z = projectPoints_all(corners, K, R, T)

        pts_img_bottom = pts_img[0:4]
        pts_img_top = pts_img[4:8]
        z_bottom = z[0:4]
        z_top = z[4:8]
        mask = (
            (pts_img_bottom[:, 1] > 0)
            * (pts_img_bottom[:, 0] > 0)
            * (pts_img_bottom[:, 0] < self.mask.shape[1])
            * (pts_img_bottom[:, 1] < self.mask.shape[0])
        )
      
        if np.sum(mask) < 2:
            return box

        pts_img_bottom = pts_img_bottom[mask]
        pts_img_top = pts_img_top[mask]

        z_bottom = z_bottom[mask]
        z_top = z_top[mask]

        if is_person:
            index = np.argmin(pts_img_bottom[:, 1])
            pt_bottom = pts_img_bottom[index]
            pt_top = pts_img_top[index]
            z_mid = (z_bottom[index] + z_top[index]) / 2.0
        else:
            index = np.argpartition(pts_img_bottom[:, 1], -2)[-2:]
            pt_bottom = (pts_img_bottom[index[0]] + pts_img_bottom[index[1]]) / 2.0
            pt_top = (pts_img_top[index[0]] + pts_img_top[index[1]]) / 2.0
            z_mid = (
                (z_bottom[index[0]] + z_bottom[index[1]]) / 2.0
                + (z_top[index[0]] + z_top[index[1]]) / 2.0
            ) / 2.0

        mid_point = (pt_bottom + pt_top) / 2.0
        
        if pt_top[1] < 0:
            return box

        if is_person:
            min_v = self.box2d[instance_idx][1]
            max_v = self.box2d[instance_idx][3]
        else:
            col_instance = mask_idx[:, int(pt_bottom[0])]
            mask_i = np.where(col_instance == True)
            if np.size(mask_i) == 0:
                return box

            min_v = min(mask_i[0])
            max_v = max(mask_i[0])
           

        h = box[5]
        dh1 = 0
        dh2 = 0
      
        if pt_top[1] > min_v:
            dv = mid_point[1] - min_v
            dh1 = dv * z_mid / K[1, 1] - h / 2
           
        if pt_bottom[1] < max_v:
            dv = max_v - mid_point[1]
            dh2 = dv * z_mid / K[1, 1] - h / 2
           
       
        if (dh1 + dh2) > h:
            return box
        
        box[5] += dh1 + dh2
        box[2] += dh1 / 2.0 - dh2 / 2.0

        return box

    def check_minimum_size(self, instance_idx):
        """
        Checks if the instance at the given index meets the minimum size requirements for its class.

        Args:
            instance_idx (int): The index of the instance to check.

        Returns:
            bool: True if the instance meets the minimum size requirements, False otherwise.
        """
        instance_class = self.classes[self.classes_index[instance_idx] - 1]

        if instance_class == "pedestrian" and self.heigth(instance_idx) > 150:
            return True
        if instance_class == "truck" and self.heigth(instance_idx) > 70:
            return True
        if instance_class == "car" and self.heigth(instance_idx) > 70:
            return True
        if instance_class == "bicycle" and self.heigth(instance_idx) > 70:
            return True
        if instance_class == "tractor" and self.heigth(instance_idx) > 60:
            return True
        if instance_class == "smallVehicle" and self.heigth(instance_idx) > 70:
            return True
        if instance_class == "utilityVehicle" and self.heigth(instance_idx) > 60:
            return True

        return False

    def getIntersectionValue(self, box, box_pts, box_class_name, instance_idx, instances_centers=None):
        """
        Computes the intersection value between a list of points in the 3d box projected on the image and a mask instance.

        Args:
            box (numpy.ndarray): the 3D bounding box .
            box_pts (numpy.ndarray): Array of shape (N, 2) containing the u,v coordinates of the projected points.
            box_class_name (str): The class name of the 3d object.
            instance_idx (int): The index of the mask instance.
            instances_centers (numpy.ndarray, optional): Array of shape (N, 3) containing the 3d point center of the mask instances. Defaults to None.

        Returns:
            float: The intersection value between the bounding box and the mask instance.
        """

        # first , compare classses
        if not self.valid[instance_idx]:
            return 0.0
        instance_class = self.classes[self.classes_index[instance_idx] - 1]
        class_check = False

        if instance_class == "pedestrian" and box_class_name == "pedestrian":
            class_check = True
        if (
            instance_class == "truck" or instance_class == "utilityVehicle"
        ) and box_class_name == "truck":
            class_check = True
        if instance_class == "truck" and box_class_name == "bus":
            class_check = True
        if (
            instance_class == "car"
            or instance_class == "utilityVehicle"
            or instance_class == "truck"
        ) and box_class_name == "car":
            class_check = True
            if instance_class == "truck":
                box_class_name = "truck"

        if instance_class == "bicycle" and box_class_name == "bicycle":
            class_check = True
        if instance_class == "bicycle" and box_class_name == "motorcycle":
            class_check = True
        if instance_class == "smallVehicle" and box_class_name == "motorcycle":
            class_check = True

        if class_check == False:
            return 0.0

        # third, compute the area of projected points on mask
        mask_idx = self.mask == instance_idx + 1
        inliers = mask_idx[box_pts[:, 1], box_pts[:, 0]] == True

        if box_pts.shape[0] == 0:
            return 0.0
        convexhull1 = cv2.convexHull(box_pts)
        area1 = cv2.contourArea(convexhull1)
        pts = box_pts[inliers]

        if area1 == 0.0:
            return 0.0
        if pts.shape[0] == 0:
            return 0.0

        convexhull2 = cv2.convexHull(pts)
        area2 = cv2.contourArea(convexhull2)

        area_mask = np.sum(mask_idx)
      #  if float(area1)
        ratio1 = area_mask / float(area1) # à corriger
        ratio2 = float(area2) / area_mask

      
        if ratio2 > 1.2 or ratio1 < 0.15:
            return 0.0

        return ratio2 

    def getIntersectionScore(
            self, box, box_pts, box_class_name, instances_centers=None
        ):
            """
            Computes the intersection score between the given bounding box and all other classes.

            Args:
                box (list): The bounding box .
                box_pts (list): The bounding box corners.
                box_class_name (str): The class name of the bounding box.
                instances_centers (list, optional): The 3d centers of the instances. Defaults to None.

            Returns:
                list: A list of intersection scores between the given bounding box and all other classes.
            """
            intersection_scores = []

            for i in range(len(self.classes_index)):
                intersection_scores.append(
                    self.getIntersectionValue(
                        box, box_pts, box_class_name, i, instances_centers
                    )
                )
            return intersection_scores


def getCameraParams(infos, idx, camera):
    K = infos[idx]["cams"][camera]["camera_intrinsics"]
    R = infos[idx]["cams"][camera]["sensor2ego_rotation"]
    T = infos[idx]["cams"][camera]["sensor2ego_translation"]

    R = R.T
    T = -R @ T

    return K, R, T


def visualize_lidar(
    fpath,
    center_instances,
    lidar=None,
    corners3d=None,
    classe_names=None,
    boxes=np.array([]),
    xlim=(0, 100),
    ylim=(-50, 50),
    color=None,
    radius: float = 15,
    thickness: float = 25,
) -> None:
    fig = plt.figure(figsize=(xlim[1] - xlim[0], ylim[1] - ylim[0]))

    ax = plt.gca()
    ax.set_xlim(*ylim)
    ax.set_ylim(*xlim)
    ax.set_aspect(1)
    ax.set_axis_off()

    if lidar is not None:
        plt.scatter(
            -lidar[:, 1],
            lidar[:, 0],
            s=radius,
            c="white",
        )
    if len(center_instances) > 0:
        center_instances = [
            center_instances[i]
            for i in range(len(center_instances))
            if np.size(center_instances[i]) > 0
        ]
        center_instances = np.asarray(center_instances)

        plt.scatter(
            -center_instances[:, 1],
            center_instances[:, 0],
            s=15,
            c="red",
        )
    if np.size(boxes) > 0:
        for k in range(boxes.shape[0]):
            name = classe_names[k]
            center = boxes[k, 0:3]
            direction = [
                math.cos(boxes[k, 6] ),
                math.sin(boxes[k, 6] ),
            ]
            pts = np.asarray(
                [
                    [center[0], center[1]],
                    [center[0] + direction[0], center[1] + direction[1]],
                ]
            )
            plt.plot(
                -pts[:, 1],
                pts[:, 0],
                linewidth=thickness,
                color=np.array(color or OBJECT_PALETTE[name])[::-1] / 255,
            )
    if corners3d is not None and len(corners3d) > 0:
        coords = corners3d[:, [0, 1, 2, 3, 0], :2]
        for index in range(coords.shape[0]):
            name = classe_names[index]
            plt.plot(
                -coords[index, :, 1],
                coords[index, :, 0],
                linewidth=thickness,
                color=np.array(color or OBJECT_PALETTE[name])[::-1] / 255,
            )

    fig.savefig(
        fpath,
        dpi=10,
        facecolor="black",
        format="png",
        bbox_inches="tight",
        pad_inches=0,
    )
    plt.close()


def filter_pointcloud(pointcloud, K, R, T, image_shape):
    uv, mask_p, z = projectPoints(pointcloud[:, 0:3], K, R, T)
    in_fov = (
        (uv[:, 1] >= 0.0)
        * (uv[:, 0] >= 0.0)
        * (uv[:, 0] < image_shape[1] - 1)
        * (uv[:, 1] < image_shape[0] - 1)
        * (z[:] > 0)
    )

    pointcloud_ = pointcloud[mask_p]
    pointcloud_ = pointcloud_[in_fov]

    return pointcloud_


def checkValidBoxes(boxes, box_pts, lidar, shape,K,R,T):
    valid_boxes = []
    valid_boxes_img = []
    for k in range(boxes.shape[0]):
        box_points, _ = get_points_in_box(lidar[:, :3], boxes[k])

        if len(box_points) == 0:
            continue

        box_pts_img, z = projectPoints_all(box_points, K, R, T)
        box_pts_img = box_pts_img[z > 0]

        if np.size(box_pts_img) == 0:
            continue

        in_fov = (
            (box_pts_img[:, 0] >= 0)
            * (box_pts_img[:, 1] >= 0)
            * (box_pts_img[:, 0] < shape[1])
            * (box_pts_img[:, 1] < shape[0])
        )

        box_pts_img = box_pts_img[in_fov]

        valid_boxes.append(boxes[k])
        valid_boxes_img.append(box_pts_img)

    return valid_boxes, valid_boxes_img

def processframe(i, cfg, maskdir, ps_dict, camera, step, outdir,debug=True):
    """
    Processes several frames of the dataset by checking the labels.

    Args:
        i (int): The iteration number.
        cfg (dict): The configuration dictionary of the dataset.
        maskdir (str): The directory containing the mask files.
        ps_dict (dict): The dictionary containing the predicted boxes.
        camera (str): The camera name.
        step (int): The number of frames to process.
        debug (bool, optional): Whether to enable debug mode. Defaults to True.

    Returns:
        tuple: A tuple containing the number of frames keeped and a dictionary containing the corrected labels.
    """

    print("Start Iteration ", i)
    dataset = ms3d_utils.load_dataset(cfg, split="train")
    start_idx = i * step
    count = 0
    results_dict = {}

    for idx in range(start_idx, start_idx + step):    
        if idx >= len(dataset.infos):
            break

        seq_id = dataset.infos[idx]["sequence"]
        frame_id = dataset.infos[idx]["frame_id"]        
        #load data and boxes 
        ##############################################################################
        if frame_id not in ps_dict:
            continue
        boxes = ps_dict[frame_id]["gt_boxes"]
        if boxes.shape[0] == 0:
         #   print('prob')
            continue
        K, R, T = getCameraParams(dataset.infos, idx, camera)
        img_path = dataset.infos[idx]["cams"][camera]["data_path"]
        img = cv2.imread(img_path)
        lidar = dataset.get_lidar_with_sweeps(idx)
        box_pts = common_vis.boxes_to_corners_3d(boxes)

       
        #load mask infos
        ###############################################################################
        mask_filename_png = maskdir + "/{}_{}.png".format(seq_id, frame_id)
        mask = cv2.imread(mask_filename_png, cv2.IMREAD_GRAYSCALE)
        mask_filename_json = maskdir + "/{}_{}.json".format(seq_id, frame_id)
        #img[mask > 0] = [255, 255, 255]
        with open(mask_filename_json) as f:
            gt_dict_instance = json.load(f)

        instances = InstanceObjects(gt_dict_instance, mask)

        #filter pointcloud
        ###############################################################################
    
        pointcloud = filter_pointcloud(lidar[:, :3], K, R, T, img.shape)
        uv, z = projectPoints_all(pointcloud[:, 0:3], K, R, T)

        #compute instance 3d center point
        ###############################################################################
        instances_centers = [
            instances.computeMeanPointForInstance(k, pointcloud, uv, img)
            for k in range(len(instances.classes_index))
        ]

        #compute valid boxes
        ###############################################################################
        valid_boxes = []
        valid_boxes_img = []

        # if debug:
        #     color = (0, 0, 255)
        #     for k in range(boxes.shape[0]):
        #         box_pts_img, z = projectPoints_all(box_pts[k], K, R, T)
        #         plot3DBoxes(img, box_pts_img, z, color, k)

        valid_boxes,valid_boxes_img=checkValidBoxes(boxes,box_pts,lidar,img.shape,K,R,T)

        if len(valid_boxes) == 0:
            continue
       # boxes=np.asarray(valid_boxes)
       # box_pts=common_vis.boxes_to_corners_3d(boxes)
        # if debug:
        #      color = (0, 0, 255)
        #      for k in range(len(valid_boxes)):
        #          box_pts_img, z = projectPoints_all(box_pts[k], K, R, T)
        #          plot3DBoxes(img, box_pts_img, z, color, k)
        #compute association scores betwenn boxes and image instances
        ###############################################################################
        final_boxes = []
        final_boxes_img = []
        score_matrix = []

        for k in range(len(valid_boxes)):
            class_name = dataset.class_names[valid_boxes[k][8].astype(int)]
            scores = instances.getIntersectionScore(
                valid_boxes[k], valid_boxes_img[k], class_name, instances_centers
            )
            #print(scores)
            score_matrix.append(scores)

        score_matrix = np.asarray(score_matrix)

        #compute final boxes based on association scores
        ###############################################################################
        if np.size(score_matrix) != 0:
            while np.max(score_matrix) > 0.10:
                ind = np.unravel_index(
                    np.argmax(score_matrix, axis=None), score_matrix.shape
                )
                final_boxes.append(valid_boxes[ind[0]])
                final_boxes_img.append(valid_boxes_img[ind[0]])
                score_matrix[ind[0], :] = 0.0
                score_matrix[:, ind[1]] = 0.0
                instances.matched[ind[1]] = True

        #check if we keep the image
        ###############################################################################
        skipImage = False

        if len(final_boxes) == 0:
            skipImage = True
            continue

        for k in range(len(instances.matched)):
            if instances.matched[k] == False and instances.valid[k] ==True and instances.check_minimum_size(k):
                skipImage = True
                break

        if skipImage == True:
            # if debug:
            #     outdir_rejected='./out_rejected'
            #     class_names = [
            #         dataset.class_names[final_boxes[k][8].astype(int)]
            #         for k in range(final_boxes.shape[0])
            #     ]

            #     for k in range(final_boxes.shape[0]):
            #         class_name = class_names[k]
            #         color = OBJECT_PALETTE[class_name]
            #         box_pts_img, z = projectPoints_all(box_pts[k], K, R, T)
                
            #         plot3DBoxes(img, box_pts_img, z, color, k)
            #     img_file = outdir_rejected+"/"+frame_id + ".png"
            #     cv2.imwrite(img_file, img)
            continue

        final_boxes = np.asarray(final_boxes)
        box_pts = common_vis.boxes_to_corners_3d(final_boxes)

        #correct boxes
        ###############################################################################
        for k in range(final_boxes.shape[0]):
            class_name = dataset.class_names[final_boxes[k][8].astype(int)]
            scores = instances.getIntersectionScore(
                final_boxes[k], final_boxes_img[k], class_name
            )
            valid = np.asarray(scores) > 0.05
          #  print(scores)
            if np.sum(valid) == 1 or class_name=="pedestrian":  # only one instance matched  or pedestrian
                index = np.argmax(scores)
                final_boxes[k] = instances.correctBox3dHeight(
                    final_boxes[k],
                    box_pts[k],
                    index,
                    K,
                    R,
                    T,
                    is_person=(class_name == "pedestrian"),
                    img=img,
                )
                # print("begin ", k)
                final_boxes[k] = instances.correctBox3dWidthandLength(
                    final_boxes[k],
                    index,
                    pointcloud,
                    uv,
                    is_person=(class_name == "pedestrian"),
                    img=img,
                )
                instance_class = instances.classes[instances.classes_index[index] - 1]
                if instance_class == "truck" and class_name == "car":
                    final_boxes[k][8] = dataset.class_names.index("truck")
                if instance_class == "bicycle" and class_name == "motorcycle":    
                    final_boxes[k][8] = dataset.class_names.index("bicycle")
                  
        #visualization for  debug
        ###############################################################################

        img_file = outdir+"/"+frame_id + ".png"
        lidar_file = outdir+"/"+ frame_id + "_lidar_.png"

        box_pts = common_vis.boxes_to_corners_3d(final_boxes)
        
        if debug:
            class_names = [
                dataset.class_names[final_boxes[k][8].astype(int)]
                for k in range(final_boxes.shape[0])
            ]

            for k in range(final_boxes.shape[0]):
                class_name = class_names[k]
                color = OBJECT_PALETTE[class_name]
                box_pts_img, z = projectPoints_all(box_pts[k], K, R, T)
               
                plot3DBoxes(img, box_pts_img, z, color, k)
           # visualize_lidar(
           #     lidar_file, instances_centers, lidar, box_pts, class_names, final_boxes
           # )
            
            
            cv2.imwrite(img_file, img)

         #save results
        ###############################################################################
        results_dict[frame_id] = {"gt_boxes": final_boxes}
        count += 1
    print("Finish Iteration ", i)
    return count, results_dict


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="arg parser")
    parser.add_argument("--ps_cfg", type=str, help="cfg file with MS3D parameters")
    parser.add_argument("--ps_pkl", type=str, help="gt file to correct")
    parser.add_argument("--outviz", type=str, help="folder for visualization")
    parser.add_argument("--nodebug", action="store_false")

    args = parser.parse_args()
    debug = args.nodebug
    print("Debug mode :", debug)
    outdir=args.outviz
    print("outviz :",outdir)
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    ms3d_configs = yaml.load(open(args.ps_cfg, "r"), Loader=yaml.Loader)
    cfg_from_yaml_file(ms3d_configs["DATA_CONFIG_PATH"], cfg)
    dataset = ms3d_utils.load_dataset(cfg, split="train")

    with open(args.ps_pkl, "rb") as f:
        ps_dict = pickle.load(f)

    camera = ms3d_configs["CAMERA"]
    maskdir = ms3d_configs["MASK_DIR"]

    color = (0, 255, 0)

    final_ps_dict = {}
    count = 0

    from joblib import Parallel, delayed
    from joblib_progress import joblib_progress

    # step=1
    # results=processframe(4000, cfg, maskdir, ps_dict.copy(), camera, step, outdir, debug=debug)
    step = 1000
    ntasks = int(len(dataset.infos) / step) + 1
    with joblib_progress("Correcting boxes...", total=ntasks):
           results = Parallel(n_jobs=32)(delayed(processframe)(i,cfg,maskdir,ps_dict.copy(),camera,step,outdir,debug=debug) for i in range(ntasks))
    final_ps_dict={}
    keeped=0
    for i in range(len(results)):
        final_ps_dict.update(results[i][1])
        keeped+=results[i][0]
    print("N frames keeped :",keeped)
    ms3d_utils.save_data(final_ps_dict, str(Path(ms3d_configs["SAVE_DIR"])), name="corrected_ps_dict.pkl")
  