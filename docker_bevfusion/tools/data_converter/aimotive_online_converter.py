import os
from collections import OrderedDict
from os import path as osp
from typing import List, Tuple, Union

import mmcv
import numpy as np
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.geometry_utils import view_points
from pyquaternion import Quaternion
from shapely.geometry import MultiPoint, box
import json
from mmdet3d.core.bbox.box_np_ops import points_cam2img
from mmdet3d.datasets import NuScenesDataset
from datatools.loader import Loader
from datatools.boxutils import Boxe3d

from sklearn.model_selection import train_test_split
from create_gt_database import create_groundtruth_database
import pathlib

map_name={
    "CAR":'car',
    "PEDESTRIAN":'pedestrian',
    "TRUCK":'truck',
    "BICYCLE":'bicycle',
    "MOTORCYCLE":'motorcycle',
    "BUS": 'bus',
    "TRAIN":'bus',
    "RIDER":'pedestrian'
}
def create_aimotive_infos(root_path, info_prefix, reduced=False, max_sweeps=10):
    """
    Create aimotive infos.

    Args:
        root_path (str): Root path of the dataset.
        info_prefix (str): Prefix of the output info file.
        reduced (bool, optional): Whether to use reduced dataset. Defaults to False.
        max_sweeps (int, optional): Maximum number of lidar sweeps. Defaults to 10.

    Returns:
        None
    """
    loader = Loader(root_path, 'box_processed_ped', 'F_MIDRANGECAM_C')

    ids = list(range(len(loader.seq_list)))
    train_ids, test_ids = train_test_split(ids,
                                           test_size=0.1,
                                           random_state=4240)
    count = 0

    loader2 = Loader(root_path, 'box', 'F_MIDRANGECAM_C')

    val_infos = []
    for id in test_ids:
        seq = loader.seq_list[id]
        val_infos = _fill_infos(loader2, seq, count, val_infos, max_sweeps, reduced)
        count += 1
        if count > 2:
            break

    metadata = dict(version=1.0)

    print('val sample: {}'.format(len(val_infos)))
    data = dict(infos=val_infos, metadata=metadata)
    info_val_path = './{}_infos_val.pkl'.format(info_prefix)
    mmcv.dump(data, info_val_path)

def _fill_infos(loader,seq_path,count,infos,max_sweeps,reduced):

    
    cam_infos={}
    sensorconfig, vehicle_geometry, vehiclepose = loader.load_calibration(seq_path)
    R,T=loader.getCameraExtrinsics(sensorconfig)
    K,D,xi=loader.getCameraIntrinsics(sensorconfig)
   
    R=R.T
    T=(-R@T).reshape((3,))
   
    

    cam_info = {
                'data_path': [],
                'type': 'CAM_FRONT',
                'sensor2lidar_rotation' : R, 
                'sensor2lidar_translation' : T,
                'sensor2ego_rotation' : R, 
                'sensor2ego_translation' : T,
                'timestamp': 0,
                'camera_intrinsics' : K
       }
       
    cam_infos['CAM_FRONT']=cam_info
  
    TLidar=np.zeros((3,1),dtype=np.float64)
    RLidar=np.eye(3,dtype=np.float64)
   
    filelidar_time=seq_path+"/timestamps_lidar.json"
    ego_file = seq_path + '/egopose.json'

    with open(filelidar_time) as f:
        lidar_infos = json.load(f)
    
    with open(ego_file) as f:
           pose_infos = json.load(f)
    print(seq_path)
    for idx in range(loader.get_nannots(seq_path)):
        
        gt_box,frameid,path= loader.load_gt(seq_path,idx)
       # print("frameid : ",frameid)
        path=loader.get_im_undist_path(seq_path,frameid)
        
        if reduced:
            lidar_path=loader.get_lidar_path_reduced(seq_path,frameid)
        else:
            lidar_path=loader.get_lidar_path(seq_path,frameid)
              
        ids_lidar=lidar_infos["frames_ids"]
        times_lidar=lidar_infos["Timestamp"]
        index_lidar=ids_lidar.index(frameid)
        size_lidar=len(ids_lidar)
        
        sequencename = pathlib.PurePath(seq_path).name
        
        token=str(count)+frameid
        timestamp=gt_box["Timestamp"]*1e-3


        ref_pose=np.asarray(pose_infos[frameid]['pose'])
        R_start= ref_pose[:3,:3].T
        T_start = ref_pose[:3,-1]

        info = {
            "lidar_path": lidar_path,
            "token": token,
            "sweeps": [],
            "cams": dict(),
            "lidar2ego_translation": TLidar,
            "lidar2ego_rotation": RLidar,
            "ego2global_translation": T_start,
            "ego2global_rotation": R_start,
            "timestamp": timestamp, 
            "sequence": str(sequencename),
            "frame_id": frameid,
        }
        
        
        cam_info_ = cam_infos['CAM_FRONT'].copy()
        cam_info_['data_path']=path   
        cam_info_['timestamp']=timestamp
        info["cams"].update({'CAM_FRONT': cam_info_})


       
        sweeps = []
        idx_s=-9
        
        RT_start_inv = np.eye(4)
        R_start_inv = ref_pose[:3,:3].T
        T_start_inv = -R_start_inv @ ref_pose[:3,-1]
        RT_start_inv[:3,:3] = R_start_inv
        RT_start_inv[:3,-1] = T_start_inv
       
        while len(sweeps) < max_sweeps and idx_s<0:
            index_current=idx_s +index_lidar
            if index_current>0 and index_current<size_lidar and index_current!=index_lidar:
                
                frameid_sweep=ids_lidar[index_current]
                timestamp_sweep=times_lidar[index_current]*1e-3
                
                sweep_pose=np.asarray(pose_infos[frameid_sweep]['pose'])
          
                file=loader.get_lidar_path(seq_path,frameid_sweep)
                
                RT_stop_relative = RT_start_inv @ sweep_pose

                R=RT_stop_relative[:3,:3]
                T=RT_stop_relative[:3,-1]
                sweep = {
                      'data_path': file,
                      'token' : str(count)+frameid_sweep,
                      'type': 'lidar',
                      'sensor2lidar_rotation' : R, 
                      'sensor2lidar_translation' : T,
                      'timestamp': timestamp_sweep,
                    
                 }
                sweeps.append(sweep)
            idx_s+=1   

        info["sweeps"] = sweeps
        
        nboxes=1
        locs=np.zeros((nboxes,3),dtype=np.float64)
        dims = np.zeros((nboxes,3),dtype=np.float64) # wlh
        rots = np.zeros((nboxes,1),dtype=np.float64)
        velocity = np.zeros((nboxes,2),dtype=np.float64)
        valid_flag = np.zeros((nboxes,),dtype=bool)     
        valid_flag[0]=True        
        names=[]
        
        names.append('car')
        names = np.array(names)
        gt_boxes = np.concatenate([locs, dims, rots], axis=1)
        
        info['gt_boxes'] = gt_boxes
        info['gt_names'] = names
        info['gt_velocity'] = velocity.reshape(-1, 2)
        info['num_lidar_pts'] = np.ones(locs.shape[0],dtype=np.int32)*30
        info['num_radar_pts'] = np.ones(locs.shape[0],dtype=np.int32)*30
        info['valid_flag'] = valid_flag
        info['location'] = "NA"
        infos.append(info)
    return infos

import argparse

def parse_args():

    parser = argparse.ArgumentParser(
        description='Create postprocessed lane annotations')
    parser.add_argument('--rootdir', help='root dir of sequences',default='/mnt/data/VisionDataBases/PSA/PSA_2023_dataset/gt_anonymized/')
    
   
    args = parser.parse_args()
    
    return args

if __name__=='__main__':
    args = parse_args()


    root_data_path=args.rootdir
   
    create_aimotive_infos(root_data_path,'./aimotive3dLidarCamera_reduced_online',reduced=True)

   