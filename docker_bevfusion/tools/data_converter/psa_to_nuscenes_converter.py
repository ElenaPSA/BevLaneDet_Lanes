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

nus_categories = (
    "car",
    "truck",
    "trailer",
    "bus",
    "construction_vehicle",
    "bicycle",
    "motorcycle",
    "pedestrian",
    "traffic_cone",
    "barrier",
)

nus_attributes = (
    "cycle.with_rider",
    "cycle.without_rider",
    "pedestrian.moving",
    "pedestrian.standing",
    "pedestrian.sitting_lying_down",
    "vehicle.moving",
    "vehicle.parked",
    "vehicle.stopped",
    "None",
)

def create_psa_infos(root_path,train_seq,val_seq,info_prefix,max_sweeps=9):

    psa_train_infos = [] #fill_trainval_infos(root_path,train_file)
    psa_val_infos = _fill_trainval_infos(root_path,val_seq[0],max_sweeps)
    
    
    metadata = dict(version=1.0)
   
    print('train sample: {}, val sample: {}'.format(len(psa_train_infos), len(psa_val_infos)))
    data = dict(infos=psa_train_infos, metadata=metadata)
  
    data['infos'] = psa_val_infos
    info_val_path = osp.join(root_path,'{}_infos_val.pkl'.format(info_prefix))
    mmcv.dump(data, info_val_path)


def _fill_trainval_infos(root_path,val_seq,max_sweeps):

    """Generate the train/val infos from the raw data.

    Args:
       
        scenes (list[str]): Basic information of scenes.
       
    Returns:
        tuple[list[dict]]: Information of training set or validation set
            that will be saved to the info file.
    """
    
    seqdir= os.path.join(root_path,val_seq)
    lidar_dir=os.path.join(seqdir,"TOP_LIDAR")
    fileLidarInfo=os.path.join(lidar_dir, "infos.json")
    infoLidarInput= json.load(open(fileLidarInfo))
    
    infoCamsInput={}
    psainfos=[]

    camera_types = [
            'CAM_FRONT',
            #'CAM_FRONT_RIGHT',
            #'CAM_FRONT_LEFT',
            #'CAM_BACK'
    ]

    cam_infos={}

    for cam in camera_types:
       cam_dir=os.path.join(seqdir,cam)
       fileCamInfo=os.path.join(cam_dir, "infos.json")
       infoCamsInput[cam]=json.load(open(fileCamInfo))

       camera_matrix= np.asarray(infoCamsInput[cam]['calibration']['cam_mat'])
       rmat=np.asarray(infoCamsInput[cam]['calibration']['rmat'])
       tvec=np.asarray(infoCamsInput[cam]['calibration']['tvec'])
       R=rmat.T
       T=(-rmat.T@tvec).reshape((3,))
       
       T[0] += 1.5
       T[2] += 1.7
      

       cam_info = {
                 'data_path': [],
                 'type': cam,
                 'sensor2lidar_rotation' : R, 
                 'sensor2lidar_translation' : T,
                 'sensor2ego_rotation' : R, 
                 'sensor2ego_translation' : T,
                 'timestamp': 0,
                 'camera_intrinsics' : camera_matrix,
                 
                 
        }
       
       cam_infos[cam]=cam_info
    
    TLidar=np.zeros((3,),dtype=np.float64)
    RLidar=np.eye(3,dtype=np.float64)
    print(cam_info)
    i=0
    for data_info in infoLidarInput:
        #if i>6000:
        #    break
        i=i+1
        if not data_info['valid']:
            continue
        lidar_path=os.path.join(root_path,data_info['file'])
        Rg=np.asarray(data_info['lidar2globalRotation'])
        Tg=np.asarray(data_info['lidar2globalTranslation']).squeeze()                      
        info = {
            "lidar_path": lidar_path,
            "token": data_info['id'],
            "sweeps": [],
            "cams": dict(),
            "lidar2ego_translation": TLidar,
            "lidar2ego_rotation": RLidar,
            "ego2global_translation": Tg,
            "ego2global_rotation": Rg,
            "timestamp": data_info["timestamp"],    
            "sequence": val_seq,
            "frame_id": str(data_info['id']),        
        }
       
        for cam in camera_types:
            cam_token = data_info['camerasIds'][cam]    
            cam_info_ = cam_infos[cam].copy()
            file=os.path.join(root_path,infoCamsInput[cam]['data'][cam_token]['file'])
            cam_info_['data_path']=file
            print(file)
            cam_info_['timestamp']=infoCamsInput[cam]['data'][cam_token]['timestamp']
            info["cams"].update({cam: cam_info_})
        
        sweeps = []
        idx_s=1
        while len(sweeps) < max_sweeps and idx_s<len(data_info['sweeps']):
            
            infosSweep=data_info['sweeps'][idx_s]
            file=os.path.join(root_path,infosSweep['file'])
            R=np.asarray(infosSweep['sensor2LidarRotation'])
            T=np.asarray(infosSweep['sensor2LidarTranslation'])
            sweep = {
                 'data_path': file,
                 'token' : infosSweep['id'],
                 'type': 'lidar',
                 'sensor2lidar_rotation' : R, 
                 'sensor2lidar_translation' : T,
                 'timestamp': infosSweep['timestamp'],
                 
            }
            sweeps.append(sweep)
            idx_s+=1

        info["sweeps"] = sweeps[::-1]
       
        locs = np.array([[8.0,0.0,0.0]]).reshape(1, 3)
        dims = np.array([[1.0,2.0,1.0]]).reshape(1, 3) # wlh
        rots = np.array([[0.0]]).reshape(1, 1)
        velocity = np.array([[0.0,0.0]]).reshape(1, 2)
        valid_flag = np.array([True],dtype=bool).reshape(1)             
        names = ['car']
        names = np.array(names)
        
        gt_boxes = np.concatenate([locs, dims, rots], axis=1)
            
        info['gt_boxes'] = gt_boxes
        info['gt_names'] = names
        info['gt_velocity'] = velocity.reshape(-1, 2)
        info['num_lidar_pts'] = np.ones(locs.shape[0],dtype=np.int)*30
        info['num_radar_pts'] = np.ones(locs.shape[0],dtype=np.int)*30
        info['valid_flag'] = valid_flag
        info['location'] = "France"
        psainfos.append(info)
    
    return psainfos

if __name__ == '__main__':

    root_data_path='/mnt/data/VisionDataBases/PSA/DatabaseCamLidar'
    seq_paths = ['20220217_110707_Rec_JLAB09'] # If more than 1 sequence, data will be merged to the final pkl files 
   
    create_psa_infos(root_data_path,[],seq_paths,'psa3dLidarOneCamera_onestepsweep')

