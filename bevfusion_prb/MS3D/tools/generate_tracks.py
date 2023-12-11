"""
MS3D Step 2

DESCRIPTION:
    Generate tracks for the fused detection set. Saves a pkl file containing a dictionary where
    each key is an object's track_id

EXAMPLES:
    python generate_tracks.py --ps_cfg /MS3D/tools/cfgs/target_nuscenes/ms3d_ps_config_rnd1.yaml --cls_id 1
    python generate_tracks.py --ps_cfg /MS3D/tools/cfgs/target_nuscenes/ms3d_ps_config_rnd1.yaml --cls_id 1 --static_veh
    python generate_tracks.py --ps_cfg /MS3D/tools/cfgs/target_nuscenes/ms3d_ps_config_rnd1.yaml --cls_id 2
"""
import sys

sys.path.append('../')
import argparse
import pickle
from pathlib import Path

import numpy as np
import yaml
from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.utils import ms3d_utils, tracker_utils

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='arg parser')                       
    parser.add_argument('--ps_cfg', type=str, help='cfg file with MS3D parameters')
    parser.add_argument('--cls_id', type=int, help='1: vehicle, 2: pedestrian, 3: cyclist')
    parser.add_argument('--static_veh', action='store_true', default=False)
    parser.add_argument('--save_name', type=str, default=None, help='overwrite default save name')

    args = parser.parse_args()

    ms3d_configs = yaml.load(open(args.ps_cfg,'r'), Loader=yaml.Loader)
    cfg_from_yaml_file(ms3d_configs["DATA_CONFIG_PATH"], cfg)
    dataset = ms3d_utils.load_dataset(cfg, split='train')

    ps_dict_pth = Path(ms3d_configs["SAVE_DIR"]) / f'initial_pseudo_labels.pkl'
    local_tracking = 'LOCAL' in ms3d_configs['TRACKING'].keys() and ms3d_configs['TRACKING']['LOCAL']
    use_imm = 'IMM' in ms3d_configs['TRACKING'].keys() and ms3d_configs['TRACKING']['IMM']
    #ps_dict_pth = '../../bevfusion/viz_aimotive/predictions.pkl'
    with open(ps_dict_pth, 'rb') as f:
        ps_dict = pickle.load(f)
   
    if args.cls_id == 1:        
        if args.static_veh:
            trk_cfg = tracker_utils.prepare_track_cfg(ms3d_configs['TRACKING']['VEH_STATIC'],use_imm=use_imm)
            save_fname = f"tracks_world_veh_static.pkl"                    
        else:
            trk_cfg = tracker_utils.prepare_track_cfg(ms3d_configs['TRACKING']['VEH_ALL'],use_imm=use_imm)
            save_fname = f"tracks_world_veh_all.pkl"
    elif args.cls_id == 2:
        trk_cfg = tracker_utils.prepare_track_cfg(ms3d_configs['TRACKING']['PEDESTRIAN'],use_imm=False)
        save_fname = f"tracks_world_ped.pkl"
    elif args.cls_id == 3:        
        if args.static_veh:
            trk_cfg = tracker_utils.prepare_track_cfg(ms3d_configs['TRACKING']['BIC_STATIC'],use_imm=use_imm)
            save_fname = f"tracks_world_bic_static.pkl"                    
        else:
            trk_cfg = tracker_utils.prepare_track_cfg(ms3d_configs['TRACKING']['BIC_ALL'],use_imm=use_imm)
            save_fname = f"tracks_world_bic_all.pkl"
    else:
        print('Only support 3 classes at the moment (1: vehicle, 2: pedestrian,3: bicycle)')
        raise NotImplementedError    
    
    save_fname = args.save_name if args.save_name is not None else save_fname
    
    tracks_world = tracker_utils.get_tracklets(dataset, ps_dict, trk_cfg,cls_id=args.cls_id,local_tracking=local_tracking)
   
    print(ms3d_configs["SAVE_DIR"])
    ms3d_utils.save_data(tracks_world, ms3d_configs["SAVE_DIR"], name=save_fname)
    print(f"saved: {save_fname}\n")