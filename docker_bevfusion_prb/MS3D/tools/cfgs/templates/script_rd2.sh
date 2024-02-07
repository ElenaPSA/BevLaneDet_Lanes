#!/bin/bash
python generate_instancesegmentation.py --ps_cfg CONFIGFILE 
python convertformat.py --ps_cfg CONFIGFILE --det_pkl DET_PKL

python generate_tracks.py --ps_cfg CONFIGFILE --cls_id 1
python generate_tracks.py --ps_cfg CONFIGFILE --cls_id 1 --static_veh
python generate_tracks.py --ps_cfg CONFIGFILE --cls_id 2
python generate_tracks.py --ps_cfg CONFIGFILE --cls_id 3
python generate_tracks.py --ps_cfg CONFIGFILE --cls_id 3 --static_veh

python smooth_tracks.py --ps_cfg CONFIGFILE --cls_id 1
python smooth_tracks.py --ps_cfg CONFIGFILE --cls_id 1 --static_veh
python smooth_tracks.py --ps_cfg CONFIGFILE --cls_id 3 
python smooth_tracks.py --ps_cfg CONFIGFILE --cls_id 3 --static_veh

python temporal_refinement.py --ps_cfg CONFIGFILE

python correct_gt.py --ps_cfg CONFIGFILE --ps_pkl PKLFILE --outviz OUTVIZ

##for visu : 
# python visualize_bev_all.py --cfg_file=cfgs/dataset_configs/stellantis_dataset_jtlab_sync.yaml --ps_pkl=cfgs/target_stellantis_jtlab_sync/label_generation/round2/ps_labels/initial_pseudo_labels.pkl --tracks_pkl=cfgs/target_stellantis_jtlab_sync/label_generation/round2/ps_labels/tracks_world_veh_all.pkl
