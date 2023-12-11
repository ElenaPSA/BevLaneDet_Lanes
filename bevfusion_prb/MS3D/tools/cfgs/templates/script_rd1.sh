#!/bin/bash
python generate_instancesegmentation.py --ps_cfg CONFIGFILE 
python ensemble_kbf.py --ps_cfg CONFIGFILE

python generate_tracks.py --ps_cfg CONFIGFILE --cls_id 1
python generate_tracks.py --ps_cfg CONFIGFILE --cls_id 1 --static_veh
python generate_tracks.py --ps_cfg CONFIGFILE --cls_id 2
python generate_tracks.py --ps_cfg CONFIGFILE --cls_id 3
python generate_tracks.py --ps_cfg CONFIGFILE --cls_id 3 --static_veh

python temporal_refinement.py --ps_cfg CONFIGFILE

python correct_gt.py --ps_cfg CONFIGFILE --ps_pkl PKLFILE --outviz OUTVIZ