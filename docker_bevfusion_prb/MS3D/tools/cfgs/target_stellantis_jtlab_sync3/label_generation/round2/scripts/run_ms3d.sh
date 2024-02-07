#!/bin/bash
python convertformat.py --ps_cfg cfgs/target_stellantis_jtlab_sync3/label_generation/round2/cfgs/ps_config.yaml --det_pkl=../../bevfusion/results_bidir3/predictions.pkl

python generate_tracks.py --ps_cfg cfgs/target_stellantis_jtlab_sync3/label_generation/round2/cfgs/ps_config.yaml --cls_id 1
python generate_tracks.py --ps_cfg cfgs/target_stellantis_jtlab_sync3/label_generation/round2/cfgs/ps_config.yaml --cls_id 1 --static_veh
python generate_tracks.py --ps_cfg cfgs/target_stellantis_jtlab_sync3/label_generation/round2/cfgs/ps_config.yaml --cls_id 2
python generate_tracks.py --ps_cfg cfgs/target_stellantis_jtlab_sync3/label_generation/round2/cfgs/ps_config.yaml --cls_id 3
python generate_tracks.py --ps_cfg cfgs/target_stellantis_jtlab_sync3/label_generation/round2/cfgs/ps_config.yaml --cls_id 3 --static_veh

python smooth_tracks.py --ps_cfg cfgs/target_stellantis_jtlab_sync3/label_generation/round2/cfgs/ps_config.yaml --cls_id 1
python smooth_tracks.py --ps_cfg cfgs/target_stellantis_jtlab_sync3/label_generation/round2/cfgs/ps_config.yaml --cls_id 1 --static_veh
python smooth_tracks.py --ps_cfg cfgs/target_stellantis_jtlab_sync3/label_generation/round2/cfgs/ps_config.yaml --cls_id 3 
python smooth_tracks.py --ps_cfg cfgs/target_stellantis_jtlab_sync3/label_generation/round2/cfgs/ps_config.yaml --cls_id 3 --static_veh

python temporal_refinement.py --ps_cfg cfgs/target_stellantis_jtlab_sync3/label_generation/round2/cfgs/ps_config.yaml 

python correct_gt.py --ps_cfg cfgs/target_stellantis_jtlab_sync3/label_generation/round2/cfgs/ps_config.yaml --ps_pkl=cfgs/target_stellantis_jtlab_sync3/label_generation/round2/ps_labels/final_ps_dict.pkl --outviz=outviz_corrected3

##for visu : 
# python visualize_bev_all.py --cfg_file=cfgs/dataset_configs/stellantis_dataset_jtlab_sync.yaml --ps_pkl=cfgs/target_stellantis_jtlab_sync/label_generation/round2/ps_labels/initial_pseudo_labels.pkl --tracks_pkl=cfgs/target_stellantis_jtlab_sync/label_generation/round2/ps_labels/tracks_world_veh_all.pkl
