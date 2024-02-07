"""
MS3D Step 3 (final)

DESCRIPTION:
    Temporally refines all tracks and detection sets using object characteristics.

EXAMPLES:
    python temporal_refinement.py --ps_cfg /MS3D/tools/cfgs/target_nuscenes/ms3d_ps_config_rnd3_2.yaml 
"""
import sys

import torch  # not used but prevents a bug

sys.path.append('../')
import argparse
from pathlib import Path

import yaml
from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.utils import ms3d_utils


def remove_unused_data(track_dict):
    clean_tracks={}
    for key,val in track_dict.items():
        
        clean_tracks[key]={}
        clean_tracks[key]['boxes'] = val['boxes']
        clean_tracks[key]['frame_id'] = val['frame_id']
       
    return clean_tracks
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='arg parser')                   
    parser.add_argument('--ps_cfg', type=str, help='cfg file with MS3D parameters')
    parser.add_argument('--save_veh_intermediate_tracks', action='store_true', default=False, help='Save static vehicle tracks at each stage: motion refinement, rolling_kde, and propagate boxes')
    args = parser.parse_args()

    ms3d_configs = yaml.load(open(args.ps_cfg,'r'), Loader=yaml.Loader)
    cfg_from_yaml_file(ms3d_configs["DATA_CONFIG_PATH"], cfg)
    dataset = ms3d_utils.load_dataset(cfg, split='train')
    use_smooth='USE_SMOOTH' in ms3d_configs['TEMPORAL_REFINEMENT'] and ms3d_configs['TEMPORAL_REFINEMENT']['USE_SMOOTH']
    print(use_smooth)
    # Load pkls
    ps_pth = Path(ms3d_configs["SAVE_DIR"]) / f'initial_pseudo_labels.pkl'
    if use_smooth:
       tracks_veh_all_pth = Path(ms3d_configs["SAVE_DIR"]) / f'tracks_world_veh_all_smoothed.pkl'
       tracks_veh_static_pth = Path(ms3d_configs["SAVE_DIR"]) / f'tracks_world_veh_static_smoothed.pkl'

       tracks_bic_all_pth = Path(ms3d_configs["SAVE_DIR"]) / f'tracks_world_bic_all_smoothed.pkl'
       tracks_bic_static_pth = Path(ms3d_configs["SAVE_DIR"]) / f'tracks_world_bic_static_smoothed.pkl'

       tracks_ped_pth = Path(ms3d_configs["SAVE_DIR"]) / f'tracks_world_ped.pkl'
    
    else:
        
        tracks_veh_all_pth = Path(ms3d_configs["SAVE_DIR"]) / f'tracks_world_veh_all.pkl'
        tracks_veh_static_pth = Path(ms3d_configs["SAVE_DIR"]) / f'tracks_world_veh_static.pkl'

        tracks_bic_all_pth = Path(ms3d_configs["SAVE_DIR"]) / f'tracks_world_bic_all.pkl'
        tracks_bic_static_pth = Path(ms3d_configs["SAVE_DIR"]) / f'tracks_world_bic_static.pkl'

        tracks_ped_pth = Path(ms3d_configs["SAVE_DIR"]) / f'tracks_world_ped.pkl'
        
    ps_dict = ms3d_utils.load_pkl(ps_pth)

    tracks_veh_all = remove_unused_data(ms3d_utils.load_pkl(tracks_veh_all_pth))
    tracks_veh_static = remove_unused_data(ms3d_utils.load_pkl(tracks_veh_static_pth))

    tracks_bic_all = remove_unused_data(ms3d_utils.load_pkl(tracks_bic_all_pth))
    tracks_bic_static = remove_unused_data(ms3d_utils.load_pkl(tracks_bic_static_pth))

    tracks_ped = remove_unused_data(ms3d_utils.load_pkl(tracks_ped_pth))


    # Get vehicle labels
    print('Refining vehicle labels')
    tracks_veh_all, tracks_veh_static = ms3d_utils.refine_veh_labels(dataset,list(ps_dict.keys()),
                                                                    tracks_veh_all, 
                                                                    tracks_veh_static, 
                                                                    static_trk_score_th=ms3d_configs['TRACKING']['VEH_STATIC']['RUNNING']['SCORE_TH'],
                                                                    veh_pos_th=ms3d_configs['PS_SCORE_TH']['POS_TH'][0],
                                                                    refine_cfg=ms3d_configs['TEMPORAL_REFINEMENT'],
                                                                    save_dir=None,cls_id=1)

       # Get bicycle labels
    print('Refining bicycle labels')
    tracks_bic_all, tracks_bic_static = ms3d_utils.refine_veh_labels(dataset,list(ps_dict.keys()),
                                                                     tracks_bic_all, 
                                                                     tracks_bic_static, 
                                                                     static_trk_score_th=ms3d_configs['TRACKING']['BIC_STATIC']['RUNNING']['SCORE_TH'],
                                                                     veh_pos_th=ms3d_configs['PS_SCORE_TH']['POS_TH'][2],
                                                                     refine_cfg=ms3d_configs['TEMPORAL_REFINEMENT'],
                                                                     save_dir=None,cls_id=3)


    
    # Get pedestrian labels
    print('Refining pedestrian labels')
    tracks_ped = ms3d_utils.refine_ped_labels(tracks_ped, 
                                               ped_pos_th=ms3d_configs['PS_SCORE_TH']['POS_TH'][1],
                                               track_filtering_cfg=ms3d_configs['TEMPORAL_REFINEMENT']['TRACK_FILTERING'])

    # # Combine pseudo-labels for each class and filter with NMS
    print('Combining pseudo-labels for each class')

    final_ps_dict = ms3d_utils.update_ps(dataset, ps_dict, tracks_veh_all, tracks_veh_static,tracks_bic_all, tracks_bic_static, tracks_ped, 
              veh_pos_th=ms3d_configs['PS_SCORE_TH']['POS_TH'][0], 
              veh_nms_th=0.05, ped_nms_th=0.5, 
              frame2box_key_static='frameid_to_propboxes', 
              frame2box_key='frameid_to_box', frame_ids=list(ps_dict.keys()))

   # final_ps_dict = ms3d_utils.select_ps_by_th(final_ps_dict, ms3d_configs['PS_SCORE_TH']['POS_TH'])
    ms3d_utils.save_data(final_ps_dict, str(Path(ms3d_configs["SAVE_DIR"])), name="final_ps_dict.pkl")

    print('Finished generating pseudo-labels')