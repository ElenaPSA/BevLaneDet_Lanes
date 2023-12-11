
import sys

sys.path.append('../')
import argparse
import os
import pickle
from pathlib import Path

import yaml
from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.utils import ms3d_utils, tracker_utils

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='arg parser')                       
    parser.add_argument('--sequence', type=str, required=True,help='sequence name')
    parser.add_argument('--inference_folder', type=str, required=True,help='folder of inference results')
    parser.add_argument('--data_folder', type=str, required=True, help='folder of dataset pkl')
    parser.add_argument('--round', type=int, required=True, help='round number')
    args = parser.parse_args()
    template_dataset_cfg='./cfgs/templates/stellantis_dataset_jtlab_template.yaml'

    if args.round==1:
        template_ps_cfg='./cfgs/templates/ps_config_rd1_template.yaml'
    else:
        template_ps_cfg='./cfgs/templates/ps_config_rd2_template.yaml'

    
    with open(template_dataset_cfg,'r') as f:
        str_cfg=f.read()
   
    str_cfg=str_cfg.replace('DEFAULTNAME','data_{}_infos_val.pkl'.format(args.sequence))
    str_cfg=str_cfg.replace('DEFAULTDATAPATH',args.data_folder)
    
    outname='cfgs/dataset_configs/stellantis_dataset_jtlab_{}.yaml'.format(args.sequence)
    with open(outname,'w') as f:
        f.write(str_cfg)


    with open(template_ps_cfg,'r') as f:
        str_cfg=f.read()

    target_dir='cfgs/target_{}'.format(args.sequence)

    if not os.path.exists(target_dir):
        os.makedirs(target_dir,exist_ok=True)
    
    ps_dir=os.path.join(target_dir,'label_generation/round{}/cfgs'.format(args.round))
    ps_label_dir=os.path.join(target_dir,'label_generation/round{}/ps_labels'.format(args.round))
    scripts_dir=os.path.join(target_dir,'label_generation/round{}/scripts'.format(args.round))
    maskdir=os.path.join(target_dir,'ps_mask')
    os.makedirs(ps_dir,exist_ok=True)
    os.makedirs(ps_label_dir,exist_ok=True)
    os.makedirs(maskdir,exist_ok=True)
    os.makedirs(ps_label_dir,exist_ok=True)
    os.makedirs(scripts_dir,exist_ok=True)
    str_cfg=str_cfg.replace('DEFAULTDETSTXT',ps_dir+'/ensemble_detections.txt')
    str_cfg=str_cfg.replace('DEFAULTSAVE_DIR',ps_label_dir)
    str_cfg=str_cfg.replace('DEFAULTMASK_DIR',maskdir)
    str_cfg=str_cfg.replace('DEFAULTCONFIG_PATH',outname)
   

    outname_psconfig=os.path.join(ps_dir,'ps_config.yaml')
    with open(outname_psconfig,'w') as f:
        f.write(str_cfg)

    if args.round==1:
        list_detections=[os.path.join(args.inference_folder,'normal/{}/predictions.pkl'.format(args.sequence)),
                      os.path.join(args.inference_folder,'reduced/{}/predictions.pkl'.format(args.sequence)),
                      os.path.join(args.inference_folder,'lidar/{}/predictions.pkl'.format(args.sequence)),
                      os.path.join(args.inference_folder,'sweep5/{}/predictions.pkl'.format(args.sequence)),
                      os.path.join(args.inference_folder,'sweep5_reduced/{}/predictions.pkl'.format(args.sequence))]
        outname= os.path.join(ps_dir,'ensemble_detections.txt')             
        with open(outname,'w') as f:
            for det in list_detections:
                f.write(det)
                f.write('\n')
    else:
        det_pkl=os.path.join(args.inference_folder,'{}/predictions.pkl'.format(args.sequence))

    if args.round==1:
        template_script='./cfgs/templates/script_rd1.sh'
    else:
        template_script='./cfgs/templates/script_rd2.sh'

    with open(template_script,'r') as f:
        str_script=f.read()
    str_script=str_script.replace('CONFIGFILE',outname_psconfig)
    str_script=str_script.replace('PKLFILE',os.path.join(ps_label_dir,'final_ps_dict.pkl'))
    str_script=str_script.replace('OUTVIZ','./outviz_{}'.format(args.sequence))
    if args.round>1:
        str_script=str_script.replace('DET_PKL',det_pkl)
    outname=os.path.join(scripts_dir,'run_pseudo_label_pipeline.sh')
    with open(outname,'w') as f:
        f.write(str_script)