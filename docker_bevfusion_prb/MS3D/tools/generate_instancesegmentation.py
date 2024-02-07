
import torch #not used but prevents a bug 
import sys
sys.path.append('../')
from pcdet.utils import ms3d_utils
import argparse
from pcdet.config import cfg, cfg_from_yaml_file
import yaml
from pathlib import Path
from datatools.create_segmentationmasks_m2f import create_infer_model
import cv2
import os 
from tqdm import tqdm
import json

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='arg parser')                   
    parser.add_argument('--ps_cfg', type=str, help='cfg file with MS3D parameters')
    parser.add_argument('--confidence', type=float, default=0.4,help='threslod on instance segmentation confidence')
    parser.add_argument('--modelpath', default="../../datatools/pretrained_models/swin_tiny_02.pt")  
    args = parser.parse_args()


    backbone_size = args.modelpath.split('/')[-1].split('_')[1]
    ms3d_configs = yaml.load(open(args.ps_cfg,'r'), Loader=yaml.Loader)
    cfg_from_yaml_file(ms3d_configs["DATA_CONFIG_PATH"], cfg)
    dataset = ms3d_utils.load_dataset(cfg, split='train')

    camera=ms3d_configs["CAMERA"]
    model=create_infer_model(backbone_size,args.modelpath,args.confidence)
    outdir=ms3d_configs["MASK_DIR"]
    if not os.path.exists(outdir):
        os.makedirs(outdir, exist_ok=True)
    for i in tqdm(range(len(dataset.infos))):
        seq_id = dataset.infos[i]['sequence']
        frame_id = dataset.infos[i]['frame_id']
        imagepath=dataset.infos[i]['cams'][camera]['data_path']
        outfilename_png=outdir+'/{}_{}.png'.format(seq_id,frame_id)
        outfilename_json=outdir+'/{}_{}.json'.format(seq_id,frame_id)
        if os.path.exists(outfilename_json):
            continue
        g_mask, classes_dict, img = model.segment_mask(imagepath)
        cv2.imwrite(outfilename_png, g_mask)
        with open(outfilename_json, 'w') as file:
            json.dump(classes_dict, file)
        