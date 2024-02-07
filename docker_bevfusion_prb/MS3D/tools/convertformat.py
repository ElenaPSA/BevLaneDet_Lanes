
import sys

import torch

sys.path.append('../')
import argparse
import pickle as pkl
from pathlib import Path

import numpy as np
import yaml
from pcdet.ops.iou3d_nms import iou3d_nms_utils
from pcdet.utils import box_fusion_utils, ms3d_utils
from tqdm import tqdm


def nms(boxes, score, thresh=0.05):
    boxs_gpu = torch.from_numpy(boxes.astype(np.float32) ).cuda()
    scores_gpu = torch.from_numpy(score.astype(np.float32) ).cuda()

    nms_inds = iou3d_nms_utils.nms_gpu(boxs_gpu, scores_gpu, thresh=thresh)
    nms_mask = np.zeros(boxs_gpu.shape[0], dtype=bool)
    nms_mask[nms_inds[0].cpu().numpy()] = 1
    return nms_mask            

def load_src_paths_pkl(src_paths_pkl):
    # with open(src_paths_txt, 'r') as f:
    #     pkl_pths = [line.split('\n')[0] for line in f.readlines()]
    pkl_pths=[src_paths_pkl]
    det_annos = {}
    det_annos['det_cls_weights'] = {}
    for idx, pkl_pth_w in enumerate(pkl_pths):
        split_pth_w = pkl_pth_w.split(',')
        pkl_pth = split_pth_w[0]        
       
        det_cls_weights = np.ones(3, dtype=np.int32) # hardcoded for 3 supercategories (should still work if only vehicle class for inference)

        with open(pkl_pth, 'rb') as f:
            if not Path(pkl_pth).is_absolute():
                pkl_pth = Path(pkl_pth).resolve()     
            det_annos[pkl_pth] = pkl.load(f)
            det_annos['det_cls_weights'][pkl_pth] = det_cls_weights
    return det_annos

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='arg parser')                   
    parser.add_argument('--ps_cfg', type=str, help='cfg file with MS3D parameters')
    parser.add_argument('--save_dir', type=str, default=None, help='Overwrite save dir in the cfg file')
    parser.add_argument('--det_pkl', type=str, default=None, required=True)
    args = parser.parse_args()
    
    ms3d_configs = yaml.load(open(args.ps_cfg,'r'), Loader=yaml.Loader)

    
    # Load detection sets
    det_annos = load_src_paths_pkl(args.det_pkl)
 
    detection_sets = box_fusion_utils.get_detection_sets(det_annos, score_th=0.01)
   
     # Get class specific config
    cls_config = {}
    for enum, cls in enumerate(box_fusion_utils.SUPERCATEGORIES): # Super categories for MS3D compatibility across datasets
        if cls in cls_config.keys():
            continue
        cls_config[cls] = {}
        cls_config[cls]['cls_id'] = enum+1 # in OpenPCDet, cls_ids enumerate fr
        cls_config[cls]['nms'] = ms3d_configs['ENSEMBLE_KBF']['NMS'][enum]
        cls_config[cls]['neg_th'] = ms3d_configs['PS_SCORE_TH']['NEG_TH'][enum]

    ps_dict = {}

    for frame_boxes in tqdm(detection_sets, total=len(detection_sets), desc='get initial ps labels'):

        boxes_lidar = np.hstack([frame_boxes['boxes_lidar'],
                                 frame_boxes['class_ids'][...,np.newaxis],
                                 frame_boxes['ori_class_ids'][...,np.newaxis],
                                 frame_boxes['score'][...,np.newaxis]])

    
        ps_label_nms = []
        for class_name in box_fusion_utils.SUPERCATEGORIES:
            cls_mask = (frame_boxes['names'] == class_name)
           
            cls_boxes = boxes_lidar[cls_mask]
            
            if cls_boxes.shape[0] == 0:
                continue
            score_mask = cls_boxes[:,9] > cls_config[class_name]['neg_th'] 
            cls_boxes = cls_boxes[score_mask]

            if cls_boxes.shape[0] == 0:
                continue
            nms_mask = nms(cls_boxes[:,:7], cls_boxes[:,9], thresh=cls_config[class_name]['nms'])
            cls_boxes = cls_boxes[nms_mask]   
            ps_label_nms.extend(cls_boxes)


        if ps_label_nms:
            ps_label_nms = np.array(ps_label_nms)
        else:
            ps_label_nms = np.empty((0,10))
            
        pred_boxes = ps_label_nms[:,:7]
        pred_labels = ps_label_nms[:,7]
        pred_ori_labels = ps_label_nms[:,8]

        pred_scores = ps_label_nms[:,9]
        gt_box = np.concatenate((pred_boxes,
                                pred_labels.reshape(-1, 1),
                                pred_ori_labels.reshape(-1, 1),
                                pred_scores.reshape(-1, 1)), axis=1)    
        
        # Currently we only store pseudo label information for each frame
        gt_infos = {
            'gt_boxes': gt_box,
        } 
        ps_dict[frame_boxes['frame_id']] = gt_infos

   
    save_dir = ms3d_configs['SAVE_DIR'] if args.save_dir is None else args.save_dir
    ms3d_utils.save_data(ps_dict, save_dir, name=f"initial_pseudo_labels.pkl")
    print(f"saved: initial_pseudo_labels.pkl\n")