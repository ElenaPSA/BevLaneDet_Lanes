import argparse
import copy
import json
import os
import sys

import mmcv
import numpy as np
import torch
from mmcv import Config
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import load_checkpoint, wrap_fp16_model
from mmdet3d.core import LiDARInstance3DBoxes
from mmdet3d.core.utils import (export_results_aimotive, export_results_dict,
                                visualize_camera, visualize_lidar,
                                visualize_map)
from mmdet3d.datasets import build_dataloader, build_dataset
from mmdet3d.models import build_model
from torchpack import distributed as dist
from torchpack.utils.config import configs


def tqdm(iterable, **kwargs):
    from tqdm import tqdm

    return tqdm(iterable, **kwargs, file=sys.stdout)


def recursive_eval(obj, globals=None):
    if globals is None:
        globals = copy.deepcopy(obj)

    if isinstance(obj, dict):
        for key in obj:
            obj[key] = recursive_eval(obj[key], globals)
    elif isinstance(obj, list):
        for k, val in enumerate(obj):
            obj[k] = recursive_eval(val, globals)
    elif isinstance(obj, str) and obj.startswith("${") and obj.endswith("}"):
        obj = eval(obj[2:-1], globals)
        obj = recursive_eval(obj, globals)

    return obj

def load_refined_boxes(path):

    boxes=[]
    labels=[]
    import pickle
    ps_dict={}
    out_dict={}
    with open(path,'rb') as f:
        ps_dict = pickle.load(f)

    for k,v in ps_dict.items():
        boxes=v['gt_boxes'][:,0:7]
        boxes[:,6]=-boxes[:,6]
        boxes[:,2]-=boxes[:,5]/2
        labels=(v['gt_boxes'][:,8]).astype(np.int32)
        out_dict[k]={}
        out_dict[k]['boxes']=LiDARInstance3DBoxes(torch.from_numpy(boxes),box_dim=7)
        out_dict[k]['labels']=torch.from_numpy(labels)

    return out_dict

def main() -> None:
    dist.init()

    parser = argparse.ArgumentParser()
    parser.add_argument("config", metavar="FILE")
    parser.add_argument("--mode", type=str, default="pred", choices=["gt", "pred"])
    parser.add_argument("--checkpoint", type=str, default=None)

    parser.add_argument("--refined", type=str, default=None)

    parser.add_argument(
        "--split", type=str, default="val", choices=["train", "val", "test"]
    )
    parser.add_argument("--bbox-classes", nargs="+", type=int, default=None)
    parser.add_argument("--bbox-score", type=float, default=None)
    parser.add_argument("--map-score", type=float, default=0.5)
    parser.add_argument("--out-dir", type=str, default="viz")
    parser.add_argument("--maxImages", type=int, default=None)
    parser.add_argument("--noviz", action="store_true")
    parser.add_argument("--sweep", type=int,default=None)
    args, opts = parser.parse_known_args()

    
    configs.load(args.config, recursive=True)
    print(opts)
    configs.update(opts)
  
    cfg = Config(recursive_eval(configs), filename=args.config)

    pipeline_cfg=cfg['test_pipeline']
    print("sweep ",args.sweep)
    if args.sweep !=None:
        for task_cfg in pipeline_cfg:
            if task_cfg['type']=='LoadPointsFromMultiSweeps':
                task_cfg['sweeps_num']=args.sweep
                print(task_cfg)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.set_device(dist.local_rank())

    # build the dataloader
    dataset = build_dataset(cfg.data[args.split])
    dataflow = build_dataloader(
        dataset,
        samples_per_gpu=1,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=True,
        shuffle=False,
    )

    # build the model and load checkpoint
    if args.mode == "pred":
        model = build_model(cfg.model)
        load_checkpoint(model, args.checkpoint, map_location="cpu")
        
        # model = MMDistributedDataParallel(
        #     model.cuda(),
        #     device_ids=[torch.cuda.current_device()],
        #     broadcast_buffers=False,
        # )
        model = MMDataParallel(model, device_ids=[0])
        model.eval()
    i = 0
    gt_boxes_ms3d = []

    if args.refined is not None:
        boxes_refined=load_refined_boxes(args.refined)
       
    for data in tqdm(dataflow):
        metas = data["metas"].data[0][0]
        #print(metas['frame_id'])
        
        frameid = metas['frame_id']
        
        name = "{}".format(frameid)
        sequence = "default"
        if args.maxImages is not None and i >= args.maxImages:
            break
        i+=1
        bboxes_r=None
        labels_r=None
        if args.refined is not None:
            boxes_refined_frame=boxes_refined[frameid]
            bboxes_r=boxes_refined_frame['boxes']
            labels_r=boxes_refined_frame['labels']
        if "sequence" in metas:
            sequence = metas["sequence"]
        pose = metas["ego2global"]
        
        # image=data['img'].data[0][0].cpu().numpy()
        # print(image.shape)
        # image=image[0].transpose(1,2,0)
        
        
        # std = [0.229, 0.224, 0.225]
        # image = ((image * std) + np.array([0.485, 0.456, 0.406]))*255
        # image=image.astype(np.uint8)
        # import cv2
        # image=cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
       # cv2.imwrite('./test.png', image)
        if args.mode == "pred":
            with torch.inference_mode():
                outputs = model(**data)
       
        if args.mode == "gt" and "gt_bboxes_3d" in data:
            bboxes = data["gt_bboxes_3d"].data[0][0].tensor.numpy()
            labels = data["gt_labels_3d"].data[0][0].numpy()
            
            if args.bbox_classes is not None:
                indices = np.isin(labels, args.bbox_classes)
                bboxes = bboxes[indices]
                labels = labels[indices]

            bboxes[..., 2] -= bboxes[..., 5] / 2
            bboxes = LiDARInstance3DBoxes(bboxes, box_dim=9)
        elif args.mode == "pred" and "boxes_3d" in outputs[0]:
            
            bboxes = outputs[0]["boxes_3d"].tensor.numpy()
            scores = outputs[0]["scores_3d"].numpy()
            labels = outputs[0]["labels_3d"].numpy()
            
            if args.bbox_classes is not None:
                indices = np.isin(labels, args.bbox_classes)
                bboxes = bboxes[indices]
                scores = scores[indices]
                labels = labels[indices]

            if args.bbox_score is not None:
                indices = scores >= args.bbox_score
                bboxes = bboxes[indices]
                scores = scores[indices]
                labels = labels[indices]
          
            bboxes[..., 2] -= bboxes[..., 5] / 2
            bboxes = LiDARInstance3DBoxes(bboxes, box_dim=9)
        else:
            bboxes = None
            labels = None

        if args.mode == "gt" and "gt_masks_bev" in data:
            masks = data["gt_masks_bev"].data[0].numpy()
            masks = masks.astype(np.bool)
        elif args.mode == "pred" and "masks_bev" in outputs[0]:
            masks = outputs[0]["masks_bev"].numpy()
            masks = masks >= args.map_score
        else:
            masks = None
       
        if "img" in data and not args.noviz:
            for k, image_path in enumerate(metas["filename"]):
                image = mmcv.imread(image_path)
                visualize_camera(
                    os.path.join(args.out_dir, sequence, f"camera-{k}", f"{name}.png"),
                    image,
                    bboxes=bboxes,
                    labels=labels,
                    bboxes_r=bboxes_r,
                    labels_r=labels_r,
                    transform=metas["lidar2image"][k],
                    classes=cfg.object_classes,
                )

        if "points" in data and not args.noviz:
            lidar = data["points"].data[0][0].numpy()
            xlim = [cfg.point_cloud_range[d] for d in [0, 3]]
            ylim = [cfg.point_cloud_range[d] for d in [1, 4]]

            visualize_lidar(
                os.path.join(args.out_dir, sequence, "lidar", f"{name}.png"),
                lidar,
                bboxes=bboxes,
                labels=labels,
                xlim=[cfg.point_cloud_range[d] for d in [0, 3]],
                ylim=[cfg.point_cloud_range[d] for d in [1, 4]],
                classes=cfg.object_classes,
            )

        if args.mode == "pred":
            export_results_aimotive(
                os.path.join(args.out_dir, sequence, "predictions", f"{name}.json"),
                bboxes,
                labels,
                scores,
                cfg.object_classes,
                frameid,
                metas["timestamp"],
            )
          
            gt_boxes_ms3d = export_results_dict(
                gt_boxes_ms3d,
                bboxes,
                labels,
                scores,
                cfg.object_classes,
                sequence,
                frameid,
                metas["timestamp"],
                pose
            )
        
        if masks is not None:
            visualize_map(
                os.path.join(args.out_dir, "map", f"{name}.png"),
                masks,
                classes=cfg.map_classes,
            )

    
   
 
    pkl_dict_path=os.path.join(args.out_dir, "predictions.pkl")
    mmcv.dump(gt_boxes_ms3d, pkl_dict_path)

if __name__ == "__main__":
    main()
