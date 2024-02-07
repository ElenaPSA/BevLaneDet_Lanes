import copy
import pickle
from pathlib import Path

import numpy as np
from tqdm import tqdm

from ...ops.roiaware_pool3d import roiaware_pool3d_utils
from ...utils import box_utils, common_utils
from ..dataset import DatasetTemplate


class StellantisDataset(DatasetTemplate):
    def __init__(self, dataset_cfg, class_names, training=True, root_path=None, logger=None):
       
        root_path = (root_path if root_path is not None else Path(dataset_cfg.DATA_PATH)) 
        super().__init__(
            dataset_cfg=dataset_cfg, class_names=class_names, training=training, root_path=root_path, logger=logger
        )
        self.infos = []
        self.split = dataset_cfg.DATA_SPLIT['train'] if training else dataset_cfg.DATA_SPLIT['test']
        self.frameid_to_idx = {}
        self.custom_train_scenes = None
     

        self.seq_name_to_infos = self.include_stellantis_data()
       

    def include_stellantis_data(self):
        
        if self.logger is not None:
            self.logger.info('Loading Stellantis dataset')
        stellantis_infos = []
        for info_path in self.dataset_cfg.INFO_PATH[self.split]:
            info_path = self.root_path / info_path
            if not info_path.exists():
                continue
            with open(info_path, 'rb') as f:
                pkl_dict = pickle.load(f)
                stellantis_infos.extend(pkl_dict['infos'])
        
        self.infos.extend(stellantis_infos)

        prev_seq = ''
        for idx, data in enumerate(self.infos):
            cur_seq = self.infos[idx]['sequence']
            if cur_seq != prev_seq:
                prev_seq = cur_seq
                sample_idx = 0            
            else:
                sample_idx += 1
            self.infos[idx]['sample_idx']=sample_idx
            self.frameid_to_idx[data['frame_id']] = idx

        seq_name_to_infos = {}
        seq_name_to_len = {}
        for i in range(len(self.infos)):
            seq_id = self.infos[i]['sequence']
            if seq_id not in seq_name_to_infos.keys():
                seq_name_to_infos[seq_id] = []            
            seq_name_to_infos[seq_id].append(self.infos[i])
            seq_name_to_len[seq_id] = len(self.infos[i])

        if self.logger is not None:
            self.logger.info('Total samples for Stellantis dataset: %d' % (len(stellantis_infos)))

        if self.dataset_cfg.get('SAMPLED_INTERVAL', None) and \
            self.dataset_cfg.SAMPLED_INTERVAL[self.mode] > 1:
            sampled_nusc_infos = []
            for k in range(0, len(self.infos), self.dataset_cfg.SAMPLED_INTERVAL[self.mode]):
                sampled_nusc_infos.append(self.infos[k])
            self.infos = sampled_nusc_infos

            seq_name_to_len = {}
            for i in range(len(self.infos)):
                seq_id = self.infos[i]['scene_name']
                if seq_id not in seq_name_to_infos.keys():
                    seq_name_to_infos[seq_id] = 0
                seq_name_to_infos[seq_id] += 1

            if self.logger is not None:
                self.logger.info('Total sampled samples for Stellantis dataset: %d' % len(self.infos))        

        self.seq_name_to_len = seq_name_to_len
        return seq_name_to_infos

  

    @staticmethod
    def remove_ego_points(points, center_radius=1.0):
        mask = ~((np.abs(points[:, 0]) < center_radius) & (np.abs(points[:, 1]) < center_radius))
        return points[mask]

    def get_sweep(self, sweep_info,curr_time):
        lidar_path = self.root_path / sweep_info['data_path']
        points_sweep = np.load(lidar_path).reshape([-1, 5])[:, :4]
        points_sweep = self.remove_ego_points(points_sweep).T
      #  print(points_sweep.shape)
        if sweep_info['sensor2lidar_rotation'] is not None:
            num_points = points_sweep.shape[1]
            transform_matrix = np.eye(4).astype(np.float32)
            transform_matrix[:3, :3] = sweep_info['sensor2lidar_rotation']
            transform_matrix[3, :3] = sweep_info['sensor2lidar_translation']
            points_sweep[:3, :] = transform_matrix.dot(np.vstack((points_sweep[:3, :], np.ones(num_points))))[:3, :]

        cur_times = (sweep_info['timestamp']-curr_time)* np.ones((1, points_sweep.shape[1]))
        return points_sweep.T, cur_times.T
   
   
    def get_lidar_with_sweeps(self, index, max_sweeps=1):
        
        info = self.infos[index]
      #  print(info)
        lidar_path = self.root_path / info['lidar_path']
      #  print(lidar_path)
        points=np.load(lidar_path)
       
        points = points[:, :4]
       
        currtime=info['timestamp']
        
        points = self.remove_ego_points(points, center_radius=1.5)
        sweep_points_list = [points]
        sweep_times_list = [np.zeros((points.shape[0], 1))]

        # points_sweep, times_sweep = self.get_sweep(info['sweeps'][-1],currtime)
        # sweep_points_list.append(points_sweep)
        # sweep_times_list.append(times_sweep)
      
        points = np.concatenate(sweep_points_list, axis=0)
        times = np.concatenate(sweep_times_list, axis=0).astype(points.dtype)

        points = np.concatenate((points, times), axis=1)
        return points

    def __len__(self):

        if self._merge_all_iters_to_one_epoch:
            return len(self.infos) * self.total_epochs

        return len(self.infos)

    def __getitem__(self, index):

        if self._merge_all_iters_to_one_epoch:
            index = index % len(self.infos)

        info = copy.deepcopy(self.infos[index])
        points = self.get_lidar_with_sweeps(index, max_sweeps=self.dataset_cfg.MAX_SWEEPS)
        if self.dataset_cfg.get('SHIFT_COOR', None):
            points[:, 0:3] += np.array(self.dataset_cfg.SHIFT_COOR, dtype=np.float32)

        input_dict = {
            'points': points,
            'frame_id': info['frame_id'],
            'metadata': {'token': info['token']}
        }
      #  print('frame_id : ',info['frame_id'])
        if 'gt_boxes' in info:
            if self.dataset_cfg.get('FILTER_MIN_POINTS_IN_GT', False):
                mask = (info['num_lidar_pts'] > self.dataset_cfg.FILTER_MIN_POINTS_IN_GT - 1)
            else:
                mask = None

            input_dict.update({
                'gt_names': info['gt_names'] if mask is None else info['gt_names'][mask],
                'gt_boxes': info['gt_boxes'] if mask is None else info['gt_boxes'][mask]
            })
            if self.dataset_cfg.get('SHIFT_COOR', None):
                input_dict['gt_boxes'][:, 0:3] += self.dataset_cfg.SHIFT_COOR

            if self.dataset_cfg.get('USE_PSEUDO_LABEL', None) and self.training:
                input_dict['gt_boxes'] = None

        if self.dataset_cfg.get('USE_PSEUDO_LABEL', None) and self.training:
            # Remap indices from pseudo-label 1-3 to order of det head classes; pseudo-labels ids are always 1:Vehicle, 2:Pedestrian, 3:Cyclist
            # Make sure DATA_CONFIG_TAR.CLASS_NAMES is same order/length as DATA_CONFIG.CLASS_NAMES (i.e. the pretrained class indices)
            
            psid2clsid = {}
            if 'car' in self.class_names:
                psid2clsid[1] = self.class_names.index('car') + 1
            if 'pedestrian' in self.class_names:
                psid2clsid[2] = self.class_names.index('pedestrian') + 1                
            self.fill_pseudo_labels(input_dict, psid2clsid)

        if self.dataset_cfg.get('SET_NAN_VELOCITY_TO_ZEROS', False) and not self.dataset_cfg.get('USE_PSEUDO_LABEL', None):
            gt_boxes = input_dict['gt_boxes']
            gt_boxes[np.isnan(gt_boxes)] = 0
            input_dict['gt_boxes'] = gt_boxes

        if not self.dataset_cfg.PRED_VELOCITY and 'gt_boxes' in input_dict and not self.dataset_cfg.get('USE_PSEUDO_LABEL', None):
            input_dict['gt_boxes'] = input_dict['gt_boxes'][:, [0, 1, 2, 3, 4, 5, 6]]

        data_dict = self.prepare_data(data_dict=input_dict)

        return data_dict

    

    def evaluation(self, det_annos, class_names, **kwargs):
        if kwargs['eval_metric'] == 'kitti':
            # eval_det_annos = copy.deepcopy(det_annos)
            # eval_gt_annos = copy.deepcopy(self.infos)
            # return self.kitti_eval(eval_det_annos, eval_gt_annos, class_names)
            return False
        elif kwargs['eval_metric'] == 'nuscenes':
            # return self.nuscene_eval(det_annos, class_names, **kwargs)
            return False
        else:
            raise NotImplementedError

    def kitti_eval(self, eval_det_annos, eval_gt_annos, class_names):
        from ..kitti.kitti_object_eval_python import eval as kitti_eval

        map_name_to_kitti = {
            'car': 'Car',
            'truck': 'Car',
            'bus': 'Car',
            'motorcycle': 'Cyclist',
            'bicycle': 'Cyclist',
            'pedestrian': 'Pedestrian'
        }
        
        def transform_to_kitti_format(annos, info_with_fakelidar=False, is_gt=False):
            for anno in annos:
                if 'name' not in anno:
                    anno['name'] = anno['gt_names']
                    anno.pop('gt_names')

                for k in range(anno['name'].shape[0]):
                    if anno['name'][k] in map_name_to_kitti:
                        anno['name'][k] = map_name_to_kitti[anno['name'][k]]
                    else:
                        anno['name'][k] = 'Person_sitting'

                if 'boxes_lidar' in anno:
                    gt_boxes_lidar = anno['boxes_lidar'].copy()
                else:
                    gt_boxes_lidar = anno['gt_boxes'].copy()

                # filter by fov
                if is_gt and self.dataset_cfg.get('GT_FILTER', None):
                    if self.dataset_cfg.GT_FILTER.get('FOV_FILTER', None):
                        fov_gt_flag = self.extract_fov_gt(
                            gt_boxes_lidar, self.dataset_cfg['FOV_DEGREE'], self.dataset_cfg['FOV_ANGLE']
                        )
                        gt_boxes_lidar = gt_boxes_lidar[fov_gt_flag]
                        anno['name'] = anno['name'][fov_gt_flag]

                anno['bbox'] = np.zeros((len(anno['name']), 4))
                anno['bbox'][:, 2:4] = 50  # [0, 0, 50, 50]
                anno['truncated'] = np.zeros(len(anno['name']))
                anno['occluded'] = np.zeros(len(anno['name']))

                if len(gt_boxes_lidar) > 0:
                    if info_with_fakelidar:
                        gt_boxes_lidar = box_utils.boxes3d_kitti_fakelidar_to_lidar(gt_boxes_lidar)

                    gt_boxes_lidar[:, 2] -= gt_boxes_lidar[:, 5] / 2
                    anno['location'] = np.zeros((gt_boxes_lidar.shape[0], 3))
                    anno['location'][:, 0] = -gt_boxes_lidar[:, 1]  # x = -y_lidar
                    anno['location'][:, 1] = -gt_boxes_lidar[:, 2]  # y = -z_lidar
                    anno['location'][:, 2] = gt_boxes_lidar[:, 0]  # z = x_lidar
                    dxdydz = gt_boxes_lidar[:, 3:6]
                    anno['dimensions'] = dxdydz[:, [0, 2, 1]]  # lwh ==> lhw
                    anno['rotation_y'] = -gt_boxes_lidar[:, 6] - np.pi / 2.0
                    anno['alpha'] = -np.arctan2(-gt_boxes_lidar[:, 1], gt_boxes_lidar[:, 0]) + anno['rotation_y']
                else:
                    anno['location'] = anno['dimensions'] = np.zeros((0, 3))
                    anno['rotation_y'] = anno['alpha'] = np.zeros(0)

        transform_to_kitti_format(eval_det_annos)
        transform_to_kitti_format(eval_gt_annos, info_with_fakelidar=False, is_gt=True)

        kitti_class_names = []
        for x in class_names:
            if x in map_name_to_kitti:
                kitti_class_names.append(map_name_to_kitti[x])
            else:
                kitti_class_names.append('Person_sitting')
        ap_result_str, ap_dict = kitti_eval.get_official_eval_result(
            gt_annos=eval_gt_annos, dt_annos=eval_det_annos, current_classes=kitti_class_names
        )
        return ap_result_str, ap_dict

    def nuscene_eval(self, det_annos, class_names, **kwargs):
        import json

        from nuscenes.nuscenes import NuScenes

        from . import nuscenes_utils
        nusc = NuScenes(version=self.dataset_cfg.VERSION, dataroot=str(self.root_path), verbose=True)
        nusc_annos = nuscenes_utils.transform_det_annos_to_nusc_annos(det_annos, nusc)
        nusc_annos['meta'] = {
            'use_camera': False,
            'use_lidar': True,
            'use_radar': False,
            'use_map': False,
            'use_external': False,
        }

        output_path = Path(kwargs['output_path'])
        output_path.mkdir(exist_ok=True, parents=True)
        res_path = str(output_path / 'results_nusc.json')
        with open(res_path, 'w') as f:
            json.dump(nusc_annos, f)

        self.logger.info(f'The predictions of NuScenes have been saved to {res_path}')

        if self.dataset_cfg.VERSION == 'v1.0-test':
            return 'No ground-truth annotations for evaluation', {}

        from nuscenes.eval.detection.config import config_factory
        from nuscenes.eval.detection.evaluate import NuScenesEval

        eval_set_map = {
            'v1.0-mini': 'mini_val',
            'v1.0-trainval': 'val',
            'v1.0-test': 'test'
        }
        try:
            eval_version = 'detection_cvpr_2019'
            eval_config = config_factory(eval_version)
        except:
            eval_version = 'cvpr_2019'
            eval_config = config_factory(eval_version)

        nusc_eval = NuScenesEval(
            nusc,
            config=eval_config,
            result_path=res_path,
            eval_set=eval_set_map[self.dataset_cfg.VERSION],
            output_dir=str(output_path),
            verbose=True,
        )
        metrics_summary = nusc_eval.main(plot_examples=0, render_curves=False)

        with open(output_path / 'metrics_summary.json', 'r') as f:
            metrics = json.load(f)

        result_str, result_dict = nuscenes_utils.format_nuscene_results(metrics, self.class_names, version=eval_version)
        return result_str, result_dict



if __name__ == '__main__':
    import argparse
    from pathlib import Path

    import yaml
    from easydict import EasyDict

    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--cfg_file', type=str, default=None, help='specify the config of dataset')
    args = parser.parse_args()

   
    dataset_cfg = EasyDict(yaml.safe_load(open(args.cfg_file)))
    ROOT_DIR = (Path(__file__).resolve().parent / '../../../').resolve()
       
    object_classes=['car','truck','bus','motorcycle','bicycle','pedestrian']
 

    stellantis_dataset = StellantisDataset(
        dataset_cfg=dataset_cfg, class_names=object_classes,
        root_path=ROOT_DIR /'..' /  'bevfusion' ,
        logger=common_utils.create_logger(), training=False
    )
    data=stellantis_dataset[20]
        # nuscenes_dataset.create_groundtruth_database(max_sweeps=dataset_cfg.MAX_SWEEPS)