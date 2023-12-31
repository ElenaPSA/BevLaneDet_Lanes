from copy import deepcopy

import numpy as np

from .. import life as life_manager
from .. import motion_model
from ..data_protos import BBox
from ..frame_data import FrameData
from ..update_info_data import UpdateInfoData


class Tracklet:
    def __init__(self, configs, id, bbox: BBox, det_type,det_ori_type, frame_index, time_stamp=None, aux_info=None):
        self.id = id
        self.time_stamp = time_stamp
        self.asso = configs['running']['asso']
        
        self.configs = configs
        self.det_type = det_type
       
        self.det_ori_type = [det_ori_type]
        self.aux_info = aux_info
        
        # initialize different types of motion model
        self.motion_model_type = configs['running']['motion_model']
        self.motion_model_imm= configs['running']['motion_imm']
       
        # simple kalman filter
        if self.motion_model_type == 'kf':
            if self.det_type==2 or not self.motion_model_imm: #pedestrian
                self.motion_model = motion_model.KalmanFilterMotionModel(
                bbox=bbox, inst_type=self.det_type, time_stamp=time_stamp, covariance=configs['running']['covariance'])
              
            else:
                self.motion_model = motion_model.IMMFilterMotionModel(
                bbox=bbox, inst_type=self.det_type, time_stamp=time_stamp, covariance=configs['running']['covariance'])
               
        # life and death management
        self.life_manager = life_manager.HitManager(configs, frame_index)
        # store the score for the latest bbox
        self.latest_score = bbox.s
        self.det_boxes = [bbox]
        bbox_=deepcopy(bbox)
        bbox_.c=det_ori_type
        self.last_detbox=bbox_
      
        
    def predict(self, time_stamp=None, is_key_frame=True,delta_pose=[]):
        """ in the prediction step, the motion model predicts the state of bbox
            the other components have to be synced
            the result is a BBox

            the ussage of time_stamp is optional, only if you use velocities
        """
        result = self.motion_model.get_prediction(time_stamp=time_stamp,delta_pose=delta_pose)
        
        self.life_manager.predict(is_key_frame=is_key_frame)
        self.latest_score = self.latest_score * 0.01
        result.s = self.latest_score
        self.last_detbox=None
        
        return result

    def update(self, update_info: UpdateInfoData):
        """ update the state of the tracklet
        """
        self.latest_score = update_info.bbox.s
        is_key_frame = update_info.aux_info['is_key_frame']
        
        # only the direct association update the motion model
        # TODO: Try mode == 1 or mode == 3 vs just mode == 1
        # when mode == 1 or 3, the mode=3 is association with non-key frame so it might be less reliable to update motion model
        if update_info.mode == 1 or update_info.mode == 3:
            self.motion_model.update(update_info.bbox, update_info.aux_info)
        else:
            pass
        self.life_manager.update(update_info, is_key_frame)
        if update_info.mode==1:
            self.det_boxes.append(deepcopy(update_info.bbox))
        else:
            self.det_boxes.append(None)

        self.det_ori_type.append(update_info.aux_info['ori_class_id'])
        bbox_=deepcopy(update_info.bbox)
        bbox_.c=update_info.aux_info['ori_class_id']
        
        if update_info.mode == 1 or update_info.mode == 3:
            self.last_detbox=bbox_
            self.last_detbox.c=update_info.aux_info['ori_class_id']
            self.last_det_ori_type=update_info.aux_info['ori_class_id']
        
        return

    def get_state(self):
        """ current state of the tracklet
        """
        result = self.motion_model.get_state()
        result.s = self.latest_score
        return result
    
    def get_detection(self):
       
        result = self.last_detbox
        return result

    def valid_output(self, frame_index):
        return self.life_manager.valid_output(frame_index)
    
    def death(self, frame_index):
        return self.life_manager.death(frame_index)
    
    def state_string(self, frame_index):
        """ the string describes how we get the bbox (e.g. by detection or motion model prediction)
        """
        return self.life_manager.state_string(frame_index)
    
    def compute_innovation_matrix(self):
        """ compute the innovation matrix for association with mahalonobis distance
        """
        return self.motion_model.compute_innovation_matrix()
    
    def sync_time_stamp(self, time_stamp):
        """ sync the time stamp for motion model
        """
        self.motion_model.sync_time_stamp(time_stamp)
        return
    
    def get_main_class(self):
        """Compute the main class (original)
        """ 
        unique, counts = np.unique(self.det_ori_type, return_counts=True)
        index = np.argmax(counts)
        return unique[index]
