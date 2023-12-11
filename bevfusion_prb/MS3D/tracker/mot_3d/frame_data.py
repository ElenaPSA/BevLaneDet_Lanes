""" input form of the data in each frame
"""
import mot_3d.utils as utils
import numpy as np

from .data_protos import BBox


class FrameData:
    def __init__(self, dets, ego,delta_ego, time_stamp=None, pc=None, det_types=None, det_ori_types=None,aux_info=None, input_opd_format=False):
        self.dets = dets         # detections for each frame
        self.ego = ego           # ego matrix information
        self.delta_ego = delta_ego      #trans, rot of ego vehicule
        self.pc = pc
        self.det_types = det_types
        self.det_ori_types = det_ori_types
        self.time_stamp = time_stamp
        self.aux_info = aux_info
   
        for i, det in enumerate(self.dets):
            self.dets[i] = BBox.array2bbox(det, input_opd_format=input_opd_format)
        
        # if not aux_info['is_key_frame']:
        #     self.dets = [d for d in self.dets if d.s >= 0.5]