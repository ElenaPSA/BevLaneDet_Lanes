import numpy as np
import TrackingUtilsPy

from ..data_protos import BBox


class IMMFilterMotionModel:
    def __init__(self, bbox: BBox, inst_type, time_stamp, covariance='default'):
        # the time stamp of last observation
        self.prev_time_stamp = time_stamp
        self.latest_time_stamp = time_stamp
        
        x0 = BBox.bbox2array(bbox)[:7]
        self.imm = TrackingUtilsPy.Filter(x0)

        self.history = [bbox]

    def update(self, det_bbox: BBox, aux_info=None): 
        """ 
        Updates the state vector with observed bbox.
        """

        z = BBox.bbox2array(det_bbox)[:7]

        if det_bbox.s is None:
            self.score = self.score * 0.01
        else:
            self.score = det_bbox.s

        self.imm.update(z)
        x = self.imm.getState()
        # x[3], x[5] = x[5], x[3]
      

        cur_bbox = x[:7].reshape(-1).tolist() 
        cur_bbox = BBox.array2bbox(cur_bbox + [self.score])
        self.history[-1] = cur_bbox
        return

    def get_prediction(self, time_stamp=None,delta_pose=[]):       
        """
        Advances the state vector and returns the predicted bounding box estimate.
        """
        time_lag = time_stamp - self.prev_time_stamp
       
        self.latest_time_stamp = time_stamp
        if len(delta_pose)==0:
             self.imm.predict(time_lag)
        else:
            self.imm.predictInEgoFrame(time_lag,delta_pose[0],delta_pose[1],delta_pose[2])
        pred_x = self.imm.getState()
        self.prev_time_stamp = self.latest_time_stamp
        pred_bbox = BBox.array2bbox(pred_x)
        self.history.append(pred_bbox)
        return pred_bbox

    def get_state(self):
        """
        Returns the current bounding box estimate.
        """
        return self.history[-1]
    
    def compute_innovation_matrix(self):
        """ compute the innovation matrix for association with mahalonobis distance
        """
        raise NotImplementedError('Who dis (compute_innovation_matrix)?')
        return np.matmul(np.matmul(self.kf.H, self.kf.P), self.kf.H.T) + self.kf.R
    
    def sync_time_stamp(self, time_stamp):
        self.time_stamp = time_stamp
        return