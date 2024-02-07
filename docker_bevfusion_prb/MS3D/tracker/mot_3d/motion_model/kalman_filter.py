""" Many parts are borrowed from https://github.com/xinshuoweng/AB3DMOT
"""

import numpy as np
from filterpy.kalman import KalmanFilter

from ..data_protos import BBox


class KalmanFilterMotionModel:
    def __init__(self, bbox: BBox, inst_type, time_stamp, covariance='default'):
        # the time stamp of last observation
        self.prev_time_stamp = time_stamp
        self.latest_time_stamp = time_stamp
        # define constant velocity model
        self.score = bbox.s
        self.inst_type = inst_type

        self.kf = KalmanFilter(dim_x=10, dim_z=7)  # x: state vector, z: measurement vector (i.e. new bbox)
        self.kf.x[:7] = BBox.bbox2array(bbox)[:7].reshape((7, 1))
        self.kf.F = np.array([[1,0,0,0,0,0,0,1,0,0],      # state transition matrix
                              [0,1,0,0,0,0,0,0,1,0],
                              [0,0,1,0,0,0,0,0,0,1],
                              [0,0,0,1,0,0,0,0,0,0],  
                              [0,0,0,0,1,0,0,0,0,0],
                              [0,0,0,0,0,1,0,0,0,0],
                              [0,0,0,0,0,0,1,0,0,0],
                              [0,0,0,0,0,0,0,1,0,0],
                              [0,0,0,0,0,0,0,0,1,0],
                              [0,0,0,0,0,0,0,0,0,1]])     

        self.kf.H = np.array([[1,0,0,0,0,0,0,0,0,0],      # measurement function,
                              [0,1,0,0,0,0,0,0,0,0],
                              [0,0,1,0,0,0,0,0,0,0],
                              [0,0,0,1,0,0,0,0,0,0],
                              [0,0,0,0,1,0,0,0,0,0],
                              [0,0,0,0,0,1,0,0,0,0],
                              [0,0,0,0,0,0,1,0,0,0]])
        
        self.kf.B = np.zeros((10, 1))                     # dummy control transition matrix

        self.covariance_type = covariance
        self.kf.R[0:3,0:3] = 1.   # measurement uncertainty position
        self.kf.R[3,3] =(np.pi/4)*(np.pi/4)   # measurement uncertainty angle
        # kf.P initializes to an identity matrix with shape (dim_x, dim_x)
        # Here they set the first 7 diagonal elements = 10, and the last 3 = 1000
        
        self.kf.P[7:, 7:] *= 1000. 	# state uncertainty, give high uncertainty to the unobservable initial velocities, covariance matrix
        self.kf.P[0:3,0:3] = 1.0
        self.kf.P[3,3]=(np.pi/4)*(np.pi/4) 

        self.kf.Q[4:7,4:7] *= 0.0001    # process uncertainty w,l,h
      
        self.kf.Q[0:3,0:3] *= 0.25
        self.kf.Q[7:,7:] *= 0.25
        self.kf.Q[0,7] *= 0.5
        self.kf.Q[7,0] *= 0.5
        self.kf.Q[1,8] *= 0.5
        self.kf.Q[8,1] *= 0.5
        self.kf.Q[2,9] *= 0.5
        self.kf.Q[9,2] *= 0.5
        
        self.Q_init=self.kf.Q.copy()
        self.history = [bbox]
       
    
    def predict(self, time_stamp=None):
        """ For the motion prediction, use the get_prediction function.
        """
        self.kf.predict()
        if self.kf.x[3] >= np.pi: self.kf.x[3] -= np.pi * 2
        if self.kf.x[3] < -np.pi: self.kf.x[3] += np.pi * 2
        return

    def update(self, det_bbox: BBox, aux_info=None): 
        """ 
        Updates the state vector with observed bbox.
        """
        bbox = BBox.bbox2array(det_bbox)[:7]

        # full pipeline of kf, first predict, then update
        self.predict()

        ######################### orientation correction
        if self.kf.x[3] >= np.pi: self.kf.x[3] -= np.pi * 2    # make the theta still in the range
        if self.kf.x[3] < -np.pi: self.kf.x[3] += np.pi * 2

        new_theta = bbox[3]
        if new_theta >= np.pi: new_theta -= np.pi * 2    # make the theta still in the range
        if new_theta < -np.pi: new_theta += np.pi * 2
        bbox[3] = new_theta

        predicted_theta = self.kf.x[3]
        if np.abs(new_theta - predicted_theta) > np.pi / 2.0 and np.abs(new_theta - predicted_theta) < np.pi * 3 / 2.0:     # if the angle of two theta is not acute angle
            self.kf.x[3] += np.pi       
            if self.kf.x[3] > np.pi: self.kf.x[3] -= np.pi * 2    # make the theta still in the range
            if self.kf.x[3] < -np.pi: self.kf.x[3] += np.pi * 2

        # now the angle is acute: < 90 or > 270, convert the case of > 270 to < 90
        if np.abs(new_theta - self.kf.x[3]) >= np.pi * 3 / 2.0:
            if new_theta > 0: self.kf.x[3] += np.pi * 2
            else: self.kf.x[3] -= np.pi * 2

        #########################     # flip

        self.kf.update(bbox)
        self.prev_time_stamp = self.latest_time_stamp

        if self.kf.x[3] >= np.pi: self.kf.x[3] -= np.pi * 2    # make the theta still in the rage
        if self.kf.x[3] < -np.pi: self.kf.x[3] += np.pi * 2

        if det_bbox.s is None:
            self.score = self.score * 0.01
        else:
            self.score = det_bbox.s
        
        # Here is where the det box is updated to be consistent with previous tracks (using KF) in terms of dims/heading
        cur_bbox = self.kf.x[:7].reshape(-1).tolist() 

        cur_bbox = BBox.array2bbox(cur_bbox + [self.score])
        self.history[-1] = cur_bbox
        return

    def get_prediction(self, time_stamp=None,delta_pose=[]):       
        """
        Advances the state vector and returns the predicted bounding box estimate.
        """
        time_lag = time_stamp - self.prev_time_stamp
      
        self.latest_time_stamp = time_stamp
        self.kf.F = np.array([[1,0,0,0,0,0,0,time_lag,0,0],      # state transition matrix 
                              [0,1,0,0,0,0,0,0,time_lag,0],     # first 3 is x,y,z. Time lag is added to compute movement of centroid
                              [0,0,1,0,0,0,0,0,0,time_lag],
                              [0,0,0,1,0,0,0,0,0,0],  
                              [0,0,0,0,1,0,0,0,0,0],
                              [0,0,0,0,0,1,0,0,0,0],
                              [0,0,0,0,0,0,1,0,0,0],
                              [0,0,0,0,0,0,0,1,0,0],
                              [0,0,0,0,0,0,0,0,1,0],
                              [0,0,0,0,0,0,0,0,0,1]])
        
        self.kf.Q = self.Q_init.copy()
        self.kf.Q[0,0] *=(time_lag*time_lag*time_lag*time_lag)
        self.kf.Q[1,1] *=(time_lag*time_lag*time_lag*time_lag)
        self.kf.Q[2,2] *=(time_lag*time_lag*time_lag*time_lag)
        self.kf.Q[7,7]*=time_lag*time_lag
        self.kf.Q[8,8]*=time_lag*time_lag
        self.kf.Q[9,9]*=time_lag*time_lag

        self.kf.Q[0,7] *= time_lag*time_lag*time_lag
        self.kf.Q[7,0] *= time_lag*time_lag*time_lag
        self.kf.Q[1,8] *= time_lag*time_lag*time_lag
        self.kf.Q[8,1] *= time_lag*time_lag*time_lag
        self.kf.Q[2,9] *= time_lag*time_lag*time_lag
        self.kf.Q[9,2] *= time_lag*time_lag*time_lag

        
        pred_x = self.kf.get_prediction()[0]
        
        if len(delta_pose)!=0:
            self.correct_pose(pred_x,self.kf.P,delta_pose)
        
        if pred_x[3] >= np.pi: pred_x[3] -= np.pi * 2
        if pred_x[3] < -np.pi: pred_x[3] += np.pi * 2
        pred_bbox = BBox.array2bbox(pred_x[:7].reshape(-1))

        self.history.append(pred_bbox)
        return pred_bbox

    def correct_pose(self, pred_x, P, delta_pose):
        """
        Corrects the predicted pose using the given delta pose and updates the covariance matrix.

        Args:
            pred_x (numpy.ndarray): The predicted state vector.
            P (numpy.ndarray): The covariance matrix.
            delta_pose: The difference between the predicted and actual pose [dx,dy,dtheta].

        Returns:
            None
        """
        x = pred_x[0] - delta_pose[0]
        y = pred_x[1] - delta_pose[1]
        dtheta = delta_pose[2]

        c = np.cos(dtheta)
        s = np.sin(dtheta)

        R = np.array([[c, s], [-s, c]]).reshape(2,2)

        pred_x[0] = x * R[0, 0] + y * R[0, 1]
        pred_x[1] = x * R[1, 0] + y * R[1, 1]

        pred_x[3] = pred_x[3] - delta_pose[2]

        P[0:2, 0:2] = R @ P[0:2, 0:2] @ R.T


    def get_state(self):
        """
        Returns the current bounding box estimate.
        """
        return self.history[-1]
    
    def compute_innovation_matrix(self):
        """ compute the innovation matrix for association with mahalonobis distance
        """
        return np.matmul(np.matmul(self.kf.H, self.kf.P), self.kf.H.T) + self.kf.R
    
    def sync_time_stamp(self, time_stamp):
        self.time_stamp = time_stamp
        return

