"""
MS3D Step 2

DESCRIPTION:
    Generate tracks for the fused detection set. Saves a pkl file containing a dictionary where
    each key is an object's track_id

EXAMPLES:
    python generate_tracks.py --ps_cfg /MS3D/tools/cfgs/target_nuscenes/ms3d_ps_config_rnd1.yaml --cls_id 1
    python generate_tracks.py --ps_cfg /MS3D/tools/cfgs/target_nuscenes/ms3d_ps_config_rnd1.yaml --cls_id 1 --static_veh
    python generate_tracks.py --ps_cfg /MS3D/tools/cfgs/target_nuscenes/ms3d_ps_config_rnd1.yaml --cls_id 2
"""
import sys

sys.path.append("../")
import argparse
import math
import pickle
from pathlib import Path

import numpy as np
import TrackingUtilsPy
import yaml
from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.utils import ms3d_utils, tracker_utils
from pcdet.utils.transform_utils import ego_to_world
from tqdm import tqdm
from tracker.mot_3d.data_protos import BBox

VMIN = 15.0
vmax_static=0.1
vmin_running=2.0

def gaussianpdf(x, mu, sigma):
    """
    Compute the probability density function of a Gaussian distribution.

    Args:
        x (numpy.ndarray): The input vector.
        mu (numpy.ndarray): The mean vector.
        sigma (numpy.ndarray): The covariance matrix.

    Returns:
        float: The probability density function of the Gaussian distribution.
    """
   
   
    n = x.shape[0]
    x=x.reshape(-1,1)
    mu=mu.reshape(-1,1)
    det = np.linalg.det(sigma)
    inv = np.linalg.inv(sigma)
   
    norm = 1 / ((2 * np.pi) ** (n / 2) * np.sqrt(det))
    exp = np.exp(-0.5 * (x - mu).T @ inv @ (x - mu))

    return max(0.00000000000000001,norm * exp)

def smooth_probabilities(Xsub_t1_k, Xsub_t1_t, Psub_t1_t, u_t, p):
    """
    Smooths the probabilities based on the given inputs.

    Args:
        Xsub_t1_k (ndarray): The input array Xsub_t1_k.
        Xsub_t1_t (ndarray): The input array Xsub_t1_t.
        Psub_t1_t (ndarray): The input array Psub_t1_t.
        u_t (ndarray): The input array u_t (probabilities).
        p (ndarray): The input array (transition matrix).

    Returns:
        ndarray: The smoothed probabilities.

    """
    n = p.shape[0]
    delta = np.zeros((n,), dtype=float)
    u = np.zeros((n), dtype=float)
    Xsub_t1_k = Xsub_t1_k.reshape(n, -1, 1)
    Xsub_t1_t = Xsub_t1_t.reshape(n, -1, 1)

    for j in range(n):
        for i in range(n):
            delta[j] += p[j, i] * gaussianpdf(Xsub_t1_k[i], Xsub_t1_t[j], Psub_t1_t[j])

    for j in range(n):
        u[j] = delta[j] * u_t[j]

    
    u /= np.sum(u)

    return u


def backward_interaction(Xsub_t1_k, Psub_t1_k, u_t1_k, u_t_t, p):
    """
    Perform backward interaction for state smoothing.

    Args:
        Xsub_t1_k (numpy.ndarray): The  states vector at time t+1 for each sub filter.
        Psub_t1_k (numpy.ndarray): The covariance matrix at time t+1 for each sub filter.
        u_t1_k (numpy.ndarray): The probability at time t+1 for each sub filter.
        u_t_t (numpy.ndarray): The forward interaction probability at time t for each sub filter..
        p (numpy.ndarray): The transition probability matrix.

    Returns:
        tuple: A tuple containing the smoothed state vector and covariance matrix at time t+1 for each track.
    """

    n = p.shape[0]
    Xsub_t1_k=Xsub_t1_k.reshape((n, -1))
    d= Xsub_t1_k.shape[1]

    bij = np.zeros((n, n), dtype=float)
    uij_t_1_k = np.zeros((n, n), dtype=float)

 
   
    for i in range(n):
        for j in range(n):
            bij[i, j] = p[j, i] * u_t_t[j]
        bij[i, :] /= np.sum(bij[i, :])
   
    for j in range(n):
        for i in range(n):
            uij_t_1_k[i, j] = bij[i, j] * u_t1_k[i]
        uij_t_1_k[:, j] /= np.sum(uij_t_1_k[:, j])

   
    Xo_sub_t1_k = np.zeros((n,d),dtype=float)
    Po_sub_t1_k = np.zeros((n,d,d),dtype=float)

   
    for j in range(n):
     
        for i in range(n):
            Xo_sub_t1_k[j] = Xo_sub_t1_k[j] + uij_t_1_k[i,j]*Xsub_t1_k[i]
          
    for j in range(n):
        for i in range(n):
            Po_sub_t1_k[j] = Po_sub_t1_k[j]+uij_t_1_k[i,j]*(Psub_t1_k[i]+(Xsub_t1_k[i]-Xo_sub_t1_k[j])@(Xsub_t1_k[i]-Xo_sub_t1_k[j]).T)
    
  

    return Xo_sub_t1_k, Po_sub_t1_k


def ukf_smoothing(imm_filter, index, X_t, P_t, Xo_t1_k, Po_t1_k, dt, dp):
    """
    Performs Unscented Kalman Filter (UKF) smoothing using RTS algorithm.

    Args:
        imm_filter (object): The IMM filter object.
        index (int): The index of the subfilter.
        X_t (numpy.ndarray): The state estimate at time t.
        P_t (numpy.ndarray): The state covariance at time t.
        Xo_t1_k (numpy.ndarray): The estimated and corrected state at time t+1.
        Po_t1_k (numpy.ndarray): The estimated and corrected state covariance at time t+1.
        dt (float): The time step.
        dp (list): delta pose at time t.

    Returns:
        tuple: A tuple containing the smoothed state and covariance at time t.
    """
    
    
    imm_filter.predictSubFilter(X_t, P_t, index, dt, dp[0])
    X_t1 = imm_filter.getSubFilterState(index)

    P_t1 = imm_filter.getSubFilterCovariance(index)
    C_t1 = imm_filter.getSubFilterCrossCovariance(index)
    
   
    Xo_t1_k = Xo_t1_k.reshape(-1, 1)
    X_t1 = X_t1.reshape(-1, 1)

    D = C_t1 @ np.linalg.inv(P_t1)
   
    X_t_ = X_t + D @ (Xo_t1_k- X_t1)
    P_t_ = P_t + D @ (Po_t1_k - P_t1) @ D.T
   
    return X_t_, P_t_, X_t1.reshape(-1), P_t1


    
def correct_direction(track):
    """
    Corrects the direction of the track based on the update boxes.

    Args:
        track (dict): The track containing boxes and update_boxes.

    Returns:
        dict: The track with corrected direction.
    """
    count = 0
    count_inv = 0

    for box, det in zip(track["boxes"], track["update_boxes"]):
        if np.size(det) != 0:
            count += 1
            theta1 = box[6]
            theta2 = det[6]
            scalar = math.cos(theta1 - theta2)
            if scalar < 0:
                count_inv += 1

    if count_inv > count / 2 + 1:
        for box in track["boxes"]:
            box[6] = box[6] + np.pi
            box[6] = box[6] % (2 * np.pi)
            if box[6] >= np.pi:
                box[6] -= np.pi * 2
            if box[6] < -np.pi:
                box[6] += np.pi * 2

    return track



def correct_pose(X, P, dp):
    """
    Corrects the state and covariance based on the pose change.

    Args:
        X (numpy.ndarray): Sate matrix
        P (numpy.ndarray): Covariance matrix.
        dp (list): List representing the pose correction [dx, dy, dtheta].

    Returns:
        tuple: Tuple containing the updated pose (X) and covariance matrix (P).
    """

    x = X[0]
    y = X[1]
   
    X[0] = dp[0] + x * math.cos(dp[2]) - y * math.sin(dp[2])
    X[1] = dp[1] + x * math.sin(dp[2]) + y * math.cos(dp[2])
    X[4] += dp[2]
    G=np.eye(6,dtype=float)
    G[0:2,0:2] = np.array([[math.cos(dp[2]), -math.sin(dp[2])], [math.sin(dp[2]), math.cos(dp[2])]])
    P = G @ P @ G.T

    return X, P


def smooth_track(track):
    """
    Smooths the track by applying a filtering and smoothing algorithm.

    Args:
        track (dict): The track data containing update boxes, timestamps, delta poses, and boxes.

    Returns:
        dict: The smoothed track data with updated box positions.
    """


    tprev = -1
    numState = 6  # imm.getNumState()

    states = []
    sub_states = []
    states_mix = []
    covars_mix = []
    covars = []
    sub_covars = []

    probs = []
    delta_times = []
    delta_poses = []

    final_state = []
    final_covar = []

    for box, timestamp, delta_pose, pred ,frame_id in zip(
        track["update_boxes_ego"],
        track["timestamp"],
        track["delta_pose"],
        track["boxes_ego"],
        track["frame_id"]
    ):
        
        if tprev == -1:
            bbox = BBox.array2bbox(track["update_boxes_ego"][0], input_opd_format=True)
            x0 = BBox.bbox2array(bbox)[:7]
            imm = TrackingUtilsPy.Filter(x0)
            numFilters = imm.getNumFilters()
            delta_times.append(0)
           
            states_mix.append([imm.getMixState(i) for i in range(numFilters)])
            covars_mix.append([imm.getMixCov(i) for i in range(numFilters)])
        else:
            time_lag = timestamp - tprev
            update = box[7] >= 0
            delta_times.append(time_lag)
            V=delta_pose[0]/time_lag
            
            imm.predictInEgoFrame(time_lag, delta_pose[0], delta_pose[1], delta_pose[2])
            states_mix.append([imm.getMixState(i) for i in range(numFilters)])
            covars_mix.append([imm.getMixCov(i) for i in range(numFilters)])
           
            if update:
                bbox = BBox.array2bbox(box, input_opd_format=True)
                z = BBox.bbox2array(bbox)[:7]
                imm.update(z)

        prob_current = imm.getModelProbabilities()
        state_current = imm.getFilterState()
        sub_states_current = [imm.getSubFilterState(i) for i in range(numFilters)]
        covar_current = imm.getFilterCovariance()
        sub_covars_current = [imm.getSubFilterCovariance(i) for i in range(numFilters)]

        states.append(state_current)
     #   final_state.append(state_current)
        sub_states.append(sub_states_current)

        covars.append(covar_current)
        sub_covars.append(sub_covars_current)

        delta_poses.append(delta_pose)
        probs.append(prob_current)
        
        tprev = timestamp

    numberofSteps = len(states)
    pij = imm.getTransitionMatrix()
   
    for i in range(numberofSteps):
        idx = numberofSteps - i - 1
      
        X_t = states[idx]
        P_t = covars[idx]
        X_mix = np.asarray(states_mix[idx])
        P_mix = np.asarray(covars_mix[idx])
        Xsub_t = np.asarray(sub_states[idx])
        Psub_t = np.asarray(sub_covars[idx])
        u_t = probs[idx]
        dp_t = delta_poses[idx]
        dt_t = delta_times[idx]

        if idx == numberofSteps - 1:
            
            X_mix_t1=X_mix
            P_mix_t1=P_mix
            X_t1_k = X_t
            P_t1_k = P_t
            Xsub_t1_k = Xsub_t
            Psub_t1_k = Psub_t
            dt_t1 = dt_t
            dp_t1 = dp_t
            u_t1_k = u_t

            final_state.append(X_t1_k.copy())
            final_covar.append(P_t1_k.copy())
          
            continue
        else:
           # computations are in ego frame
            X_t1_k, P_t1_k = correct_pose(X_t1_k, P_t1_k, dp_t1)
           
            for j in range(numFilters):
                Xsub_t1_k[j], Psub_t1_k[j] = correct_pose(Xsub_t1_k[j], Psub_t1_k[j], dp_t1)
             
            #  IMM backward_interaction
            Xo_sub_t1_k, Po_sub_t1_k = backward_interaction(Xsub_t1_k,Psub_t1_k,u_t1_k,u_t,pij)

           
            #  Sub filter UKF smoothing
            Xsub_t1_t=Xo_sub_t1_k.copy()
            Psub_t1_t=Po_sub_t1_k.copy()
            Xsub_t_k=Xsub_t.copy()
            Psub_t_k=Psub_t.copy()

            for j in range(numFilters):
                Xsub_t_k[j], Psub_t_k[j],Xsub_t1_t[j],Psub_t1_t[j] = ukf_smoothing(imm,j,X_mix_t1[j],P_mix_t1[j], Xo_sub_t1_k[j], Po_sub_t1_k[j],dt_t1,dp_t1)

            # update probabilities
            u_t_k=smooth_probabilities(Xsub_t1_k,Xsub_t1_t,Psub_t1_t,u_t,pij)

            # Final state and covariance estimation
            X_t_k=np.zeros((numState,1),dtype=float)
            P_t_k=np.zeros((numState,numState),dtype=float)

            for j in range(numFilters):
                X_t_k=X_t_k+u_t_k[j]*Xsub_t_k[j]
                
            for j in range(numFilters):
                P_t_k=P_t_k+u_t_k[j]*(Psub_t_k[j]+(Xsub_t_k[j]-X_t_k)@(Xsub_t_k[j]-X_t_k).T)            
           
            X_t1_k=X_t_k.copy()
            P_t1_k=P_t_k.copy()
            Xsub_t1_k=Xsub_t_k.copy()
            Psub_t1_k=Psub_t_k.copy()
            X_mix_t1=X_mix.copy()
            P_mix_t1=P_mix.copy()
            u_t1_k = u_t_k.copy()
            dt_t1 = dt_t
            dp_t1 = dp_t

            
            final_state.append(X_t_k.copy())
        
           
    final_state.reverse()
   
    vmean=0.0
    vmean_abs=0.0
    for i,state in enumerate(final_state):
        state=state.reshape(-1)
        vmean+=state[3]
        vmean_abs+=abs(state[3])
    vmean/=len(final_state)
    vmean_abs/=len(final_state)

    if vmean_abs<vmax_static:
        track['static']=True
    else:
        track['static']=False    

    for i,state in enumerate(final_state):
        state=state.reshape(-1)
        if vmean <0.0 and abs(vmean)>vmin_running:
            state[3]=-state[3]
            state[4]=state[4]+np.pi
            state[4]=state[4]%(2*np.pi)
            if state[4]>=np.pi:
                state[4]-=np.pi*2
            if state[4]<-np.pi:
                state[4]+=np.pi*2
            
        track['boxes'][i][:3]=state[:3]
        track['boxes'][i][6]=state[4]

        frame_id=track['frame_id'][i]
            
        _,boxes_world=ego_to_world(track['pose'][i],boxes=np.asarray([track['boxes'][i]]))
        track['boxes'][i]=boxes_world[0]

    
    return track

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="arg parser")
    parser.add_argument("--ps_cfg", type=str, help="cfg file with MS3D parameters")
    parser.add_argument('--cls_id', type=int, help='1: vehicle, 2: pedestrian, 3: cyclist')
    parser.add_argument('--static_veh', action='store_true', default=False)
    
    args = parser.parse_args()
    ms3d_configs = yaml.load(open(args.ps_cfg, "r"), Loader=yaml.Loader)
  #  tracks_dict_pth = Path(ms3d_configs["SAVE_DIR"]) / f"tracks_world_veh_all.pkl"
    
   

    if args.cls_id == 1:        
        if args.static_veh:
            tracks_dict_pth = Path(ms3d_configs["SAVE_DIR"]) /  f"tracks_world_veh_static.pkl" 
            save_fname = "tracks_world_veh_static_smoothed.pkl"                   
        else:
            tracks_dict_pth = Path(ms3d_configs["SAVE_DIR"]) /  f"tracks_world_veh_all.pkl"
            save_fname = "tracks_world_veh_all_smoothed.pkl"   
    elif args.cls_id == 2:
        print("Pedestrian smoothing not implemented")
        raise NotImplementedError  
        tracks_dict_pth = Path(ms3d_configs["SAVE_DIR"]) /  f"tracks_world_ped.pkl"
    elif args.cls_id == 3:        
        if args.static_veh:
            tracks_dict_pth = Path(ms3d_configs["SAVE_DIR"]) /  f"tracks_world_bic_static.pkl"  
            save_fname = "tracks_world_bic_static_smoothed.pkl"                     
        else:
            tracks_dict_pth = Path(ms3d_configs["SAVE_DIR"]) /  f"tracks_world_bic_all.pkl"
            save_fname = "tracks_world_bic_all_smoothed.pkl"   
    else:
        print('Only support 3 classes at the moment (1: vehicle, 2: pedestrian,3: bicycle)')
        raise NotImplementedError    
    
    with open(tracks_dict_pth, "rb") as f:
        tracks_dict = pickle.load(f)

    for key in tqdm(tracks_dict.keys()):
      
        track = tracks_dict[key]
        track = smooth_track(track)
    
    # track_id=89
    # track = tracks_dict[track_id]
    # track = smooth_track(track,pose_dict)

    for key in tracks_dict.keys():
         track = tracks_dict[key]
         if track['static']:
             track = correct_direction(track)
             print('correct track ',key)
             tracks_dict[key] = track
   
    ms3d_utils.save_data(tracks_dict, ms3d_configs["SAVE_DIR"], name=save_fname)
