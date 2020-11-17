import torch
from typing import Any, Dict, List, Tuple
import pandas as pd
import numpy as np
import argparse
import math
from numpy import cos, sin


RAW_DATA_FORMAT = {
    "TIMESTAMP": 0,
    "TRACK_ID": 1,
    "OBJECT_TYPE": 2,
    "X": 3,
    "Y": 4,
    "CITY_NAME": 5,
}

VELOCITY_THRESHOLD = 1.0  # Velocity threshold for stationary
EXIST_THRESHOLD = (
    15
)  # Number of timesteps the track should exist to be considered in social context
STATIONARY_THRESHOLD = (
    13)  # index of the sorted velocity to look at, to call it as stationary

NEARBY_DISTANCE_THRESHOLD = 30

def transform(theta, tracks_obs):
    th = -theta + np.pi/2
    Rot = np.array([[cos(th),-sin(th)],[sin(th),cos(th)]])
    new_pos = (np.dot(Rot, tracks_obs.reshape(-1,2).T)).T
    return new_pos


def compute_velocity(track_df: pd.DataFrame) -> List[float]:
    """Compute velocities for the given track.
    Returns:
        vel (list of float): Velocity at each timestep
    """
    x_coord = track_df["X"].values
    y_coord = track_df["Y"].values
    timestamp = track_df["TIMESTAMP"].values
    vel_x, vel_y = zip(*[(
        x_coord[i] - x_coord[i - 1] /
        (float(timestamp[i]) - float(timestamp[i - 1])),
        y_coord[i] - y_coord[i - 1] /
        (float(timestamp[i]) - float(timestamp[i - 1])),
    ) for i in range(1, len(timestamp))])
    vel = [np.sqrt(x**2 + y**2) for x, y in zip(vel_x, vel_y)]

    return vel



def get_is_track_stationary(track_df: pd.DataFrame) -> bool:
    """Check if the track is stationary.
    Args:
        track_df (pandas Dataframe): Data for the track
    Return:
        _ (bool): True if track is stationary, else False 
    """
    vel = compute_velocity(track_df)
    sorted_vel = sorted(vel)
    threshold_vel = sorted_vel[STATIONARY_THRESHOLD]
    return True if threshold_vel < VELOCITY_THRESHOLD else False



def pad_track(
        track_df: pd.DataFrame,
        seq_timestamps: np.ndarray,
        obs_len: int,
        raw_data_format: Dict[str, int],
) -> np.ndarray:
    """Pad incomplete tracks.
    Args:
        track_df (Dataframe): Dataframe for the track
        seq_timestamps (numpy array): All timestamps in the sequence
        obs_len (int): Length of observed trajectory
        raw_data_format (Dict): Format of the sequence
    Returns:
            padded_track_array (numpy array): Track data padded in front and back
    """
    track_vals = track_df.values
    track_timestamps = track_df["TIMESTAMP"].values
    track_x = track_df["X"].values
    track_y = track_df["Y"].values

    # start and index of the track in the sequence
    start_idx = np.where(seq_timestamps == track_timestamps[0])[0][0]
    end_idx = np.where(seq_timestamps == track_timestamps[-1])[0][0]

    # Edge padding in front and rear, i.e., repeat the first and last coordinates
    padded_track_array = np.pad(track_vals,
                                ((start_idx, obs_len - end_idx - 1),
                                 (0, 0)), "edge")
    # padded_track_x = np.pad(track_x,
    #                             (start_idx, obs_len - end_idx - 1),
    #                              "constant")
    # padded_track_y = np.pad(track_y,
    #                             (start_idx, obs_len - end_idx - 1),
    #                              "constant")

    # Overwrite the x y in padded part
    # for i in range(padded_track_array.shape[0]):
    #     padded_track_array[i, 3] = padded_track_x[i]
    #     padded_track_array[i, 4] = padded_track_y[i]

    if padded_track_array.shape[0] < obs_len:
        padded_track_array = fill_track_lost_in_middle(
            padded_track_array, seq_timestamps, raw_data_format)

    # Overwrite the timestamps in padded part
    for i in range(padded_track_array.shape[0]):
        padded_track_array[i, 0] = seq_timestamps[i]
    return padded_track_array



def fill_track_lost_in_middle(
        track_array: np.ndarray,
        seq_timestamps: np.ndarray,
        raw_data_format: Dict[str, int],
) -> np.ndarray:
    """Handle the case where the object exited and then entered the frame but still retains the same track id. It'll be a rare case.
    Args:
        track_array (numpy array): Padded data for the track
        seq_timestamps (numpy array): All timestamps in the sequence
        raw_data_format (Dict): Format of the sequence
    Returns:
        filled_track (numpy array): Track data filled with missing timestamps
    """
    curr_idx = 0
    filled_track = np.empty((0, track_array.shape[1]))
    for timestamp in seq_timestamps:
        filled_track = np.vstack((filled_track, track_array[curr_idx]))
        if timestamp in track_array[:, raw_data_format["TIMESTAMP"]]:
            curr_idx += 1
    return filled_track



def filter_tracks(seq_df: pd.DataFrame, obs_len: int, agent_track_obs: np.ndarray,
                  raw_data_format: Dict[str, int] = RAW_DATA_FORMAT) -> np.ndarray :
    """Pad tracks which don't last throughout the sequence. Also, filter out non-relevant tracks.
    Args:
        seq_df (pandas Dataframe): Dataframe containing all the tracks in the sequence
        obs_len (int): Length of observed trajectory
        raw_data_format (Dict): Format of the sequence
    Returns:
        social_tracks (numpy array): Array of relevant tracks
    """
    social_tracks = np.empty((0, obs_len, len(raw_data_format)))

    # Timestamps in the sequence
    seq_timestamps = np.unique(seq_df["TIMESTAMP"].values)

    # Track groups
    df_groups = seq_df.groupby("TRACK_ID")

    for group_name, group_data in df_groups:

        # Check if the track is long enough
        if len(group_data) < EXIST_THRESHOLD:
            continue

        # Skip if agent track
        if group_data["OBJECT_TYPE"].iloc[0] == "AGENT":
            continue

        if group_data["OBJECT_TYPE"].iloc[0] == "AV":
            padded_track_array = pad_track(group_data, seq_timestamps,
                                    obs_len, raw_data_format).reshape(
                                    (1, obs_len, -1))
            social_tracks = np.vstack((social_tracks, padded_track_array))
            continue


        # Check if the track is stationary
        if get_is_track_stationary(group_data):
            continue

        if (group_data["X"].iloc[-1] - agent_track_obs[-1,raw_data_format["X"]])**2 + \
            (group_data["Y"].iloc[-1] - agent_track_obs[-1,raw_data_format["Y"]])**2 > NEARBY_DISTANCE_THRESHOLD**2:
            continue

        padded_track_array = pad_track(group_data, seq_timestamps,
                                            obs_len,
                                            raw_data_format).reshape(
                                                (1, obs_len, -1))

        social_tracks = np.vstack((social_tracks, padded_track_array))

    return social_tracks



def compute(seq_path: str, obs_len) -> np.ndarray:

    """
    social_tracks_obs: NxT(20)x6
    agent_track_obs: T(20)x6
    """

    df = pd.read_csv(seq_path, dtype={"TIMESTAMP": str})

    agent_track = df[df["OBJECT_TYPE"] == "AGENT"].values

    agent_ts = np.sort(np.unique(df["TIMESTAMP"].values))


    if agent_ts.shape[0] == obs_len:
        df_obs = df
        agent_track_obs = agent_track
        agent_track_pred = None
    else:
        # Get obs dataframe and agent track
        df_obs = df[df["TIMESTAMP"] < agent_ts[obs_len]]
        assert (np.unique(df_obs["TIMESTAMP"].values).shape[0] == obs_len
                ), "Obs len mismatch"
        agent_track_obs = agent_track[:obs_len]
        agent_track_pred = agent_track[obs_len:]



    social_tracks_obs = filter_tracks(df_obs, obs_len, agent_track_obs)



    tracks_obs = np.concatenate((np.expand_dims(agent_track_obs,axis=0), social_tracks_obs),axis = 0)
    return tracks_obs, agent_track_pred


def process_data(data_path, obs_len, pred_len):
    # argo_data = argo_create(data_path)

    # count = 0
    # for scene_name in argo_data.keys():
        
    # read data
    tracks_obs, agent_track_pred = compute(data_path, obs_len)

    # create edge_index
    # num_obj = tracks_obs.shape[0]
    # edge_index_row2 = torch.linspace(1, num_obj-1, steps=num_obj-1).view(1,-1)
    # edge_index_row1 = torch.zeros(num_obj - 1).view(1,-1)
    # edge_index = torch.cat((edge_index_row1, edge_index_row2),0)
    # edge_index = edge_index.long()

    # 
    tracks_obs = tracks_obs[0,:,3:5].astype('float64')
    tracks_obs = np.expand_dims(tracks_obs, axis=0) 
    agent_track_pred = agent_track_pred[:,3:5].astype('float64')


    # normalize            
    agent_track_pred = agent_track_pred - tracks_obs[0,-1,:]
    tracks_obs = tracks_obs - tracks_obs[0,-1,:]  

    # cal velocity
    v = np.zeros_like(tracks_obs)
    for i in range(tracks_obs.shape[0]):
        for j in range(tracks_obs.shape[1]-1):
            v[i,j,:] = tracks_obs[i,j+1,:] - tracks_obs[i,j,:]
        v[i,-1,:] = v[i,-2,:]
    
    # cal acceleration
    a = np.zeros_like(tracks_obs)
    for i in range(v.shape[0]):
        for j in range(a.shape[1]-1):
            a[i,j,:] = v[i,j+1,:] - v[i,j,:]
        a[i,-1,:] = a[i,-2,:]

    # cal theta
    theta = np.zeros(v.shape[1])
    for k in range(v.shape[1]):
        if v[0,k,1]>0 and v[0,k,0] >0:
            theta[k] = math.atan(v[0,k,1]/v[0,k,0])
        if v[0,k,1]>0 and v[0,k,0] <0:
            theta[k] = np.pi - math.atan(v[0,k,1]/(-v[0,k,0]))
        if v[0,k,1]<0 and v[0,k,0] <0:
            theta[k] = math.atan((-v[0,k,1])/(-v[0,k,0])) + np.pi
        if v[0,k,1]<0 and v[0,k,0] >0:
            theta[k] = -math.atan((-v[0,k,1])/v[0,k,0])

        if theta[k]>2*np.pi:
            theta[k,0] = theta[k] - 2*np.pi
        if theta[k]<0:
            theta[k] = theta[k] + 2*np.pi

        if k>0:
            if theta[k]-theta[k-1]>np.pi:
                theta[k] -= 2*np.pi
            elif theta[k]-theta[k-1]<-np.pi:
                theta[k] += 2*np.pi        
    

    # transform coodinate
    # obs
    th = np.mean(theta[-4:])
    track_obs = transform(th, tracks_obs).reshape(-1,obs_len,2)
    v = transform(th, v).reshape(-1,obs_len,2)
    a = transform(th, a).reshape(-1,obs_len,2)
    new_tracks_obs = np.concatenate((track_obs,v,a),2)
    # pred
    new_agent_pred = transform(th, agent_track_pred).reshape(pred_len,2)



    x = new_tracks_obs
    y = new_agent_pred

    return x, y 


def window_shift(inputs, label):
    # inputs: 20x6 [x, y, vx, vy, ax, ay]
    # label: 1x2 [x, y]
    vx, vy = label[0] - inputs[0,-1,0],  label[1] - inputs[0,-1,1]
    ax, ay = vx - inputs[0,-1,2], vy - inputs[0,-1,3]

    label = torch.tensor([[[label[0], label[1], vx, vy, ax, ay]]])
    inputs = torch.cat((inputs,label),1)

    #inputs = torch.from_numpy(inputs[1:,:]).float()
    return inputs[:,1:,:]



if __name__ == '__main__':
	#dataset = MyOwnDataset(root='/home/junanchen/anaconda3/envs/GNN/argo',coord = 'track_ego')
    df = compute('/home/junanchen/anaconda3/envs/GNN/argo/data/train/data/2.csv')
