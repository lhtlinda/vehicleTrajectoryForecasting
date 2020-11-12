import sys
import os
import numpy as np
import pandas as pd
import argparse
import math


def calculate_heading(start, end):
    dx = end[0]-start[0]
    dy = end[1]-start[1]
    if dy>=0 and dx>0:
        theta = math.atan(dy/dx)
    if dy>=0 and dx <0:
        theta = np.pi - math.atan(dy/(-dx))
    if dy<=0 and dx <0:
        theta = math.atan((-dy)/(-dx)) + np.pi
    if dy<=0 and dx >0:
        theta = -math.atan((-dy)/dx)
    if dx == 0:
        if dy>0:
            theta = np.pi/2
        elif dy == 0:
            theta = 0
        else:
            theta = - np.pi/2
    if theta < -np.pi:
        theta += 2*np.pi
    if theta > np.pi:
        theta -= 2*np.pi
        
    return theta

def process_scene(argo_scene):
    # every scene is a batch
    data = pd.DataFrame(columns=['frame_id',
                                 'object_type',
                                 'track_id',
                                 'x', 'y'
                                 ])
    
    
    min_frame = argo_scene['TIMESTAMP'][0]
    for index, row in argo_scene.iterrows():
        if row['OBJECT_TYPE'] == 'AV' or row['OBJECT_TYPE'] == 'AGENT':
            data_point = pd.Series({'frame_id': int((argo_scene['TIMESTAMP'][index]-min_frame)*10),
                                'object_type': row['OBJECT_TYPE'] ,
                                'track_id': row['TRACK_ID'],
                                'x': row['X'],
                                'y': row['Y']
                                   })
            data = data.append(data_point, ignore_index=True)

    if len(data.index) == 0:
        return None
    objects = data.track_id.unique()
    
    data.sort_values('frame_id', inplace=True)
   
#    origin_idx = index[data['frame_id']==0 and data['object_type'] == 'AV'].tolist()[0]
    origin_idx = data.index[(data['frame_id'] == 0) & (data['object_type'] == 'AV')][0]
    #print(origin_idx)
    origin = np.array([data.iloc[origin_idx]['x'],data.iloc[origin_idx]['y']])
    next_idx = data.index[(data['frame_id'] == 1) & (data['object_type'] == 'AV')]
    if len(next_idx)> 0:
        next_idx = next_idx[0]
        next = [data.iloc[next_idx]['x'],data.iloc[next_idx]['y']]
        heading = calculate_heading(origin, next)
    else:
        heading = 0
    
    # 0-indexed frame ids
    max_timesteps = data['frame_id'].max()+1
    scene_array = np.zeros((len(objects),max_timesteps,6))
    headers = dict()
    scene_idx = dict()
    index = 0
    for obj in objects:
        headers[obj] = 0
        scene_idx[obj] = index
        index += 1
    for index, row in data.iterrows():
        track = row['track_id']
        frame = row['frame_id']
        if headers[track] != (frame-1):
            #occlusion
            #duplicate previous element
            scene_array[scene_idx[track], headers[track]+1:frame, :] = scene_array[scene_idx[track], headers[track], :]
        xy =  np.array([row['x'],row['y']])
        xy = xy - origin
        rot = np.array([[math.cos(heading),-math.sin(heading)],[math.sin(heading), math.cos(heading)]])
        xy = rot @ xy
        scene_array[scene_idx[track], frame, :2] = xy
    scene_array[:,:,2] = np.concatenate((scene_array[:,1:,0],np.expand_dims(scene_array[:,-1,0], axis = 1)), axis = 1)-scene_array[:,:,0]
    scene_array[:,:,3] = np.concatenate((scene_array[:,1:,1],np.expand_dims(scene_array[:,-1,1], axis = 1)), axis = 1)-scene_array[:,:,1]
    scene_array[:,:,4] = np.concatenate((scene_array[:,1:,2],np.expand_dims(scene_array[:,-1,2], axis = 1)), axis = 1)-scene_array[:,:,2]
    scene_array[:,:,5] = np.concatenate((scene_array[:,1:,3],np.expand_dims(scene_array[:,-1,3], axis = 1)), axis = 1)-scene_array[:,:,3]
    
    
    return scene_array

def argo_create(path):
    scenes = dict()

    for fname in os.listdir(path):
        if fname.endswith('csv'):
            scenes[fname[:len(fname)-4]] = pd.read_csv(path+"/"+fname)
    return scenes
            
def process_data(data_path):
    argo_data = argo_create(data_path)
    scenes = []
    for scene_name in argo_data.keys():
        scene = process_scene(argo_data[scene_name])
        scenes.append(scene)
    print(f'Processed {len(scenes):.2f} scenes')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, required=True)
    #parser.add_argument('--output_path', type=str, required=True)
    args = parser.parse_args()
    process_data(args.data)
