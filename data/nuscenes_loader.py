import sys
import os
import numpy as np
import pandas as pd
import argparse
import math
from nuscenes.nuscenes import NuScenes
from nuscenes.map_expansion.map_api import NuScenesMap
from nuscenes.utils.splits import create_splits_scenes
from pyquaternion import Quaternion
from sklearn.model_selection import train_test_split
from tqdm import tqdm
scene_blacklist = [499, 515, 517]

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

def process_scene(ns_scene, nusc):
    data = pd.DataFrame(columns=['frame_id',
                                 'type',
                                 'node_id',
                                 'x', 'y',
                                 'heading'])
    sample_token = ns_scene['first_sample_token']
    sample = nusc.get('sample', sample_token)
    frame_id = 0
    while sample['next']:
        annotation_tokens = sample['anns']
        for annotation_token in annotation_tokens:
            annotation = nusc.get('sample_annotation', annotation_token)
            category = annotation['category_name']
            if len(annotation['attribute_tokens']):
                attribute = nusc.get('attribute', annotation['attribute_tokens'][0])['name']
            else:
                continue

            if 'vehicle' in category and 'bicycle' not in category and 'motorcycle' not in category and 'parked' not in attribute:
                our_category = 'VEHICLE'
                data_point = pd.Series({'frame_id': frame_id,
                                    'type': our_category,
                                    'node_id': annotation['instance_token'],
                                    'x': annotation['translation'][0],
                                    'y': annotation['translation'][1],
                                    'heading': Quaternion(annotation['rotation']).yaw_pitch_roll[0]})
                data = data.append(data_point, ignore_index=True)

        # Ego Vehicle
        our_category = 'VEHICLE'
        sample_data = nusc.get('sample_data', sample['data']['CAM_FRONT'])
        annotation = nusc.get('ego_pose', sample_data['ego_pose_token'])
        data_point = pd.Series({'frame_id': frame_id,
                                'type': our_category,
                                'node_id': 'ego',
                                'robot': True,
                                'x': annotation['translation'][0],
                                'y': annotation['translation'][1],
                                'z': annotation['translation'][2],
                                'length': 4,
                                'width': 1.7,
                                'height': 1.5,
                                'heading': Quaternion(annotation['rotation']).yaw_pitch_roll[0],
                                'orientation': None})
        data = data.append(data_point, ignore_index=True)

        sample = nusc.get('sample', sample['next'])
        frame_id += 1
    
    
    if len(data.index) == 0:
        return None
    objects = data.node_id.unique()
    
    data.sort_values('frame_id', inplace=True)
   
#    origin_idx = index[data['frame_id']==0 and data['object_type'] == 'AV'].tolist()[0]
    origin_idx = data.index[(data['frame_id'] == 0) & (data['node_id'] == 'ego')][0]
    #print(origin_idx)
    origin = np.array([data.iloc[origin_idx]['x'],data.iloc[origin_idx]['y']])
    heading = data.iloc[origin_idx]['heading']
    #print('heading',heading)
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
        track = row['node_id']
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

def process_data(data_path, version, val_split):
    nusc = NuScenes(version=version, dataroot=data_path, verbose=True)
    splits = create_splits_scenes()
    train_scenes, val_scenes = train_test_split(splits['train' if 'mini' not in version else 'mini_train'], test_size=val_split)
    train_scene_names = splits['train' if 'mini' not in version else 'mini_train']
    val_scene_names = splits['val' if 'mini' not in version else 'mini_val']

    ns_scene_names = dict()
    ns_scene_names['train'] = train_scene_names
    ns_scene_names['val'] = val_scene_names
    scenes = []
    for data_class in ['train', 'val']:
        for ns_scene_name in tqdm(ns_scene_names[data_class]):
            ns_scene = nusc.get('scene', nusc.field2token('scene', 'name', ns_scene_name)[0])
            scene_id = int(ns_scene['name'].replace('scene-', ''))
            if scene_id in scene_blacklist:  # Some scenes have bad localization
                continue

            scene = process_scene(ns_scene, nusc)
            if scene is not None:
                scenes.append(scene)
    
    print(f'Processed {len(scenes):.2f} scenes')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    #parser.add_argument('--output_path', type=str, required=True)
    parser.add_argument('--data', type=str, required=True)
    parser.add_argument('--version', type=str, required=True)
    parser.add_argument('--val_split', type=int, default=0.15)
    args = parser.parse_args()
    process_data(args.data, args.version, args.val_split)
