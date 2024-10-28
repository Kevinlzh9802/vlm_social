import os
from pathlib import Path
import json
import pickle
from typing import List, Tuple

import torch
import numpy as np
from tqdm import tqdm


dataset_path = '/home/zonghuan/tudelft/projects/datasets/conflab/'
def vid_seg_to_segment(vid: int, seg: int) -> int:
    return {
        (2,8): 0,
        (2,9): 1,
        (3,1): 2,
        (3,2): 3,
        (3,3): 4,
        (3,4): 5,
        (3,5): 6,
        (3,6): 7
    }[(vid,seg)]


def read_skeletons(dataset_path):
    data_path = os.path.join(dataset_path, 'annotations/pose/coco/')
    # load file list
    all_tracks = [{} for _ in range(8)]
    # [seg:
    #    {[cam]:
    #          {tracks: {
    #              pid: [np.array shape [track_len, 17*2 + 17 + 1]]
    #          }
    #     }
    # ]

    for seg_file in tqdm(Path(data_path).glob('*.json')):
        parts = os.path.basename(seg_file).split('_')
        cam = int(parts[0][-1:])
        seg = vid_seg_to_segment(
            int(parts[1][-1:]),
            int(parts[2][-1:])
        )

        tracks = {}
        with open(seg_file) as f:
            coco_json = json.load(f)
            for frame_skeletons in coco_json['annotations']['skeletons']:
                for fs in frame_skeletons.values():
                    pid = fs['id']

                    if pid not in tracks:
                        tracks[pid] = []

                    tracks[pid].append({
                        'frame': fs['image_id'],
                        'kp': fs['keypoints'],
                        'occl': fs['occluded']
                    })
        # join list of tracks into a single track per pid
        for pid, track in tracks.items():
            new_track = [[e['frame'], *e['kp'], *e['occl']] for e in track]
            tracks[pid] = np.array(new_track, dtype=np.float64)

        all_tracks[seg][cam] = {
            'tracks': tracks
        }
    c = 0


def main():
    read_skeletons(dataset_path)


if __name__ == '__main__':
    main()
