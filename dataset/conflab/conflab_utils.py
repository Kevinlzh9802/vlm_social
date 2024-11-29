from collections import defaultdict
import os
import random
from typing import Dict, List, Mapping, Optional, Set, Tuple
import numpy as np
import json
import parse
import seaborn as sns
from sklearn.metrics.pairwise import euclidean_distances
from tqdm import tqdm
import logging

logger = logging.getLogger("detectron2")


def extract_file_info(filename: str) -> Mapping:
    # filename: cam2_vid2_seg7.json
    parsed_info = parse.parse("cam{cam}_vid{vid}_seg{seg}.json", str(filename))
    return parsed_info


def convert_kp_to_bb(keypoints: List) -> List:
    """ get bounding box keypoints """
    px = keypoints[0::3]
    py = keypoints[1::3]
    visibility = keypoints[2::3]

    n_kp = len(px)

    # remove invisible kp
    px = [px[i] for i in range(n_kp) if visibility[i] > 0]
    py = [py[i] for i in range(n_kp) if visibility[i] > 0]

    x1, y1, x2, y2 = [min(px), min(py), max(px), max(py)]
    bbox = [x1, y1, x2 - x1 + 1, y2 - y1 + 1]
    return bbox


def filter_kp_xy(kp: List[float],
                 w: int,
                 h: int,
                 valid_kv_ratio: Optional[float] = 0.5
                 ) -> Tuple[List[float], List[float]]:
    """
    convert floating point kp to w,h size
    and handle null values

    valid_kv_ratio: ratio to remove person from annotation
    """
    # v=0: not labeled (in which case x=y=0),
    # v=1: labeled but not visible,
    # v=2: labeled and visible
    N = len(kp) // 2
    new_kp = []
    for i in range(N):
        each_kp = kp[2 * i:(2 * i + 1 + 1)]
        if (each_kp[0] is not None
                and not (each_kp[0] == 0 and each_kp[1] == 0)):
            # some erronaeous annotation with (0,0)
            res = [each_kp[0], each_kp[1], 2]
        else:
            res = [0, 0, 0]
        new_kp.extend(res)

    new_kp = remove_farthest_kp(new_kp)

    #
    if valid_kv_ratio is not None:
        valid_kv = [x == 2 for x in new_kp[2::3]]
        if sum(valid_kv) <= int(np.ceil(N * valid_kv_ratio)):
            return None, None

    new_kp[0::3] = [int(x * w) for x in new_kp[0::3]]
    new_kp[1::3] = [int(x * h) for x in new_kp[1::3]]

    bbox = convert_kp_to_bb(new_kp)

    # sanity_check_kp_bb(new_kp, bbox)

    return new_kp, bbox


def remove_farthest_kp(kps, threshold: float = 4):
    """ remove kp that is very far away """
    kx, ky, kv = kps[0::3], kps[1::3], kps[2::3]
    N = len(kx)
    X = list(zip(kx, ky))  # (N, 2)
    distance = euclidean_distances(X, X)  # (N, N)
    distance.sort(axis=-1)
    distance = distance[:, 1:]

    topk = int(np.floor(N * 0.8))

    topk_distance = distance[:, :topk].mean(axis=-1)

    thresh_val = np.median(topk_distance) * threshold
    outlier_indices = np.where(topk_distance > thresh_val)[0]

    for ind in outlier_indices:
        if kv[ind] == 2:
            kps[3 * ind:3 * ind + 3] = [0, 0, 0]
    return kps


def sanity_check_kp_bb(kps, bb):
    # sanity check
    # all kp should be inside bb

    x1, y1, w, h = bb

    if x1 == 0 and y1 == 0:
        pass
        # import pdb
        # pdb.set_trace()

    x2, y2 = x1 + w, y1 + h
    for x, y, v in zip(kps[0::3], kps[1::3], kps[2::3]):
        if v == 0:
            continue
        if v == 2 and x1 <= x <= x2 and y1 <= y <= y2:
            pass
        else:
            pass
            # import pdb
            # pdb.set_trace()
            # raise ValueError("sanity check failed")


kpt_oks_sigmas = np.array([
    .26, .25, .25, .35, .35, .79, .79, .72, .72, .62, .62, 1.07, 1.07, .87,
    .87, .89, .89
]) / 10.0

ORIG_COCO_KEYPOINTS = [
    'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear', 'leftShoulder',
    'rightShoulder', 'leftElbow', 'rightElbow', 'leftWrist', 'rightWrist',
    'leftHip', 'rightHip', 'leftKnee', 'rightKnee', 'leftAnkle', 'rightAnkle'
]

KEYPOINTS = [
    "head", "nose", "neck", "rightShoulder", "rightElbow", "rightWrist",
    "leftShoulder", "leftElbow", "leftWrist", "rightHip", "rightKnee",
    "rightAnkle", "leftHip", "leftKnee", "leftAnkle", "rightFoot", "leftFoot"
]

OKS = np.array([
    0.026, 0.026, 0.026, 0.08, 0.072, 0.062, 0.079, 0.072, 0.062, 0.10, 0.087,
    0.089, 0.10, 0.087, .089, 0.09, 0.09
])

KP_TO_OKS = dict(zip(KEYPOINTS, OKS))

KEYPOINTS_1 = [
    "head", "nose", "neck", "rightShoulder", "leftShoulder", "rightElbow",
    "rightWrist", "leftElbow", "leftWrist", "rightHip", "leftHip"
]
KEYPOINTS_2 = [
    "head", "nose", "neck", "rightShoulder", "leftShoulder", "rightElbow",
    "rightWrist", "leftElbow", "leftWrist"
]
KEYPOINTS_3 = ["head", "nose", "neck", "rightShoulder", "leftShoulder"]


def get_keypoints(rank: int = 0):
    if not rank or rank == "0" or rank == 0:
        return get_kp_names()
    keypoints, keypoint_connection_rules, keypoint_flip_map, _ = get_kp_names()
    sm_keypoints = eval(f"KEYPOINTS_{rank}")
    original_indices = [keypoints.index(x) for x in sm_keypoints]
    # original_indices = list(idx_orig_to_now.keys())
    sm_keypoint_connection_rules = []
    for (a, b, c) in keypoint_connection_rules:
        if a in sm_keypoints and b in sm_keypoints:
            sm_keypoint_connection_rules.append((a, b, c))
    sm_keypoint_flip_map = []
    for a, b in keypoint_flip_map:
        if a in sm_keypoints and b in sm_keypoints:
            sm_keypoint_flip_map.append((a, b))

    return (sm_keypoints, sm_keypoint_connection_rules, sm_keypoint_flip_map,
            original_indices)


def get_kp_names():
    keypoints = KEYPOINTS
    connections = [[0, 1], [0, 2], [2, 3], [2, 6], [3, 4], [4, 5], [6, 7],
                   [7, 8], [2, 9], [9, 10], [10, 11], [11, 15], [2, 12],
                   [12, 13], [13, 14], [14, 16]]
    colors = [(int(r * 255), int(g * 255), int(b * 255))
              for r, g, b in sns.color_palette(n_colors=len(connections))]
    keypoint_connection_rules = []
    for i, (a, b) in enumerate(connections):
        keypoint_connection_rules.append(
            (keypoints[a], keypoints[b], colors[i]))
    keypoint_flip_map = (('leftFoot', 'rightFoot'),
                         ('leftShoulder', 'rightShoulder'), ('leftElbow',
                                                             'rightElbow'),
                         ('leftWrist', 'rightWrist'), ('leftHip', 'rightHip'),
                         ('leftKnee', 'rightKnee'), ('leftAnkle',
                                                     'rightAnkle'))

    return keypoints, keypoint_connection_rules, keypoint_flip_map, None


class AnnStat(object):
    def __init__(self) -> None:
        self.reset()

    def reset(self):
        self.info_im = defaultdict(lambda: defaultdict(int))

        self.total_image = 0
        self.total_person = 0
        self.nonnull_person = 0

        self.total_pt = 0
        self.null_pt = 0

        self.rejected_filename = []

    def update_ann(self, filename=""):
        # added a new file
        self.info_im[filename]["total_person"] += 1

    def update_pt(self, num: int = 1, null_num: int = 0, filename=""):
        # total keypoitn point, and null keypoint
        self.info_im[filename]["total_pt"] += num
        self.info_im[filename]["null_pt"] += null_num

    def update_nonnull_bb(self, filename=""):
        self.info_im[filename]["nonnull_bb"] += 1

    def update_nonnull_ann(self, filename=""):
        self.info_im[filename]["nonnull_ann"] += 1

    def update_nonnull_person(self, filename=""):
        self.info_im[filename]["nonnull_person"] += 1

    def stats(self):
        stats = defaultdict(int)
        eps = 1e-6

        for im, values in self.info_im.items():
            if im in self.rejected_filename:
                continue
            for k, v in values.items():
                stats[k] += v

        stat_str = []

        stat_str.append((
            f"images: valid {len(self.info_im)-len(self.rejected_filename)},\t"
            +
            f"rejected {len(self.rejected_filename)/(len(self.info_im)+eps) * 100:.1f}%"
        ))
        stat_str.append(
            f"Persons: valid {stats['total_person']},\t" +
            f"rejected {(1-stats['nonnull_person']/(stats['total_person']+eps)) * 100:.1f}%"
        )

        stat_str.append(
            f"\tNull points: {stats['null_pt']/(stats['total_pt']+eps) * 100 : .1f}%"
        )

        logger.info("\n" + f"Global info\n" + ("-" * 40) + "\n" +
                    "\n".join(stat_str))

    def stats_file(self, filename):
        eps = 1e-6
        stats = self.info_im[filename]
        stat_str = []
        stat_str.append(
            f"Persons: valid {stats['total_person']}\t" +
            f"rejected {(1-stats['nonnull_person']/(stats['total_person']+eps)) * 100:.1f}%"
        )

        stat_str.append(
            f"\t\tNull points: {stats['null_pt']/(stats['total_pt']+eps) * 100 : .1f}%"
        )
        logger.info("\n" + f"file: {filename}\n" + ("-" * 40) + "\n" +
                    "\n".join(stat_str))

    def remove_filename(self, filename):
        self.rejected_filename.append(filename)

    def null_kp(self, filename=""):
        if self.info_im[filename]["total_pt"] == 0:
            return 1.0
        return self.info_im[filename]["null_pt"] / self.info_im[filename][
            "total_pt"]


def save_coco(file, info, licenses, images, annotations, categories):
    with open(file, 'w', encoding='UTF-8') as coco:
        json.dump(
            {
                'info': info,
                'licenses': licenses,
                'images': images,
                'annotations': annotations,
                'categories': categories
            },
            coco,
            indent=2)


def filter_annotations(annotations,
                       images,
                       filter_kp_ids: Optional[Set] = None):
    image_ids = set(map(lambda i: int(i['id']), images))

    _list = []
    for ann in tqdm(annotations):
        if int(ann['image_id']) in image_ids:
            if filter_kp_ids:
                ann = filter_keypoints(ann, filter_kp_ids)
            _list.append(ann)
    return _list


def filter_keypoints(ann: dict, filter_ids: Set) -> dict:
    keypoints = ann["keypoints"]

    new_kp = []

    for i in filter_ids:
        new_kp.extend(keypoints[i * 3:i * 3 + 3])
    ann["num_keypoints"] = len(new_kp[2::3])
    ann["keypoints"] = new_kp
    return ann


def coco_split(annotation_file: str,
               train_file: str,
               test_file: str,
               test_cam: List[str] = ["cam6"],
               train_cam: List[str] = ["cam8"],
               filter_kp_ids: Optional[Set] = None,
               total_im_per_vid: Optional[int] = None,
               seed: int = 0):

    with open(annotation_file, 'r', encoding='UTF-8') as annotations:
        coco = json.load(annotations)

        logger.info("loaded coco json")

        info = coco['info']
        licenses = coco.get('licenses', [])
        images = coco['images']
        annotations = coco['annotations']
        categories = coco['categories']

        # TEST camera: cam6
        images_train, images_test = [], []
        if total_im_per_vid is None:
            for im in images:
                if any(im['file_name'].startswith(x) for x in test_cam):
                    images_test.append(im)
                elif any(im['file_name'].startswith(x) for x in train_cam):
                    images_train.append(im)
        else:
            rng = np.random.RandomState(seed)
            vid_info = defaultdict(list)
            for im in images:
                vidname = os.sep.join(im['file_name'].split('/')[:-1])
                vid_info[vidname].append(im)

            for vidname in vid_info:
                if len(vid_info[vidname]) > total_im_per_vid:
                    vid_info[vidname] = rng.choice(vid_info[vidname],
                                                   size=total_im_per_vid,
                                                   replace=False)
                if any(vidname.startswith(x) for x in test_cam):
                    images_test.extend(vid_info[vidname])
                elif any(vidname.startswith(x) for x in train_cam):
                    images_train.extend(vid_info[vidname])

        logger.info("splitting images...")

        save_coco(train_file, info, licenses, images_train,
                  filter_annotations(annotations, images_train, filter_kp_ids),
                  categories)
        logger.info(f"saved train to {train_file}")

        save_coco(test_file, info, licenses, images_test,
                  filter_annotations(annotations, images_test, filter_kp_ids),
                  categories)
        logger.info(f"saved test to {test_file}")

        print("Saved {} entries in {} and {} in {}".format(
            len(images_train), train_file, len(images_test), test_file))


def main():
    a = get_kp_names()
    c = 9

if __name__ == '__main__':
    main()
