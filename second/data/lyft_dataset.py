import pickle
import warnings
from pathlib import Path

import numpy as np
from pyquaternion import Quaternion

from second.core import box_np_ops
from second.data import dataset_utils as ds_utils
from second.data.dataset import Dataset, register_dataset
from second.utils.progress_bar import progress_bar_iter as prog_bar

try:
    from lyft.nuscenes import NuScenes
    from lyft.utils import splits
    from lyft.utils.data_classes import LidarPointCloud
except ImportError:
    warnings.warn("if you use LyftDataset, do 'pip install git+https://github.com/lyft/nuscenes-devkit.git'")


@register_dataset
class LyftDataset(Dataset):
    NumPointFeatures = 4

    def __init__(self,
                 root_path,
                 info_path,
                 class_names=None,
                 prep_func=None,
                 num_point_features=None):
        self._root_path = Path(root_path)
        with open(info_path, 'rb') as f:
            data = pickle.load(f)
        self._infos = list(sorted(data['infos'], key=lambda e: e['timestamp']))
        self._metadata = data['metadata']
        self.version = self._metadata['version']
        self._class_names = class_names
        self._prep_func = prep_func

    def __len__(self):
        return len(self._infos)

    def __getitem__(self, idx):
        input_dict = self.get_sensor_data(idx)
        example = self._prep_func(input_dict=input_dict)
        example["metadata"] = input_dict["metadata"]
        if "anchors_mask" in example:
            example["anchors_mask"] = example["anchors_mask"].astype(np.uint8)
        return example

    @property
    def ground_truth_annotations(self):
        if 'gt_boxes' not in self._nusc_infos[0]:
            return None
        # from lyft.eval.detection.config import eval_detection_configs
        # cls_range_map = eval_detection_configs[
        #     self.eval_version]["class_range"]
        gt_annos = []
        for info in self._infos:
            gt_names = info['gt_names']
            gt_boxes = info['gt_boxes']
            num_lidar_pts = info['num_lidar_pts']

            mask_pts = num_lidar_pts > 0
            gt_names = gt_names[mask_pts]
            gt_boxes = gt_boxes[mask_pts]
            num_lidar_pts = num_lidar_pts[mask_pts]

            mask_cls = np.isin(gt_names, self._class_names)
            gt_names = gt_names[mask_cls]
            gt_boxes = gt_boxes[mask_cls]
            num_lidar_pts = num_lidar_pts[mask_cls]

            gt_names_mapped = [self._kitti_name_mapping[n] for n in gt_names]
            det_range = np.array([cls_range_map[n] for n in gt_names_mapped])
            det_range = det_range[..., np.newaxis] @ np.array([[-1, -1, 1, 1]])
            mask = (gt_boxes[:, :2] >= det_range[:, :2]).all(1)
            mask &= (gt_boxes[:, :2] <= det_range[:, 2:]).all(1)
            gt_names = gt_names[mask]
            gt_boxes = gt_boxes[mask]
            num_lidar_pts = num_lidar_pts[mask]

            # use occluded to control easy/moderate/hard in kitti
            easy_mask = num_lidar_pts > 15
            moderate_mask = num_lidar_pts > 7
            occluded = np.zeros([num_lidar_pts.shape[0]])
            occluded[:] = 2
            occluded[moderate_mask] = 1
            occluded[easy_mask] = 0
            N = len(gt_boxes)
            gt_annos.append({
                "bbox": np.tile(np.array([[0, 0, 50, 50]]), [N, 1]),
                "alpha": np.full(N, -10),
                "occluded": occluded,
                "truncated": np.zeros(N),
                "name": gt_names,
                "location": gt_boxes[:, :3],
                "dimensions": gt_boxes[:, 3:6],
                "rotation_y": gt_boxes[:, 6],
            })
        return gt_annos

    def get_sensor_data(self, query):
        idx = query
        read_test_image = False
        if isinstance(query, dict):
            assert 'lidar' in query
            idx = query['lidar']['idx']
            read_test_image = 'cam' in query

        info = self._infos[idx]
        res = {
            'lidar': {
                'type': 'lidar',
                'points': None,
            },
            'metadata': {
                'token': info['token']
            },
        }
        pointcloud = LidarPointCloud.from_file(info['lidar_path'])
        if self._metadata['flat_vehicle_coordinates']:
            # Move box to ego vehicle coord system
            pointcloud.rotate(Quaternion(info['lidar2ego_rotation']).rotation_matrix)
            pointcloud.translate(np.array(info['lidar2ego_translation']))

            # Move box to ego vehicle coord system parallel to world z plane
            # Note: If rotation is applied simultaneously, coordinate transformation is incorrect.
            pointcloud.rotate(Quaternion(info['ego2global_rotation']).rotation_matrix)
            yaw = Quaternion(info['ego2global_rotation']).yaw_pitch_roll[0]
            quat = Quaternion(scalar=np.cos(yaw / 2), vector=[0, 0, np.sin(yaw / 2)])
            pointcloud.rotate(quat.inverse.rotation_matrix)

        points = pointcloud.points.T
        points[:, 3] /= 255
        res['lidar']['points'] = points

        if read_test_image:
            if Path(info['cam_front_path']).exists():
                with open(str(info['cam_front_path']), 'rb') as f:
                    image_str = f.read()
            else:
                image_str = None
            res['cam'] = {
                'type': 'camera',
                'data': image_str,
                'datatype': Path(info['cam_front_path']).suffix[1:],
            }

        if 'gt_boxes' in info:
            mask = info['num_lidar_pts'] > 0
            res['lidar']['annotations'] = {
                'boxes': info['gt_boxes'][mask].copy(),
                'names': info['gt_names'][mask].copy(),
            }

        return res

    def evaluation_nusc(self, detections, output_dir):
        version = self.version
        # eval_set_map = {
        #     "v1.0-mini": "mini_train",
        #     "v1.0-trainval": "val",
        # }
        gt_annos = self.ground_truth_annotations
        if gt_annos is None:
            return None

        nusc_annos = {}
        mapped_class_names = self._class_names
        token2info = {info['token']: info for info in self._infos}

        for det in detections:
            annos = []
            boxes = _second_det_to_nusc_box(det)
            for i, box in enumerate(boxes):
                name = mapped_class_names[box.label]
                velocity = box.velocity[:2].tolist()
                if len(token2info[det["metadata"]["token"]]["sweeps"]) == 0:
                    velocity = (np.nan, np.nan)
                box.velocity = np.array([*velocity, 0.0])
            boxes = _lidar_nusc_box_to_global(
                token2info[det["metadata"]["token"]], boxes,
                mapped_class_names, "cvpr_2019")
            for i, box in enumerate(boxes):
                name = mapped_class_names[box.label]
                velocity = box.velocity[:2].tolist()
                nusc_anno = {
                    "sample_token": det["metadata"]["token"],
                    "translation": box.center.tolist(),
                    "size": box.wlh.tolist(),
                    "rotation": box.orientation.elements.tolist(),
                    "velocity": velocity,
                    "detection_name": name,
                    "detection_score": box.score,
                    "attribute_name": NuScenesDataset.DefaultAttribute[name],
                }
                annos.append(nusc_anno)
            nusc_annos[det["metadata"]["token"]] = annos
        nusc_submissions = {
            "meta": {
                "use_camera": False,
                "use_lidar": False,
                "use_radar": False,
                "use_map": False,
                "use_external": False,
            },
            "results": nusc_annos,
        }
        res_path = Path(output_dir) / "results_nusc.json"
        with open(res_path, "w") as f:
            json.dump(nusc_submissions, f)
        eval_main_file = Path(__file__).resolve().parent / "nusc_eval.py"
        # why add \"{}\"? to support path with spaces.
        cmd = f"python {str(eval_main_file)} --root_path=\"{str(self._root_path)}\""
        cmd += f" --version={self.version} --eval_version={self.eval_version}"
        cmd += f" --res_path=\"{str(res_path)}\" --eval_set={eval_set_map[self.version]}"
        cmd += f" --output_dir=\"{output_dir}\""
        # use subprocess can release all nusc memory after evaluation
        subprocess.check_output(cmd, shell=True)
        with open(Path(output_dir) / "metrics_summary.json", "r") as f:
            metrics = json.load(f)
        detail = {}
        res_path.unlink()  # delete results_nusc.json since it's very large
        result = f"Nusc {version} Evaluation\n"
        for name in mapped_class_names:
            detail[name] = {}
            for k, v in metrics["label_aps"][name].items():
                detail[name][f"dist@{k}"] = v
            tp_errs = []
            tp_names = []
            for k, v in metrics["label_tp_errors"][name].items():
                detail[name][k] = v
                tp_errs.append(f"{v:.4f}")
                tp_names.append(k)
            threshs = ', '.join(list(metrics["label_aps"][name].keys()))
            scores = list(metrics["label_aps"][name].values())
            scores = ', '.join([f"{s * 100:.2f}" for s in scores])
            result += f"{name} Nusc dist AP@{threshs} and TP errors\n"
            result += scores
            result += "\n"
            result += ', '.join(tp_names) + ": " + ', '.join(tp_errs)
            result += "\n"
        return {
            'results': {
                'lyft': result
            },
            'detail': {
                'lyft': detail
            },
        }

    def evaluation(self, detections, output_dir):
        # res_lyft = self.evaluation_lyft(detections, output_dir)
        # res = {
        #     'lyft': {
        #         'lyft': res_lyft['results']['lyft'],
        #     },
        #     'detail': {
        #         'eval.lyft': res_lyft['detail']['lyft'],
        #     },
        # }
        res = {
            'lyft': {
                'lyft': '',
            },
            'detail': {
                'eval.lyft': {},
            },
        }
        return res


def _fill_trainval_infos(lyftdata,
                         use_flat_vehicle_coordinates,
                         use_second_format_direction,
                         calc_num_points,
                         is_test=False):
    train_infos, val_infos = [], []
    train_scene_tokens = [
        scene['token'] for scene in lyftdata.scene
        if scene['name'] in splits.train
    ]
    for sample in prog_bar(sorted(lyftdata.sample, key=lambda s: s['timestamp'])):
        lidar_token = sample['data']['LIDAR_TOP']
        cam_front_token = sample['data']['CAM_FRONT']

        sd_record = lyftdata.get('sample_data', lidar_token)
        cs_record = lyftdata.get('calibrated_sensor', sd_record['calibrated_sensor_token'])
        pose_record = lyftdata.get('ego_pose', sd_record['ego_pose_token'])

        lidar_path, boxes_lidar, _ = lyftdata.get_sample_data(lidar_token)
        cam_path, _, cam_intrinsic = lyftdata.get_sample_data(cam_front_token)

        info = {
            'lidar_path': lidar_path,
            'cam_front_path': cam_path,
            'token': sample['token'],
            'lidar2ego_translation': cs_record['translation'],
            'lidar2ego_rotation': cs_record['rotation'],
            'ego2global_translation': pose_record['translation'],
            'ego2global_rotation': pose_record['rotation'],
            'timestamp': sample['timestamp'],
        }

        if not is_test:
            locs = np.array([b.center for b in boxes_lidar]).reshape(-1, 3)
            dims = np.array([b.wlh for b in boxes_lidar]).reshape(-1, 3)
            rots = np.array([b.orientation.yaw_pitch_roll[0] for b in boxes_lidar]).reshape(-1, 1)
            gt_boxes = np.concatenate([locs, dims, rots], axis=1)
            gt_names = np.array([b.name for b in boxes_lidar])

            if calc_num_points:
                try:
                    pointcloud = LidarPointCloud.from_file(lidar_path)
                except Exception as e:
                    print('ERROR:', e, lidar_path)
                    continue

                indices = box_np_ops.points_in_rbbox(pointcloud.points.T[:, :3], gt_boxes)
                num_points_in_gt = indices.sum(0)
                info['num_lidar_pts'] = num_points_in_gt.astype(np.int32)

            if use_flat_vehicle_coordinates:
                _, boxes_flat_vehicle, _ = lyftdata.get_sample_data(
                    lidar_token,
                    use_flat_vehicle_coordinates=True
                )
                locs = np.array([b.center for b in boxes_flat_vehicle]).reshape(-1, 3)
                dims = np.array([b.wlh for b in boxes_flat_vehicle]).reshape(-1, 3)
                rots = np.array([b.orientation.yaw_pitch_roll[0] for b in boxes_flat_vehicle]).reshape(-1, 1)
                gt_boxes = np.concatenate([locs, dims, rots], axis=1)
                gt_names = np.array([b.name for b in boxes_flat_vehicle])

            if use_second_format_direction:
                gt_boxes[:, 6] = np.apply_along_axis(
                    lambda r: -ds_utils.wrap_to_pi(r + np.pi / 2),
                    axis=1,
                    arr=gt_boxes[:, 6:])

            annotations = [
                lyftdata.get('sample_annotation', token)
                for token in sample['anns']
            ]
            assert len(gt_boxes) == len(annotations), f"{len(gt_boxes)} != {len(annotations)}"

            info['gt_boxes'] = gt_boxes
            info['gt_names'] = gt_names

        if sample['scene_token'] in train_scene_tokens:
            train_infos.append(info)
        else:
            val_infos.append(info)

    return train_infos, val_infos


def create_lyft_infos(root_path,
                      version,
                      use_flat_vehicle_coordinates,
                      use_second_format_direction,
                      calc_num_points=True):
    """

    Args:
        root_path (str):
        version (str):
        use_flat_vehicle_coordinates (bool): Move box to ego vehicle coord system parallel to world z plane
        use_second_format_direction (bool):
        calc_num_points (bool): The number of points in ground-truth is not calculated in annotations

    Returns:

    """

    available_vers = ['v1.01-train']
    assert version in available_vers, f'{version} not in {available_vers}'

    root_path = Path(root_path)
    lyftdata = NuScenes(version=version, dataroot=root_path, verbose=True)
    train_infos, val_infos = _fill_trainval_infos(
        lyftdata,
        use_flat_vehicle_coordinates=use_flat_vehicle_coordinates,
        use_second_format_direction=use_second_format_direction,
        calc_num_points=calc_num_points,
        is_test=False,
    )

    data = {
        'metadata': {
            'version': version,
            'flat_vehicle_coordinates': use_flat_vehicle_coordinates,
            'second_format_direction': use_second_format_direction
        },
        'infos': train_infos
    }

    with open(root_path / "infos_train.pkl", 'wb') as f:
        pickle.dump(data, f)

    data['infos'] = val_infos
    with open(root_path / "infos_val.pkl", 'wb') as f:
        pickle.dump(data, f)

    data['infos'] = train_infos + val_infos
    with open(root_path / "infos_trainval.pkl", 'wb') as f:
        pickle.dump(data, f)
