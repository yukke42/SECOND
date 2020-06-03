import base64
import os
import pickle
from pathlib import Path

import numpy as np
from flask import Flask, render_template, jsonify, request
from flask_cors import CORS

from second.data.dataset import get_dataset_class

app = Flask(__name__)
CORS(app)


class SecondBackend:
    def __init__(self):
        self.root_path = None
        self.image_idxes = None
        self.dt_annos = None
        self.dataset = None
        self.net = None
        self.device = None


BACKEND = SecondBackend()


def error_response(msg):
    response = {"status": "error", "message": "[ERROR]" + msg}
    print("[ERROR]" + msg)
    return response


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/loadtexture')
def load_texture():
    filepath = os.path.dirname(__file__)
    with open(f'{filepath}/static/images/disc.png', 'rb') as f:
        img_ = f.read()
    return img_


@app.route('/api/readinfo', methods=['POST'])
def readinfo():
    global BACKEND
    print(BACKEND)
    instance = request.json
    root_path = Path(instance["root_path"])
    response = {"status": "normal"}
    BACKEND.root_path = root_path
    info_path = Path(instance["info_path"])
    dataset_class_name = instance["dataset_class_name"]
    BACKEND.dataset = get_dataset_class(dataset_class_name)(
        root_path=root_path, info_path=info_path)
    BACKEND.image_idxes = list(range(len(BACKEND.dataset)))
    response["image_indexes"] = BACKEND.image_idxes
    app.logger.info(f'{len(BACKEND.image_idxes)} data')

    response = jsonify(results=[response])
    response.headers['Access-Control-Allow-Headers'] = '*'
    return response


@app.route('/api/read_detection', methods=['POST'])
def read_detection():
    global BACKEND
    instance = request.json
    det_path = Path(instance["det_path"])
    response = {"status": "normal"}
    if BACKEND.root_path is None:
        return error_response("root path is not set")

    with open(det_path, "rb") as f:
        dt_annos = pickle.load(f)

    BACKEND.dt_annos = dt_annos
    response = jsonify(results=[response])
    response.headers['Access-Control-Allow-Headers'] = '*'
    return response


@app.route('/api/get_pointcloud', methods=['POST'])
def get_pointcloud():
    global BACKEND
    if BACKEND.root_path is None:
        return error_response("root path is not set")

    instance = request.json
    response = {"status": "normal"}
    image_idx = instance["image_idx"]
    enable_int16 = instance["enable_int16"]

    idx = BACKEND.image_idxes.index(image_idx)
    query = {'lidar': {'idx': idx, 'use_second_fmt_dir': True}}
    sensor_data = BACKEND.dataset.get_sensor_data(query)

    # img_shape = image_info["image_shape"] # hw
    if 'annotations' in sensor_data["lidar"]:
        annos = sensor_data["lidar"]['annotations']
        gt_boxes = annos["boxes"].copy()
        response["labels"] = annos["names"].tolist()
        response["locs"] = gt_boxes[:, :3].tolist()
        response["dims"] = gt_boxes[:, 3:6].tolist()
        rots = np.concatenate([
            np.zeros([gt_boxes.shape[0], 2], dtype=np.float32),
            -gt_boxes[:, 6:7]
        ],
            axis=1)
        response["rots"] = rots.tolist()

    response["num_features"] = 3
    points = sensor_data["lidar"]["points"][:, :3]
    if enable_int16:
        int16_factor = instance["int16_factor"]
        points *= int16_factor
        points = points.astype(np.int16)
    pc_str = base64.b64encode(points.tobytes())
    response["pointcloud"] = pc_str.decode("utf-8")
    app.logger.debug(f"load pointcloud! size={len(response['pointcloud'])}")

    if BACKEND.dt_annos is not None:
        dt_anno = BACKEND.dt_annos[idx]
        dt_boxes = dt_anno['box3d_lidar'].cpu().numpy().copy()
        dt_labels = dt_anno['label_preds'].cpu().numpy().copy()
        dt_scores = dt_anno['scores'].cpu().numpy().copy()

        mask = dt_scores >= 0.5
        dt_boxes = dt_boxes[mask]
        dt_labels = dt_labels[mask]
        dt_scores = dt_scores[mask]

        response['dt_boxes'] = dt_boxes.tolist()
        response['dt_locs'] = dt_boxes[:, :3].tolist()
        response['dt_dims'] = dt_boxes[:, 3:6].tolist()
        rots = np.concatenate([
            np.zeros([dt_boxes.shape[0], 2], dtype=np.float32),
            dt_boxes[:, 6:7] + np.pi / 2
        ],
            axis=1)
        response['dt_rots'] = rots.tolist()
        response['dt_scores'] = dt_scores.tolist()

    response = jsonify(results=[response])
    response.headers['Access-Control-Allow-Headers'] = '*'
    return response


@app.route('/api/get_image', methods=['POST'])
def get_image():
    global BACKEND
    instance = request.json
    response = {"status": "normal"}
    if BACKEND.root_path is None:
        return error_response("root path is not set")
    image_idx = instance["image_idx"]
    idx = BACKEND.image_idxes.index(image_idx)
    query = {"lidar": {"idx": idx}, "cam": {}}
    sensor_data = BACKEND.dataset.get_sensor_data(query)

    if "cam" in sensor_data \
            and "data" in sensor_data["cam"] \
            and sensor_data["cam"]["data"] is not None:
        image_str = sensor_data["cam"]["data"]
        image_b64 = f'data:image/{sensor_data["cam"]["datatype"]};base64,'
        image_b64 += base64.b64encode(image_str).decode("utf-8")
        response["image_b64"] = image_b64
        app.logger.debug(f"load image!: size={len(response['image_b64'])}")
    else:
        response["image_b64"] = ""

    response = jsonify(results=[response])
    response.headers['Access-Control-Allow-Headers'] = '*'
    return response


if __name__ == "__main__":
    app.run(host='0.0.0.0')
